"""Offline NumPy end-to-end optimization and dyadic quantization-aware training."""

import json
import math
from pathlib import Path

import numpy as np

from .scenes import BOX_SCALE, IMAGE_SCALE, SEED, training_data

FEATURE_SCALE = 1/16
WEIGHT_SCALE = 1/32
GAMMA_SCALE = 1/16
WEIGHT_SCALES = {"body": 1.0}
EPSILON = 1/16
LAYERS = {"stem": (3,4,4,2), "body": (1,4,4,1), "down": (3,4,4,2),
          "fine": (1,4,8,1), "coarse": (1,4,8,1)}


def _clip(x, step, lo=-128, hi=127):
    return (np.clip(np.rint(x/step),lo,hi)*step).astype(np.float32)


def _conv(x, w, b, stride):
    kernel, _, ci, co = w.shape
    pad = kernel//2
    area = np.pad(x, ((0,0),(pad,pad),(pad,pad),(0,0))) if pad else x
    z = np.lib.stride_tricks.sliding_window_view(area,(kernel,kernel),axis=(1,2))[:,::stride,::stride]
    z = z.transpose(0,1,2,4,5,3)
    rows = np.ascontiguousarray(z).reshape(-1,kernel*kernel*ci)
    out = (rows @ w.reshape(-1,co) + b).reshape(*z.shape[:3],co)
    return out, (rows,x.shape,area.shape,w,stride,pad)


def _conv_backward(dy, saved):
    rows, x_shape, area_shape,w,stride,pad = saved
    kernel, _, ci, co = w.shape
    d = dy.reshape(-1, co)
    dw = (rows.T @ d).reshape(w.shape)
    db = d.sum(axis=0)
    nh, nw = dy.shape[1:3]
    sections = (d @ w.reshape(-1,co).T).reshape(dy.shape[0],nh,nw,kernel,kernel,ci)
    dx = np.zeros(area_shape, np.float32)
    for y in range(kernel):
        for x in range(kernel):
            dx[:,y:y+nh*stride:stride,x:x+nw*stride:stride] += sections[:,:,:,y,x]
    return (dx[:,pad:pad+x_shape[1],pad:pad+x_shape[2]],dw,db)


def _fixed_rms(x,gamma):
    """Vectorized QAT version of the same documented int8 reciprocal-square-root format."""
    codes = np.rint(x/FEATURE_SCALE).astype(np.int32)
    gcodes = np.rint(gamma/GAMMA_SCALE).astype(np.int32)
    total = np.sum(codes.astype(np.int64)**2,axis=-1,keepdims=True)+round(EPSILON*4/FEATURE_SCALE**2)
    exponent = np.floor(np.log2(total)*.5).astype(np.int64)
    shift = 9-2*exponent
    mantissa = np.where(shift >= 0, total << np.maximum(shift,0), total >> np.maximum(-shift,0)) & 2047
    index, fraction = (mantissa>>4)-32,mantissa&15
    endpoints = np.rint(32768*np.sqrt(32/np.arange(32,129))).astype(np.int64)
    base = endpoints[index]
    interp = base-(((base-endpoints[index+1])*fraction+8)>>4)
    inv = (interp*64)>>(8+exponent)
    return (np.clip((codes*gcodes*inv)>>12,-128,127)*FEATURE_SCALE).astype(np.float32)


def forward(images, p, qat=False):
    local = {}

    def conv(name,x):
        step = BOX_SCALE if name in ("fine","coarse") else FEATURE_SCALE
        input_step = IMAGE_SCALE if name == "stem" else FEATURE_SCALE
        weight_step = WEIGHT_SCALES.get(name, WEIGHT_SCALE)
        w = _clip(p[name+"_w"],weight_step) if qat else p[name+"_w"]
        b = _clip(p[name+"_b"],weight_step*input_step,-(1<<30),1<<30) if qat else p[name+"_b"]
        z, local[name] = _conv(x,w,b,LAYERS[name][3])
        return _clip(z,step) if qat else z

    x = _clip(images,IMAGE_SCALE) if qat else images
    stem = conv("stem",x)
    positive = np.maximum(stem,0)
    residual = conv("body",positive)
    joined = positive+np.maximum(residual,0)
    if qat:
        joined = _clip(joined,FEATURE_SCALE)
    gamma = _clip(p["gamma"],GAMMA_SCALE) if qat else p["gamma"]
    inv = 1/np.sqrt(np.mean(joined*joined,axis=-1,keepdims=True)+EPSILON)
    norm = joined*inv*gamma
    hw_norm = _fixed_rms(joined,gamma) if qat else norm
    down = conv("down",hw_norm)
    fine = conv("fine",hw_norm).reshape(-1,16,8)
    coarse = conv("coarse",np.maximum(down,0)).reshape(-1,4,8)
    return np.concatenate((fine,coarse),axis=1), (local,stem,residual,joined,inv,gamma,down)


def backward(d, saved):
    local,stem,residual,joined,inv,gamma,down = saved
    g = {}

    def back(name,grad):
        dx,g[name+"_w"],g[name+"_b"] = _conv_backward(grad,local[name])
        return dx

    dx = back("fine",d[:,:16].reshape(-1,4,4,8))
    dc = back("coarse",d[:,16:].reshape(-1,2,2,8))
    dx += back("down",dc*(down>0))
    g["gamma"] = np.sum(dx*joined*inv,axis=(0,1,2))
    dx *= gamma
    dx = inv*(dx-joined*np.mean(dx*joined,axis=-1,keepdims=True)*inv*inv)
    dx = (dx+back("body",dx*(residual>0)))*(stem>0)
    back("stem",dx)
    return g


def loss(pred, target):
    positive = target[:,:,0]
    logistic = 1/(1+np.exp(-np.clip(pred[:,:,0],-32,32)))
    weight = np.where(positive>0,3.,1.)
    loss_obj = np.logaddexp(0,pred[:,:,0])-pred[:,:,0]*positive
    cls = pred[:,:,1:3]
    exps = np.exp(cls-cls.max(axis=2,keepdims=True))
    probs = exps/exps.sum(axis=2,keepdims=True)
    cls_true = target[:,:,1:3]
    boxes = pred[:,:,4:8]-target[:,:,4:8]
    norm = max(1.,float(positive.sum()))
    grad = np.zeros_like(pred)
    grad[:,:,0] = (logistic-positive)*weight/weight.sum()
    grad[:,:,1:3] = (probs-cls_true)*positive[:,:,None]/norm
    grad[:,:,4:8] = .5*boxes*positive[:,:,None]/norm
    mean = np.sum(loss_obj*weight)/weight.sum()
    mean -= np.sum(cls_true*np.log(np.clip(probs,1e-12,1.)))/norm
    mean += .25*np.sum(boxes*boxes*positive[:,:,None])/norm
    return float(mean), grad


def metrics(pred, target):
    actual = target[:,:,0]>0
    detected = pred[:,:,0]>0
    hits = actual & detected
    return {"object_precision":float(hits.sum()/max(1,detected.sum())),
            "object_recall":float(hits.sum()/max(1,actual.sum())),
            "positive_class_accuracy":float(np.mean(np.argmax(pred[:,:,1:3],axis=-1)[actual] ==
                                                          np.argmax(target[:,:,1:3],axis=-1)[actual])),
            "offset_mae_px":float(np.mean(np.abs(pred[:,:,4:8]-target[:,:,4:8])[actual]))}


def train(steps=900,size=768):
    images,labels = training_data(size)
    rng = np.random.default_rng(SEED+2)
    p = {name+"_w":rng.normal(0,math.sqrt(2/(k*k*ci)),size=(k,k,ci,co)).astype(np.float32)
         for name,(k,ci,co,_) in LAYERS.items()}
    p.update({name+"_b":np.zeros(co,np.float32) for name,(_,_,co,_) in LAYERS.items()})
    p["gamma"] = np.ones(4,np.float32)
    m = {k:np.zeros_like(v) for k,v in p.items()}
    v = {k:np.zeros_like(v) for k,v in p.items()}
    records = []
    for step in range(1,steps+1):
        batch = rng.integers(size,size=24)
        qat = step > steps//2
        pred,saved = forward(images[batch],p,qat=qat)
        score,dy = loss(pred,labels[batch])
        grad = backward(dy,saved)
        lr = .016 if not qat else .006
        for name in p:
            m[name] = .9*m[name]+.1*grad[name]
            v[name] = .999*v[name]+.001*grad[name]**2
            p[name] -= lr*(m[name]/(1-.9**step))/(np.sqrt(v[name]/(1-.999**step))+1e-7)
            # Keep weights representable; the floating oracle uses the trained
            # prequantization real values.
            bound = (127*WEIGHT_SCALE if name.endswith("_w") else
                     127*GAMMA_SCALE if name=="gamma" else 12.)
            np.clip(p[name],-bound,bound,out=p[name])
        if step % 150 == 0 or step == steps:
            records.append({"step":step,"qat":qat,"minibatch_loss":score})
    report = {"seed":SEED,"training_images":size,"steps":steps,"training_trace":records,
              "fp32_train":metrics(forward(images,p)[0],labels),
              "qat_train":metrics(forward(images,p,qat=True)[0],labels),
              "trained_parameters":sum(x.size for x in p.values())}
    return p,report


def save_parameters(path,p):
    path.write_text(json.dumps({k:v.tolist() for k,v in p.items()},sort_keys=True,separators=(",",":"))+"\n")


def load_parameters(path):
    return {k:np.asarray(v,np.float32) for k,v in json.loads(Path(path).read_text()).items()}
