"""Export and qualify the official COCO YOLO26n under a static input contract."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import ultralytics
from ultralytics import YOLO

CHECKPOINT = "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"
URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"


def letterbox(rgb, size):
    """Square static padding equivalent to Ultralytics LetterBox(auto=False)."""
    h, w = rgb.shape[:2]
    ratio = min(size/h, size/w)
    nw, nh = round(w*ratio), round(h*ratio)
    x = round((size - nw)/2 - .1)
    y = round((size - nh)/2 - .1)
    dst = np.full((size, size, 3), 114, dtype=np.uint8)
    dst[y:y+nh, x:x+nw] = cv2.resize(rgb, (nw,nh), interpolation=cv2.INTER_LINEAR)
    return dst, (ratio,x,y,w,h)


def _session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def export(root, sizes):
    weights = root/"yolo26n.pt"
    digest = sha256(weights.read_bytes()).hexdigest()
    if digest != CHECKPOINT:
        raise ValueError("checkpoint hash did not match the Ultralytics release")
    input_info = json.loads((root/"images.json").read_text(encoding="utf-8"))
    versions = {"python":platform.python_version(),"torch":torch.__version__,
                "ultralytics":ultralytics.__version__,"onnx":onnx.__version__,
                "onnxruntime":ort.__version__,"opencv":cv2.__version__}
    metadata = {"checkpoint":weights.name,"checkpoint_sha256":digest,"source":URL,
                "versions":versions,"exports":{}}
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    for size in sizes:
        model = YOLO(str(weights))
        names = {int(i):label for i,label in model.names.items()}
        if len(names) != 80 or names[0] != "person" or names[5] != "bus":
            raise RuntimeError("checkpoint must use the 80 COCO labels")
        have = {"format":"onnx","imgsz":size,"batch":1,"dynamic":False,"simplify":True,
                "nms":False,"opset":19,"device":"cpu","half":False}
        onnx_path = root/f"yolo26n_{size}.onnx"
        produced = Path(model.export(**have))
        exported = onnx.load(produced)
        onnx.checker.check_model(exported, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(exported,strict_mode=True,data_prop=True)
        onnx.checker.check_model(inferred, full_check=True)
        onnx.save(inferred,onnx_path)
        session = _session(onnx_path)
        if len(session.get_inputs())!=1 or session.get_inputs()[0].shape!=[1,3,size,size]:
            raise RuntimeError("ONNX did not expose the requested static FP32 input")
        if session.get_inputs()[0].type != "tensor(float)" or len(session.get_outputs())!=1:
            raise RuntimeError("unexpected YOLO input type or output count")
        if session.get_outputs()[0].shape != [1,300,6]:
            raise RuntimeError(f"expected end-to-end YOLO output, got {session.get_outputs()[0].shape}")
        data = {}
        view = root/f"fp32_overlay_{size}"
        view.mkdir(exist_ok=True)
        for name, provenance in sorted(input_info["eval"].items()):
            img_path=root/"eval"/name
            img=cv2.imdecode(np.frombuffer(img_path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"{name} does not decode")
            rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            fixed,(ratio,x,y,w,h)=letterbox(rgb,size)
            result=session.run(None,{session.get_inputs()[0].name:np.ascontiguousarray(
                fixed.transpose(2,0,1)[None],dtype=np.float32)/np.float32(255)})[0][0]
            if result.shape != (300,6) or not np.isfinite(result).all():
                raise RuntimeError(f"unexpected detections in {name}")
            det=[]
            for x1,y1,x2,y2,score,cid in result:
                if score < .25:
                    continue
                cls=int(cid)
                if cls not in names or cid!=cls:
                    raise RuntimeError(f"invalid class ID {cid}")
                box=[max(0,min(limit,float((v-offset)/ratio))) for v,offset,limit in
                     zip((x1,y1,x2,y2),(x,y,x,y),(w,h,w,h),strict=True)]
                det.append({"class_id":cls,"class":names[cls],"score":float(score),"xyxy":box,
                            "network_xyxy":[float(v) for v in (x1,y1,x2,y2)]})
                xa,ya,xb,yb=map(round,box)
                cv2.rectangle(img,(xa,ya),(xb,yb),(0,180,255),2)
                cv2.putText(img,f"{names[cls]} {score:.2f}",(xa,max(12,ya-4)),
                            cv2.FONT_HERSHEY_SIMPLEX,.45,(0,180,255),1,cv2.LINE_AA)
            (view/name).write_bytes(cv2.imencode(".jpg",img)[1].tobytes())
            data[name]={"source":provenance,"letterbox":{"gain":ratio,"pad_left":x,"pad_top":y,
                        "source_width":w,"source_height":h},"detections":det}
            print(size,name,len(det),[d["class"] for d in det if d["score"]>=.4],flush=True)
        if size==320:
            bus=[d["class"] for d in data["bus.jpg"]["detections"] if d["score"]>=.4]
            if "bus" not in bus or bus.count("person")<2:
                raise RuntimeError("FP32 320 bus scene failed recognition check")
        (root/f"fp32_{size}.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
        metadata["exports"][str(size)] = {"onnx":onnx_path.name,
             "sha256":sha256(onnx_path.read_bytes()).hexdigest(),"parameters":int(sum(v.numel() for v in model.model.parameters())),
             "arguments":have,"labels":names,
             "input":{"shape":[1,3,size,size],"name":session.get_inputs()[0].name,"dtype":"float32"},
             "output":{"shape":[1,300,6],"name":session.get_outputs()[0].name}}
        (root/"model.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root",type=Path,default=Path.cwd())
    cli.add_argument("--sizes",type=int,nargs="+",default=[320,128])
    args=cli.parse_args()
    export(args.root.resolve(),args.sizes)
