"""Encode trained quantized detector and no-NMS decode as standard ONNX."""

import argparse
from pathlib import Path

import numpy as np

from examples.generate_quantized_onnx import FLOAT, INT8, INT32, INT64, Model

from .scenes import BOX_SCALE, CELLS, CHANNELS, GRID, IMAGE_SCALE, SIZE, TOP_K, held_out, quantized_image
from .train import EPSILON, FEATURE_SCALE, GAMMA_SCALE, WEIGHT_SCALE, WEIGHT_SCALES, load_parameters, train, save_parameters


def encode(params, inputs, independent=False):
    model = Model("YOLO26_Tiny_Style_C5")
    for name, value in (("image_scale",IMAGE_SCALE),("weight_scale",WEIGHT_SCALE),
                        ("feature_scale",FEATURE_SCALE),("head_scale",BOX_SCALE),
                        ("gamma_scale",GAMMA_SCALE),("body_weight_scale",WEIGHT_SCALES["body"]),("epsilon",EPSILON)):
        model.tensor(name,[],FLOAT,[value])
    model.tensor("zero",[],INT8,[0])
    model.tensor("axes_last",[1],INT64,[-1])
    model.tensor("topk",[1],INT64,[TOP_K])
    model.tensor("grid",[len(CELLS),4],INT8,np.rint(GRID/BOX_SCALE).astype(np.int8).ravel().tolist())
    g = np.clip(np.rint(params["gamma"]/GAMMA_SCALE),-128,127).astype(np.int8).tolist()
    model.tensor("gamma",[4],INT8,g)
    weights = {}
    for name in ("stem","body","down","fine","coarse"):
        step = WEIGHT_SCALES.get(name, WEIGHT_SCALE)
        weight = np.clip(np.rint(params[name+"_w"]/step),-128,127).astype(np.int8)
        weight = weight.transpose(3,2,0,1)
        bias_scale = (IMAGE_SCALE if name == "stem" else FEATURE_SCALE)*step
        bias = np.rint(params[name+"_b"]/bias_scale).astype(np.int32)
        weights[name] = (model.tensor(name+"_w",list(weight.shape),INT8,weight.ravel().tolist()),
                         model.tensor(name+"_b",list(bias.shape),INT32,bias.tolist()))
    for key,values in (("fine_matrix",[16,8]),("coarse_matrix",[4,8]),("fine_features",[16,4]),
                       ("restore_feature",[1,4,4,4]),("candidate_flat",[len(CELLS)]),
                       ("score_start",[0]),("score_end",[1]),("class_start",[1]),("class_end",[3]),
                       ("box_start",[4]),("box_end",[8]),("slice_col",[1])):
        model.tensor(key,[len(values)],INT64,values)

    for number, image in enumerate(inputs):
        prefix = f"im{number:02d}_" if independent else ""

        def op(name, kind, operands, **attrs):
            return model.op(prefix+name,kind,operands,prefix+name,**attrs)

        def conv(name, source, *, stride=1, pad=0, src="feature_scale", dst="feature_scale", relu=False):
            nchw = op(name+"_to_nchw","Transpose",[source],perm=[0,3,1,2])
            w,b = weights[name]
            y = op(name+"_conv","QLinearConv",[nchw,src,"zero",w,"body_weight_scale" if name=="body" else "weight_scale","zero",dst,"zero",b],
                   strides=[stride,stride], pads=[pad]*4)
            if relu:
                y = op(name+"_relu","Relu",[y])
            return op(name+"_to_nhwc","Transpose",[y],perm=[0,2,3,1])

        def quant(name, operand):
            return op(name, "QuantizeLinear",[operand,"feature_scale","zero"])

        def dq(name,operand,scale="feature_scale"):
            return op(name,"DequantizeLinear",[operand,scale,"zero"])

        def rms(source):
            x = dq("norm_dq",op("as_vectors","Reshape",[source,"fine_features"]))
            gamma = dq("norm_gamma","gamma","gamma_scale")
            square = op("norm_square","Mul",[x,x])
            mean = op("norm_mean","ReduceMean",[square,"axes_last"],keepdims=1)
            root = op("norm_sqrt","Sqrt",[op("norm_epsilon","Add",[mean,"epsilon"])])
            norm = op("norm_gamma_mul","Mul",[op("norm_div","Div",[x,root]),gamma])
            return op("norm_nhwc","Reshape",[quant("norm_quant",norm),"restore_feature"])

        x = model.input(prefix+"image",[1,SIZE,SIZE,CHANNELS],INT8,quantized_image(image).ravel().tolist())
        s = conv("stem",x,stride=2,pad=1,src="image_scale",relu=True)
        r = conv("body",s,relu=True)
        residual = quant("residual",op("residual_sum","Add",[dq("skip_dq",s),dq("residual_dq",r)]))
        f = rms(residual)
        c = conv("down",f,stride=2,pad=1,relu=True)
        fine = op("fine_rows","Reshape",[conv("fine",f,dst="head_scale"),"fine_matrix"])
        coarse = op("coarse_rows","Reshape",[conv("coarse",c,dst="head_scale"),"coarse_matrix"])
        packed = op("all_cells","Concat",[fine,coarse],axis=0)
        obj = op("score_slice","Slice",[packed,"score_start","score_end","slice_col"])
        obj = op("objectness","Reshape",[obj,"candidate_flat"])
        model.op(prefix+"topk_objectness","TopK",[obj,"topk"],[prefix+"scores",prefix+"indices"],axis=0)
        selected = op("selected_rows","Gather",[packed,prefix+"indices"],axis=0)
        cl = op("class_slice","Slice",[selected,"class_start","class_end","slice_col"])
        op("classes","ArgMax",[cl],axis=1,keepdims=0)
        corner = op("offset_slice","Slice",[selected,"box_start","box_end","slice_col"])
        corner = op("extend_offsets","Cast",[corner],to=INT32)
        grid = op("selected_grid","Gather",["grid",prefix+"indices"],axis=0)
        op("boxes","Add",[corner,op("extend_grid","Cast",[grid],to=INT32)])
        for output,shape,dtype in (("scores",[TOP_K],INT8),("indices",[TOP_K],INT64),
                                   ("classes",[TOP_K],INT64),("boxes",[TOP_K,4],INT32)):
            model.output(prefix+output,shape,dtype)
    return model.encode()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--parameters",type=Path)
    args = cli.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.parameters:
        params = load_parameters(args.parameters)
    else:
        params, report = train()
        import json
        save_parameters(args.out/"parameters.json",params)
        (args.out/"training.json").write_text(json.dumps(report,indent=2)+"\n")
    images = [item[0] for item in held_out().values()]
    (args.out/"YOLO26_Tiny_Style_INT8.onnx").write_bytes(encode(params,images[:1]))
    (args.out/"YOLO26_Tiny_Style_4screen_INT8.onnx").write_bytes(encode(params,images,independent=True))


if __name__ == "__main__":
    main()
