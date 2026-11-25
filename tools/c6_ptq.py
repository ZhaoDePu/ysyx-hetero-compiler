"""Calibrate official YOLO26n and qualify a portable Z15 integer IR.

The calibration command uses onnx, onnxruntime and cv2, typically in the
Windows export environment. The resulting quant.json/weights.npz can be
evaluated by NumPy alone after copying this module and the IR to Linux.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from hetero_compiler.c6_quant import QuantGraph, multiplier_shifts

_LINK = {"Identity", "Transpose", "Reshape", "Slice", "Split", "Resize", "MaxPool", "Gather",
         "GatherElements", "Squeeze", "Unsqueeze", "Flatten", "Expand"}


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_onnx(path):
    import onnx
    from onnx import numpy_helper

    source = onnx.load(path)
    onnx.checker.check_model(source)
    model = onnx.shape_inference.infer_shapes(source, strict_mode=True, data_prop=True)
    graph = model.graph
    if any(n.op_type == "BatchNormalization" for n in graph.node):
        raise ValueError("export must fold batch normalization before PTQ")
    constants = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    tensors = {}
    for v in [*graph.input, *graph.value_info, *graph.output]:
        spec = v.type.tensor_type
        if spec.elem_type <= 0 or any(not d.HasField("dim_value") or d.dim_value < 0 for d in spec.shape.dim):
            raise ValueError(f"ONNX tensor {v.name}: expected a static element type and shape")
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(spec.elem_type))
        tensors[v.name] = {"shape": [d.dim_value for d in spec.shape.dim], "dtype": dtype.name}
    nodes = []
    for index, node in enumerate(graph.node):
        attrs = {}
        for attr in node.attribute:
            value = onnx.helper.get_attribute_value(attr)
            if isinstance(value, bytes):
                value = value.decode("ascii")
            if isinstance(value, onnx.TensorProto):
                value = numpy_helper.to_array(value)
            attrs[attr.name] = value
        if node.op_type == "Constant":
            name = node.output[0]
            if "value" in attrs:
                constants[name] = attrs["value"]
            elif len(attrs) == 1:
                key, value = next(iter(attrs.items()))
                constants[name] = np.asarray(value, np.int64 if key.startswith("value_int") else np.float32)
            else:
                raise ValueError(f"unsupported ONNX Constant {name}")
            continue
        nodes.append({"id": node.name or f"{index}:{node.op_type}", "op": node.op_type,
                      "inputs": list(node.input), "outputs": list(node.output), "attrs": attrs})
    for n, val in constants.items():
        tensors[n] = {"shape": list(val.shape), "dtype": val.dtype.name}
    ins = [v.name for v in graph.input if v.name not in constants]
    outs = [v.name for v in graph.output]
    if len(ins) != 1 or len(outs) != 1 or tensors[ins[0]]["shape"][:2] != [1, 3]:
        raise ValueError("PTQ requires the frozen batch-one YOLO26 RGB interface")
    for node in nodes:
        for name in [*node["inputs"], *node["outputs"]]:
            if name and name not in tensors:
                raise ValueError(f"missing inferred shape for {name}")
    return model, dict(nodes=nodes, tensors=tensors, inputs=ins, outputs=outs), constants


def frame(path, size):
    import cv2

    img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode {path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    ratio = min(size / h, size / w)
    nw, nh = round(w * ratio), round(h * ratio)
    x = round((size - nw) / 2 - .1)
    y = round((size - nh) / 2 - .1)
    dst = np.full((size, size, 3), 114, dtype=np.uint8)
    dst[y:y+nh, x:x+nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return dst.transpose(2, 0, 1)[None], (ratio, x, y, w, h)


def _images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def capture_ranges(model, graph, paths, percentiles, out):
    import onnx
    import onnxruntime as ort

    ins = graph["inputs"][0]
    names = list(dict.fromkeys(n for row in graph["nodes"] for n in row["outputs"]
                               if graph["tensors"][n]["dtype"] == "float32"))
    size = graph["tensors"][ins]["shape"][-1]
    existing = {v.name for v in model.graph.output}
    for n in names:
        if n not in existing:
            model.graph.output.append(onnx.helper.make_tensor_value_info(
                n, onnx.TensorProto.FLOAT, graph["tensors"][n]["shape"]))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), sess_options=opts, providers=["CPUExecutionProvider"])
    sample = defaultdict(list)
    peaks = defaultdict(float)
    provenance = []
    rng = np.random.default_rng(6026)
    for path in paths:
        value = frame(path, size)[0].astype(np.float32) / np.float32(255)
        values = session.run(names, {ins: value})
        for n, val in zip(names, values, strict=True):
            flat = np.abs(val).ravel()
            if not len(flat) or not np.isfinite(flat).all():
                raise ValueError(f"{path.name} has invalid {n} activations")
            peaks[n] = max(peaks[n], float(np.max(flat)))
            if len(flat) > 4096:
                idx = rng.integers(0, len(flat), size=4096)
                flat = flat[idx]
            sample[n].append(flat)
        provenance.append({"name": path.name, "sha256": sha256(path.read_bytes()).hexdigest()})
        print("calibrated",size,path.name,flush=True)
    ranges = {}
    for n in names:
        values = np.concatenate(sample[n])
        ranges[n] = {str(p): float(v) for p,v in zip(percentiles,np.percentile(values,percentiles),strict=True)}
        ranges[n]["max"] = peaks[n]
    _write(out, {"images": provenance, "sample_points_per_image_tensor": 4096, "seed": 6026, "ranges": ranges})
    return ranges


def _act_lut(src, dst, fn):
    x = np.arange(256, dtype=np.uint8).view(np.int8).astype(np.float64) * src
    return np.clip(np.rint(fn(x)/dst), -128, 127).astype(np.int8).astype(int).tolist()


def _sigmoid(x):
    return 1/(1+np.exp(-np.clip(x, -80, 80)))


def _silu(x):
    return x*_sigmoid(x)


def _core_and_tail(nodes):
    """The decode ABI starts at last quantized dense output(s), never a core layer."""
    providers = {out: index for index,node in enumerate(nodes) for out in node["outputs"]}
    dense = {i for i,n in enumerate(nodes) if n["op"] in {"Conv", "MatMul"}}
    core = dense.copy()
    todo = list(dense)
    while todo:
        i = todo.pop()
        for k in nodes[i]["inputs"]:
            j = providers.get(k)
            if j is not None and j not in core:
                core.add(j)
                todo.append(j)
    return core


def _conv_options(attrs, source, weight):
    h, w = source["shape"][-2:]
    kh, kw = weight.shape[-2:]
    sy,sx = attrs.get("strides", [1,1])
    dy,dx = attrs.get("dilations",[1,1])
    pads = attrs.get("pads",[0,0,0,0])
    if attrs.get("auto_pad", "NOTSET") in {"SAME_UPPER", "SAME_LOWER"}:
        hp = max(0,(math.ceil(h/sy)-1)*sy + (kh-1)*dy+1-h)
        wp = max(0,(math.ceil(w/sx)-1)*sx + (kw-1)*dx+1-w)
        before_h = (hp + (attrs["auto_pad"]=="SAME_LOWER"))//2
        before_w = (wp + (attrs["auto_pad"]=="SAME_LOWER"))//2
        pads = [before_h, before_w, hp-before_h, wp-before_w]
    return dict(kernel=[kh,kw], strides=[sy,sx], pads=pads, dilations=[dy,dx],
                groups=attrs.get("group",1))


def _extent(tensor):
    shape = tensor["shape"]
    return int(math.prod(shape))


def _placement(graph, arrays):
    counts = {}
    roles = Counter()
    for node in graph["nodes"]:
        name, dtype = node["outputs"][0], node["device"]
        ops = node["op"]
        roles[dtype] += 1
        out = _extent(graph["tensors"][name])
        if ops == "Conv":
            w = arrays[node["attrs"]["weight"]]
            mac = out * int(math.prod(w.shape[:-1]))
            work = mac * 2 + 4*out
        elif ops == "MatMul":
            mac = out * graph["tensors"][node["inputs"][0]]["shape"][-1]
            work = mac * 2 + 3*out
        elif ops == "TopK":
            mac = 0
            source, k = node["inputs"]
            work = _extent(graph["tensors"][source]) * int(arrays[k].item())
        elif ops in {"Softmax", "VpuSoftmax", "Q23Softmax", "Sigmoid", "SiLU"}:
            mac,work = 0,8*out
        elif ops == "MaxPool":
            mac,work = 0,out*math.prod(node["attrs"]["kernel_shape"])
        else:
            mac,work = 0,out*max(1,len(node["inputs"])-1)
        counts[dtype] = counts.get(dtype,0) + work
        node["estimated_macs"] = int(mac)
        node["scalar_work_upper_estimate"] = int(work)
    total = sum(counts.values())
    return {"operators":dict(roles), "scalar_work_estimate":counts, "ps_fraction":counts.get("PS",0)/total,
            "convention":"2 scalar operations/MAC; output epilogues and TopK linear scans counted separately; planning estimate, not hardware evidence"}


def build_quant(graph, constants, ranges, percentile, softmax_mode, classifier_bound=0, *, first_scale_multiplier=1.0):
    tensors = {k:dict(v) for k,v in graph["tensors"].items()}
    original = graph["nodes"]
    init = dict(constants)
    core = _core_and_tail(original)
    consumers = defaultdict(list)
    providers = {n:i for i,x in enumerate(original) for n in x["outputs"]}
    for i,row in enumerate(original):
        for name in row["inputs"]:
            if name:
                consumers[name].append(i)
    silu = {}
    skipped = set()
    fused = {}
    for i,node in enumerate(original):
        if node["op"] != "Mul" or i not in core:
            continue
        for a,b in (node["inputs"], node["inputs"][::-1]):
            provider = providers.get(b,-1)
            if provider>=0 and original[provider]["op"]=="Sigmoid" and original[provider]["inputs"]==[a] and len(consumers[b])==1:
                silu[i] = a
                skipped.add(provider)
                parent = providers.get(a,-1)
                if parent>=0 and original[parent]["op"]=="Conv" and consumers[a]==[i,provider]:
                    fused[parent] = i
                elif parent>=0 and original[parent]["op"]=="Conv" and set(consumers[a])=={i,provider}:
                    fused[parent] = i
                break
    skipped.update(fused.values())

    def scale(n):
        if n in ranges:
            data = ranges[n]
            # Floor of sampled peak guards small histograms with an isolated, large feature.
            return max(1e-9,float(data[str(percentile)]),float(data["max"])*0.25)/127
        if n in constants:
            obj = constants[n]
            return max(1e-9,float(np.max(np.abs(obj))))/127
        raise ValueError(f"no calibration for {n}")

    def assign(n, s, dtype="int8"):
        tensors[n].update(scale=float(s), dtype=dtype)

    def current(n):
        if "scale" not in tensors[n]:
            if n not in constants or not np.issubdtype(constants[n].dtype,np.floating):
                raise ValueError(f"expected quantized {n}")
            # Raw initializer names can also feed the FP32 tail; scaled metadata
            # lets the interpreter dequantize those edges consistently.
            assign(n,scale(n))
            init[n] = np.round(np.clip(constants[n]/tensors[n]["scale"],-128,127)).astype(np.int8)
        return tensors[n]["scale"]

    nodes = []

    def append(node, op, ins, outs, attrs, dev):
        nodes.append(dict(id=node["id"],op=op,inputs=ins,outputs=outs,attrs=attrs,device=dev))

    def rescale(n, s, owner, slot):
        prev = current(n)
        if prev == s and tensors[n]["dtype"] == "int8":
            return n
        name = f"@c6/edge/{owner}/{slot}"
        tensors[name] = {"shape": tensors[n]["shape"], "dtype": "int8", "scale": s}
        if tensors[n]["dtype"]=="uint8":
            # The opcode indexes unsigned bytes. Codes above 127 must retain
            # their positive interpretation for a probability conversion.
            lut = np.clip(np.rint(np.arange(256)*prev/s),-128,127).astype(int).tolist()
        else:
            lut = _act_lut(prev,s,lambda x:x)
        append({"id":name},"RequantLUT",[n],[name],{"lut":lut},"VPU")
        return name

    assign(graph["inputs"][0], 1/127)
    first_conv = True
    for i,node in enumerate(original):
        if i in skipped:
            continue
        op,attrs,ins,outs=(node[k] for k in ("op","attrs","inputs","outputs"))
        a, out = ins[0], outs[0]
        if i not in core:
            append(node,op,ins,outs,attrs,"PS")
            continue
        if op == "Conv":
            w = constants[ins[1]]
            channels = w.shape[0]
            b = constants[ins[2]] if len(ins)>2 and ins[2] else np.zeros(channels)
            if not np.issubdtype(w.dtype,np.floating):
                raise ValueError("PTQ requires the original FP32 folded weights")
            s_in = current(a)
            s_w = np.max(np.abs(w.reshape(channels,-1)),axis=1)/127
            s_w = np.where(s_w>0,s_w,1).astype(np.float64)
            s_out = scale(out)
            sigmoid_tail = (channels == 80 and consumers[out] and
                            all(j not in core for j in consumers[out]))
            if classifier_bound and sigmoid_tail:
                # The ONNX tail applies sigmoid before ranking class scores.
                # Very negative logits from absent classes cannot materially
                # change confidence, yet dominate a symmetric INT8 range.
                s_out = min(s_out, classifier_bound / 127)
            if first_conv:
                s_out *= first_scale_multiplier
                first_conv = False
            ws = f"@c6/weight/{i}"
            bs = f"@c6/bias/{i}"
            init[ws] = np.clip(np.rint(w/s_w[:,None,None,None]),-127,127).astype(np.int8).transpose(2,3,1,0).copy()
            # Native epilogue uses arithmetic right shift. Express half a
            # post-requant LSB as a small accumulator bias to avoid repeated
            # negative floor error across the network.
            m,s = multiplier_shifts(s_w*s_in/s_out)
            if np.any(m == 0):
                raise ValueError(f"{node['id']}: requant scale is below frozen hardware precision")
            half_lsb = np.ldexp(np.full(channels, .5), s.astype(int)) / m
            bias = np.rint(b/(s_w*s_in) + half_lsb)
            if np.max(np.abs(bias)) >= 1<<31:
                raise ValueError(f"{node['id']}: signed bias overflows int32")
            init[bs] = bias.astype(np.int32)
            opts = dict(_conv_options(attrs,tensors[a],w),weight=ws,bias=bs,mul=m.tolist(),shift=s.tolist(),
                        requant_input_scale=float(s_out),weight_quant_scales=s_w.tolist())
            if classifier_bound and sigmoid_tail:
                opts["terminal_class_logits_clip"] = classifier_bound
            if i in fused:
                activation = original[fused[i]]
                out = activation["outputs"][0]
                s_final = scale(out)
                opts["lut"] = _act_lut(s_out,s_final,_silu)
                opts["fused_onnx_ids"] = [x["id"] for x in original if x["outputs"][0] in
                                          {activation["inputs"][0],activation["inputs"][1],out} and x["op"] in {"Sigmoid","Mul"}]
            else:
                s_final = s_out
            assign(out,s_final)
            append(node,"Conv",[a],[out],opts,"NPU")
        elif op == "MatMul":
            assign(out,scale(out))
            m,s=multiplier_shifts(current(a)*current(ins[1])/tensors[out]["scale"])
            append(node,op,ins,outs,{"mul":int(m),"shift":int(s)},"NPU")
        elif i in silu or op in {"Sigmoid","Relu"}:
            if i in silu:
                a = silu[i]
            assign(out,scale(out))
            fn = _silu if i in silu else _sigmoid if op=="Sigmoid" else lambda x:np.maximum(x,0)
            append(node,"Lookup",[a],outs,{"lut":_act_lut(current(a),tensors[out]["scale"],fn)},"VPU")
        elif op == "Add":
            s_out = max(scale(out),min(current(n) for n in ins))
            assign(out,s_out)
            append(node,"EWAdd",[rescale(n,s_out,i,j) for j,n in enumerate(ins)],outs,{},"VPU")
        elif op == "Mul":
            assign(out,scale(out))
            m,s=multiplier_shifts(current(a)*current(ins[1])/tensors[out]["scale"])
            append(node,"EWMul",ins,outs,{"mul":int(m),"shift":int(s)},"VPU")
        elif op == "Softmax":
            assign(out,1/256,"uint8")
            axis = attrs.get("axis",-1)
            if softmax_mode == "native":
                shift = int(np.clip(round(math.log2(math.log(2)/current(a))),0,31))
                append(node,"VpuSoftmax",ins,outs,{"axis":axis,"shift":shift},"VPU")
            else:
                table=np.rint((1<<23)*np.exp(-np.arange(256)*current(a))).astype(np.int64).tolist()
                append(node,"Q23Softmax",ins,outs,{"axis":axis,"lut":table},"GPU")
        elif op == "Concat":
            s_out = max([scale(out),*map(current,ins)])
            assign(out,s_out)
            append(node,op,[rescale(n,s_out,i,j) for j,n in enumerate(ins)],outs,attrs,"GPU")
        elif op in _LINK:
            if np.issubdtype(np.dtype(tensors[a]["dtype"]),np.floating) or "scale" in tensors[a]:
                for name in outs:
                    assign(name,current(a),tensors[a]["dtype"])
            append(node,op,ins,outs,attrs,"VPU" if op=="MaxPool" else "GPU")
        else:
            raise ValueError(f"unimplemented quantized core operator {node['id']}: {op}")
    use = Counter(n for row in nodes for n in row["inputs"] if n)
    weights = {row["attrs"][k] for row in nodes for k in ("weight","bias") if row["op"]=="Conv"}
    present = set(use) | weights | set(graph["inputs"]+graph["outputs"]) | {n for row in nodes for n in row["outputs"]}
    tensors = {n:t for n,t in tensors.items() if n in present}
    for n,t in tensors.items():
        shape = t["shape"]
        t["physical_shape"] = [shape[0],shape[2],shape[3],shape[1]] if len(shape)==4 else shape
        t["layout"] = "NHWC" if len(shape)==4 else "row-major"
        t["bytes"] = _extent(t)*np.dtype(t["dtype"]).itemsize
    init = {n:v for n,v in init.items() if n in present}
    quant = {**graph,"format":"c6-z15-quant-v1","nodes":nodes,"tensors":tensors,"use_counts":dict(use),
             "input_preprocess":{"letterbox_fill":114,"rgb_to_code":"round_even(uint8 * 127 / 255)",
                                 "ONNX_input":"NCHW","chip_input":"NHWC"},
             "percentile":percentile,"first_scale_multiplier":first_scale_multiplier,"softmax_mode":softmax_mode,
             "terminal_class_logits_bound":classifier_bound,
             "conv_rounding":"ASR floor with half-output-LSB encoded in the int32 accumulator bias"}
    quant["placement_estimate"] = _placement(quant,init)
    return quant, init


def detections(pred, meta, minimum=0.25):
    ratio, x, y, w, h = meta
    flat = np.asarray(pred).reshape(-1, 6)
    if not np.isfinite(flat).all():
        raise ValueError("non-finite YOLO26 output")
    result = []
    for row in flat:
        x1, y1, x2, y2, score, cls = map(float, row)
        if score < minimum:
            continue
        result.append({"class_id": int(cls), "score": score, "network_xyxy": [x1,y1,x2,y2],
                       "xyxy": [min(w,max(0,(x1-x)/ratio)), min(h,max(0,(y1-y)/ratio)),
                                min(w,max(0,(x2-x)/ratio)), min(h,max(0,(y2-y)/ratio))]})
    return result


def iou(a, b):
    x1,y1,x2,y2 = a
    u1,v1,u2,v2 = b
    overlap = max(0,min(x2,u2)-max(x1,u1))*max(0,min(y2,v2)-max(y1,v1))
    union = max(0,x2-x1)*max(0,y2-y1) + max(0,u2-u1)*max(0,v2-v1) - overlap
    return overlap / union if union else 0.0


def compare(fp32, int8, confidence):
    ref = [d for d in fp32 if d["score"] >= confidence]
    pred = [d for d in int8 if d["score"] >= confidence]
    ranked = sorted((-iou(a["xyxy"], b["xyxy"]), i,j)
                    for i,a in enumerate(ref) for j,b in enumerate(pred))
    left, right, pairs = set(), set(), []
    for neg_overlap, i, j in ranked:
        if neg_overlap > -.5:
            break
        if i in left or j in right:
            continue
        left.add(i)
        right.add(j)
        pairs.append({"fp32_index": i, "int8_index": j, "iou": -neg_overlap,
                      "class_equal": ref[i]["class_id"] == pred[j]["class_id"]})
    matched = len(pairs)
    agrees = sum(p["class_equal"] for p in pairs)
    return {"fp32_count": len(ref), "int8_count": len(pred), "spatial_matches": matched,
            "classes_equal": agrees, "unmatched_fp32":len(ref)-matched,
            "unmatched_int8":len(pred)-matched, "pairs":pairs,
            "class_agreement":agrees/len(ref) if ref else None,
            "mean_iou":sum(p["iou"] for p in pairs)/matched if matched else None}


def aggregate(rows):
    refs = sum(r["fp32_count"] for r in rows)
    class_ok = sum(r["classes_equal"] for r in rows)
    pairs = [p for r in rows for p in r["pairs"]]
    preds = sum(r["int8_count"] for r in rows)
    matches = len(pairs)
    return {"fp32_count":refs, "int8_count":preds, "spatial_matches":matches,
            "classes_equal":class_ok,"unmatched_fp32":refs-matches,"unmatched_int8":preds-matches,
            "class_agreement":class_ok/refs if refs else None,
            "fp32_spatial_miss_rate":(refs-matches)/refs if refs else None,
            "int8_spatial_extra_rate":(preds-matches)/preds if preds else None,
            "mean_iou":sum(p["iou"] for p in pairs)/matches if matches else None}


def emit_quant(out, quant, arrays):
    _write(out / "quant.json", quant)
    np.savez_compressed(out / "weights.npz", **arrays)


def _overlay(path, result, directory):
    import cv2

    img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    layers = []
    for kind, color in (("fp32",(210,115,20)), ("int8",(30,170,30))):
        layer = img.copy()
        cv2.putText(layer, kind.upper() + " conf >= 0.25", (10,25), cv2.FONT_HERSHEY_SIMPLEX, .65,
                    color,2,cv2.LINE_AA)
        for d in result[kind]:
            x1,y1,x2,y2 = map(round,d["xyxy"])
            cv2.rectangle(layer,(x1,y1),(x2,y2),color,2,cv2.LINE_AA)
            label = f"{d.get('class', d['class_id'])} {d['score']:.2f}"
            cv2.putText(layer,label,(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2,cv2.LINE_AA)
        layers.append(layer)
    ok, encoded = cv2.imencode(".jpg",np.concatenate(layers,axis=1),[cv2.IMWRITE_JPEG_QUALITY,88])
    if not ok:
        raise ValueError(f"cannot encode detection overlay for {path}")
    (directory / (path.stem + ".jpg")).write_bytes(encoded.tobytes())


def _pl_outputs(graph):
    used_by_ps = {name for n in graph["nodes"] if n["device"] == "PS" for name in n["inputs"]}
    outputs = [name for n in graph["nodes"] if n["device"] != "PS" for name in n["outputs"]
               if name in used_by_ps]
    return {name: {"suffix":f".pl_out_{i}.bin", **graph["tensors"][name]} for i,name in enumerate(outputs)}


def _run(g, images, size, baselines, out=None, labels=None):
    result = {}
    boundaries = _pl_outputs(g.graph) if out is not None else {}
    if out is not None:
        _write(out / "pl_outputs.json",boundaries)
    for path in images:
        rgb, box = frame(path, size)
        if out is None:
            capture = None
        else:
            def capture(node, name, value):
                if name in boundaries:
                    meta = boundaries[name]
                    if meta["layout"] == "NHWC":
                        value = value.transpose(0, 2, 3, 1)
                    np.ascontiguousarray(value).tofile(out / (path.stem + meta["suffix"]))
        pred = next(iter(g.infer(rgb.astype(np.float32)/np.float32(255), capture=capture).values()))
        pred = np.asarray(pred).reshape(-1, 6)
        rows = detections(pred, box)
        if labels is not None:
            for row in rows:
                row["class"] = labels[str(row["class_id"])]
        base = baselines[path.name]["detections"]
        result[path.name] = {"fp32":base, "int8":rows, "at_0_25":compare(base, rows, .25),
                             "at_0_40":compare(base, rows, .4)}
        if out is not None:
            rgb_nhwc = np.ascontiguousarray(rgb.transpose(0,2,3,1))
            qinput = np.rint(rgb_nhwc.astype(np.float64)*(127/255)).astype(np.int8)
            stem = out / path.stem
            rgb_nhwc.tofile(stem.with_suffix(".rgb_nhwc_u8.bin"))
            (out / (path.stem + ".ppm")).write_bytes(f"P6\n{size} {size}\n255\n".encode() + rgb_nhwc.tobytes())
            qinput.tofile(stem.with_suffix(".input_nhwc_s8.bin"))
            np.rint(pred[:,:4].astype(np.float64)*65536).astype("<i4").tofile(stem.with_suffix(".coords_q16_16.bin"))
            np.rint(pred[:,4].astype(np.float64)*65536).astype("<i4").tofile(stem.with_suffix(".scores_q16_16.bin"))
            pred[:,5].astype("<i8").tofile(stem.with_suffix(".classes_i64.bin"))
            np.save(stem.with_suffix(".npy"), pred)
            overlays = out.parent / "overlays"
            overlays.mkdir(exist_ok=True)
            _overlay(path,result[path.name],overlays)
        print("compared",size,path.name,result[path.name]["at_0_40"],flush=True)
    return {"scenes":result, "aggregate":{k:aggregate([r[k] for r in result.values()])
                                            for k in ("at_0_25","at_0_40")}}


def calibrate_classifier_bias(model_path, quant, arrays, quant_graph, images, size):
    """Fit integer bias correction to unsaturated foreground logits."""
    import onnx
    import onnxruntime as ort

    heads = {n["outputs"][0]:n for n in quant["nodes"] if n["attrs"].get("terminal_class_logits_clip")}
    if len(heads) != 3:
        raise ValueError("expected one 80-class output for each detection feature map")
    model, _, _ = load_onnx(model_path)
    for name in heads:
        model.graph.output.append(onnx.helper.make_tensor_value_info(
            name, onnx.TensorProto.FLOAT, quant["tensors"][name]["shape"]))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), sess_options=opts, providers=["CPUExecutionProvider"])
    sums = {name:np.zeros(80, dtype=np.float64) for name in heads}
    counts = {name:np.zeros(80, dtype=np.float64) for name in heads}
    ins = quant["inputs"][0]
    for path in images:
        rgb = frame(path, size)[0].astype(np.float32)/np.float32(255)
        targets = session.run(list(heads), {ins:rgb})
        predicted = {}

        def capture(node, name, output):
            if name in heads:
                predicted[name] = output.astype(np.float64) * quant["tensors"][name]["scale"]

        quant_graph.infer(rgb, capture=capture)
        for name,target in zip(heads,targets,strict=True):
            f = target.astype(np.float64)
            confidence = _sigmoid(f)
            # Suppress abundant background logits. Use threshold-independent
            # soft weights in the range where score precision matters.
            weights = confidence*(1-confidence)*(confidence >= .12)
            error = np.clip(f-predicted[name],-2,2)
            sums[name] += np.sum(weights*error,axis=(0,2,3))
            counts[name] += np.sum(weights,axis=(0,2,3))
    result = {}
    for name,node in heads.items():
        total = counts[name].sum()
        mean = float(sums[name].sum() / total) if total else 0.
        offset = np.clip((sums[name] + 4*mean)/(counts[name]+4),-.75,.75)
        attrs = node["attrs"]
        s_out = quant["tensors"][name]["scale"]
        step = s_out * np.ldexp(attrs["mul"],-np.array(attrs["shift"], dtype=int))
        acc_delta = np.rint(offset/step).astype(np.int64)
        key = attrs["bias"]
        value = arrays[key].astype(np.int64) + acc_delta
        if np.max(np.abs(value)) >= 1<<31:
            raise ValueError("classifier bias calibration overflows signed int32")
        arrays[key] = value.astype(np.int32)
        result[node["id"]] = {"weighted_branch_mean":mean, "accumulator_delta":acc_delta.tolist(),
                              "fitted_logit_offset":offset.tolist(), "calibration_count":len(images)}
    return result


def _baseline(model_path, images, size):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    ins = session.get_inputs()[0].name
    result = {}
    for path in images:
        rgb, box = frame(path,size)
        pred = session.run(None, {ins:rgb.astype(np.float32)/np.float32(255)})[0]
        result[path.name] = {"detections":detections(pred,box)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, help="official export model.json containing the COCO label map")
    parser.add_argument("--percentiles", nargs="+", type=float, default=[99.99])
    parser.add_argument("--softmax", nargs="+", choices=["native","q23"], default=["native"])
    parser.add_argument("--classifier-bounds", nargs="+", type=float, default=[12],
                        help="symmetric terminal class-logit limits; zero retains the full calibrated range")
    parser.add_argument("--classifier-bias-correction", action="store_true")
    parser.add_argument("--reuse-calibration", action="store_true")
    parser.add_argument("--selection-count", type=int, default=6,
                        help="deterministic calibration subset used to choose among multiple candidates")
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    model, graph, constants = load_onnx(args.onnx)
    size = graph["tensors"][graph["inputs"][0]]["shape"][-1]
    labels = (json.loads(args.metadata.read_text(encoding="utf-8"))["exports"][str(size)]["labels"]
              if args.metadata else None)
    cal = _images(args.calibration)
    eval_images = _images(args.eval)
    cal_hash = {sha256(p.read_bytes()).hexdigest() for p in cal}
    eval_hash = {sha256(p.read_bytes()).hexdigest() for p in eval_images}
    if len(cal_hash) < 32 or len(eval_hash) < 4 or cal_hash & eval_hash:
        raise ValueError("require 32 distinct calibration and four disjoint evaluation images")
    baselines = json.loads(args.golden.read_text(encoding="utf-8"))
    if baselines.keys() != {p.name for p in eval_images}:
        raise ValueError("FP32 golden and evaluation file names differ")
    ranges_path = args.out / "calibration.json"
    if args.reuse_calibration:
        saved = json.loads(ranges_path.read_text(encoding="utf-8"))
        if cal_hash != {p["sha256"] for p in saved["images"]}:
            raise ValueError("calibration source changed")
        ranges = saved["ranges"]
    else:
        ranges = capture_ranges(model,graph,cal,args.percentiles,ranges_path)
    best = None
    ledger = []
    if any(bound < 0 for bound in args.classifier_bounds):
        raise ValueError("classifier bound must be nonnegative")
    candidates = [(p,m,cl) for p in args.percentiles for m in args.softmax for cl in args.classifier_bounds]
    subset = (cal[::math.ceil(len(cal)/args.selection_count)][:args.selection_count]
              if len(candidates)>1 or args.classifier_bias_correction else [])
    selection_fp32 = _baseline(args.onnx,subset,size) if subset else {}
    for params in candidates:
        percentile, mode, bound = params
        quant, arrays = build_quant(graph,constants,ranges,*params)
        quant["source_onnx_sha256"] = sha256(args.onnx.read_bytes()).hexdigest()
        emit_quant(args.out,quant,arrays)
        if subset:
            selection = _run(QuantGraph(args.out),subset,size,selection_fp32)["aggregate"]["at_0_40"]
            agreement = selection["class_agreement"] or 0
            mean = selection["mean_iou"] or 0
            score = (agreement >= .85 and mean >= .7, agreement+mean, agreement, mean)
            ledger.append({"percentile":percentile,"softmax":mode,"terminal_class_logits_bound":bound,"validation":selection})
        else:
            score = (True, 0, 0, 0)
        if best is None or score > best[0]:
            best = (score,params)
        print("candidate",params,score,flush=True)
    quant, arrays = build_quant(graph,constants,ranges,*best[1])
    quant["source_onnx_sha256"] = sha256(args.onnx.read_bytes()).hexdigest()
    if args.classifier_bias_correction:
        emit_quant(args.out,quant,arrays)
        fit = [p for p in cal if p not in subset]
        quant["classifier_bias_correction"] = calibrate_classifier_bias(
            args.onnx,quant,arrays,QuantGraph(args.out),fit,size)
        quant["classifier_bias_fitting_images"] = [p.name for p in fit]
        quant["classifier_selection_images"] = [p.name for p in subset]
    if labels is not None:
        quant["labels"] = labels
    quant["output_abi"] = {"source":"[1,N,6] ordered xyxy,score,class",
                          "coords":"little-endian int32 in network pixels, Q16.16",
                          "scores":"little-endian int32 Q16.16","classes":"little-endian int64"}
    emit_quant(args.out,quant,arrays)
    vectors = args.out / "vectors"
    vectors.mkdir(exist_ok=True)
    quality = _run(QuantGraph(args.out),eval_images,size,baselines,vectors,labels)
    high = quality["aggregate"]["at_0_40"]
    quality["gate"] = {"confidence":.4,"class_threshold":.85,"mean_iou_threshold":.7,
                       "passed":(high["class_agreement"] or 0)>=.85 and (high["mean_iou"] or 0)>=.7}
    quality.update(size=size, percentile=best[1][0], softmax=best[1][1], terminal_class_logits_bound=best[1][2],
                   classifier_bias_correction=quant.get("classifier_bias_correction"),
                   candidate_ledger=ledger,
                   comparison="one-to-one spatial matching at IoU >=0.5, descending IoU; unmatched FP32 detections count against class agreement",
                   placement=quant["placement_estimate"])
    _write(args.out / "accuracy.json",quality)
    print("RESULT",json.dumps({k:v for k,v in quality.items() if k not in {"scenes","classifier_bias_correction"}}),flush=True)
    if not quality["gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
