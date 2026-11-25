"""Audit a shape-inferred static ONNX graph without loading the weights.

This reports graph liveness, separate from device placement and DMA traffic.
Run as python3 -m tools.audit_c6_yolo MODEL --out-dir reports.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from hetero_compiler import onnx as wire
from hetero_compiler.ir import GraphError

HALF = 1 << 20
TOTAL = 2 * HALF
# TensorProto fixed-size types used in numeric ONNX graphs.
DTYPES = {
    1: ("float32", 4), 2: ("uint8", 1), 3: ("int8", 1), 4: ("uint16", 2),
    5: ("int16", 2), 6: ("int32", 4), 7: ("int64", 8), 9: ("bool", 1),
    10: ("float16", 2), 11: ("float64", 8), 12: ("uint32", 4), 13: ("uint64", 8),
    14: ("complex64", 8), 15: ("complex128", 16), 16: ("bfloat16", 2),
}
FLOATS = {"float16", "bfloat16", "float32", "float64"}
NODE_FIELDS = (
    "index", "name", "scope", "domain", "op_type", "inputs", "outputs",
    "input_onnx_bytes", "output_onnx_bytes", "input_int8_bytes", "output_int8_bytes",
    "live_onnx_bytes", "live_int8_bytes", "attributes", "conv_family", "kernel",
    "stride", "padding", "dilation", "groups", "macs", "structure",
)
TENSOR_FIELDS = (
    "name", "shape", "dtype", "kind", "onnx_bytes", "int8_bytes", "producer",
    "first_step", "last_consumer", "consumer_count", "consumers", "onnx_limit",
    "int8_limit", "cross_layer", "producer_scope", "consumer_scopes", "ddr_candidate",
)
SCOPE = re.compile(r"(?:^|/)(model\.\d+)(?:/|\.|$)")


def _shape_decl(message):
    info = wire._message(message)
    name = wire._text(wire._one(info, 1, 2))
    typ = wire._one(wire._message(wire._one(info, 2, 2)), 1, 2)
    if typ is None:
        raise GraphError(f"{name}: only ONNX tensor values are supported by this audit")
    tensor = wire._message(typ)
    shape = wire._one(tensor, 2, 2)
    dims = None if shape is None else [
        wire._one(wire._message(dim), 1, 0) for dim in wire._all(wire._message(shape), 1, 2)
    ]
    return name, dims, wire._one(tensor, 1, 0)


def _tensor_decl(tensor):
    return list(wire._repeated_ints(tensor, 1)), wire._one(tensor, 2, 0)


def _merge_meta(meta, name, dims, dtype):
    if not name:
        raise GraphError("empty tensor name")
    if name in meta:
        old_dims, old_type = meta[name]
        if old_type != dtype:
            raise GraphError(f"{name}: conflicting ONNX types")
        if old_dims is not None and dims is not None:
            if len(old_dims) != len(dims):
                raise GraphError(f"{name}: conflicting ONNX ranks")
            for a, b in zip(old_dims, dims):
                if a is not None and b is not None and a != b:
                    raise GraphError(f"{name}: conflicting ONNX sizes")
            dims = [a if a is not None else b for a, b in zip(old_dims, dims)]
        elif dims is None:
            dims = old_dims
    meta[name] = dims, dtype


def _node(payload, index):
    msg = wire._message(payload)
    op = wire._text(wire._one(msg, 4, 2))
    name = wire._text(wire._one(msg, 3, 2)) or f"{op}:{index}"
    attrs = {}
    for attr in wire._all(msg, 5, 2):
        data = wire._message(attr)
        attr_name = wire._text(wire._one(data, 1, 2))
        kind = wire._one(data, 20, 0)
        if kind == 1:
            value = struct.unpack("<f", wire._one(data, 2, 5))[0]
        elif kind == 2:
            value = wire._signed(wire._one(data, 3, 0, 0))
        elif kind == 3:
            value = wire._text(wire._one(data, 4, 2))
        elif kind == 4:
            dims, dtype = _tensor_decl(wire._message(wire._one(data, 5, 2)))
            value = {"shape": dims, "tensorproto_type": dtype}
        elif kind == 6:
            value = list(wire._floats(data, 7))
        elif kind == 7:
            value = [wire._signed(v) for v in wire._repeated_ints(data, 8)]
        elif kind == 8:
            value = [wire._text(v) for v in wire._all(data, 9, 2)]
        else:
            value = {"attributeproto_type": kind}
        attrs[attr_name] = value
    inputs, outputs = ([wire._text(i) for i in wire._all(msg, t, 2)] for t in (1, 2))
    match = SCOPE.search(name)
    return {
        "index": index, "name": name, "scope": match[1] if match else "(unscoped)",
        "domain": wire._text(wire._one(msg, 7, 2)) or "ai.onnx",
        "op_type": op, "inputs": inputs, "outputs": [v for v in outputs if v],
        "attributes": attrs,
    }


def _limit(size):
    if size > TOTAL:
        return "exceeds-sram"
    if size > HALF:
        return "exceeds-half"
    return "within-half"


def _make_tensor(name, meta, producer, consumer, inputs, outputs, constants, end):
    shape, typ = meta.get(name, (None, None))
    if shape is None or any(not isinstance(n, int) or n < 0 for n in shape):
        raise GraphError(f"{name}: unresolved static shape, run ONNX shape inference")
    if typ not in DTYPES:
        raise GraphError(f"{name}: unsupported tensor dtype {typ} for byte accounting")
    dtype, size = DTYPES[typ]
    elements = math.prod(shape)
    onnx_bytes = elements * size
    int8_bytes = elements if dtype in FLOATS else onnx_bytes
    kind = "constant" if name in constants and name not in inputs else "activation"
    start = producer.get(name, -1)
    uses = consumer[name]
    last = end if name in outputs else max(uses, default=start)
    cross = kind == "activation" and (len(uses) > 1 or last - start > 1)
    return {
        "name": name, "shape": shape, "dtype": dtype, "kind": kind,
        "onnx_bytes": onnx_bytes, "int8_bytes": int8_bytes, "producer": start,
        "first_step": start, "last_consumer": last, "consumer_count": len(uses),
        "consumers": uses, "onnx_limit": _limit(onnx_bytes), "int8_limit": _limit(int8_bytes),
        "cross_layer": cross,
        "ddr_candidate": kind == "activation" and (cross or int8_bytes > HALF),
    }


def _live(tensors, count, width):
    events = [0] * (count + 3)
    for t in tensors.values():
        if t["kind"] == "activation":
            events[t["first_step"] + 1] += t[width]
            events[t["last_consumer"] + 2] -= t[width]
    value = 0
    timeline = []
    for delta in events:
        value += delta
        timeline.append(value)
    return timeline


def _conv(node, tensors):
    attrs = node["attributes"]
    x, weight = (tensors[n]["shape"] for n in node["inputs"][:2])
    y = tensors[node["outputs"][0]]["shape"]
    if len(x) != 4 or len(weight) != 4 or len(y) != 4:
        raise GraphError(f"{node['name']}: expected NCHW Conv2D")
    group = attrs.get("group", 1)
    kernel = attrs.get("kernel_shape", weight[2:])
    if group < 1 or x[1] != group * weight[1] or y[1] != weight[0] or kernel != weight[2:]:
        raise GraphError(f"{node['name']}: inconsistent Conv2D groups or kernel")
    if group > 1:
        family = "depthwise" if group == x[1] and weight[1] == 1 else "grouped"
    else:
        family = "pointwise" if kernel == [1, 1] else "ordinary"
    return {
        "family": family, "kernel": kernel, "stride": attrs.get("strides", [1, 1]),
        "padding": attrs.get("auto_pad", "NOTSET") if attrs.get("auto_pad", "NOTSET") != "NOTSET"
        else attrs.get("pads", [0, 0, 0, 0]),
        "dilation": attrs.get("dilations", [1, 1]), "groups": group,
        "macs": math.prod(y) * math.prod(weight[1:]),
    }


def _structure(node, tensors):
    op = node["op_type"]
    if op == "Add":
        inputs = [tensors[name] for name in node["inputs"] if name]
        output = tensors[node["outputs"][0]]
        if (len(inputs) == 2 and inputs[0]["name"] != inputs[1]["name"]
                and all(t["kind"] == "activation" and t["shape"] == output["shape"] for t in inputs)):
            return "equal-shape residual candidate"
    return {
        "Concat": "feature join", "Split": "feature partition", "Resize": "resampling",
        "MaxPool": "spatial max pooling", "Softmax": "normalization",
        "MatMul": "matrix product", "TopK": "ranked detection candidates",
        "GatherElements": "candidate gather", "Gather": "index gather",
        "NonMaxSuppression": "graph NMS",
    }.get(op, "")


def audit_model(data):
    """Return per-node and per-tensor structural and liveness accounting."""
    model = wire._message(data)
    graph = wire._message(wire._one(model, 7, 2))
    meta, initials = {}, set()
    for payload in wire._all(graph, 5, 2):
        t = wire._message(payload)
        name = wire._text(wire._one(t, 8, 2))
        _merge_meta(meta, name, *_tensor_decl(t))
        initials.add(name)
    inputs, outputs = set(), set()
    for tag, declared in ((11, inputs), (12, outputs), (13, None)):
        for payload in wire._all(graph, tag, 2):
            name, dims, typ = _shape_decl(payload)
            _merge_meta(meta, name, dims, typ)
            if declared is not None:
                declared.add(name)
    nodes = [_node(payload, i) for i, payload in enumerate(wire._all(graph, 1, 2))]
    producer, consumer = {}, defaultdict(list)
    sigmoid_input = {}
    available, constants = initials | inputs, set(initials)
    for node in nodes:
        idx, op = node["index"], node["op_type"]
        for name in dict.fromkeys(node["inputs"]):
            if name:
                if name not in available:
                    raise GraphError(f"{node['name']}: undefined input {name}")
                consumer[name].append(idx)
        if op == "Sigmoid":
            sigmoid_input[node["outputs"][0]] = node["inputs"][0]
        for name in node["outputs"]:
            if name in available:
                raise GraphError(f"{node['name']}: duplicate definition of {name}")
            producer[name] = idx
            available.add(name)
            if op == "Constant":
                constants.add(name)
                descriptor = node["attributes"].get("value")
                if isinstance(descriptor, dict) and "tensorproto_type" in descriptor:
                    _merge_meta(meta, name, descriptor["shape"], descriptor["tensorproto_type"])
    if outputs - available:
        raise GraphError("undefined graph outputs " + ", ".join(sorted(outputs - available)))
    tensors = {name: _make_tensor(name, meta, producer, consumer, inputs, outputs, constants, len(nodes))
               for name in sorted(available)}
    for tensor in tensors.values():
        origin = tensor["producer"]
        tensor["producer_scope"] = nodes[origin]["scope"] if origin >= 0 else "(graph input or initializer)"
        tensor["consumer_scopes"] = list(dict.fromkeys(nodes[i]["scope"] for i in tensor["consumers"]))
    onnx_live = _live(tensors, len(nodes), "onnx_bytes")
    int8_live = _live(tensors, len(nodes), "int8_bytes")
    for node in nodes:
        idx, op = node["index"], node["op_type"]
        for direction in ("input", "output"):
            names = list(dict.fromkeys(n for n in node[direction + "s"] if n))
            for width in ("onnx", "int8"):
                node[f"{direction}_{width}_bytes"] = sum(tensors[n][width + "_bytes"] for n in names)
        node["live_onnx_bytes"] = onnx_live[idx + 1]
        node["live_int8_bytes"] = int8_live[idx + 1]
        node["structure"] = _structure(node, tensors)
        if op == "Mul" and len(node["inputs"]) == 2:
            left, right = node["inputs"]
            if sigmoid_input.get(left) == right or sigmoid_input.get(right) == left:
                node["structure"] = "SiLU (Sigmoid × input)"
        if op == "Conv":
            conv = _conv(node, tensors)
            node["conv"] = conv
            node.update(conv_family=conv["family"], kernel=conv["kernel"], stride=conv["stride"],
                        padding=conv["padding"], dilation=conv["dilation"], groups=conv["groups"],
                        macs=conv["macs"])
        elif op == "MatMul":
            a, b = (tensors[n]["shape"] for n in node["inputs"][:2])
            result = tensors[node["outputs"][0]]["shape"]
            if a and b and a[-1] == b[-2 if len(b) > 1 else -1]:
                node["macs"] = math.prod(result) * a[-1]
    opsets = {}
    for payload in wire._all(model, 8, 2):
        entry = wire._message(payload)
        domain = wire._text(wire._one(entry, 1, 2)) or "ai.onnx"
        opsets[domain] = wire._one(entry, 2, 0)
    return {
        "sha256": hashlib.sha256(data).hexdigest(), "opsets": opsets,
        "producer": wire._text(wire._one(model, 2, 2)),
        "producer_version": wire._text(wire._one(model, 3, 2)),
        "ir_version": wire._one(model, 1, 0),
        "inputs": sorted(inputs),
        "outputs": sorted(outputs), "nodes": nodes, "tensors": tensors,
        "peak_onnx_bytes": max(onnx_live), "peak_int8_bytes": max(int8_live),
        "peak_onnx_step": onnx_live.index(max(onnx_live)) - 1,
        "peak_int8_step": int8_live.index(max(int8_live)) - 1,
        "initializer_elements": sum(math.prod(tensors[n]["shape"]) for n in initials),
        "initializer_onnx_bytes": sum(tensors[n]["onnx_bytes"] for n in initials),
        "constant_int8_bytes": sum(t["int8_bytes"] for t in tensors.values() if t["kind"] == "constant"),
    }


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_cell(v) for v in row) + " |" for row in rows),
    ]


def render_report(audit, model_name, node_csv, tensor_csv, checkpoint_sha=None, source_sha=None):
    nodes, tensors = audit["nodes"], audit["tensors"]
    activation = [t for t in tensors.values() if t["kind"] == "activation"]
    by_op = Counter(n["op_type"] for n in nodes)
    by_scope = defaultdict(list)
    for node in nodes:
        by_scope[node["scope"]].append(node)
    lines = [
        "# M1 官方 YOLO26n ONNX 图审计",
        "",
        f"审计输入 `{model_name}`，ONNX SHA-256 为 `{audit['sha256']}`。",
        "",
    ]
    if checkpoint_sha:
        lines += [f"官方预训练权重 `yolo26n.pt` 的 SHA-256 为 `{checkpoint_sha}`。", ""]
    if source_sha:
        lines += [f"原始导出 ONNX SHA-256 为 `{source_sha}`。审计输入增加了静态形状声明。", ""]
    lines += [
        f"逐节点记录在 [{node_csv}]({node_csv})，逐张量记录在 [{tensor_csv}]({tensor_csv})。"
        "节点从 0 编号，输入在节点 −1 进入，最终输出存活至图末端。",
        "",
        "## 输入、输出与算子清单",
        "",
        f"ONNX IR 版本 `{audit['ir_version']}`。"
        f"模型生产者 `{audit['producer']} {audit['producer_version']}`。"
        f"算子集 {json.dumps(audit['opsets'], ensure_ascii=False)}。",
        "",
        *_table(["角色", "张量", "形状", "数据类型"], [
            (role, name, tensors[name]["shape"], tensors[name]["dtype"])
            for role, names in (("输入", audit["inputs"]), ("输出", audit["outputs"])) for name in names
        ]),
        "",
        f"整图 {len(nodes)} 个节点、{len(by_op)} 种 OpType、{len(tensors)} 个张量。",
        "",
        *_table(["ONNX OpType", "节点数"], sorted(by_op.items())),
        "",
        "## 卷积规格",
        "",
    ]
    shapes = Counter(
        (c["family"], str(c["kernel"]), str(c["stride"]), str(c["padding"]), str(c["dilation"]), c["groups"])
        for n in nodes if (c := n.get("conv"))
    )
    lines += _table(["卷积类别", "Kernel", "Stride", "Padding", "Dilation", "Groups", "节点数"],
                    [(*key, count) for key, count in sorted(shapes.items())])
    lines += ["", "类别由权重形状和分组推导。普通与逐点卷积的 groups 为 1，深度卷积的 groups 等于输入通道数。",
              "", "## 连接与计算结构", ""]
    summaries = []
    for scope, members in by_scope.items():
        counts = Counter(n["op_type"] for n in members)
        roles = []
        if counts["Softmax"] and counts["MatMul"] >= 2:
            roles.append("自注意力（QK、Softmax、PV）")
        if counts["MaxPool"]:
            roles.append(f"SPPF 池化链（MaxPool×{counts['MaxPool']}）")
        if counts["TopK"]:
            roles.append(f"Detect（TopK×{counts['TopK']}、GatherElements×{counts['GatherElements']}）")
        if counts["Resize"]:
            roles.append("上采样或重采样")
        residuals = sum(n["structure"] == "equal-shape residual candidate" for n in members)
        if residuals:
            roles.append(f"{residuals} 处等形残差候选")
        if counts["Split"] or counts["Concat"]:
            roles.append(f"Split×{counts['Split']}、Concat×{counts['Concat']}")
        if roles:
            summaries.append((scope, "，".join(roles), len(members)))
    lines += _table(["导出节点作用域", "可见结构", "该作用域节点数"], summaries)
    checks = [n for n in nodes if n["op_type"] in {"MatMul", "MaxPool", "Resize", "Softmax", "TopK"}]
    lines += ["", *_table(["节点", "算子", "主输入", "主输出", "关键属性"], [
        (n["name"], n["op_type"], tensors[n["inputs"][0]]["shape"], tensors[n["outputs"][0]]["shape"],
         "，".join(f"{k}={v}" for k, v in n["attributes"].items() if k != "cubic_coeff_a") or "—")
        for n in checks
    ])]
    conv_macs = sum(n.get("macs", 0) for n in nodes if n["op_type"] == "Conv")
    matmul_macs = sum(n.get("macs", 0) for n in nodes if n["op_type"] == "MatMul")
    nms = "含" if by_op.get("NonMaxSuppression") else "未出现"
    bn = by_op["BatchNormalization"]
    silu_mul = sum(n["structure"] == "SiLU (Sigmoid × input)" for n in nodes)
    lines += [
        "",
        f"ONNX 图中{nms} `NonMaxSuppression` 节点，存在 {bn} 个显式 `BatchNormalization` 节点。"
        f"检测到 {silu_mul} 处 `Sigmoid(x) × x` SiLU 模式。"
        "权重和张量形状导出的逻辑运算量为 Conv "
        f"{conv_macs:,} MAC、MatMul {matmul_macs:,} MAC。卷积 MAC 计入边缘补零，"
        "这些数值用于核对后续映射，实际硬件指令与周期由编译和 RTL 测量给出。",
        "",
        "## 活跃张量与两级存储压力",
        "",
        f"ONNX 初始化张量共 {audit['initializer_elements']:,} 个元素、{audit['initializer_onnx_bytes']:,} B。"
        f"常量按浮点转一字节、其余宽度不变的假设估算为 {audit['constant_int8_bytes']:,} B。",
        "",
        "逻辑活跃量统计包含消费节点还未释放的输入和同一节点刚产生的输出，"
        "不计入常量区、重排、对齐、宽累加或硬件内部缓冲。"
        "INT8 列仅将浮点激活按每元素 1 B 估算，保留整数索引的宽度。"
        "实际量化位宽和实板缓冲由 M2/M3 决定。",
        "",
        *_table(["逻辑表示", "峰值活跃 B", "峰值节点", "超过 1 MiB 半区", "超过 2 MiB 片上容量"], [
            (label, f"{audit[f'peak_{width}_bytes']:,}", audit[f"peak_{width}_step"],
             sum(t[f"{width}_bytes"] > HALF for t in activation),
             sum(t[f"{width}_bytes"] > TOTAL for t in activation))
            for width, label in (("onnx", "导出 ONNX"), ("int8", "INT8 容量敏感度"))
        ]),
        "",
        "FP32 单张量上限与峰值活跃量是两个容量检查。INT8 最大片上需求还受一次计算中的"
        "输入权重、输出暂存、宽累加和双半区分配影响。",
        "",
        f"INT8 假设下有 {sum(t['cross_layer'] for t in activation)} 个跨层活跃张量"
        "（存在多次消费或跨过中间节点），建议结合其拓扑编号评估 DDR 持久化。"
        "超出半区的完整张量需要分块、跨半区安排或外存暂存。"
        "CSV 提供全部张量的首末节点、消费边、两种容量分类和 DDR 候选标记，供两级规划使用。",
        "",
        *_table(["最大活跃张量（前八）", "形状", "ONNX B", "INT8 估算 B", "活跃期"], [
            (t["name"], t["shape"], f"{t['onnx_bytes']:,}", f"{t['int8_bytes']:,}",
             f"{t['first_step']}–{t['last_consumer']}")
            for t in sorted(activation, key=lambda v: -v["onnx_bytes"])[:8]
        ]),
        "",
        *_table(["跨作用域激活（按 INT8 大小前八）", "产出", "消费", "INT8 估算 B", "活跃期"], [
            (t["name"], t["producer_scope"], "、".join(t["consumer_scopes"]), f"{t['int8_bytes']:,}",
             f"{t['first_step']}–{t['last_consumer']}")
            for t in sorted((v for v in activation if v["producer"] >= 0 and any(
                scope != v["producer_scope"] for scope in v["consumer_scopes"]
            )), key=lambda v: -v["int8_bytes"])[:8]
        ]),
        "",
        "按图顺序逐层执行且每层只在自己的半区保留输入和输出时，跨作用域的激活进入 DDR 工作区。"
        "融合与跨层常驻可缩减这一集合。"
        "DMA 总量、驻留 DDR 工作区与片上峰值需按最终物理分块、权重布局、外存复用和指令调度核验。",
        "",
    ]
    return "\n".join(lines)


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                             if isinstance(v, (dict, list, bool)) else v for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--checkpoint-sha")
    parser.add_argument("--source-sha")
    parser.add_argument("--infer", action="store_true", help="run static shape inference with the optional ONNX package")
    args = parser.parse_args()
    data = args.model.read_bytes()
    if args.infer:
        import onnx
        args.source_sha = hashlib.sha256(data).hexdigest()
        data = onnx.shape_inference.infer_shapes(
            onnx.load_model_from_string(data), strict_mode=True, data_prop=True,
        ).SerializeToString()
    audit = audit_model(data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nodes = "m1-yolo26n-graph-nodes.csv"
    tensors = "m1-yolo26n-graph-tensors.csv"
    _write_csv(args.out_dir / nodes, NODE_FIELDS, audit["nodes"])
    _write_csv(args.out_dir / tensors, TENSOR_FIELDS, audit["tensors"].values())
    out = args.out_dir / "m1-yolo26n-graph-audit.md"
    out.write_text(render_report(audit, args.model.name, nodes, tensors, args.checkpoint_sha, args.source_sha),
                   encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
