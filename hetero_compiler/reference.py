"""Independent integer interpreter over original IR nodes, before fusion."""

import math

from .ir import DTYPES, GraphError
from .layout import ARITIES as LAYOUT_OPS
from .layout_reference import evaluate_layout


def _wrap(value, bits, signed=True):
    mask = (1 << bits) - 1
    value &= mask
    if signed and value > mask // 2:
        return value - (mask + 1)
    return value


def _narrow(x, typ, mode):
    width, low, high = DTYPES[typ]
    if mode == "wrap" or typ == "int32":
        return _wrap(x, width * 8, typ != "uint8")
    return max(low, min(high, x))


def _gemm(a, b, bias, adims, bdims, attrs):
    m, k = adims
    transpose = attrs.get("transpose_b", False)
    az, bz = attrs.get("a_zero_point", 0), attrs.get("b_zero_point", 0)
    n = bdims[0 if transpose else 1]
    return [sum((a[r*k+i] - az) * (b[c*k+i if transpose else i*n+c] - bz) for i in range(k)) + bias[c]
            for r in range(m) for c in range(n)]


def _conv(a, b, bias, adims, bdims, odims, attrs):
    batches, h, w, ci = adims
    kh, kw, _, co = bdims
    _, oh, ow, _ = odims
    sy, sx = attrs["strides"]
    top, left, _, _ = attrs["pads"]
    az, bz = attrs.get("a_zero_point", 0), attrs.get("b_zero_point", 0)
    for batch in range(batches):
        for y in range(oh):
            for x in range(ow):
                for ch in range(co):
                    acc = bias[ch]
                    for dy in range(kh):
                        iy = y * sy + dy - top
                        for dx in range(kw):
                            ix = x * sx + dx - left
                            if 0 <= iy < h and 0 <= ix < w:
                                off = ((batch*h + iy)*w + ix)*ci
                                wt = ((dy*kw + dx)*ci)*co + ch
                                for j in range(ci):
                                    acc += (a[off + j] - az) * (b[wt + j*co] - bz)
                    yield _wrap(acc, 32) >> attrs.get("shift", 0)


def _attention(q, k, v, qdims, kdims, vdims, attrs):
    m, d = qdims
    s, n = vdims
    for r in range(m):
        active = attrs["position"] + r + 1 if attrs["causal"] else s
        scores = [sum(q[r*d+i]*k[c*d+i] for i in range(kdims[1]))
                  for c in range(active)]
        max_score = max(scores)
        weights = [256 >> ((max_score - value) >> attrs["exp_shift"]) for value in scores]
        probs = [value * 255 // sum(weights) for value in weights]
        for c in range(n):
            yield sum(p * v[i*n+c] for i, p in enumerate(probs)) >> 8


def _gated(data, threshold, rounds):
    mask = 0xffffffff

    def rot(value, bits):
        return ((value << bits) | (value >> (32 - bits))) & mask

    for value in data:
        byte = value & 255
        x = byte + 0x01234567
        y = x ^ 0x9E3779B9
        for k in range(rounds):
            if value >= threshold:
                x = (rot(x, 5) ^ (y + byte + k)) & mask
                y = (rot(y, 29) + (x ^ 0xA5A5A5A5)) & mask
                x = ((x + (y >> 7)) ^ (byte << 11)) & mask
            else:
                x = (rot(x, 25) + (y ^ byte ^ k)) & mask
                y = (rot(y, 3) ^ (x + 0x3C6EF372)) & mask
                x = ((x ^ (y << 9)) + (byte >> 1)) & mask
        yield x ^ y


def _exp_weights(row):
    peak = max(row)
    return [int(65536 * math.exp(-((peak - x + 8) // 16)) + 0.5) for x in row]


def _probability(x):
    return int(256 / (1 + math.exp(-((x + 8) // 16))) + 0.5)


def _sppf(data, shape, kernel):
    batches, h, w, channels = shape
    levels = [data]
    radius = kernel // 2
    for _ in range(3):
        previous = levels[-1]
        levels.append([
            max(previous[((b*h+yy)*w+xx)*channels+c]
                for yy in range(max(0, y-radius), min(h, y+radius+1))
                for xx in range(max(0, x-radius), min(w, x+radius+1)))
            for b in range(batches) for y in range(h) for x in range(w) for c in range(channels)
        ])
    return [level[i*channels+c] for i in range(batches*h*w)
            for level in levels for c in range(channels)]


def _detect(dfl, cls, shape, classes, stride):
    _, height, width, _ = shape
    for i in range(height * width):
        logits = cls[i*classes:(i+1)*classes]
        best = max(range(classes), key=logits.__getitem__)
        dist = []
        for side in range(4):
            start = (i * 4 + side) * 4
            weights = _exp_weights(dfl[start:start + 4])
            dist.append(sum(j*w for j, w in enumerate(weights)) * stride // sum(weights))
        x, y = (i % width)*stride + stride//2, (i // width)*stride + stride//2
        yield best
        yield min(255, _probability(logits[best]))
        yield from (max(0, min(255, c)) for c in (x-dist[0], y-dist[1], x+dist[2], y+dist[3]))


def _overlap(a, b):
    def area(r):
        return max(0, r[4]-r[2]) * max(0, r[5]-r[3])

    shared = (max(0, min(a[4], b[4])-max(a[2], b[2])) *
              max(0, min(a[5], b[5])-max(a[3], b[3])))
    union = area(a) + area(b) - shared
    return shared * 256 // union if union else 0


def _nms(data, score_threshold, iou_threshold):
    rows = [data[i:i+6] for i in range(0, len(data), 6)]
    accepted = []
    output = [0] * (len(rows) * 7)
    for i in sorted(range(len(rows)), key=lambda index: (-rows[index][1], index)):
        a = rows[i]
        if a[1] >= score_threshold and all(a[0] != b[0] or _overlap(a, b) <= iou_threshold
                                          for b in accepted):
            accepted.append(a)
            output[i*7:(i+1)*7] = a + [1]
    return output


def _requant(x, attrs):
    x = (x - attrs.get("input_zero_point", 0)) * attrs["multiplier"]
    if attrs.get("rounding") == "nearest_even":
        divisor = attrs.get("divisor", 1 << attrs["shift"])
        q, r = divmod(x, divisor)
        q += 2*r > divisor or (2*r == divisor and q & 1)
    else:
        q = _wrap(x, 32) >> attrs["shift"]
    return q + attrs["zero_point"]


def evaluate(graph, inputs=None):
    values = {name: spec["data"][:] for name, spec in graph["tensors"].items() if "data" in spec}
    if inputs:
        if not isinstance(inputs, dict) or inputs.keys() - set(graph["inputs"]):
            raise GraphError("reference inputs contain unknown or constant tensors")
        for name, data in inputs.items():
            tensor = graph["tensors"][name]
            _, lo, hi = DTYPES[tensor["dtype"]]
            if (not isinstance(data, list) or len(data) != math.prod(tensor["shape"]) or
                    any(type(x) is not int or not lo <= x <= hi for x in data)):
                raise GraphError(f"{name}: reference input shape or data range mismatch")
            values[name] = data
    for node in graph["nodes"]:
        op, names, dst, attrs = (node[k] for k in ("op", "inputs", "output", "attrs"))
        tensors = graph["tensors"]
        if any(i not in values for i in names):
            raise GraphError(f"{node['id']}: reference input data is required")
        operands = [values[i] for i in names]
        shapes = [tensors[i]["shape"] for i in names]
        dtype, out_shape = tensors[dst]["dtype"], tensors[dst]["shape"]
        a = operands[0]
        if op in LAYOUT_OPS:
            result = evaluate_layout(op, operands, shapes, out_shape, attrs)
        elif op in ("Gemm", "Conv"):
            bias = operands[2] if len(operands) == 3 else [0] * out_shape[-1]
            if op == "Gemm":
                result = _gemm(a, operands[1], bias, *shapes[:2], attrs)
            else:
                result = _conv(a, operands[1], bias, *shapes[:2], out_shape, attrs)
        elif op in ("Add", "Mul"):
            b = operands[1]
            if op == "Add":
                result = (x + b[i % len(b)] for i, x in enumerate(a))
            else:
                result = (x * y for x, y in zip(a, b, strict=True))
        elif op == "Relu":
            result = (max(0, x) for x in a)
        elif op == "Requant":
            result = (_requant(x, attrs) for x in a)
        elif op == "SiLU":
            if attrs.get("mode") == "legacy_q4_knots":
                result = (math.trunc(x * _probability(x) / 256) for x in a)
            else:
                result = (round(x/(1 + math.exp(-x/16))) for x in a)
        elif op == "GELU":
            result = (round(x/2 * (1 + math.erf(x/(16*math.sqrt(2))))) for x in a)
        elif op == "RMSNorm":
            rows, cols = shapes[0]
            output = []
            for i in range(rows):
                row = a[i*cols:(i+1)*cols]
                square_sum = sum(x*x for x in row)
                exp = sum(square_sum >= 2 * cols * 4**j for j in range(7))
                output.extend((16*x) >> exp for x in row)
            result = output
        elif op == "RowSoftmax":
            rows, cols = shapes[0]
            output = []
            for i in range(rows):
                count = min(cols, i + 1) if attrs["causal_mask"] else cols
                row = a[i*cols:i*cols + count]
                weights = _exp_weights(row)
                output.extend(min(255, 256*x//sum(weights)) for x in weights)
                output.extend([0] * (cols - count))
            result = output
        elif op == "C2f":
            ca, cb = (shape[-1] for shape in shapes)
            result = [value for i in range(len(a)//ca)
                      for value in a[i*ca:(i+1)*ca] + operands[1][i*cb:(i+1)*cb]]
        elif op == "SPPF":
            result = _sppf(a, shapes[0], attrs["kernel"])
        elif op == "YoloDetectHead":
            result = _detect(a, operands[1], shapes[0], shapes[1][-1], attrs["stride"])
        elif op == "YoloNMS":
            result = _nms(a, attrs["score_threshold"], attrs["iou_threshold"])
        elif op == "Attention":
            result = _attention(*operands, *shapes, attrs)
        elif op == "GatedBranch":
            result = _gated(a, attrs["threshold"], attrs["rounds"])
        else:
            raise GraphError(f"no reference for {op}")
        mathematical = op in LAYOUT_OPS or (op == "Requant" and attrs.get("rounding") == "nearest_even")
        values[dst] = [_narrow(x if mathematical else _wrap(x, 32), dtype,
                               attrs.get("narrowing", "saturate")) for x in result]
    return values
