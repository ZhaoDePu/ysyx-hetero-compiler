"""Small independent NumPy interpreter for the actual exported FP32 ONNX artifacts."""

import numpy as np

from hetero_compiler.onnx import _all, _initializer, _message, _node_attrs, _one, _text, _value_info


def _nchw_conv(input_, kernel, bias, pads, strides):
    batch, ci, ih, iw = input_.shape
    co, _, kh, kw = kernel.shape
    top, left, bottom, right = pads
    sy, sx = strides
    height, width = (ih+top+bottom-kh)//sy+1, (iw+left+right-kw)//sx+1
    padded = np.pad(input_, ((0, 0), (0, 0), (top, bottom), (left, right)))
    output = np.empty((batch, co, height, width), dtype=np.float32)
    for row in range(height):
        for col in range(width):
            source = padded[:, :, row*sy:row*sy+kh, col*sx:col*sx+kw]
            output[:, :, row, col] = source.reshape(batch, ci*kh*kw) @ (
                kernel.reshape(co, ci*kh*kw).T) + bias
    return output


class Fp32Onnx:
    def __init__(self, blob):
        model = _message(blob)
        graph = _message(_one(model, 7, 2))
        self.inputs = dict(_value_info(val) for val in _all(graph, 11, 2))
        self.outputs = dict(_value_info(val) for val in _all(graph, 12, 2))
        self.constants = {}
        for payload in _all(graph, 5, 2):
            name, tensor = _initializer(payload)
            if tensor["dtype"] not in {"float32", "int64"}:
                raise ValueError("FP32 reference accepts only float tensors and int64 shape initializers")
            self.constants[name] = np.asarray(tensor["data"], dtype=tensor["dtype"]).reshape(tensor["shape"])
        self.nodes = []
        for payload in _all(graph, 1, 2):
            node = _message(payload)
            if _text(_one(node, 7, 2)) or len(_all(node, 2, 2)) != 1:
                raise ValueError("FP32 reference requires standard singleton-output operators")
            ins = tuple(_text(x) for x in _all(node, 1, 2))
            self.nodes.append((_text(_one(node, 4, 2)), ins, _text(_one(node, 2, 2)), _node_attrs(node)))

    def evaluate(self, **inputs):
        if set(inputs) != set(self.inputs):
            raise ValueError("mismatched FP32 input symbols")
        values = dict(self.constants)
        for name, arr in inputs.items():
            if arr.dtype != np.float32 or arr.shape != tuple(self.inputs[name]["shape"]):
                raise ValueError(f"{name}: mismatched FP32 dtype or shape")
            values[name] = arr
        for op, names, dest, attr in self.nodes:
            args = [values[n] for n in names]
            if op == "Conv":
                out = _nchw_conv(*args, attr["pads"], attr["strides"])
            elif op == "MatMul":
                out = np.matmul(*args)
            elif op == "Add":
                out = np.add(*args)
            elif op == "Mul":
                out = np.multiply(*args)
            elif op == "Relu":
                out = np.maximum(args[0], np.float32(0))
            elif op == "Transpose":
                out = args[0].transpose(attr["perm"])
            elif op == "Softmax":
                dim = attr.get("axis", -1)
                stable = args[0] - args[0].max(axis=dim, keepdims=True)
                pow_ = np.exp(stable)
                out = pow_ / pow_.sum(axis=dim, keepdims=True)
            elif op == "Reshape":
                out = args[0].reshape(args[1])
            elif op == "ArgMax":
                out = args[0].argmax(axis=attr["axis"], keepdims=bool(attr["keepdims"])).astype(np.int64)
            else:
                raise ValueError(f"unsupported FP32 reference operator {op}")
            values[dest] = out
        result = {name: values[name] for name in self.outputs}
        for name, arr in result.items():
            if arr.dtype != self.outputs[name]["dtype"] or arr.shape != tuple(self.outputs[name]["shape"]):
                raise ValueError(f"{name}: FP32 result disagrees with declared dtype or shape")
        return result
