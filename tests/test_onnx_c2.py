from hashlib import sha256
from pathlib import Path
import struct
import unittest

from hetero_compiler.ir import normalize, GraphError
from hetero_compiler.onnx import parse_model
from hetero_compiler.reference import evaluate
from test_frontends import field, info


def scalar(name, dtype, data):
    return field(2, dtype) + field(8, name) + field(9, data)


def node(op, ins, outs, *, attrs=(), domain=""):
    return (b"".join(field(1, x) for x in ins) + b"".join(field(2, x) for x in outs) +
            field(4, op) + b"".join(field(5, x) for x in attrs) +
            (field(7, domain) if domain else b""))


def model(nodes, inputs, outputs, initializers=(), opset=18):
    graph = b"".join(field(1, n) for n in nodes)
    graph += b"".join(field(11, x) for x in inputs) + b"".join(field(12, y) for y in outputs)
    graph += b"".join(field(5, v) for v in initializers)
    return (field(1, 8) + field(7, graph) + field(8, field(2, opset)) +
            field(8, field(1, "com.microsoft") + field(2, 1)) +
            field(8, field(1, "com.ysyx") + field(2, 1)))


class QuantizedOnnx(unittest.TestCase):
    def test_qlinear_matmul_ties_and_nonzero_points(self):
        names = ["x", "xs", "xz", "w", "ws", "wz", "ys", "yz"]
        w = field(1, b"\x01\x01") + scalar("w", 3, b"\x05")
        initializers = [w] + [scalar(n, 1, struct.pack("<f", v)) for n, v in
                             [("xs", 0.5), ("ws", 0.5), ("ys", 1.0)]]
        initializers += [scalar(n, 3, bytes([v])) for n, v in [("xz", 2), ("wz", 3), ("yz", 7)]]
        parsed = normalize(parse_model(model([node("QLinearMatMul", names, ["y"])],
            [info("x", [4, 1])], [info("y", [4, 1])], initializers)))
        self.assertEqual(evaluate(parsed, {"x": [-1, 1, 3, 5]})["y"], [5, 7, 7, 9])

    def test_conv_integer_uses_nchw_and_zero_point_for_padding(self):
        w = field(1, b"\x01\x01\x01\x02") + scalar("w", 3, b"\x02\xff")
        xz = scalar("xz", 2, b"\x05")
        pad = field(1, "pads") + field(8, b"\x00\x01\x00\x00") + field(20, 7)
        parsed = normalize(parse_model(model([node("ConvInteger", ["x", "w", "xz"], ["y"], attrs=[pad])],
            [info("x", [1, 1, 1, 3], 2)], [info("y", [1, 1, 1, 3], 6)], [w, xz])))
        self.assertEqual(evaluate(parsed, {"x": [8, 7, 9]})["y"], [-3, 4, 0])

    def test_split_transpose_concat_reshape_argmax(self):
        split = field(1, b"\x02") + scalar("parts", 7, struct.pack("<2q", 1, 1))
        shape = field(1, b"\x02") + scalar("shape", 7, struct.pack("<2q", 3, 2))
        axis = lambda n: field(1, "axis") + field(3, n) + field(20, 2)
        keep = field(1, "keepdims") + field(3, 0) + field(20, 2)
        parsed = normalize(parse_model(model([
            node("Split", ["x", "parts"], ["a", "b"], attrs=[axis(0)]),
            node("Concat", ["b", "a"], ["c"], attrs=[axis(0)]),
            node("Transpose", ["c"], ["t"]),
            node("Reshape", ["t", "shape"], ["r"]),
            node("ArgMax", ["r"], ["y"], attrs=[axis(1), keep]),
        ], [info("x", [2, 3])], [info("y", [3], 7)], [split, shape])))
        self.assertEqual(parsed["tensors"]["y"]["dtype"], "int64")
        self.assertEqual(evaluate(parsed, {"x": [7, 2, 5, 3, 8, 5]})["y"], [1, 0, 0])

    def test_node_specific_error_for_unsupported_conv_group(self):
        g = field(1, "group") + field(3, 2) + field(20, 2)
        w = field(1, b"\x01\x01\x01\x01") + scalar("w", 3, b"\x01")
        with self.assertRaisesRegex(GraphError, "onnx_0.*group"):
            parse_model(model([node("ConvInteger", ["x", "w"], ["y"], attrs=[g])],
                              [info("x", [1, 1, 1, 1])], [info("y", [1, 1, 1, 1], 6)], [w]))


class QuantizationIslands(unittest.TestCase):
    def test_qlinear_add_uses_single_round_with_asymmetric_zero_points(self):
        initializers = [scalar(n, 1, struct.pack("<f", v)) for n, v in
                        [("sx", 0.5), ("sb", 0.25), ("sy", 0.5)]]
        initializers += [scalar(n, 3, struct.pack("<b", v)) for n, v in
                         [("xz", -1), ("bz", 1), ("yz", 3)]]
        graph = model([node("QLinearAdd", ["x", "sx", "xz", "b", "sb", "bz", "sy", "yz"],
                            ["y"], domain="com.microsoft")],
                      [info("x", [3]), info("b", [3])], [info("y", [3])], initializers)
        self.assertEqual(evaluate(normalize(parse_model(graph)), {"x": [0, 2, 3], "b": [1, -2, 0]})["y"],
                         [4, 5, 7])

    def test_quantized_float_conv_with_static_bias(self):
        initializers = [scalar(n, 1, struct.pack("<f", v)) for n, v in
                        [("xs", 0.25), ("ws", 0.5), ("ys", 0.25)]]
        initializers += [scalar("z", 3, b"\x00"),
                         field(1, b"\x01\x01\x01\x01") + scalar("w", 3, b"\x02"),
                         field(1, b"\x01") + scalar("bias", 1, struct.pack("<f", 0.25))]
        graph = model([
            node("DequantizeLinear", ["x", "xs", "z"], ["xf"]),
            node("DequantizeLinear", ["w", "ws", "z"], ["wf"]),
            node("Conv", ["xf", "wf", "bias"], ["yf"]),
            node("QuantizeLinear", ["yf", "ys", "z"], ["y"]),
        ], [info("x", [1, 1, 1, 3])], [info("y", [1, 1, 1, 3])], initializers)
        imported = normalize(parse_model(graph))
        self.assertEqual(evaluate(imported, {"x": [-3, 1, 5]})["y"], [-2, 2, 6])
        self.assertEqual([n["op"] for n in imported["nodes"]], ["Transpose", "Conv", "Requant", "Transpose"])

    def test_standard_sigmoid_gelu_silu_quantized_islands(self):
        initializers = [scalar("q4", 1, struct.pack("<f", 1/16)), scalar("z", 3, b"\x00")]
        graph = model([
            node("DequantizeLinear", ["x", "q4", "z"], ["f"]),
            node("Sigmoid", ["f"], ["sig"]),
            node("Gelu", ["f"], ["gel"]),
            node("Mul", ["sig", "f"], ["silu"]),
            *[node("QuantizeLinear", [op, "q4", "z"], [op + "_q"]) for op in ("sig", "gel", "silu")],
        ], [info("x", [5])], [info(op + "_q", [5]) for op in ("sig", "gel", "silu")], initializers, opset=20)
        imported = normalize(parse_model(graph))
        result = evaluate(imported, {"x": [-32, -16, 0, 16, 32]})
        self.assertEqual([n["op"] for n in imported["nodes"]], ["Lookup", "Lookup", "Lookup"])
        self.assertEqual([result[n] for n in imported["outputs"]],
                         [[2, 4, 8, 12, 14], [-1, -3, 0, 13, 31], [-4, -4, 0, 12, 28]])

    def test_standard_softmax_quantized_island(self):
        initializers = [scalar("q4", 1, struct.pack("<f", 1/16)),
                        scalar("q8", 1, struct.pack("<f", 1/256)), scalar("z", 3, b"\x00")]
        graph = model([
            node("DequantizeLinear", ["x", "q4", "z"], ["f"]),
            node("Softmax", ["f"], ["s"]),
            node("QuantizeLinear", ["s", "q8"], ["y"]),
        ], [info("x", [2, 2])], [info("y", [2, 2], 2)], initializers)
        self.assertEqual(evaluate(normalize(parse_model(graph)), {"x": [0, 0, 16, -16]})["y"], [128, 128, 225, 31])

    def test_legacy_upsample_and_slice(self):
        scales = field(1, b"\x02") + scalar("scales", 1, struct.pack("<2f", 1, 2))
        graph = model([node("Upsample", ["x", "scales"], ["y"])], [info("x", [1, 2])],
                      [info("y", [1, 4])], [scales], opset=9)
        self.assertEqual(evaluate(normalize(parse_model(graph)), {"x": [9, -3]})["y"], [9, 9, -3, -3])
        v = lambda name, value: field(1, b"\x01") + scalar(name, 7, struct.pack("<q", value))
        graph = model([node("Slice", ["x", "start", "end", "axis", "step"], ["y"])],
                      [info("x", [1, 6])], [info("y", [1, 2])],
                      [v("start", -5), v("end", 5), v("axis", -1), v("step", 2)])
        self.assertEqual(evaluate(normalize(parse_model(graph)), {"x": [9, 7, 5, 3, 1, -1]})["y"], [7, 3])


class EndToEndBenchmarks(unittest.TestCase):
    def test_reference_benchmarks_with_runtime_overrides(self):
        from examples.generate_quantized_onnx import image_values, token_values, transformer, vision
        identities = [
            ("yolo_block_int8", vision, "image", image_values, {
                0: ["d9df05ae8480dbee88e6a10b07cd78e5cbc160d6b99202c43e462d3e5213f9aa",
                    "e59a54937bce700deac456da537b77aa6704b10b85764e06779149f44fd4a834"],
                9: ["e17036c6b6b721f7b9f5237bf9bad982fe7f60818911d1096f3214d8e9bd70b2",
                    "e7dde13717263cccb0f91cf49c7634bdafd2863d83f969798a0d32b83ebed4e2"],
            }),
            ("transformer_layer_int8", transformer, "tokens", token_values, {
                0: ["1895f9df6b12cc3801eed8c28420d7ef0dd3262caa0257caf3d9effec716d0d9",
                    "e756ad5fc07858108e8f5825b1865bfe42b2a1a9c948951ba65a2814f1f43d19"],
                9: ["66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925",
                    "eb7785a509b5fac630569edfe8f308f19982d9869c36c24e5c612a917fbc1f1c"],
            }),
        ]
        examples = Path(__file__).resolve().parents[1] / "examples"
        for name, generate, source, values, expected in identities:
            data = (examples / (name + ".onnx")).read_bytes()
            self.assertEqual(data, generate())
            imported = normalize(parse_model(data))
            self.assertEqual(imported["inputs"], [source])
            for seed, golden in expected.items():
                with self.subTest(model=name, seed=seed):
                    results = evaluate(imported, {source: values(seed)})
                    digests = []
                    for out in imported["outputs"]:
                        code = "q" if imported["tensors"][out]["dtype"] == "int64" else "b"
                        payload = struct.pack(f"<{len(results[out])}{code}", *results[out])
                        digests.append(sha256(payload).hexdigest())
                    self.assertEqual(digests, golden)


if __name__ == "__main__":
    unittest.main()
