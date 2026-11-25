import unittest
from hetero_compiler.ir import GraphError, normalize
from hetero_compiler.onnx import parse_model
from hetero_compiler.profile import load_profile


def varint(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def field(index, value):
    if isinstance(value, int):
        return varint(index << 3) + varint(value)
    if isinstance(value, str):
        value = value.encode()
    return varint((index << 3) | 2) + varint(len(value)) + value


def info(name, dims, dtype=3):
    shape = b"".join(field(1, field(1, d)) for d in dims)
    return field(1, name) + field(2, field(1, field(1, dtype) + field(2, shape)))


class Frontends(unittest.TestCase):
    def test_legacy_and_toposort(self):
        g = normalize({
            "name": "legacy", "memory_mode": "arena4k",
            "tensors": {"x": {"shape": [1, 2], "dtype": "int8", "data": [1, 2]},
                        "w": {"shape": [2, 1], "dtype": "int8", "data": [3, 4]}},
            "nodes": [{"name": "relu", "op": "ReLU", "inputs": ["h"], "output": "y"},
                      {"name": "mat", "op": "MatMul", "inputs": ["x", "w"], "output": "h"}],
            "outputs": ["y"],
        })
        self.assertEqual([n["op"] for n in g["nodes"]], ["Gemm", "Relu"])
        self.assertEqual(g["nodes"][0]["attrs"]["narrowing"], "wrap")
        self.assertEqual(g["tensors"]["y"]["shape"], [1, 1])

    def test_integer_onnx_matmul(self):
        node = field(1, "x") + field(1, "w") + field(2, "y") + field(4, "MatMulInteger")
        w = field(1, b"\x02\x01") + field(2, 3) + field(8, "w") + field(9, b"\x03\xfe")
        graph = field(1, node) + field(5, w) + field(11, info("x", [1, 2]))
        graph += field(12, info("y", [1, 1], 6))
        model = field(1, 8) + field(7, graph) + field(8, field(2, 13))
        parsed = normalize(parse_model(model))
        self.assertEqual(parsed["tensors"]["w"]["data"], [3, -2])
        self.assertEqual(parsed["tensors"]["y"]["dtype"], "int32")
        self.assertEqual(parsed["nodes"][0]["op"], "Gemm")

    def test_cycle_and_truncated_onnx(self):
        with self.assertRaisesRegex(GraphError, "cycle"):
            normalize({"tensors": {}, "outputs": ["x"], "nodes": [
                {"id": "a", "op": "Relu", "inputs": ["x"], "output": "x"}]})
        with self.assertRaisesRegex(GraphError, "truncated"):
            parse_model(b"\x3a\x07\x12")

    def test_profile_capacities(self):
        expected = {"fpga_zu15eg_balanced": (16, 16, 4, 2 << 20),
                    "fpga_zu15eg_full": (24, 8, 8, 2 << 20),
                    "asic_nangate45_p8": (4, 8, 1, 4096),
                    "asic_nangate45_ip": (4, 8, 1, 65536)}
        for name, spec in expected.items():
            p = load_profile(name)
            self.assertEqual((p["npu"]["rows"], p["gpu"]["lanes"], p["gpu"]["warps"],
                              p["sram"]["capacity"]), spec)


if __name__ == "__main__":
    unittest.main()
