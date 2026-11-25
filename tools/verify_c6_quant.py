"""Verify Windows-exported C6 INT8 and PL boundary vectors with NumPy only."""
import argparse
import json
from pathlib import Path

import numpy as np

from hetero_compiler.c6_quant import QuantGraph


def verify(root):
    runtime = QuantGraph(root)
    vectors = root / "vectors"
    contracts = json.loads((vectors / "pl_outputs.json").read_text(encoding="utf-8"))
    count = 0
    for rgb_path in sorted(vectors.glob("*.rgb_nhwc_u8.bin")):
        stem = rgb_path.name.removesuffix(".rgb_nhwc_u8.bin")
        spec = runtime.tensors[runtime.graph["inputs"][0]]
        rgb = np.fromfile(rgb_path,dtype=np.uint8).reshape(spec["physical_shape"])
        expected_input = ((rgb.astype(np.int32)*127+127)//255).astype(np.int8)
        actual_input = np.fromfile(vectors/(stem+".input_nhwc_s8.bin"),dtype=np.int8).reshape(rgb.shape)
        np.testing.assert_array_equal(expected_input, actual_input, err_msg=stem + " RGB quantization")
        verified_outputs = set()

        def capture(node, name, value):
            if name not in contracts:
                return
            row = contracts[name]
            path = vectors / (stem + row["suffix"])
            got = value.transpose(0,2,3,1) if row["layout"] == "NHWC" else value
            raw = np.fromfile(path,dtype=row["dtype"]).reshape(row["physical_shape"])
            np.testing.assert_array_equal(got,raw,err_msg=stem + " " + name)
            verified_outputs.add(name)

        inputs = rgb.transpose(0,3,1,2).astype(np.float32) / np.float32(255)
        out = next(iter(runtime.infer(inputs,capture=capture).values())).reshape(-1,6)
        if verified_outputs != contracts.keys():
            raise ValueError("missing PL boundary outputs")
        for suffix, values, dtype in [
            (".coords_q16_16.bin",np.rint(out[:,:4].astype(np.float64)*65536),"<i4"),
            (".scores_q16_16.bin",np.rint(out[:,4].astype(np.float64)*65536),"<i4"),
            (".classes_i64.bin",out[:,5],"<i8"),
        ]:
            expected = np.fromfile(vectors/(stem+suffix),dtype=dtype).reshape(values.shape)
            np.testing.assert_array_equal(values.astype(dtype),expected,err_msg=stem + suffix)
        count += 1
        print(stem, f"{len(verified_outputs)} PL heads, RGB input, Q16.16 boxes/scores and classes byte-exact")
    if count < 4:
        raise ValueError("four captured scenes required")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact",type=Path)
    verify(parser.parse_args().artifact)
