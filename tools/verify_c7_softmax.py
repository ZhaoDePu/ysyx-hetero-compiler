"""Measure both C7 Softmax kernels on scores captured from a fixed 320 frame."""

import argparse
from pathlib import Path

import numpy as np

from hetero_compiler.c6_attention import lower_matmul, lower_q23_softmax, lower_vpu_softmax_words
from hetero_compiler.c6_quant import QuantGraph
from hetero_compiler.c7_softmax import log_softmax, log_softmax_tables, lower_log_softmax
from tools.prepare_c7_softmax import apply_modes
from tools.verify_c6_attention import replay


def compare(quant, runner, output, backend, layer, prepared=None):
    if prepared is not None and backend != "vpu-raw":
        raise ValueError("the prepared calibrated graph is for the native raw path")
    source = QuantGraph(prepared or quant)
    op = "VpuSoftmax" if prepared is not None else "Q23Softmax"
    nodes = [node for node in source.graph["nodes"] if node["op"] == op]
    score_name = nodes[layer]["inputs"][0]
    values = {}
    expected_names = {score_name}
    if backend == "vpu-raw":
        input_qk, = (n["inputs"] for n in source.graph["nodes"] if n["outputs"] == [score_name])
        output_softmax, = nodes[layer]["outputs"]
        expected_names.update((*input_qk, output_softmax))
        if prepared is None:
            apply_modes(source.graph, ("vpu-raw", "log8") if layer == 0 else ("log8", "vpu-raw"))

    def capture(node, name, array):
        if name in expected_names:
            values[name] = array.copy()

    path = quant / "vectors/bus.rgb_nhwc_u8.bin"
    image = np.fromfile(path, dtype=np.uint8).reshape(1, 320, 320, 3)
    source.infer(image.transpose(0, 3, 1, 2).astype(np.float32) / np.float32(255), capture=capture)
    scores = values[score_name]
    scale = source.tensors[score_name]["scale"]
    if backend == "vpu-raw":
        query, key = (values[name] for name in input_qk)
        replay(output / "dot", runner, {"query": query, "key": key}, scores,
               lambda at: lower_matmul("qk-raw", at["query"], at["key"], at["result"],
                                       query.shape, key.shape, 1, 0, raw_output=True))
        replay(output / "norm", runner, {"source": scores}, values[output_softmax],
               lambda at: lower_vpu_softmax_words(nodes[layer]["id"], at["source"], at["result"],
                                                  scores.shape, nodes[layer]["attrs"]["shift"]))
        return
    weights = np.asarray(nodes[layer]["attrs"]["lut"], dtype="<u4")
    if backend == "log8":
        tables = log_softmax_tables(weights, scale, 8, scores.shape[-1])
        expected = log_softmax(scores, -1, weights, scale, 8)
        packed = np.frombuffer(tables.payload, dtype="uint8")
        replay(output, runner, {"source": scores, "lut": packed}, expected,
               lambda at: lower_log_softmax(nodes[layer]["id"], at["source"], at["result"],
                                            at["lut"], scores.shape, tables))
    elif backend == "q23":
        weight = weights[scores.max(-1, keepdims=True).astype(np.int16) - scores].astype(np.uint64)
        denominator = weight.sum(-1, keepdims=True)
        q, r = np.divmod(256 * weight, denominator)
        expected = np.minimum(255, q + ((2*r > denominator)
                              | ((2*r == denominator) & (q & 1) != 0))).astype(np.uint8)
        replay(output, runner, {"source": scores, "lut": weights}, expected,
               lambda at: lower_q23_softmax(nodes[layer]["id"], at["source"], at["result"],
                                            at["lut"], scores.shape))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quant", type=Path, required=True)
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", choices=("log8", "q23", "vpu-raw"), required=True)
    p.add_argument("--layer", choices=(0, 1), type=int, required=True)
    p.add_argument("--prepared", type=Path, help="capture native operands from the final calibrated graph")
    args = p.parse_args()
    compare(args.quant, args.runner, args.output, args.case, args.layer, args.prepared)


if __name__ == "__main__":
    main()
