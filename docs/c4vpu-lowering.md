# Phase C4-VPU lowering on Z15 Balanced

The optimized Z15 pipeline selects native vector instructions by default.
`--vpu-lowering off` retains the GPU implementation of the new fusions.
The A0 and A1 baselines leave these fusions disabled. Use
`--vpu-attention exact|fast` to select the native reciprocal operation.

Whole attention fusion recognizes rank-2 signed Q, K and V, a private
MatMul/requantization, optional causal mask, Q/DQ Softmax and
MatMul/requantization to a signed byte. The supported path uses zero-centered
operands, probability scale 1/256, compact PV with shift 8, aligned row
widths, at most 128 keys, and an affine causal prefix. A dynamic V is
transposed to the native column-major layout. The scratch planner reserves
the six ordered ATTN operands within one ownership half. Supported
standalone Q/DQ Softmax uses the native vector reduction on int32 scores.

The importer recognizes both a last-axis RMSNormalization and the
square/mean/epsilon/sqrt/divide/gamma decomposition inside a static Q/DQ
island. The hardware and GPU fallback use the same fixed-point reciprocal
square-root convention. For SwiGLU, a quantized `x * Sigmoid(x)` or SiLU
uses the quantizer-generated 256-entry activation table. A private,
zero-centered signed multiply with a supported requantization uses EW_MUL.
Equivalent saturating signed residual Add uses EW_ADD. Unsupported layouts
retain the existing portable routes where the GPU provides a lowering.

The native base-two attention approximation and compact PV rounding change
some outputs relative to the GPU Q23 and FP32 paths. Scaled EW_MUL rounds
half up. The exact RTL gate uses the selected fixed-point contract, and
the accuracy ledger measures differences on the same workload inputs.

```sh
python3 -m hetero_compiler \
  --graph examples/c4vpu/TransformerLayer_INT8_C4VPU.onnx \
  --profile fpga_zu15eg_balanced --pipeline-tiles --pipeline-scheduler bank-aware \
  --vpu-lowering on --vpu-attention exact --out /dev/shm/ysyx_compiler_c5/layer
```

See [the native runtime format](yhb-format.md) for the Step 2 MMIO and
DMA requirements.

Use the causal KV entry point to run the compound native ATTN with a
resident row-major K/V state. The fixed-pitch packed V cache updates new
columns using native strided transpose. The recompute comparator updates
the full prefix during decode. Absolute causal positions exclude padded
and future keys. The published K/V outputs retain logical unpadded shapes.

```sh
python3 -m hetero_compiler.kv_schedule --graph examples/c4a_kv/causal_kv32.json   --vpu-lowering on --vpu-attention exact --out /dev/shm/ysyx_compiler_c5/kv
```
