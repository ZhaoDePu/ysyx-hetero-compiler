# C3 standard ONNX Q/DQ Softmax on frozen RTL

The importer recognizes standard-domain ONNX opset 13 and later graphs
containing DequantizeLinear → Softmax → QuantizeLinear. Optional Add
between dequantization and Softmax accepts a static float32 causal plane
over the last two dimensions. A leading singleton dimension may broadcast
this plane. Each row exposes a nonempty key prefix, growing by one per
query up to the key count. A blocked entry can be negative infinity or a
negative sentinel that remains at least 128 below the stable row maximum
for all byte logits after float32 rounding. An arbitrary additive mask
requires another implementation.

The lowered QSoftmax operation accepts int8 or uint8 logits, scalar
power-of-two scales, and at most 255 keys. The output scale is 1/2^b,
where 0 ≤ b ≤ 16. Unmasked reductions can use any axis of the static
rank-1 to rank-4 tensor. A causal reduction uses the final axis.
The compiler keeps the existing floating reference route for scales and
row lengths outside the GPU arithmetic contract.

## Arithmetic and hardware boundary

The input zero point cancels when the per-row maximum is subtracted.
The compiler generates a 256-entry integer exponential table using
round-to-nearest-even on 2^23 × exp(−input_scale × delta). The RV32
GPU kernel reads the quantized logits, excludes blocked columns,
accumulates Q23 weights, and performs exact integer round-to-nearest-even
division. The denominator is at most 255 × 2^23, so the doubled
remainder fits in unsigned 32 bits. The kernel adds the output zero point
and saturates to the requested byte type. Blocked positions use the zero
point.

The frozen VPU row Softmax implements a distinct exponential and
normalization contract used by C1/C2. C3 uses the frozen GPU's existing
RV32 instructions for standard Q/DQ Softmax. The Python fixed-point
oracle evaluates the exponential table independently using Decimal and
uses unrestricted integer quotient/remainder for normalization.

For n included keys, the maximum stable exponential is exactly 1.
Each other Q23 exponent is within 2^-24 of its real value. Before output
rounding, the probability error due to this table is at most
(n + 1) × 2^-24. At output scale 1/256 and n ≤ 255 the bound is
1/256 of one output count. Near a half-count boundary, a fixed-point
result can round differently from an FP32 implementation. RTL byte
comparison uses the Q23 oracle. FP32-vs-quantized model accuracy and
tensor drift are evaluated independently against the trained model.

## Reproduction

After building the frozen Balanced Verilator runner, execute:

```sh
nice -n 10 ionice -c2 -n7 python3 -B -m tools.run_c3_qsoftmax
```

The replay compares the complete aligned DMA output region, including
padding, for causal masks, a finite blocked sentinel, a strided
nonlast-axis uint8 reduction, nearest-even ties, and a tiny input scale.
