# Phase C2 static quantized ONNX subset

The importer reads protobuf wire data using Python's standard library. Source
opset and node attributes determine the mathematical operation. Lowering to a
device happens after this import. The importer attaches source node names to
diagnostics and rejects symbolic shapes, unsupported domains, sparse/external
initializers, or runtime quantization parameters.

| Source operators | Import path and current scope |
| --- | --- |
| MatMulInteger, ConvInteger | Rank-2 matrices or 2-D NCHW/OIHW convolution, scalar zero points, int32 accumulators. |
| QLinearMatMul, QLinearConv | Per-tensor static parameters and optional int32 Conv bias. Requantization uses exact static rational scales, nearest-even when needed, and byte saturation. |
| QLinearAdd (com.microsoft) | Static scalar parameters, broadcast, common integer denominator and a single nearest-even output round. |
| DequantizeLinear–Conv–QuantizeLinear | Static dyadic scales within the exact float32 mantissa envelope. Optional float bias must be exactly representable at the accumulator scale. |
| Add, Mul, Relu | Integer data path. NumPy broadcasting for Add/Mul becomes explicit Expand. |
| Sigmoid, Gelu, Sigmoid×input | Static scalar Q/DQ boundaries with dyadic scales become a per-byte integer lookup table. `com.ysyx` also declares fixed-Q4 Sigmoid, SiLU and GELU. |
| DequantizeLinear–Softmax–QuantizeLinear | Reference evaluation over static row dimensions and Q/DQ scales. No native backend currently binds this form. |
| Softmax, RMSNorm (com.ysyx) | C1 fixed-point row algorithms. The causal variant masks columns beyond the current query. |
| MaxPool | 2-D sliding windows, integer activations, floor output sizing and value output. |
| Resize, Upsample | Static nearest-neighbor transforms using asymmetric coordinates and floor indices. Explicit scales must produce exact integral extents. |
| Concat, Split, Slice | Static axis and nonempty intervals. Slice steps are positive. Split becomes one Slice per output. |
| Reshape, Transpose, Cast, ArgMax | Static shape transformations and integer casts. ArgMax evaluates the requested tie rule and writes physical int64 indices. |

The core graph uses positive, static ranks 1 through 4. Scalar scale and
zero-point initializers are supported at import. Grouped or dilated convolution,
per-axis quantization, and arbitrary floating-point subgraphs remain outside this
subset. A quantization ratio is accepted when its numerator/denominator and the
worst-case product fit the exact int32 envelope. Backend eligibility is checked
separately on the normalized graph. The canonical Requant operation uses a
signed wrapping 32-bit numerator and zero-point addition. The ONNX importer
proves a conservative envelope that keeps accepted ONNX quantization within
that range. An explicit divisor takes priority over the power-of-two shift.

## Fixtures

Run from the repository root to reproduce the binary files.

```shell
nice -n 15 ionice -c 3 python3 -B examples/generate_quantized_onnx.py
nice -n 15 ionice -c 3 python3 -B -m unittest discover -s tests -p test_onnx_c2.py
```

- `yolo_block_int8.onnx` includes a 3×3 and a 1×1 quantized convolution, bias,
  split/reorder/concat, residual, SiLU, 5×5 max pooling, nearest ×2 upsampling and
  per-pixel class ArgMax. NCHW activations are explicitly transformed to NHWC
  at the convolution boundary. Static weights are transposed at import.
- `transformer_layer_int8.onnx` includes two quantized RMSNorms, learned-style
  (deterministically generated) Q/K/V projections, dynamic V, causal row
  softmax, output projection and residual, SwiGLU, a second residual, and a
  vocabulary projection with ArgMax.

Both are deterministic end-to-end benchmark blocks with synthetic weights and
runtime-overridable input defaults. Seeds 0 and 9 change both logits and index
outputs in each block. Reproduction requires no ONNX package.
