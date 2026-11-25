# C4-VPU ONNX and frozen RTL benchmarks

`TransformerLayer_INT8_C4VPU.onnx` contains four independent tiles of four
tokens each. Each tile implements two RMSNorm blocks with explicit
`Mul → ReduceMean → Add → Sqrt → Div → Mul`, causal Q/DQ attention, SwiGLU
using `Sigmoid(x) × x` and gated multiplication, residual adds, and a
vocabulary head with ArgMax. The model uses standard ONNX operators with
static inputs and initializer weights. Activation and gamma scales use Q4,
attention probabilities use unsigned units of 1/256, and RMS epsilon is
2⁻¹⁰. Regenerate it with `python3 -B -m examples.c4vpu.transformer`.

`tools.run_c4vpu_eval` measures this model, the C4-A 24×48 causal attention
graph, and the C4-A single-sequence KV case. A0 runs the GPU only. A1 adds
heterogeneous placement. A2 enables operator fusion and memory reuse. A3 also
enables the two-half scheduler. Hardware VPU lowering is disabled in A0 and
A1 of both ladders to isolate the fusion change. The KV benchmark crosses
the VPU setting with cached append and full-prefix recompute.

```sh
export PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -B -m tools.run_c4vpu_eval \
  --out /dev/shm/ysyx_compiler_c4vpu/bench/release
```

The runner checks each output in frozen RTL against the fixed-point oracle
for its lowering mode. `performance.csv` and `performance.json` retain
raw AXI payload counts, SRAM plan sizes, and RTL occupancy counters. Cycle
conversions at 164.555 MHz and 125.00 MHz use the same measured NPU-domain
tick count. `accuracy.csv` and `accuracy.json` further compare the published
GPU and VPU bytes and an independent NumPy FP32 reference. The FP32 graph
uses the original model weights, epsilon and causal mask, and excludes
intermediate integer narrowing. It generates KV decode tokens from its own
previous predictions. Attention and layer errors are in dequantized Q4
units. The KV model uses continuous code units. Token agreements are
computed separately from numeric errors. Each case has deterministic
synthetic parameters and inputs.
