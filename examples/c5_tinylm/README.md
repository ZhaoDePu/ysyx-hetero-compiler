# TinyLM-Chat

The fixture trains a two-layer causal word Transformer with learned
embeddings and LM head, two attention heads per layer, width eight,
RMSNorm and SwiGLU. Adam trains all weights against next-token labels
for 240 deterministic NumPy steps. Weight bounds limit activation
outliers before symmetric dyadic INT8 PTQ. The corpus contains five
authored technical sentences. The three published prompts are prefixes
from that corpus and demonstrate deterministic continuation within
this small vocabulary of 64 IDs. The training accuracy and RTL
generation agreement apply to that closed corpus.

`TinyLM-Chat.onnx` is standard static ONNX. It uses Gather, integer
matrix multiplication, explicit DequantizeLinear and QuantizeLinear,
a causal mask and standard floating islands for RMSNorm, Softmax and
SwiGLU. Export uses the learned parameters in `weights.fp32.npz`.

The specialized compiler imports and checks the ONNX graph, then
rewrites the static causal DAG into one-token tiles. Four heads each
have independent row-major K and V resident in scratch. Each token
projects one new row in each head. Native attention additionally uses
EW9 to append to a fixed-pitch packed V buffer. The predicted int64
token passes directly from ArgMax to the next device Gather within
the same scratch allocation. The host supplies four initial IDs once.
Checkpoint zero exposes the prefill prediction. The next eight
checkpoints expose the tokens produced by incremental decode.

- A0 uses only GPU. All stages are the same with VPU on or off.
- A1 places static integer matmuls on the NPU. The on configuration
  also uses saturating native residual Add.
- A2 adds interval-based transient reuse and fuses the GPU score/PV
  requantizations, or fuses the attention chain with native ATTN
  when VPU is enabled. Native mode also lowers RMSNorm and SwiGLU.
- A3 schedules the independent requests across opposite ownership
  halves while retaining checkpoints between successive tokens.
- Off disables all VPU and ATTN commands in this fixture. Each mode
  is checked against its own fixed-point full-causal graph evaluation.

Per-lane scratch reservation counts the allocated address span
inside that lane's physical 4 KiB page. Live peak conservatively sums
all resident arenas and the largest active transient footprint per
scheduled half. Reported multi-request ms/token and tokens/s are
amortized across the three independent prompts. Single-request
performance is available from the deployment replay.

Reproduce the learned graph and hardware replay on the isolated host:

```sh
export TMPDIR=/dev/shm/ysyx_compiler_c5
export PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
learned="$TMPDIR/lm-checkpoint"
nice -n 10 python3 -B -m examples.c5_tinylm.train --out "$learned"
nice -n 10 python3 -B -m examples.c5_tinylm.export \
    --weights "$learned" --out "$learned/TinyLM-Chat.onnx"
nice -n 10 python3 -B -m tools.run_c5_tinylm \
    --learned "$learned" --out "$TMPDIR/lm-replay"
nice -n 10 python3 -B -m tools.run_c5_tinylm \
    --learned "$learned" --out "$TMPDIR/lm-board" --kit
```

The RTL runner requires the compiler's frozen Step 2 Verilator build.
The two latency projections divide the same observed NPU wall
counter by 164.555 MHz and 125 MHz.
