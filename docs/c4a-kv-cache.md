# Phase C4-A single-sequence KV cache

The causal IR fixture is `examples/c4a_kv/causal_kv32.json`. It describes
pin-compatible static geometry, Q/K/V, output projection, two FFN projections
with ReLU, residual paths, a vocabulary projection and greedy token selection.
The example has 32 prefix positions, four prefill Tiles of eight positions
and four decode steps. The sequence shares unrestricted causal history across
all Tile boundaries. The fixture uses deterministic synthetic weights and
inputs for schedule validation.

The planner reserves disjoint K and V regions and carves separate transient
lifetime-colored buffers for even and odd Tiles. Every cached Tile or step
writes only its new rows and scores against the complete prefix. Q/K/V and the
other fixed-weight projections use the native NPU, and QK, Q23 Softmax, PV and
ArgMax use the GPU. The mask uses absolute query position within the sequence.
The attention convention is signed int8 Q/K/V and unsigned probabilities
quantized to 1/128 using nearest-even rounding. The NumPy oracle computes the
entire prefill as one matrix and uses Decimal-generated exponent weights
independent of the device code generator.

Actual GPU token selection writes a DDR feedback symbol via DMA, and every
decode step reads that slot before creating its embedding. The benchmark CPU
only compares output bytes; it does not select or inject the generated
tokens. All logits and tokens have distinct DDR outputs. The C99 checkpointed
entry points can optionally replace `token_feedback` between steps with a
little-endian signed 64-bit vocabulary index in 0 through 3. The original
prompt and fixed weights are loaded before prefill and remain available.

The baseline uses the same initial Tile sequence and the same attention
and output path. Its four decode steps intentionally reproject K and V over
their complete prefix instead of appending new rows. It holds previous token
embeddings in resident scratch, so the comparison isolates reprojection cost
and startup, feedback and result DMA bytes are equal. Both modes reserve the
same address budget for valid fixed-size K/V buffers. In the baseline the
K/V workspace is rewritten completely between steps. GPU memory operands
share page 0 and wait on the frozen Half-0 ownership fence. The transient bank
alternation in this workload is for lifetime reuse; the streaming benchmark
measures concurrent physical half use.

Compile a checkpointed YHB without running the simulator:

```sh
python3 -m hetero_compiler.kv_schedule --graph examples/c4a_kv/causal_kv32.json --out kv-bundle
```

The frozen lowering supports width four, prefill lengths 32 and 64, tiles of
four or eight positions, and four through eight decode steps.

Reproduce the byte-exact MMIO comparison with:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 10 python3 -B -m tools.run_c4a_kv --out /dev/shm/ysyx_compiler_c4a/kv/replay
```

The transcript contains frozen RTL wall ticks and actual AXI payload bytes.
The CSV and JSON retain the planned SRAM footprint and projected K/V row
counts as separate fields. Folded latency uses the Phase 12C NPU-domain
164.555 MHz Fmax.
