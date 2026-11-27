# Valence C15 Phase 18 CMDQ

The runtime keeps the public C99 ABI and the YHB1 format. Phase 17 YOLO graphs
continue to use the existing YHP1 marker and explicit p17 target. Select the
execution mode with an optional CLI argument.

| Mode | Behavior |
| --- | --- |
| auto | Use CMDQ for EXT_VERSION 0x0c0c0200 with CAP_FLAGS bit 4. Otherwise use legacy MMIO. |
| cmdq | Require CMDQ. An unavailable capability or a submission failure interrupts the invocation. |
| mmio | Use the existing MMIO sequence. This mode also retains per-stage trace output. |

The linked C99 runtime uses the same setting from YSYX_EXEC_MODE, which defaults
to auto. Use --exec-stats on the CLI to write dispatch counts to stderr. The
counts describe the dispatch path. Graph upload, preflight and output reads
have their own measurements. Polling reads depend on host pacing. The
Verilator HAL advances on AXI accesses and polls without wall-clock sleeps.

## Command queue lifetime

The recorder starts from the command sequence validated by the YHB loader.
It emits little-endian 64-bit commands and fences and places their DDR regions
above the graph's initial images, symbols and DMA extents. Each region
reserves full 32-byte fetch beats. The max-count split and each GPU image
switch drain existing engines and append WAIT_ALL_IDLE and END_BATCH. ARM
alone uploads GPU instruction images while the GPU is idle. Tile scale and
activation commits use the Turbo direct tables. The recorder omits redundant
writes of unchanged entries within the program.

A resident session owns its DDR placement and GPU instruction state for its
lifetime. The first frame uploads constants and the queue. Subsequent frames
transfer dynamic inputs and reuse the cached commands. The GPU-free C14
YOLO graph has one resident command batch.

The UIO HAL probes the optional extension in a child process, so an old
firmware's synchronous SIGBUS on the optional register can be handled without
terminating the application. Use mmio to bypass this probe. The runtime
synchronizes DDR writes for the device and orders queue bytes before START.
Failure diagnostics capture status, the retired count, queue error and legacy
error before CLEAR/STOP. A runtime failure never replays an already started
batch via MMIO.

## Qwen batches

Qwen records contiguous PL work and submits it before a host read, a host
write that may alias queued work, a counter snapshot or an IMEM update.
This preserves the C14 mixed-precision layer's causal dependencies. The host
performs dynamic activation quantization, high-precision epilogues and other
arithmetic on intermediate results. The execution statistics expose the
resulting multiple batches per layer.

## Local verification

Run make -C runtime OUT=YOUR_SCRATCH/native check c10-session-check
c12-check c14-check c15-check. The C15 reference tests exercise all six
opcodes, all ten error codes, binary validation, old capability fallback and
safe splitting. Reference HAL command ticks are functional diagnostics.

Use YSYX_REFERENCE_CMDQ=1 to enable Phase 18 capability on the C99 software
reference HAL. For backend A/B comparison, run z15_c10_benchmark with the
same case, frequency profile and frame count and use --exec-mode mmio or
--exec-mode cmdq along with --exec-stats. For a C14 YHP1 case, use
--target p17 and store the unchanged YHP1 program as program.yhp next
to the case's io.tsv. The on-board C10 service accepts --target p17
with the unchanged C14 model file.

## Complete Qwen CLI reference replay

The native chat CLI accepts --backend cmdq-reference --target p17. This
offline mode executes the same command recorder and DDR descriptors as Z15.
The functional HAL interprets the scalar programs and computes packed W4
dot products, including Phase 17 ties-to-even scaling, before retiring
commands. Its counters do not represent RTL timing.

Use --mode bench --trace-steps on --dump-logits FILE to run the three
built-in ChatML prompts. The dump contains each full result vector in
little-endian Q16, starting with the last prefill vector and continuing
through the scores used for greedy decode. The CPU backend accepts the
same dump option for byte comparisons.
