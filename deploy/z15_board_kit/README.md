# Z15 Balanced deployment kit

Copy this directory to the PetaLinux build or target machine. It contains
the complete C99 loader and Linux UIO backend in `source/`, plus Balanced
YHB1 programs, little-endian input tensors, expected output bytes and
bindings in `cases/`. Each case supplies explicit inputs. The default first
case is a short mixed NPU/GPU smoke.

Build on x86-64 to check compiler portability:

```sh
make -j4 CC=gcc AR=ar
```

With the PetaLinux cross toolchain configured:

```sh
make -j4 CC=aarch64-xilinx-linux-gcc AR=aarch64-xilinx-linux-ar OUT=build-aarch64
file build-aarch64/ysyx_run
readelf -h build-aarch64/ysyx_run
```

Deploy the executable with the unchanged kit cases. On the board, boot the
matching frozen Balanced PL and PetaLinux package. The Linux HAL needs the
64 KiB control UIO at physical `0x80000000` and the noncached reserved
512 MiB DDR pool starting at `0x40000000`. The loader validates their UIO
map metadata and the frozen hardware magic and version.

```sh
YSYX_RUN=./build-aarch64/ysyx_run ./run_z15_bench.sh 00-mixed
YSYX_RUN=./build-aarch64/ysyx_run ./run_z15_bench.sh
```

The script can use auto-discovery or `YSYX_CONTROL_UIO` and `YSYX_DMA_UIO`
with the UIO name or exact `/dev/uioN` path. Run as an account with access
to those UIO devices. Set `YSYX_LOG_DIR` to keep raw logs. Each case checks
the YHB CRC, compares all output bytes and prints monotonic host elapsed time
and item throughput. This clock covers runtime execution and output writes,
including requested statistics, after opening the backend. Hardware counter
values are printed separately. The source build uses
`-std=c99 -O2 -Wall -Wextra -Werror`.

The same script accepts `YSYX_BACKEND=verilator` with the compiler repository's
C99/Verilator executable. Reports identify the backend. Verilator host
throughput and simulated NPU ticks describe the simulation.

## C4-A KV sessions and streaming

`40-kv-cache` contains a 32-token prefill split into four tiles and four
autoregressive decode steps. Its `kv.tsv` has five ordered checkpoint
commands. The runner keeps the same HAL and SRAM allocation through each
command and installs the initial weight image once. Existing C3 cases
use the same one-shot path as before.

```sh
YSYX_RUN=./build-aarch64/ysyx_run ./run_z15_bench.sh 40-kv-cache
```

For integration code, use the `ysyx_kv_begin`, `ysyx_kv_step`,
`ysyx_kv_read_output` and `ysyx_kv_close` API, or run `ysyx_run --kv-script`
with explicit `input`, `output` and `feedback` checkpoint commands.
The included fixture feeds back actual generated tokens automatically.

`41-stream-vision` runs 16 Conv/residual tiles and their GPU tails.
`42-stream-llm` runs 24 causal attention tiles with quantized GPU Softmax.
Both replay the compiled bank-aware YHB through the regular C99 dispatch
path. The optional instruction preload used by the MMIO ablation runner is
specific to that replay tool.

## C4-VPU and Step 2 MMIO

The C99 loader validates the Step 2 Z15 ABI family, major revision, ready
status, command ownership, aligned DMA extents and explicit LUT commits.
The included VPU cases use the same frozen ABI and YHB1 loader.

- `50-vpu-kv` uses incremental native packed V, compound causal ATTN and
  five host-visible checkpoints.
- `51-vpu-attention` uses 24 windows with 48 causal keys each.
- `52-vpu-transformer` runs four complete layers with RMSNorm, ATTN,
  activation-table SwiGLU, EW_MUL and residual EW_ADD.

Keep the noncoherent DDR UIO reserved mapping and the matching Step 2
bitstream. The loader reports the final ABI error word and the number of
invalid DMA descriptors for each successful execution.

## C5 task demonstrations

The standard build also produces `z15_tinylm_demo` and
`z15_yolo26_demo`. Use the case directory from the same kit as the model.
TinyLM accepts four whitespace-delimited vocabulary words and uses one
C99 session through prefill and every hardware feedback step. The detector
accepts a name from the packaged input samples. The detector ranks and
decodes the candidates in the device program. The demo converts the
integer coordinates and displays the scores using the case calibration.

```sh
./build/z15_tinylm_demo cases/60-tinylm-chat --prompt 'our compiler schedules work'
./build/z15_yolo26_demo cases/61-yolo26-tiny --image dense-overlap
```

The demos report NPU, GPU, wall and overlap ticks read through the Step 2
MMIO runtime. They also report throughput from the same ticks at 164.555
and 125.00 MHz. Set `YSYX_CONTROL_UIO` and `YSYX_DMA_UIO` to select
explicit UIO devices if needed.

The compiler repository's `tools/build_c5_demos.sh` supports `x86`,
`aarch64` and `verilator`. Build the frozen runner and the runtime
bridge before the Verilator variant. Pass `--backend verilator` to run
that variant against the frozen shell.
