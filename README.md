# YSYX heterogeneous graph compiler · Phase C1

One Python standard-library compiler consumes static integer JSON graphs, the
legacy O4 graph format, or an ONNX subset. It emits an arena image, a launch
schedule, NPU/VPU descriptors or MMIO commands, and RV32EM ELF plus word HEX
for GPU stages.

## Quick start on xu4090

```sh
cd /home/zdp/ysyx-migration/work-compiler
mkdir -p /dev/shm/ysyx_compiler_c5
case_dir=$(mktemp -d /dev/shm/ysyx_compiler_c5/quick.XXXXXX)
nice -n 10 ionice -c2 -n7 python3 -B -m hetero_compiler \
  --graph examples/mixed.json --profile fpga_zu15eg_balanced --out "$case_dir/bundle"
```

Use `--format legacy` to select legacy arithmetic explicitly. JSON auto mode
detects legacy `memory_mode` or node `name` fields. The canonical form uses
`tensors`, topologically sortable `nodes` with `id/op/inputs/output/attrs`,
and `outputs`. Tensor shapes are positive and static; supported dtypes are
`int8`, `uint8`, and `int32`. Use `kind: "input"` for replaceable inputs.
Input `data` provides a fixture value. Non-input data is a constant.

The ONNX importer requires static tensor shapes and supported operator
semantics. See [C2 ONNX coverage](docs/c2-onnx.md) for quantized formats,
layouts and exactness limits. The reader derives its message fields from the
[ONNX protobuf schema](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3).

The canonical dense and NHWC/HWIO convolution contracts use signed 32-bit
wrapping accumulators. Requant uses a wrapping 32-bit multiply, arithmetic
right shift, wrapping zero-point addition and explicit byte narrowing.
Canonical Q4 SiLU/GELU use nearest-even rounding. Legacy byte narrowing,
legacy SiLU knots, and integer row operations retain the old model behavior.

## Targets and evidence

| Profile | Hardware contract | RTL evidence |
| --- | --- | --- |
| `fpga_zu15eg_balanced` | 16×16 NPU, 16 threads/4 warps, VPU, 2 MiB, 256-bit DMA | Frozen AXI/MMIO shell and shared scratch, with same-instance NPU→GPU→NPU |
| `fpga_zu15eg_full` | 24×24 NPU, 8 threads/8 warps, VPU, 2 MiB, 256-bit DMA | 8T/8W GPU RTL exercised independently. The board MMIO binding is projected |
| `asic_nangate45_p8` | Phase 8 4 KiB EXT/PROGRAM descriptors and v6 GPU | Actual 4 KiB page relay between independent pinned NPU and GPU RTL models |
| `asic_nangate45_ip` | 64 KiB capacity model, 4 KiB Phase-8 compatibility window and reserved VPU | Compatibility-form bundle emission. The 64 KiB native ABI and VPU binding remain open |

The frozen Z15 shell encodes a two-bit warp ID and four status slots. An
8-warp shell and corresponding frozen generated cluster are required for
full-profile integration. The Phase 8 replay preserves every observed byte
across models. A same-instance ASIC model containing both the legacy NPU
interface and the v6 scalar-issue path remains a separate hardware integration.

## Bundle contract

- `bundle.json` contains the graph, profile, launch/fence plan, input
  bindings, native commands and DMA extents.
- `initial.bin` contains the initial scratch arena. Load only the listed
  startup DMA extents on Z15, and patch input bindings before upload.
- `npu.bin` and `npu.hex` contain Phase 8 relative descriptors at
  `binary_base`, or Z15 little-endian MMIO register/value pairs.
- `gpu.json` and `gpu/*.elf|hex` contain per-stage code and job waves.
  Finish the preceding GPU stage before overwriting instruction RAM.
- `memory_plan.json` records inclusive tensor lifetimes, scratch addresses,
  physical replacements, ownership zones and native tile/bank information.
  `analysis.json` reports theoretical compute cycles, live bytes and DMA
  transfer demand. RTL timing is recorded by the replay logs.

A fused stage preserves original-node narrowing. Fusion requires private,
single-use intermediates and an exact native or GPU lowering. A connected
scratch component stays in one Z15 ownership half. GPU-addressed tensors share a 4 KiB page for each stage. ASIC targets
address the first page through the Phase 8 compatibility contract. Independent tensor-only components can
use the other half. Constant weights in native packed format are loaded once
and remain resident through their last use. The planner honors the target's
internal A/B/output ping-pong banks and allocates an explicit partial
accumulator when the native packed K split needs it. The Z15 attention path
uses ordered Q/K/V/score/probability/output storage and native QK→softmax→PV.
The schedule records same-engine predecessors and scratch write-drain
fences across owners. The stage-check replay serializes stages to locate mismatches. The Z15
schedule replay executes the actual launch and fence records and reads back
the planned final DMA extents. It can overlap independent engines in distinct
ownership halves.

## Replaying RTL

From the repository root, build the required runner for the target and
verify the generated bundle. Use a fresh case directory inside compiler RAM.

```sh
bash tools/build_npu_z15_rtl.sh
# Phase 8: bash tools/build_npu_p8_rtl.sh
nice -n 10 ionice -c2 -n7 python3 -B -m tools.verify_bundle \
  --bundle "$case_dir/bundle" --case "$case_dir/check"
nice -n 10 ionice -c2 -n7 python3 -B -m unittest discover -s tests -v
```

Add `--inputs <json mapping>` to replay with runtime inputs. Add
`--execute-plan` on Z15 Balanced to exercise the emitted schedule and final
DMA extents, including concurrent ownership where applicable. The Z15
runner compares stage results over actual AXI DMA and checks the emitted
MMIO commands and GPU machine code. Phase 8 replays actual full arena images
through the native descriptor and v6 GPU models. Build scripts read pinned
hardware files into `/dev/shm/ysyx_compiler_c5`, apply low-priority scheduling,
and limit each build to four jobs.

Standalone validation commands are `python3 -B -m tools.verify_npu_rtl`
(`p8|z15`, executable, scratch directory) and
`python3 -B tools/validate_gpu_backends.py` (`v6|g4-16t4w|g4-8t8w`).

## Phase C2 optimizer and measurements

See [the ONNX coverage](docs/c2-onnx.md) for exact static quantization contracts
and instructions to regenerate the visual and Transformer examples.

Use `-O3` for the default pipeline and select an ablation with
`--device-policy auto|gpu-only|npu-greedy`, `--no-fusion`, `--no-overlap`,
and `--no-mem-reuse`. Legacy flag names `--no-fuse` and `--no-reuse` remain.
The reserved address footprint counts allocator spans in each ownership half.
The live peak counts overlapping allocation lifetimes. These values measure
different allocation costs.

Z15 can select a different physical 4 KiB page for each GPU stage. All
outstanding warps of that stage share the page. The planner first tries a
resident layout, then packs page-local GPU components into the larger arena.
When those groups do not fit, supported stages can use DDR-backed staging.
Elementwise operations can tile along the linear index and GEMM can tile M
while reusing the same K×N weights. Arbitrary oversized Conv, reductions
and layout groups require an explicit additional tiling contract.
`--stream-ddr` requests the staged schedule for experiments. It alternates
ownership halves and prefetches an independent next tile during compute.
Disabling overlap drains the active owner before that transfer.

Run the four configurations over the same inputs:

```sh
nice -n 10 ionice -c2 -n7 python3 -B -m tools.run_c2_ablation \
  --out /dev/shm/ysyx_compiler_c5/ablation \
  examples/yolo_block_int8.onnx examples/transformer_layer_int8.onnx examples/mixed.json
```

The replay records raw Z15 MMIO simulation wall ticks, AXI transactions and
busy-overlap counters. The reported conversion divides wall ticks by
164.555 MHz. The GPU busy projection separately divides observed GPU clock
edges under the synchronized GPU busy flag by 124.239 MHz. The conversion
uses Phase 12C routed Fmax and does not retime the simulator clocks.

## Single-file programs and Z15 runtime

Standard profiles emit `program.yhb` by default. The [YHB1 format](docs/yhb-format.md)
describes the byte layout, CRC, inputs and both backends. Use `--no-yhb` for
native/textual artifacts alone. Build the portable C99 library and the Linux
UIO CLI with a warning gate:

```sh
nice -n 10 ionice -c2 -n7 make -C runtime -j2 CC=gcc
bash runtime/build_verilator.sh
nice -n 10 ionice -c2 -n7 python3 -B -m runtime.verify_yhb \
  --bundle /dev/shm/ysyx_compiler_c5/quick.example/bundle \
  --scratch /dev/shm/ysyx_compiler_c5/runtime/yhb-check
```

The Verilator binary uses the pinned Balanced AXI/MMIO shell. The Linux
binary requires the matching UIO control and noncached reserved DDR devices
on the board. All four standard targets support packing and inspection.

The C1-preserving regression is reproducible with
`bash tools/run_c2_regression.sh /dev/shm/ysyx_compiler_c5/regression`.
It checks all software tests, 42 legacy GPU full-page results, 13 native
NPU/VPU outputs, all 13 frozen O4 examples, the other C1 complete bundles
and YHB emission for all four targets. The C2 ablation and C99 runtime
commands above exercise additional ONNX and hardware paths.

## Phase C3 qualification and deployment

- [Two-half graph tiling and RTL ablation](docs/c3-tiled-pipeline.md)
- [Offline training and held-out FP32/INT8 accuracy](docs/c3-trained-models.md)
- [Standard quantized causal Softmax](docs/c3-qsoftmax.md)
- [Frozen Full tensor comparison](docs/c3-full-qualification.md)
- [64 KiB ASIC IP software contract and 4 KiB compatibility target](docs/asic_ip_v3_contract.md)
- [Z15 Balanced C99 board kit with checked programs and IO](deploy/z15_board_kit/README.md)

## Phase C4-A persistent KV and bank-aware dispatch

- [Compile and replay a causal KV session](docs/c4a-kv-cache.md).
- [Run strict A0–A3 ablations on standard streaming ONNX](examples/c4a_streams/README.md).
- [Drive the C99 KV checkpoints and feed back device-produced tokens](docs/yhb-format.md#optional-kv-checkpoints).
- [Run the standalone Z15 deployment fixtures](deploy/z15_board_kit/README.md).

## Phase C4-VPU and Step 2 Z15 ABI

- [Native quantized ONNX fusion and resident KV](docs/c4vpu-lowering.md)
- [Frozen latency and accuracy benchmarks](examples/c4vpu/README.md)
- [YHB1 runtime format and MMIO preflight](docs/yhb-format.md)
- [Step 2 standalone C99 deployment kit](deploy/z15_board_kit/README.md)

## Phase C5 trained task workloads

- [TinyLM-Chat training, ONNX and stateful evaluation](examples/c5_tinylm/README.md)
- [YOLO26-Tiny-Style training and hardware detection](examples/c5_yolo/README.md)
- [ASIC A3 IPV3 native VPU bindings](docs/asic-ip-vpu-abi.md)
- [Step 2 C99 demonstrations and deployment](deploy/z15_board_kit/README.md)

Use the dedicated compiler RAM directory `/dev/shm/ysyx_compiler_c5`.
Frozen RTL builds take one compiler build slot and select two or four
jobs from the available RAM. The C5 workflows use only offline Python
dependencies and the frozen Verilator sources.

## Phase C6 official YOLO26n

The C6 compiler reads the calibrated, static COCO 80-class model and
generates a Z15 Balanced YHB1 with static DDR liveness and double-buffered
scratch tiles. See the [C6 lowering and validation flow](docs/c6-z15-lowering.md)
and [C99 board kit](deploy/z15_yolo26n_320_kit/README-YOLO26N-Z15.md).
