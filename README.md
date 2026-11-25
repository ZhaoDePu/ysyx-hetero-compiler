# YSYX heterogeneous graph compiler · Phase C1

One Python standard-library compiler consumes static integer JSON graphs, the
legacy O4 graph format, or an ONNX subset. It emits an arena image, a launch
schedule, NPU/VPU descriptors or MMIO commands, and RV32EM ELF plus word HEX
for GPU stages.

## Quick start on xu4090

```sh
cd /home/zdp/ysyx-migration/work-compiler
mkdir -p /dev/shm/ysyx_compiler
case_dir=$(mktemp -d /dev/shm/ysyx_compiler/quick.XXXXXX)
nice -n 15 ionice -c 3 python3 -B -m hetero_compiler \
  --graph examples/mixed.json --profile fpga_zu15eg_balanced --out "$case_dir/bundle"
```

Use `--format legacy` to select legacy arithmetic explicitly. JSON auto mode
detects legacy `memory_mode` or node `name` fields. The canonical form uses
`tensors`, topologically sortable `nodes` with `id/op/inputs/output/attrs`,
and `outputs`. Tensor shapes are positive and static; supported dtypes are
`int8`, `uint8`, and `int32`. Use `kind: "input"` for replaceable inputs.
Input `data` provides a fixture value. Non-input data is a constant.

ONNX parsing supports static, inline integer initializers, rank-2
`MatMulInteger` with implicit zero points, and same-shape integer
`Add/Mul/Relu`. Static tensor types, dimensions, and supported operator
semantics are required. The reader derives its message fields from the
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
scratch component stays in one Z15 ownership half. GPU-addressed tensors are
restricted to the first 4 KiB page. Independent tensor-only components can
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
nice -n 15 ionice -c 3 python3 -B -m tools.verify_bundle \
  --bundle "$case_dir/bundle" --case "$case_dir/check"
nice -n 15 ionice -c 3 python3 -B -m unittest discover -s tests -v
```

Add `--inputs <json mapping>` to replay with runtime inputs. Add
`--execute-plan` on Z15 Balanced to exercise the emitted schedule and final
DMA extents, including concurrent ownership where applicable. The Z15
runner compares stage results over actual AXI DMA and checks the emitted
MMIO commands and GPU machine code. Phase 8 replays actual full arena images
through the native descriptor and v6 GPU models. Build scripts read pinned
hardware files into `/dev/shm/ysyx_compiler`, apply low-priority scheduling,
and limit each build to two jobs.

Standalone validation commands are `python3 -B -m tools.verify_npu_rtl`
(`p8|z15`, executable, scratch directory) and
`python3 -B tools/validate_gpu_backends.py` (`v6|g4-16t4w|g4-8t8w`).
