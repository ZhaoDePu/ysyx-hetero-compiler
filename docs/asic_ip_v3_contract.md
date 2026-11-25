# ASIC IP v3 software target and integration contract

`asic_nangate45_ip` is a software projection for a unified 64 KiB scratchpad.
It uses YHB1 target ID 5. The Phase A2 frozen source and its SRAM interface
at `a4dbe9329170f2a250b07c7d2034c7e01e91ddfc` expose four 4096×32-bit data
banks and a 16 KiB instruction window. Their host port takes 16-bit byte
addresses. The Phase A2 GPU still addresses a selected 4 KiB page per launch.
The [Phase A3 native vector binding](asic-ip-vpu-abi.md) specifies the
additional descriptors and their frozen hardware host-port mapping. This
document defines the baseline YHB projection and its software contract.

## Execution and storage

Native NPU and VPU descriptors carry byte offsets across the 64 KiB arena.
Offsets are 8-byte aligned, ranges stay in bounds, and the destination cannot
alias live source ranges. There is one ownership zone: each successive stage
has a fence. NPU and VPU commands live in YHB DATA rather than consuming
scratch storage. The initializer records install resident graph inputs and
constants. An external driver would upload these extents via the host port,
load native descriptors, dispatch after fences and read back output symbols.

The v6 compiler fallback keeps its first 4 KiB data page and 16 KiB
instruction window. Target-5 software replay covers the native NPU and shared
VPU command records. The portable C99 loader validates and inspects ID 5,
with execution currently bound to Z15 Balanced. The projected ASIC path has
no DMA section.

`asic_nangate45_ip_compat4k` explicitly selects the previous Profile and
keeps target ID 4, Phase 8 descriptors and the 4 KiB engine window.
`asic_nangate45_p8` remains target ID 3.

## Native descriptor layout

Each native stage has 32 consecutive little-endian u32 words (128 bytes), with
an 8-byte-aligned start in YHB DATA. Unused words and shape slots are zero;
an absent operand address is `0xffffffff`. Rank is 1–4, element types are
`1=int8`, `2=uint8`, `3=int32`. Bias is int32 with one value per output
channel. Dimensions specify dense row-major storage.

| Word | Meaning |
| ---: | --- |
| 0–2 | Magic `IPV3` (`0x33565049`), opcode, flags |
| 3–5 | A/B/C ranks (B rank 0 for unary) |
| 6–9 | A/B/C/bias byte addresses |
| 10–12 | A/B/C element types (B type 0 for unary) |
| 13–16 | A shape |
| 17–20 | B shape |
| 21–24 | C shape |
| 25–31 | Operation parameters from the table below |

Flag 0 selects wrap narrowing. Flag 1 selects GEMM transpose or final-axis Add
broadcast. Flag 2 selects row-causal Softmax. Flag 3 selects the legacy Q4
SiLU knot approximation. Flag 4 selects nearest-even Requant. Bits not used
by that opcode are zero.

| Opcode | Engine | Format and parameters starting at word 25 |
| ---: | --- | --- |
| 1 GEMM | NPU | A and B zero points |
| 2 Conv | NPU | NHWC/HWIO; A and B zero points, shift, strides (Y and X), top/left pads, bottom/right pads. Two-axis parameters take two u16 halves, first axis low |
| 3 Requant | VPU | Multiplier, shift, divisor, output zero point, input zero point |
| 4 ReLU | VPU | Width and narrowing from descriptors |
| 5 SiLU | VPU | Signed Q4, nearest-even or legacy knots |
| 6 GELU | VPU | Signed Q4, nearest-even |
| 7 Softmax | VPU | Lattice row format and optional causal mask, signed int8 → uint8 |
| 8 RMSNorm | VPU | Quantized row RMSNorm, signed int8 → int8 |
| 9 EW_ADD | VPU | Exact integer add with equal shape or last-axis broadcast |
| 10 EW_MUL | VPU | Exact integer multiply with equal shape |

Integer accumulation follows the IR's int32 wrap semantics. The proposed
Softmax opcode implements the legacy integer lattice row algorithm. The
general standard-ONNX Q/DQ Softmax uses its distinct shape/scale-qualified
GPU lowering.

## Software acceptance

Run from the compiler repository on xu4090:

```sh
nice -n 10 ionice -c2 -n7 python3 -B -m tools.verify_asic_c3_target \
  --out /dev/shm/ysyx_compiler_c5/targets/ip-native
```

The fixture maps all six requested shared VPU arithmetic operations. It uses
an initial tensor range above 0x1000 and compares every output byte after
applying fresh host input bytes. The replay reconstructs each operation from
its emitted binary descriptor and checks the physical extents during execution.
