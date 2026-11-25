# ASIC A3 host-port binding for IPV3

The target-5 `asic_nangate45_ip` compiler format uses the 64 KiB data arena and
16 KiB program window of the frozen `asic_soc_sram_top` at
`6fe0028a7b7def51c1f2754a058c1984830ee6d0`. See
[the IPV3 descriptor layout](asic_ip_v3_contract.md) for the container and
the pre-existing opcodes 1–10. Canonical VPU geometry uses 8-byte-aligned
SRAM byte offsets, dense row-major tensors and disjoint output spans.

The A3 top exposes a host port and a separate vector command port. The host
writes word-aligned data or instruction requests, waits for acknowledgement,
and releases the request. For a vector command the host waits for
`start_ready`, presents `vpu_op`, A/B/C byte addresses, length, rows,
columns, channels, scale, shift and epsilon, then pulses `vpu_start` once.
Each command completes until `done` or `fault` before the next engine can
acquire the SRAM. LUT data uses 256 indexed `vpu_table_write` operations
while idle. These are wires on the frozen ASIC top. The Linux UIO runtime's
(Z15) MMIO register map applies to the FPGA only.

The C99 target-5 path inspects YHB structure, CRC, operand spans and the
new descriptor settings. `hardware_vector_records` is the executable A3
host adapter for the compatible VPU and v6 subset. It consumes actual YHB
initializers, stages, launch/fence records and output symbols. The
hardware replay gates each requested output against independently
computed fixed-point bytes.

## Native vector descriptors

Parameter words and tensor fields retain the usual 32-word record.
Unsigned fields are little-endian u32, with narrower hardware ranges
checked before submission. The shared compiler IR retains its C4-VPU
operation names because both lowerings consume the same fixed-point
parameters.

| IPV3 opcode | Operation | A3 wire opcode | Words 25 onward | Tensor format |
| ---: | --- | ---: | --- | --- |
| 11 | Gamma RMSNorm | 7 | gain u16, epsilon u24, shift u5 | signed int8 A/C `[rows,width]`, signed int8 gamma B `[width]` |
| 12 | Base-two Softmax | 12 | exponent shift u5, causal position | signed int32 A and unsigned byte C, `[rows,width]` |
| 13 | Indexed activation LUT | 11 | SRAM table address, 8-byte aligned | equal-shape byte A/C, 256 consecutive table bytes |
| 14 | Scaled EW_MUL | 6 | scale u16, shift u5 | same-shape signed int8 A/B/C |

Unused parameter words are zero. Opcode 12 flags bit 2 and bit 3 select
causality and fast reciprocal. The position is zero without causality and
`position + rows <= width` with causality. This binding accepts word-aligned
row widths (a multiple of four) so physical row bytes equal logical row
bytes. Softmax clamps usable geometry to 511 rows and 128 columns. RMSNorm
uses at most 1,023 rows and at most 1,020 columns. The SRAM extent may tighten
both limits.

### Fixed point across targets

- **RMSNorm.** Sum signed squares and epsilon over one input row. Clamp
  the sum to at least one. Normalize the sum to the frozen reciprocal
  square-root mantissa domain and linearly interpolate between fixed ROM
  entries. The reciprocal is the low 21 bits of the gain product shifted
  right by `8 + floor(log4(sum))`. Each output is the signed input times
  signed gamma times this reciprocal, arithmetically shifted by the
  descriptor shift and saturated to int8. Static Q/DQ RMSNorm lowering
  converts epsilon to input-code square units and gain to the ABI fields.
- **Softmax.** For the visible prefix of each row, subtract from the row
  maximum using a 33-bit nonnegative difference. The weight is
  `256 >> (difference >> exp_shift)`, with zero for shifts beyond 8.
  Normalize as `floor(255*weight/sum)`. Fast reciprocal multiplies by
  `floor(65280/sum)` and shifts by 8, which yields identical bytes for these
  power-of-two weights. Causal positions past the visible prefix write
  zero. Static standard Q/DQ Softmax with byte scores can use an explicit
  int32 Cast before this operation. This hardware approximation differs
  from continuous FP32 and the portable Q23 exponential path.
- **SwiGLU.** The compiler evaluates each Q/DQ SiLU byte code once to
  construct the 256-entry activation table. Lookup uses `A & 255` as its
  index and writes the table byte. When the following multiply and
  requantization are an eligible private, zero-centered, signed-int8
  path, the compiler emits scaled EW_MUL. It computes the signed
  product times the unsigned scale, adds `2^(shift-1)` for nonzero shift,
  arithmetically shifts, and saturates. This mode rounds ties toward
  positive infinity. Native EW_ADD (IPV3 9 to wire 0) supports equal-shape,
  saturating int8 residuals.
- **Requant.** IPV3 opcode 3 remains the original logical descriptor.
  A byte-input/byte-output Requant is evaluated across its entire 256-code
  domain and sent as wire opcode 11. The table preserves floor or
  nearest-even, the signed multiplier, zero points, and wrap or saturate.
  The A3 vector port has no general int32 Requant. The optimized compiler
  places wider Requant on the v6 GPU. The legacy software-only opcode
  remains available under `--vpu-lowering off`.

The A3 vector port has no compound ATTN or EW9 strided transpose. ASIC
lowering enables the compatible standalone stages without emitting either
wire opcode. Old software-only lattice RowSoftmax (opcode 7), unary
RMSNorm (opcode 8), and general NPU descriptors (opcodes 1–2) retain the
existing software replay. This host adapter checks for a precise
supported mapping and rejects such descriptors at the hardware boundary.
The Phase 8 4 KiB compatibility target remains separate.

## Reproduce

Build from compiler-owned scratch using the common build lock and memory
guard. Frozen RTL is extracted from its pinned Git object through a
read-only archive. The acceptance uses two standard ONNX graphs, a native
RMSNorm/SwiGLU segment and a mixed extension with causal Softmax and
wide-Requant v6 fallback. Both reciprocal modes run through the mixed
graph. The C99 gate also tries checksummed invalid shape, dtype, extent,
aliasing, flag and scale fields.

```sh
bash tools/build_c5_asic_a3_rtl.sh
source tools/c5_resources.sh
compiler_build_slots
nice -n 10 make -C runtime -j"$compiler_jobs" CC=gcc AR=ar \
  OUT=/dev/shm/ysyx_compiler_c5/asic/c99
nice -n 10 python3 -B -m tools.verify_c5_asic_vpu \
  --out /dev/shm/ysyx_compiler_c5/asic/verified
```
