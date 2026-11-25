# YSYX Hetero Bundle v1

A `.yhb` is one little-endian byte stream. Section starts and referenced DATA
spans are 8-byte aligned. The producer fills alignment padding with zero.
Integer types below have fixed widths. A reader decodes bytes explicitly
and does not cast a file location to a host struct.

## Header and directory

| Bytes | Field |
| --- | --- |
| 0–3 | ASCII `YHB1` |
| 4–5 | Version `1` |
| 6–7 | Header bytes `32` |
| 8–11 | Target ID |
| 12–15 | Total file bytes |
| 16–19 | IEEE CRC32 |
| 20–23 | Section count `7` |
| 24–27 | Target scratch capacity bytes |
| 28–31 | Reserved, zero |

CRC32 covers the entire file, including header and directory, with bytes
16–19 treated as zero. The loader checks it before applying any records.
The CRC detects corruption. Load programs only from a trusted compiler.

Target IDs are 1 for `fpga_zu15eg_balanced`, 2 for `fpga_zu15eg_full`, 3 for
`asic_nangate45_p8`, and 4 for `asic_nangate45_ip`. All four IDs support
packing and inspection. The C99 executor currently binds target 1 to the
frozen Z15 MMIO version `0x0C0C0100`.

The directory begins at byte 32. Its seven 16-byte entries contain four
u32 fields `[id, file_offset, byte_length, record_count]`, in increasing ID
order. An empty section has length and count zero. Strings and data use a
one-byte record stride. Fixed-record sections use `byte_length = stride *
record_count`.

## Sections

| ID | Section and record | u32 fields in order |
| --- | --- | --- |
| 1 | STRINGS, bytes | Counted UTF-8 name bytes |
| 2 | SYMBOLS, 48 B | Name offset, name length, flags, dtype, rank, four dimensions, DDR offset, byte size, reserved |
| 3 | DATA, bytes | Initial spans, native word arrays, GPU job records |
| 4 | INITIAL, 16 B | DDR offset, byte size, DATA offset, reserved |
| 5 | STAGES, 32 B | Name offset, name length, engine, DATA word offset, word count, DATA job offset, job count, scratch ownership half |
| 6 | DMA, 24 B | Direction, DDR offset, scratch offset, bytes, flags, reserved |
| 7 | COMMANDS, 16 B | Opcode, operand, reserved, reserved |

Name offsets are relative to STRINGS. Native data offsets are relative
to DATA. Physical addresses are byte offsets from the start of their
respective DDR and scratch apertures. Reserved fields are zero.

A symbol's flags are input bit 0, output bit 1 and input-required bit 2.
Dtypes are `int8=1`, `uint8=2`, `int32=3`, `int64=4`, `uint16=5` and
`int16=6`. Rank is 1–4 and unused dimensions are zero. Host input and
output files contain dense row-major elements encoded with the same
little-endian dtype. An input replacement exactly matches its symbol size.
Initial extents preload the relevant DDR ranges before applying overrides.
A missing mandatory input is rejected.

Engine IDs are GPU 1, NPU 2 and VPU 3. GPU code is an array of 32-bit
instruction words. Job records are 16 bytes each and contain relative code
entry PC, packed `GPU_JOB` value, thread count and Warp ID. Before uploading
new code, the host waits for the prior GPU stage to drain. Warp submissions
can overlap when the slot and pending queue permit it. A page change
waits for outstanding work because the frozen shell admits only one
active page. The Full Profile limits MMIO dispatch to four Warp slots,
within the frozen shell's packed job field. It retains the 8T/8W core
geometry in the Profile separately from that dispatch limit.

A Z15 NPU or VPU stage is an array of `(MMIO register offset, value)` u32
pairs. Its final pair issues the doorbell, and preceding control writes
may commit LUT or scale values. Both engines share the tensor sequencer.
Phase 8 and the explicit ASIC-IP compatibility Profile encode three native
values `[a_base, descriptor_address, program_slot_count]` for each tensor
stage. The 4 KiB descriptor image is included in the initial extents.
The native 64 KiB ASIC-IP projection instead encodes a 32-word command in
YHB DATA, and its symbol/initial offsets refer to scratch addresses. See
[the ASIC IP contract](asic_ip_v3_contract.md).

DMA direction 1 reads DDR into scratch and direction 2 writes scratch
back to DDR. Flags bit 1 allows extended AXI bursts. DDR offset, scratch
offset and length are multiples of 32 bytes. A DMA record stays within one
1 MiB scratch half and the 512 MiB reserved DDR pool.

Schedule instructions are `LAUNCH=1` with stage index, `WAIT_STAGE=2` with
stage index, `DMA=3` with DMA record index and `WAIT_DMA=4` with operand
zero. A DMA submission drains the previous DMA, so consecutive transfers
share one sequencer. Launches remain in flight until explicitly drained.
Balanced preflight checks index bounds, engine serialization, GPU page
and tensor operand base ownership, and submission order before execution.
Frozen RTL validates native operand extents. Execution also drains
the final outstanding operations before exposing results.

## Optional KV checkpoints

Target 1 programs can additionally encode `KV_YIELD=5` in COMMANDS.
The operand is the zero-based step number. A checkpoint follows each drained
step, including prefill and the final decode. It requires explicit compute
and DMA fences. The loader checks increasing step IDs and rejects empty or
undrained checkpoints. The original seven sections, CRC32 coverage and
record strides are unchanged. A C3-only loader rejects the new opcode.
The current runtime still reads and executes existing C3 streams.

One-shot execution advances across checkpoints. A session loads the initial
DDR image once and owns one exclusive HAL for the entire invocation.
Each step consumes its preplanned commands and leaves SRAM and DDR intact.
The compiler allocates persistent K/V and weights separately from transient
scratch. The runtime writes only designated input symbols. An input update
becomes visible to scratch through the next slice's scheduled DMA.
Output reads refer to symbols already materialized by a drained step.
DDR alias bindings can pass generated tokens to the next slice without
opening or reinitializing either backend.

`ysyx_kv_begin`, `ysyx_kv_step`, `ysyx_kv_write_input`, `ysyx_kv_read_output`,
`ysyx_kv_feedback` and `ysyx_kv_close` expose the session API. The caller keeps
the program image and exclusive HAL alive until close. The step count is
fixed in the program. A HAL failure while dispatching or modifying the
session invalidates it.

The CLI accepts a tab-separated `--kv-script` with the following rows.
`step N` runs the next step. `input SYMBOL PATH` writes a file-sized input,
`output SYMBOL PATH` reads an already produced output, and
`feedback OUT_SYMBOL IN_SYMBOL` copies an actual output of the same dtype
and byte size to an input. Step IDs start at zero. Explicitly run each step
in order; initial required inputs may be supplied with normal `--input`.
Zero host override commands are needed for autonomous feedback.

To reproduce a two-step hardware fixture with one-time resident weight
loading, device feedback, host feedback and a host override:

```sh
nice -n 10 python3 -B -m tools.verify_c4a_session \
  --out /dev/shm/ysyx_compiler_c4a/runtime/session-proof
```

## Build and execute

From the repository root, build the native Linux executable with
`make -C runtime -j2 CC=gcc OUT=/path/to/output`. The makefile uses
`-std=c99 -O2 -Wall -Wextra -Werror` by default and creates
`libysyx_rt.a` and `ysyx_run`. On PetaLinux use the supplied toolchain
prefix, for example `CC=aarch64-xilinx-linux-gcc AR=aarch64-xilinx-linux-ar`.
The C core uses standard-library allocation and file I/O. The Linux HAL
uses POSIX UIO discovery, memory mapping and ordering primitives.

The Linux HAL validates control map0 at physical `0x80000000` for 64 KiB,
and reserved DMA map0 at `0x40000000` for 512 MiB. It reads UIO sysfs names
and mapping metadata and accepts explicit names or `/dev/uioN` paths.
The platform must provide the boot package's `no-map` reserved pool and
a noncached or device DMA UIO mapping. Memory barriers order accesses to
that mapping. The runtime also calls `msync` and accepts the unsupported
operation errors produced by UIO drivers without a synchronization callback.
A cached alias requires kernel-managed cache maintenance.
[Linux UIO](https://github.com/torvalds/linux/blob/master/drivers/uio/uio.c),
[Linux msync implementation](https://github.com/torvalds/linux/blob/master/mm/msync.c). The CLI checks ABI magic, version and idle status before
writing data. C compiler compatibility does not establish board operation.

The Verilator build links the same C files against the frozen compiled
Z15 RTL and the C++ AXI/DDR rig. Run `bash tools/build_npu_z15_rtl.sh`
before `bash runtime/build_verilator.sh`. The bridge instruments the
copied rig and does not change frozen HDL. On xu4090 use the C2 scratch
directory and launch all compilation and simulation via low-priority
`nice` and `ionice`.

```sh
ysyx_run program.yhb --inspect
ysyx_run program.yhb --backend verilator \
    --input x=input.bin --output y=output.bin --dump-stats
```

Stage busy ticks and the wall counter are measured in the NPU fabric clock
domain. GPU-busy ticks use the hardware's NPU-domain sampling, rather than
the GPU clock counter. The runtime reports fence latency in the same
domain. DMA operations and bytes come from dispatched descriptors. With
the Verilator HAL, AXI transactions, DMA-busy ticks and the simultaneous
DMA/compute or three-engine busy ticks are observed from the simulated
interface and frozen busy signals. Native Linux supports the common
hardware MMIO counters.
