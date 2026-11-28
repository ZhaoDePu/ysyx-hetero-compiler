# C17B offline P19 preview

The ABI preview is generated from hetero_compiler/p19_abi.json. Its sync point
is v9-soc-zu15eg-phase19a-tma-tmem and v10-gpu-zu15eg-g11-warpcore after hardware
seal. The checked-in header must match c_header(). The source snapshots are
the Phase 19A draft ABI and the G11B 8T8W handoff instruction specification.

Run python -m hetero_compiler --target p19a --graph TILE_SPEC.json --out NEW.
The tile spec contains tiles, rows, row_bytes, ddr_stride and sram_stride.
Optional load_base and store_base select offsets relative to the existing
512 MiB DDR pool. The default plan alternates two 512 KiB input banks and
two 512 KiB result banks. These are logical ownership regions. The hardware
interleaves 16 physical banks per half. A store subslot releases the result
bank before it is overwritten. The auditor rejects conflicts, unsatisfied
barriers and invalid address extents. Use python -m tools.cmdq_reference_p19
NEW/plan.json to recheck the serialized transfers, token order and fixture
outputs.

For an existing YHB1/YHP1 graph, pass it to the same target. The transfer
audit folds only consecutive, order-equivalent rows and retains the stage
boundaries. It reports a full-graph transfer count. The tile-fixture plans
also have an analytical overlap ledger. Their NPU/GPU payloads are simple
byte transforms, so the transfer/barrier listings are for offline replay.

Run python -m hetero_compiler --target p19b-g11 --graph KERNEL_SPEC.json
--out NEW. Select name flashattention-pv, rmsnorm-a8 or topk, with lanes set
to a complete 4, 8 or 16 lane window. For topk specify k from one to four.
The templates consume and produce register windows as annotated in the
emitter. The assembly and little-endian binary contain the instruction
body. Integration owns warp launch, layout, address calculation, load/store
and termination. Top-K expects distinct nonnegative IDs and excludes
INT32_MIN scores. RMSNorm expects nonoverflowing Q16 square sums and an amax
whose scaled reciprocal fits int32. FlashAttention expects nonoverflowing
Q16 accumulators. The interpreter uses ideal Q16 SFU values, providing a
test of the instruction sequence and scalar recurrence. LUT bit-exactness,
shared GPU barrier wiring, and real overlap require RTL confirmation.

The actual draft CMDQ barrier word contains one slot and a zero payload,
with one token consumed per wait. Masks describe logical dependencies and
are lowered to individual slots. The AXI and bank estimates do not measure
hardware cycles. Existing hardware runtime targets continue using their
existing execution paths.
