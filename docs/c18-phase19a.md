# Phase 19A Qwen decode backend

The ABI is the Phase 19A specification at frozen RTL commit
a0c1bb8ef1c5f1c310e38cc6cdab8fde66a8103f. The exact ABI file SHA-256 is
696a23e3fa0aa291b165361aab94924e407ac6782af3702c3586ff0335884cb9.

Qwen selects this path only when CMDQ and both 0x60 extension capability
bits are present and p19a is requested. Auto uses the portable resident
path while the TMA wall-time gate remains open. The p18b mode pins that path
for comparison. The p19a mode enables TMA for wide projections on S3-H2.
The YHB1 and YHP1 container formats are retained.

A TMA slab packs strided K-group rows into one half of scratch SRAM. TC
consumes the loaded tile only after its load token. The last group signals
a result token, which gates TMA stores. The opposite half can prefetch
while the current TC slab runs. The single TMA engine drains a prior store
before modifying the associated region. The runtime validates all extents
and binds, and checks the terminal error code after submission. The
Verilator probe counts cycles in TMA TRANSFER state while the NPU is busy.
It excludes descriptor bounds and token waits.

The extension is confined to the Qwen grouped projection backend. YOLO
uses its existing YHP1 command plan.
