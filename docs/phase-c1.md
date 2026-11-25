# Phase C1 implementation and acceptance

A dependency-free Python compiler normalizes canonical JSON, legacy O4 JSON and
a static integer ONNX subset. A typed, topologically ordered tensor graph is the
input to every planner pass. Preserve the legacy narrowing at each original
node through fusion. Keep hardware formats out of the frontends.

- Frontend milestone: normalized IR with shape/type validation, tests for ONNX
  parsing and legacy input, plus four validated target profiles.
- Compiler milestone: capability-aware placement, fusion at private single-use
  boundaries, inclusive lifetimes with address reuse, and hardware-bounded
  buffers and synchronization. Emit one manifest and loadable backend artifacts.
- Verification milestone: replay the produced bytecode and command records on
  pinned read-only RTL snapshots, compare to an independent integer graph
  evaluator, and archive exact target coverage and commands. Report cycle
  estimates separately from observed cycles. Verify unbound target assumptions
  and fail unsupported operations deterministically.

Design choice: a common graph and schedule are shared across targets. Payload
encodings belong to the backends. This avoids binding the IR to the 4 KiB
program descriptor, while still allowing exact Phase 8 compatibility. Use one
GPU-addressable region for this phase. Native bank alternation and program
queuing depend on each backend ABI. Reservations or static scheduling models
do not imply overlap observed in RTL.
