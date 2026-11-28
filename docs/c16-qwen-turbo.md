# C16-Turbo Qwen runtime

Phase 18 CMDQ selects the C16 schedule for the C14 hybrid dual-plane weight
format (YSQC14H). The supported release asset is S3-H2 with KV8. The kit
launcher checks the installed asset hashes and uses the existing files.
P1/P2/P3 formats and repackers remain in the separate research candidate.

The runner groups prefill into tiles of up to 16 rows, advances each layer's KV
in token order and asks the LM head for the final position only. The head
retains its original vocabulary order. Decode queues Q/K/V, O, Gate/Up and
Down in four batches per layer. The resident full head takes ten more batches.
The fused route uses the same host Q16 RoPE and residual arithmetic as the
C99 reference. Set YSYX_QWEN_C16=0 on the CLI to compare the C15 schedule.

The hardware launcher's --target auto probes the extension through the
existing Linux UIO HAL. Phase 18 with a CMDQ capability selects the native W4
target. Step 2-v3 Turbo selects the old MMIO path and packed table writes.
If a separate --fallback-weights path is present, the old path opens those
C12 weights directly. Step 2 v1 uses the same fallback model with v1 MMIO.
Explicit step2 and p17 targets remain available for existing scripts.

For an offline replay, use --backend cmdq-reference --target p17
--exec-mode auto --exec-stats. For byte comparisons, use --dump-logits
and compare the result with --backend cpu using the same weight and prompt.
Functional HAL counters do not represent RTL wall cycles.
