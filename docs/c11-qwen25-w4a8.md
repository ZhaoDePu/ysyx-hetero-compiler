# Qwen2.5-0.5B-Instruct W4A8 reference

The exporter accepts the official BF16 checkpoint at revision 7ae557604adf67be50417f59c2c2f167def9a775 and checks the entire safetensors SHA-256 before reading tensor data. The model has 24 layers, 14 query heads, two KV heads, 64 channels per head, QKV bias, RoPE theta 1,000,000 and a shared input embedding and vocabulary projection.

The image begins with a 64-byte little-endian shape and format header. Sections are 64-byte aligned. Linear weights and grouped scales use **input group, output channel, input offset** order. A group has 128 input channels. Each byte holds two signed two's-complement nibbles with the earlier channel in the low nibble. Weight scale uses unsigned Q24. Gamma, bias, residuals and cache values use signed Q16. The image also includes the Q16 RoPE and Q24 exponential tables.

The scalar and AVX2 C99 references run the same bounded integer arithmetic. Each linear quantizes its input vector to symmetric signed int8, computes int32 group dot products, combines Q24 group scales in int64 and returns Q16. Q and K are rotated in pairs spanning the two head halves. Each KV head supplies seven query heads. The attention path uses the current causal prefix, accumulates Q24 exponential weights and appends one token of K/V per layer. The weight loader checks shape, image size and table bounds.

For 512 positions the weight image is 262,850,688 B and the software Q16 K/V reservation is 12,582,912 B. A 32 MiB working allowance yields a planning total of 308,988,032 B. The allowance is a budget, and the hardware-only DDR peak requires its own ledger.

The command-line comparison compiles the supplied C99 token-ID harness and uses the official Transformers ChatML tokenizer, pinned local checkpoint values loaded as FP32, and greedy decode. It records every prefill row's cosine and top-1 agreement. Free-running decode can diverge after the first token mismatch. The reference samples are capped at 48 generated tokens and include factual errors and truncated answers. See the archived dialogue and quality ledger for the complete outputs.

- Export with python3 -m tools.c11_qwen_export and the pinned source directory.
- Compare with python3 -m tools.c11_qwen_quality and the independently built runtime/tests/c11_qwen_reference.c harness.
- SHA-256, exact byte counts and raw samples are under reports/c11-qwen25.
