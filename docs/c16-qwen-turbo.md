# Valence · 蔚澜 C16 Qwen2.5 Turbo

Set YSYX_QWEN_C16=1 with a Phase 18 CMDQ device to select the experimental
scheduler. C15 is the default. The existing C99 API, YHB1 and C12/C14 model
read paths retain their formats.

## Decoding

Q, K and V use one staged activation and disjoint DDR result slices. The
Valence-TC projects all three before the runtime reads any of the slices.
Gate and Up likewise share one submission for both legacy grouped and
Block-16 projections.
The resident LM head uses the original group stride in DDR, which removes
the per-panel host gather and enlarges a submission to 16,384 vocab rows.
The host still applies the same Q24 epilogue. Exact Q16 RoPE and residual
operations use the host for this ablation. Verilator WALL excludes the
seconds the host spends between AXI requests.

## Prefill

The runtime evaluates up to 16 prompt positions together per layer and
handles the final smaller tile. The TC uses multiple A rows with one shared
weight transfer. For Block-16 projections, the TC returns the raw dot
products and the host applies the same per-token Q0.8 coefficient and
ties-to-even rounding as C15. The intermediate results occupy the reserved
DDR working ring. The causal pass writes and reads only the already
available history, reusing unpacked KV within the layer tile. The CLI
computes the vocabulary projection for the last prefill position.

## Quantization experiments

The C16 tagged image can mix single-plane Gate and Up with existing dual
planes. The experiment uses the pinned official BF16 source and archived
SmoothQuant calibration moments. Gate and Up share a permutation within
each Down quantization group. The corresponding Down columns are permuted
offline. The compact W4 head experiment stores a vocabulary permutation
and its loader builds the inverse for tied embedding lookup. Full logits
are restored to canonical vocab order for the C99 API.

Evaluate a candidate before changing the default model. The opt-in schedule
also runs with the existing S3-H2 weights for like-for-like comparisons.
