# C12 Qwen2.5-0.5B candidate

The quantization tools start from official revision 7ae557604adf67be50417f59c2c2f167def9a775.
The original safetensors object must match SHA-256
fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe.
The exporter uses a fixed, independent calibration corpus and records every
projection's per-channel maximum and second moment. Alpha selection uses a
separate holdout. The historical ChatML prompts are only used for evaluation.

## Quantization and image layout

The 64-byte little-endian header starts with YSQC12 followed by two zero bytes.
Its fourteen uint32 fields specify version, group size, layer count, model
width, FFN width, Q heads, KV heads, head width, vocabulary, context
capacity, policy, Q24 weight shift, exp table size, and Q32 RMS epsilon.

Policy bits 0, 1, and 2 select W8 attention, down projection, and tied
embedding/head respectively. Other matrix weights use W4. Bit 3 selects
per-token and per-group A8 scales. With this bit clear all groups in a token
share a single A8 scale. All weight matrices have G64 or G128 groups over
their input dimension. Every W4 byte packs the even input in the low
nibble and the odd input in the high nibble, in signed two's complement.
The encoder uses the range [-7, 7] and has zero point 0. W8 uses signed
[-127, 127].

Matrix payloads are ordered by input group, output channel, then input
index or pair. The corresponding uint32 Q24 scales use the same group
and output order. Matrix payloads, scales, Q16 norm and bias vectors,
RoPE, and the exponential table begin at 64-byte aligned offsets.
The image has one tied weight matrix. The runtime maps this matrix for
embedding lookup and uses it again for the LM head.

SmoothQuant folds the reciprocal of the per-channel smooth factor into
the upstream RMSNorm and the factor into the consuming matrix columns.
Q/K/V and gate/up share their upstream norm factors. For an enabled down
projection the exporter divides the up projection rows by the down
factor before quantization. SwiGLU is linear in that branch. The selective
policy applies smoothing to W4 paths and preserves the original scale
of the W8 paths.

## Cache

Keys after RoPE and projected values use one G64 record per head per
token. The record has 32 (KV4) or 64 (KV8) payload bytes followed by one
little-endian uint32 Q16 scale. Nibble order is the same as W4.
The packed cache persists between steps and attention expands only the
current layer's prefix.

Policy bit 4 adds four five-byte entries to each KV4 record. Each entry
has a channel index followed by its little-endian signed Q16 value in the
quantizer's domain. Entries cover the four largest magnitudes, ties
favoring lower indices. Policy bit 5 appends signed int8 channel
exponents to the image after the exp table, indexed by layer, K/V,
head, channel. The decoder reverses the equalization by powers of two.
Neither extension changes KV8's record size. At context 2048, the
standard KV4, protected KV4, and KV8 caches reserve 7,077,888,
11,010,048, and 13,369,344 bytes, respectively.

## Reproduction

    python3 -B -m tools.c12_smoothquant --original "$original" --out "$out/sweep"
    python3 -B -m tools.c12_smoothquant --original "$original" --out "$out/mixed" \
      --calibration "$out/sweep/calibration.npz" --alphas 0.5 \
      --grouped-a8 --kv4-outliers --smooth-w4-only --kv-equalization

The quality tool runs the C99 quantized program against official BF16 tensors
evaluated with FP32 arithmetic, with full prompt-position logits. The
free-running token match metric compares IDs at the same continuation
offset even when previous decisions differ. EOS is masked for the first 64
decode positions in the fixed-length comparison.

    make -C runtime -j4 OUT="$scratch/native" c11b-qwen
    "$scratch/native/z15_qwen25_chat" \
      --weights "$out/mixed/alpha-0.50.bin" --tokenizer tokenizer.bpe \
      --backend cpu --kv-bits 8 --mode chat --prompt 'Explain FPGA.' \
      --max-new-tokens 64 --trace-steps on

This candidate's Z15 dispatch uses the frozen NPU for exact signed integer
partial dots and GPU kernels for Q16 RoPE and residuals. Scale accumulation,
RMSNorm, causal attention, KV coding and SwiGLU run in the host Q16 path.
The v1 profile is the default. Turbo extension probing at 0x0120 requires an
explicit, trusted deployment profile because the legacy shell responds to
that address with SLVERR. The published Turbo table aperture accelerates
native activation uploads and provides DMA counters. The mixed path does
not use native VPU activation uploads. See the archive for actual replay
scope, quality results and hardware acceptance status.
