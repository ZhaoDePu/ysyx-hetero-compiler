# Valence（蔚澜） C13 Qwen2.5-0.5B offload candidate

C13 reuses the C12 SmoothQuant image byte for byte. KV8 remains the default.
The candidate adds explicit GPU modes for Q16 RMSNorm, SwiGLU, and causal GQA
with a GPU-encoded KV8 cache in PL DDR. C12 remains available as the default
dispatch mode.

| CLI mode | Added GPU operations |
| --- | --- |
| c12 | C12 schedule |
| rms | Pre-attention, pre-MLP and final RMSNorm |
| swiglu | RMSNorm and SwiGLU |
| all | RMSNorm, SwiGLU, GQA and PL DDR KV8 |

The folded SmoothQuant gamma is consumed by the Q16 RMSNorm kernel. A shared
integer reciprocal and one residual check give exactly rounded Q16 results.
SwiGLU uses the C12 exponential table and exact integer nearest rounding.
Projection group-scale accumulation and dynamic A8 preparation still execute
in C99. The beta dispatch stages most vectors through host memory. The mode
named all selects all added kernels.

## Cache and GPU memory

Each layer has one 2,048-entry DDR arena beyond the projection staging ring.
A slot has four C12-compatible 68-byte KV8 head records and zero padding to
288 bytes for the frozen DMA alignment. The total reservation is 14,155,776
bytes for 24 layers. K and V are quantized and stored by GPU and DMA. KV cache
readback occurs only when the replay harness requests a verification sample.

The GPU uses the frozen 18-bit data window and its own per-lane stack
initialization. A 128-token SRAM tile holds K and V, and the scratch layout
holds scores for 14 query heads over the maximum context. The score sweep
updates a per-head maximum across tiles. The second sweep accumulates the
unrounded 64-bit weighted sums and the denominator before one final division.
This avoids introducing partial softmax rounding at tile edges.

The program occupies a separate IMEM slot and is uploaded once after the GPU
is idle. The exponential table is also reused across dispatches. The v1
control ABI is unchanged. Turbo probing remains opt-in and requires known
image provenance, because probing the extension on the v1 shell can return
SLVERR.

## Reproduction

Use a frozen Step 2 source build in explicit single-thread simulation and
supply the C12 model through its existing path.

    make -C runtime -j4 OUT="$scratch/native" c13-qwen
    "$scratch/native/z15_qwen25_chat_c13" --backend cpu \
      --weights "$c12_kit/model.w4a8.bin" --tokenizer "$c12_kit/tokenizer.bpe" \
      --kv-bits 8 --prompt 'Explain a systolic array.' --max-new-tokens 64

The C13 host emulator uses the same freestanding kernel C sources compiled
as ordinary C instead of the RV32E image. Its replay compares entire
prompt-position logits and generated IDs to C12. Frozen RTL results cover
only the logged operator and layer runs. Published quality numbers for
bit-exact emulator results refer to the archived C12 FP32 comparison.

The candidate does not constitute full-PL acceptance. In particular, host
projection epilogues, activation quantization and host staging remain open.
