# C11 Z15 kernel and CPU chat candidate

The W4A8 projection lowers group-major 128-channel panels to YHB1. GPU unpack runs four 16-lane warps, loads packed words and stores signed bytes directly into the tensor engine's packed layout. The B2 schedule alternates the ownership halves. Q24 scale reduction combines the raw int32 group results using signed 64-bit products. The projection and reduction test tools transfer the first replay's RTL output into the second replay as fixture input.

The real Layer-0 tests capture Qwen tensors from the fixed-point reference. GPU RoPE compares the 14 query and two key heads. Native attention maintains two resident KV groups, seven queries per group, and validates the cache after positions 7, 8 and 9. Native RMSNorm and SwiGLU use the selected signed-byte VPU contracts. Their results are compared with byte-domain C99 and their dequantized error relative to the Q16 model is recorded separately.

The validated operations are independent replay sequences. The Q16 reference and byte-domain native operations still need a common end-to-end scale, activation and KV layout contract. Full Layer-0 hidden states, the autoregressive interaction across kernels and the full-model Z15 runtime are pending. The 2,048-channel projection ablation is scoped to the vocabulary slice.

Build CPU chat with make -C runtime OUT=... CFLAGS=-O2 c11-qwen. It accepts the exported weight image and the binary Byte-Level BPE asset produced by tools.c11_qwen_tokenizer. The CPU backend is standalone C99. Test with --mode bench or --mode chat. The hardware selector returns an error before any device access until the full model is signed off.
