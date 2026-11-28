# Phase C17A software changes for Phase 18B

The S3-H2 tied vocabulary projection can retain its full INT32 partial buffer
in the 10 MiB transient DDR ring. For 151,936 entries and 14 groups that buffer
uses 8,508,416 B. It starts at offset 4 KiB in the ring, after the activation.
The command queue is placed above the ring. Two copies of the 896 B activation
remain in the NPU A slices of the alternating scratch halves while the
existing 496-channel weight and output tiles reuse their B and C slices.
There is one host read and one CMDQ START for the vocabulary projection.
Set YSYX_QWEN_C17=0 to run the original C16 head tiling for comparison.

Four Decode batches retain the existing exact fixed-point dependencies.

| Batch | Host work between batches |
| --- | --- |
| Q/K/V | INT32 group epilogue, Q16 RoPE, KV update, score/softmax/PV, dynamic A8 |
| O | INT32 epilogue, residual, post-attention RMSNorm, dynamic A8 |
| Gate/Up | S3-H2 group epilogue, Q16 SiLU and multiply, dynamic A8 |
| Down | INT32 epilogue, residual and the next layer's RMSNorm and dynamic A8 |

Phase 18B CMDQ can sequence waits and MMIO writes. These host operations
depend on the preceding computed values. A layer with fewer host syncs would
need a hardware-equivalent path for these intermediate operations.
The 16-row Prefill schedule and its W4 workspace sizing are retained.

The Cortex-A53 build accelerates integer reductions, RoPE, the attention
dot and weighted value loops, dynamic A8 division, the SwiGLU product,
KV8 packing/unpacking, greedy selection, bilinear letterbox and RGB
quantization. The exponent lookup and normalization divisions retain exact
scalar arithmetic where needed. The old C12 KV4 path remains available.
Use YSYX_SCALAR_C17 at compile time for the scalar comparison.
YOLO26n uses ranked detections and has no NMS stage.
The test in runtime/tests/c17_simd.c compares Cortex-A53 emulation and
x86 outputs. Emulator CPU timings cannot establish Cortex-A53 board timings.

Linux UIO already synchronizes the actual transfer extent, adjusted to the
mapping page where required, and orders noncached DDR accesses. CMDQ memory
is retained across submissions, so allocations occur only on capacity growth.
