# C18 resident Qwen command plans

The model keeps the public S3-H2 file structure. For each selected dual-plane
W4 matrix, the device's existing DDR allocation is repacked once from panel
order to K-group order. Native TC produces unscaled group sums for wider
output panels. The host applies the same dynamic coefficient and ties-to-even
rounding that the C17 on-chip accumulation used on each individual partial.
The CPU reference and the public model weights use the original layout.

The decode input, result and weight offsets are fixed for each projection.
The complete CMDQ batch is recorded on first use and placed in a disjoint
high-address arena within the existing DDR reservation. The initial register
values are written independently of the previous batch. Later submissions
ring the same descriptor with zero newly generated command words and zero
upload bytes. There is no scale or KV patch table because those operations
remain in the exact host epilogue. The switch YSYX_QWEN_C18=0 selects the
C17 schedule for comparisons.

Warmup includes device weight repacking and one-time command uploads.
The wider raw-group path trades additional intermediate DDR reads for fewer
TC micro-operations and a static command stream. Report total DDR traffic,
steady upload bytes and host CPU time separately. The CLI uses a thread CPU
clock for host_cpu_ms and a monotonic wall clock for ms. Reference backends
provide CPU timings and functional counts.

The generic YHB runtime keeps forced CMDQ mode. Auto mode selects direct
MMIO when a KV checkpoint segment averages fewer than 128 commands per
START. The filter leaves visual graphs on their existing path.
