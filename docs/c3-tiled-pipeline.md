# Phase C3 Z15 graph-tile pipeline

The two fixtures are actual quantized ONNX graphs for throughput evaluation.
`StreamVision_MultiTile` has 24 leading-batch image tiles, a 3×3 and a 1×1
convolution with activation and residual, plus a GPU class selection tail.
`StreamLLM_MultiTile` contains 36 independent 8-token causal contexts, each
with Q/K/V, causal QK→Softmax→PV, projections, SwiGLU and residual paths.
All tokens for a context stay together so causal attention retains its full
prefix. The fixtures use deterministic synthetic weights. Accuracy evaluation
uses the separate trained C3 models.

The planner proves leading-batch tiles or disjoint runtime-input components,
then runs ordinary placement and fusion per tile. Tiles retain intermediate
values in one 1 MiB half. Native weight encodings share immutable DDR storage.
Tile loads and writebacks use the original runtime tensor bindings and
aligned DMA extents. The two halves alternate. The list scheduler starts work
only after a same-half dependency and any earlier use of the engine have
drained. The serial ablation drains each operation before dispatching another.
The two ownership halves permit two overlapping owners. Three simultaneous
owners would need another independent ownership domain.

The coverage includes one leading batch slice with supported layout operators
or multiple closed input components. A global operation across the chosen
batch axis or a byte-misaligned boundary prevents this decomposition. The
existing C2 staged lowering remains available for elementwise or GEMM tiling.

Regenerate and measure on the pinned Balanced binding:

```sh
nice -n 10 ionice -c2 -n7 python3 -B -m examples.c3_streams.generate
bash tools/build_npu_z15_rtl.sh
nice -n 10 ionice -c2 -n7 python3 -B -m tools.run_c2_ablation --pipeline-tiles \
  --out /dev/shm/ysyx_compiler_c4a/stream-rtl \
  examples/c3_streams/StreamVision_MultiTile.onnx examples/c3_streams/StreamLLM_MultiTile.onnx
```

Gate success requires four byte-exact runs per graph with strictly decreasing
measured wall ticks and real A3 overlap. DMA bytes count successful AXI R
beats and W strobes in the passive harness. AXI transactions count successful
address handshakes. The JSON and CSV retain the static cost model separately;
reported RTL latency projects measured NPU-domain ticks at 164.555 MHz.
