# Phase C3 Full qualification

The frozen Phase 12C commit
`4d410d72508a5822bc5c8014981e5b55ef8faa56` contains a 24×24 MaxNPU top
and a distinct 16×16 Balanced top. Both instantiate the same
`zu15eg_tensor_wide`, with 16 wide words and two-cycle registered reads.
The independently frozen Z15 board MMIO shell instantiates the Balanced
top and exposes four GPU doorbell slots. The available generated G3 cluster
is the Balanced one. The Phase 12C MaxNPU artifact keeps the hash of the
generated 8-slot cluster; the corresponding generated source is absent.
Its exact generator needs Yosys.

The Full Profile describes the 8T/8W core geometry and conservatively limits
MMIO launch waves to four slots. The emitted GPU job words have bits [31:26]
clear. Target-2 packages can be inspected by the C99 reader. On-board
execution still needs a frozen Full MMIO binding.

## Direct frozen tensor measurement

The C++ harness drives the unaltered frozen tensor module using compile-time
parameters from the two frozen tops. It applies the NPU register values
emitted from one standard static `MatMulInteger` ONNX with M/N/K = 72/72/64,
and models the fixed two-cycle wide memory return. It compares the entire
2 MiB scratch image after execution, including the 20,736 B int32 result.
Cycle counts cover native command launch through done. They exclude host
setup, AXI, DMA, GPU, and board clock crossing.

| Array | Measured NPU cycles | NPU Fmax conversion | Scratch comparison |
| --- | ---: | ---: | --- |
| Balanced 16×16 | 961 | 5.840 μs at 164.555 MHz | 2 MiB exact |
| Full 24×24 | 561 | 3.666 μs at 153.022 MHz | 2 MiB exact |

The observed NPU cycle ratio is 1.713×, with a 1.593× Fmax-converted ratio.
Fmax values are from the frozen Phase 12C route archive. This is a
geometry-controlled tensor comparison. The same binary's Balanced MMIO
replay passes the 20,736-B output with 2,428 wall ticks, 960 tensor busy
ticks, 29,952 DMA bytes and 9 AXI transactions. A Full board end-to-end
advantage needs its own MMIO replay.

```sh
bash tools/build_c3_target_tensor.sh
nice -n 10 ionice -c2 -n7 python3 -B -m tools.compare_full_c3_target \
  --out /dev/shm/ysyx_compiler_c4vpu/targets/full-compare
```
