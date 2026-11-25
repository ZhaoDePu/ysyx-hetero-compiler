# Phase C6 YOLO26n on Z15 Balanced

The C6 path accepts the calibrated model IR and coefficient archive produced by
`tools/c6_ptq.py`. It keeps the frozen `z15-mmio-abi-v1` register interface and
serializes the resulting schedule as YHB1. The following commands build the board
program and a replay for one quantized evaluation frame.

```sh
PYTHONPATH=. python3 tools/compile_c6_yolo.py --quant "$PTQ" --output "$BOARD"
PYTHONPATH=. python3 tools/verify_c6_model.py --quant "$PTQ" \
    --input-nhwc "$PTQ/vectors/bus.input_nhwc_s8.bin" \
    --output "$REPLAY" --runner "$RUNNER"
```

The board program takes one external `images` input. It is signed INT8 NHWC
with a zero fourth channel. The six `boxes_stride*` and `scores_stride*`
outputs expose the signed INT8 head tensors in NHWC order. The
`c6_head_scales` and `c6_score_lut` metadata outputs store IEEE binary32
scale and score bits in little-endian words. The compiler verifies the static
half-cell anchor grid and stride array against the exported graph. The host
decoder applies coordinate scaling and the two-stage Top-K from the checked
ONNX tail. The manifest records the source tensor name for each output symbol.

Immutable weights and tables, frame bindings, and intermediate activations
occupy three contiguous regions in the 512 MiB DMA aperture. Activation slots
reuse aligned ranges after their last consumer drains. Frame input storage can
be reclaimed after its last consumer within a synchronous invocation. DDR
initial contents are sparse and contain weights plus dedicated metadata. Each
dynamic DDR read follows its producer's DMA fence. Scratch tiles alternate
between two independently bounded 1 MiB ownership halves. Within a layer the
schedule uses the opposite half for the previous tile's writeback and the next
tile's prefetch.

Native dense convolution loads an aligned halo interval and programs the
stride phase with `NPU_SLICE`. Its constant layout is output-channel-major,
followed by kernel y, kernel x, and padded input channels. Per-channel
quantization uses a signed INT32 accumulator multiplied by an unsigned 16-bit
coefficient, an arithmetic right shift, saturation to signed INT8, and an
optional 256-byte activation lookup. The last output transfer can include DMA
alignment padding outside the logical symbol's byte extent.

When the reduction dimension exceeds 511, VPU `SPLIT` partitions input
channels. Each channel partition produces raw INT32 values on the NPU. The GPU
adds these partials and evaluates one final 48-bit product and optional
activation per element. Bias is included in the first partition. Depthwise
convolutions use 16-lane block-diagonal native filters with VPU splits and joins.
This preserves byte-level arithmetic while allowing the frozen DOT4 lanes to
consume standard NHWC channel blocks.

Image activations use NHWC. Attention views use contiguous logical rows and
the frozen elementwise transpose port moves data across layouts. Native MatMul
packs its right operand into per-column order. An unsigned probability operand
is moved to the native unsigned-A port through the identity
`(V P)^T = P^T V^T`. Q23 Softmax uses a graph-specific exponential lookup on the
GPU. The model manifest records actual scheduled engines for every hardware
node and keeps the ONNX host-tail operators with their input/output names.

`verify_c6_model.py --layer GLOB` replays an official layer subset with
independent fixed-point oracle inputs and outputs. With no selector, it checks
all six raw output maps for the full graph in the frozen shell. C99 decoding
can then be checked against the separate final coordinate, score and class
vectors. The direct RTL runner reports the combined native tensor/VPU counter,
GPU, DMA, and overlap counters in fabric-clock ticks. The control trace retains
individual stage and engine assignments.

The portable C99 reference HAL in `runtime/hal_reference_c6.c` also runs an
external-input board YHB through the ordinary C runtime. It executes the
emitted descriptor arithmetic and the C6 GPU instruction subset against host
memory and reports functional results. Its hardware cycle counters read zero.
The standalone test consumes the same quantized scene vectors as the RTL
checks and retains one HAL instance across frames.

```sh
gcc -std=c99 -O2 -Wall -Wextra -Werror -I runtime \
    runtime/tests/c6_reference.c runtime/hal_reference_c6.c runtime/ysyx_rt.c \
    -o "$SCRATCH/c6_reference"
"$SCRATCH/c6_reference" "$BOARD/program.yhb" "$PTQ/vectors" \
    bus zidane coco000000000241 coco000000000384
```

The C99 software reference runs the same YHB1 command stream, through its own
DMA, native arithmetic and instruction interpretation. It emits no simulated
hardware ticks. The PetaLinux integration and media formats are described in
[the deployment README](../deploy/z15_yolo26n_320_kit/README-YOLO26N-Z15.md).

The frozen Verilator rig uses a GPU:NPU clock ratio of 2:3 by default to
preserve the historic qualification runs. Set `YSYX_Z15_SIM_GPU_RATIO=4:5`
for the Step 2 100:125 ratio. The optional build writes a separate executable
beneath the selected compiler scratch directory. It only changes the input
clock stimulus.

On xu4090 the C6 build selects its dedicated scratch root explicitly.

    export YSYX_COMPILER_SCRATCH=/dev/shm/ysyx_compiler_c6
    mkdir -p "$YSYX_COMPILER_SCRATCH/npu"
    nice -n 10 bash tools/build_npu_z15_rtl.sh
    nice -n 10 env YSYX_Z15_SIM_GPU_RATIO=4:5 bash runtime/build_verilator.sh

The separate VPU softmax path widens byte scores and exercises the frozen
power-of-two exponent kernel. The release keeps the GPU Q23 exponential
lookup because the four-scene fixed-point comparison has higher agreement.
