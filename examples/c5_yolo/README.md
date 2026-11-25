# YOLO26-Tiny-Style, C5

This is a trained, two-scale detector for a controlled geometric task. It
uses an 8×8, four-channel synthetic image. The label space is square and
diamond. Radii, position, ink intensity and background vary by fixed seed.
The four named held-out placements are excluded from training. They cover
single, multiple and overlapping objects at both grid scales. The scene
generator treats labels as amodal boxes and may extend them outside the
visible image. The example is an original task model inspired by direct
YOLO-style dense prediction with one-to-one assignments and no NMS.

A stride-2 spatial convolution, 1×1 residual block, row RMSNorm and a second
stride-2 spatial convolution produce 4×4 and 2×2 feature maps. Each scale
has a learned eight-channel objectness, class and corner-offset prediction
head. Only the designated cell in the designated scale is a positive for
each object. All 400 convolution weights, biases and RMSNorm gains are
optimized with NumPy Adam. The training objective combines weighted
logistic objectness, class cross entropy at positives and corner error.
Training uses 768 images, 900 steps and seed 53026. The second half uses
dyadic int8 fake quantization, signed activation saturation and the fixed
reciprocal-square-root convention, with straight-through gradients.

The export contains standard ONNX quantized convolution, Q/DQ residual and
RMSNorm, and a no-NMS GPU tail. The tail computes a deterministic Top-K
on the 20 objectness values, gathers each selected row, takes class ArgMax
and adds its grid coordinates. GPU TopK/Gather accept a bounded integer
subset, preserving ascending original index for tied scores. Host
presentation applies sigmoid and a strict 0.5 cutoff to the six selected
scores. This filtering cannot suppress an overlapping box based on a peer.

`YOLO26_Tiny_Style_INT8.onnx` is the single-image deployable fixture.
`YOLO26_Tiny_Style_4screen_INT8.onnx` contains the same graph instantiated
for four separate runtime inputs so the scheduler can overlap frames.
Every full ablation uses all four held-out inputs. The four basic rungs use
a pure GPU, NPU-first heterogeneous placement, then compatible fusion and
reuse, then two-half overlap. The detector VPU-off benchmark disables all VPU
placement, including the preexisting residual or activation paths. The
VPU-on optimized rungs contain native normalization.

Generate into a fresh directory under Phase C5 RAM. This invocation retrains
the detector, emits both ONNX files, measures all eight distinct hardware
runs, checks each published DDR output against the selected fixed-point
reference and builds a self-contained single-image C99 case. A matching
frozen Verilator runner and C99 bridge must already be built.

```bash
export TMPDIR=/dev/shm/ysyx_compiler_c5 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 10 python3 -B -m tools.run_c5_yolo_eval \
  --out /dev/shm/ysyx_compiler_c5/detector-qualification
```

Each expected tensor also survives repeated C99 runs of the same single
YHB with changed runtime image bytes. The image is named `image`. The
outputs are signed-byte `scores` (logit scale 1/4), int64 `indices` and
`classes`, and int32 `boxes` in x1/y1/x2/y2 order (four codes per pixel).
The kit includes all four input binaries and a smoke configuration using
the single-object input.

The independent floating model reuses the trained real-valued parameters
with continuous convolution and RMSNorm. IoU, class agreement, confidence
error and IoU≥0.5 precision/recall are scored on the held-out set. These
are controlled-task measurements and are limited to this four-frame set.
Each FPS column divides the measured frozen RTL cycles by its named clock
and uses all four frames in the numerator. Physical-board performance
requires a separate run.
