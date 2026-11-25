# Phase C3 deterministic training and accuracy

Both classifiers train genuine learned weights offline with NumPy, Adam and a
fixed PCG seed. Training, model selection, range calibration and test use
independently seeded datasets. Each task uses 1,024 training examples and 256
examples in each other split. Training starts from a random initialization and
uses backpropagation through every weight, including attention projections.

| Model | Data and supervised target | Network |
| --- | --- | --- |
| Vision | Four classes of noisy translated strokes, with random per-channel gain and distractors | Three trained spatial convolutions, two ReLUs, four logits |
| Transformer | Four-step input with independent noisy token and position features. At positions 1–3, predict the preceding token | One trainable head with Q, K and V projections, scaled dot product with a standard additive causal mask, Softmax, output projection, residual, two-layer feed-forward path, residual and classifier |

The benchmark datasets are generated to exercise these tasks. The trained
parameters should not be used to make performance claims about natural images
or language corpora. The lag task uses the preceding token, which is independent
of the token at the scored position. Position 0 has no preceding token and is
excluded from accuracy metrics. All output bytes are still validated, including
position 0.

The exports use standard ONNX opset 18. The quantized models take signed INT8
inputs and return signed INT8 logits and INT64 ArgMax indices. Per-tensor scales
are symmetric powers of two from the independent calibration split, and vision
biases are INT32. The Transformer uses a standard
DequantizeLinear–Add(causal mask)–Softmax–QuantizeLinear path. The mask is float32
zero or negative infinity, score scale is 1/16 and probability scale is 1/256
unsigned. Residuals use static DQ–Add–Q.

The accuracy driver executes the exported FP32 ONNX bytes with an independent
NumPy interpreter for the model's standard operators. The RTL binary runs the
compiler-generated YHB1, applies independent held-out inputs at runtime, and
checks each result against the compiler's fixed-point oracle. The default
qualification policy is `npu-greedy`, which explicitly offloads eligible raw
convolutions and projections to the NPU and preserves exact non-native
requantization on the GPU. The independently
read output bytes determine INT8 predictions and dequantized logit errors.
Cosine, MAE and maximum absolute logit error are aggregate values over the
scored examples/positions. The wall-time counter is the frozen Verilator C99
MMIO NPU-domain counter. Frequency conversion uses 164.555 MHz from the Phase
12C Balanced routed estimate.

From the repository root on xu4090, using the existing C3 Verilator runtime:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/dev/shm/ysyx_compiler_c4vpu/training
mkdir -p "$TMPDIR"
nice -n 10 ionice -c2 -n7 python3 -B -m examples.c3_training.train --out "$TMPDIR/models"
nice -n 10 ionice -c2 -n7 python3 -B -m tools.c3_accuracy \
  --models "$TMPDIR/models" --out "$TMPDIR/acceptance" --vectors 16
```

The output contains training curves, calibration scales, model hashes,
per-sample CLI logs and binary outputs, and aggregate JSON and CSV. The checked-in
ONNX fixtures and their training manifests allow model-byte comparison against
a fresh offline training run.
