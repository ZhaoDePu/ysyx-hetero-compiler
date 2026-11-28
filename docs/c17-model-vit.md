# C17 model candidate

ViT-Tiny/16 uses the timm augreg_in21k_ft_in1k release. Its public
model.safetensors has SHA-256
fecf81b492bd13ee7a5297cb74d1d417aac8bf7e1b7d96aed89c4691984587ed.
Use the matching config.json from that release. The exporter requires
NumPy, Pillow and ONNX 1.17.0 with protobuf. It verifies the weight hash and
runs the ONNX full checker on both FP32 and opset-19 Q/DQ graphs.

Run python -m tools.c17_vit with --weights and --config, distinct image
paths following --cal and --test, and a new --out directory. Per-row W8
scales apply to the projection weights. Static A8 scales come from the
calibration images. LayerNormalization, Softmax and GELU remain in the
float domain in the interchange graph. The sample fixtures contain
the normalized image and both NumPy reference logits.

The C17 candidate has software evidence for the entire model. The existing
Valence-C integer import rejects the FP32 ONNX interface. Hardware
qualification also needs lowering for LayerNormalization, mixed-precision
attention, and the transformer layout at 197 tokens. The exported graphs
are suitable for host replay. A Phase 18B YHP1 and board CLI require that
remaining compiler work.
