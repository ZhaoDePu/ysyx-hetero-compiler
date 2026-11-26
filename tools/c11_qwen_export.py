"""Export the pinned official Qwen2.5-0.5B-Instruct checkpoint as W4A8."""

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import struct

import numpy as np


REPO = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
MODEL_BYTES = 988097824
MIB = 1024 * 1024
GROUP, LAYERS, HIDDEN, INTER, HEADS, KV_HEADS, HEAD_DIM, VOCAB = (
    128, 24, 896, 4864, 14, 2, 64, 151936,
)
ACT_SHIFT, WEIGHT_SHIFT, EXP_POINTS = 16, 24, 4097
MAGIC = b"YSQW4A8\0"
META = ("config.json", "tokenizer.json", "tokenizer_config.json",
        "vocab.json", "merges.txt", "generation_config.json", "LICENSE")
CONFIG = {
    "model_type": "qwen2", "num_hidden_layers": LAYERS, "hidden_size": HIDDEN,
    "intermediate_size": INTER, "num_attention_heads": HEADS,
    "num_key_value_heads": KV_HEADS, "vocab_size": VOCAB, "hidden_act": "silu",
    "tie_word_embeddings": True, "use_sliding_window": False, "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
}
PROJECTS = (
    ("self_attn.q_proj", HIDDEN, HIDDEN, True),
    ("self_attn.k_proj", KV_HEADS * HEAD_DIM, HIDDEN, True),
    ("self_attn.v_proj", KV_HEADS * HEAD_DIM, HIDDEN, True),
    ("self_attn.o_proj", HIDDEN, HIDDEN, False),
    ("mlp.gate_proj", INTER, HIDDEN, False),
    ("mlp.up_proj", INTER, HIDDEN, False),
    ("mlp.down_proj", HIDDEN, INTER, False),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Tensors:
    def __init__(self, image):
        self.image = image
        header_bytes, = struct.unpack_from("<Q", image)
        if not 0 < header_bytes < MIB:
            raise ValueError("Unexpected safetensors header length")
        self.base = 8 + header_bytes
        self.directory = json.loads(image[8:self.base])

    def bf16(self, name, shape):
        record = self.directory[name]
        if record["dtype"] != "BF16" or tuple(record["shape"]) != shape:
            raise ValueError(f"Unexpected type or shape for {name}")
        first, end = record["data_offsets"]
        if (first < 0 or self.base + end > len(self.image)
                or end - first != 2 * np.prod(shape)):
            raise ValueError(f"Invalid safetensors extent for {name}")
        return np.frombuffer(self.image, "<u2", count=(end - first) // 2,
                             offset=self.base + first).reshape(shape)


def fp32(bf16):
    return (bf16.astype(np.uint32) << 16).view(np.float32)


def quantize_grouped(values):
    """Signed nibbles in conventional two's complement, first value low."""
    rows, cols = values.shape
    if cols % GROUP or not np.isfinite(values).all():
        raise ValueError("W4 requires finite weights and group-aligned columns")
    groups = values.reshape(rows, cols // GROUP, GROUP)
    peak = np.max(np.abs(groups), axis=-1)
    scales = np.maximum(1, np.rint(peak.astype(np.float64) * ((1 << WEIGHT_SHIFT) / 7)))
    if np.any(scales > np.iinfo(np.uint32).max):
        raise ValueError("Weight scale exceeds Q24 uint32")
    ratio = groups.astype(np.float64) * (1 << WEIGHT_SHIFT) / scales[..., None]
    nibbles = np.rint(ratio).clip(-7, 7).astype(np.int8).view(np.uint8) & 15
    flat = nibbles.reshape(rows, cols)
    packed = flat[:, 0::2] | (flat[:, 1::2] << 4)
    return packed, scales.astype("<u4")


class Image:
    def __init__(self, file, tensors):
        self.file = file
        self.tensors = tensors
        self.extents = {}

    def align(self):
        self.file.write(bytes(-self.file.tell() % 64))
        return self.file.tell()

    def matrix(self, name, rows, cols):
        weights = self.tensors.bf16(name, (rows, cols))
        payload = self.align()
        packed_bytes = rows * cols // 2
        scales = (payload + packed_bytes + 63) & ~63
        groups = cols // GROUP
        scale_bytes = rows * groups * 4
        # A group holds contiguous NPU columns, so each tile needs one DMA.
        for first in range(0, rows, 1024):
            packed, multipliers = quantize_grouped(fp32(weights[first:first + 1024]))
            panel = packed.reshape(len(packed), groups, GROUP // 2)
            for g in range(groups):
                self.file.seek(payload + (g * rows + first) * (GROUP // 2))
                self.file.write(panel[:, g, :].tobytes())
                self.file.seek(scales + (g * rows + first) * 4)
                self.file.write(multipliers[:, g].tobytes())
        self.file.seek(scales + scale_bytes)
        self.extents[name] = {"shape": [rows, cols], "packed_offset": payload,
                              "packed_bytes": packed_bytes, "scale_offset": scales,
                              "scale_bytes": scale_bytes,
                              "order": "group, output channel, packed input pair"}

    def q16(self, name, length):
        values = fp32(self.tensors.bf16(name, (length,)))
        values = np.rint(values.astype(np.float64) * (1 << ACT_SHIFT))
        if not np.isfinite(values).all() or np.max(np.abs(values)) > 0x7fffffff:
            raise ValueError(f"Out-of-range Q16 constants in {name}")
        offset = self.align()
        self.file.write(values.astype("<i4").tobytes())
        self.extents[name] = {"offset": offset, "bytes": length * 4}

    def table(self, name, values, dtype):
        offset = self.align()
        self.file.write(values.astype(dtype).tobytes())
        self.extents[name] = {"offset": offset, "bytes": values.size * 4}


def convert(source, output, context):
    if not 32 <= context <= 2048:
        raise ValueError("Context must be between 32 and 2048")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if any(config.get(name) != expected for name, expected in CONFIG.items()):
        raise ValueError("Configuration does not match Qwen2.5-0.5B-Instruct")

    weights = source / "model.safetensors"
    if weights.stat().st_size != MODEL_BYTES or sha256(weights) != MODEL_SHA256:
        raise ValueError("Original weights differ from the pinned official LFS object")
    original = {name: sha256(source / name) for name in META}
    original[weights.name] = MODEL_SHA256
    output.mkdir(parents=True, exist_ok=True)
    target = output / "qwen25-05b.w4a8"
    temporary = target.with_suffix(".part")
    with weights.open("rb") as raw, mmap.mmap(raw.fileno(), 0, access=mmap.ACCESS_READ) as model:
        with temporary.open("wb") as file:
            tensors = Tensors(model)
            file.write(struct.pack("<8s14I", MAGIC, 1, GROUP, LAYERS, HIDDEN, INTER, HEADS,
                                   KV_HEADS, HEAD_DIM, VOCAB, context, ACT_SHIFT, WEIGHT_SHIFT,
                                   EXP_POINTS, round(CONFIG["rms_norm_eps"] * (1 << 32))))
            writer = Image(file, tensors)
            writer.matrix("model.embed_tokens.weight", VOCAB, HIDDEN)
            for layer in range(LAYERS):
                base = f"model.layers.{layer}."
                writer.q16(base + "input_layernorm.weight", HIDDEN)
                writer.q16(base + "post_attention_layernorm.weight", HIDDEN)
                for project, out_ch, in_ch, bias in PROJECTS:
                    writer.matrix(base + project + ".weight", out_ch, in_ch)
                    if bias:
                        writer.q16(base + project + ".bias", out_ch)
            writer.q16("model.norm.weight", HIDDEN)
            positions = np.arange(context, dtype=np.float64)[:, None]
            frequencies = CONFIG["rope_theta"] ** (-np.arange(0, HEAD_DIM, 2) / HEAD_DIM)
            angles = positions * frequencies
            rope = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
            writer.table("rope_q16", np.rint(rope * (1 << ACT_SHIFT)), "<i4")
            exp = np.exp(-np.arange(EXP_POINTS, dtype=np.float64) / 256.0)
            writer.table("exp_q24", np.rint(exp * (1 << WEIGHT_SHIFT)), "<u4")
            byte_count = writer.align()

    kv_bytes = LAYERS * 2 * context * KV_HEADS * HEAD_DIM * 4
    # Operational planning allowance for activation/staging/command buffers.
    allowance = 32 * MIB
    if byte_count > 320 * MIB or byte_count + kv_bytes + allowance >= 460 * MIB:
        raise ValueError("The W4 image and selected context exceed the C11 budgets")
    temporary.replace(target)
    weight_ledger = {
        "model": REPO, "revision": REVISION, "original_sha256": original,
        "file": target.name, "sha256": sha256(target), "bytes": byte_count,
        "group": GROUP, "weight_scale_fraction_bits": WEIGHT_SHIFT,
        "activation_fraction_bits": ACT_SHIFT, "context": context,
        "reference_kv_dtype": "int32 Q16", "reference_kv_bytes": kv_bytes,
        "working_allowance_bytes": allowance,
        "planning_total_bytes": byte_count + kv_bytes + allowance,
        "planning_scope": "Weight image plus Q16 KV and reserved work envelope, not an RTL measurement",
        "tensor_extents": writer.extents, "tied_lm_head": "model.embed_tokens.weight",
        "layers": LAYERS,
    }
    (output / "w4a8.json").write_text(json.dumps(weight_ledger, indent=2) + "\n", encoding="utf-8")
    (output / "OFFICIAL-SHA256SUMS.txt").write_text(
        "".join(f"{sha}  {name}\n" for name, sha in sorted(original.items())), encoding="utf-8")
    print(f"W4A8_EXPORT_PASS layers={LAYERS} bytes={byte_count} kv_bytes={kv_bytes}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", type=int, default=512)
    args = parser.parse_args()
    convert(args.original, args.out, args.context)


if __name__ == "__main__":
    main()
