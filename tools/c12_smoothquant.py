"""Calibrate and export compact, signed W4/W8 Qwen2.5 candidates offline."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import struct

import numpy as np

from tools.c11_qwen_export import (CONFIG, EXP_POINTS, HEAD_DIM, HEADS, HIDDEN, INTER,
                                  KV_HEADS, LAYERS, MODEL_BYTES, MODEL_SHA256, PROJECTS,
                                  REVISION, VOCAB, sha256)

ALPHAS = (0.5, 0.6, 0.75, 0.85)
SCALE = 1 << 24
MAGIC = b"YSQC12\0\0"
DDR_BYTES = 512 << 20
WORKSPACE_BYTES = 32 << 20
LAYOUT = ("attention", "down", "head")
# An independent corpus for activation statistics. No evaluation prompt is used.
CALIBRATION = (
    "请解释为什么光纤比普通铜线更适合跨城市传输。请举一个实际应用。",
    "如何安排一次周末的城市博物馆参观？列出三个具体步骤。",
    "请比较普通插座和带过载保护的插座的用途，用简单中文回答。",
    "请说明火车通过弯道时为什么会有横向的力，并解释怎么减轻。",
    "Give a short explanation of how a battery stores and releases energy.",
    "Draft a short email to reschedule a meeting from Monday to Wednesday.",
    "Why do consistent units matter when combining measurements from multiple sensors?",
    "Explain when an embedded application might benefit from using a ring buffer.",
    "Write Python code that normalizes whitespace in a list of strings.",
    "Implement binary search in Python and explain what it returns for missing entries.",
    "给出检测括号是否平衡的 Python 示例。请解释栈的作用。",
    "请用 Python 实现对两个直方图的逐项求和，处理长度不同的输入。",
    "Explain how a packet is forwarded through a network, including the role of a router.",
    "Describe a concrete way to reduce the memory footprint of an image processing application.",
    "请用一个小例子说明无符号整数溢出，并区分溢出和舍入误差。",
    "Explain why a capacitor blocks steady DC but can pass a changing signal.",
)
VALIDATION = (
    "请用三个步骤说明如何对采集的温度序列做去噪。",
    "Explain how a light sensor can automatically turn on a lamp.",
    "Write a Python function that merges two sorted lists. Include a short example.",
    "请说明为什么公开演讲前做试讲很有帮助。",
)


def pack4(values):
    """Store even/odd indices in low/high signed two's-complement nibbles."""
    if values.shape[-1] % 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("W4 requires an aligned, integer trailing dimension")
    if np.any(values < -8) or np.any(values > 7):
        raise ValueError("W4 is outside the signed nibble range")
    codes = values.astype(np.uint8) & 15
    return codes[..., ::2] | (codes[..., 1::2] << 4)


def unpack4(packed):
    values = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=np.int8)
    values[..., ::2] = (packed & 15).astype(np.int8)
    values[..., 1::2] = (packed >> 4).astype(np.int8)
    values[values >= 8] -= 16
    return values


def smooth_scale(peak, weights, alpha):
    """Use the maximum W magnitude across the projections sharing a norm."""
    activation = np.maximum(np.asarray(peak, dtype=np.float32), 1e-5)
    magnitude = np.maximum(np.asarray(weights, dtype=np.float32), 1e-5)
    result = activation ** alpha / magnitude ** (1 - alpha)
    return np.clip(result, 0.015625, 64.0)


def group_quantize(weight, group, bits, importance=None):
    """Choose symmetric clipping by minimizing diagonal activation-weighted error."""
    if bits not in (4, 8) or group not in (64, 128):
        raise ValueError("Unsupported bit width or weight group")
    if weight.ndim != 2 or weight.shape[1] % group or not np.isfinite(weight).all():
        raise ValueError("Weights must be finite and group aligned")
    w = weight.reshape(len(weight), -1, group)
    limit = (1 << (bits - 1)) - 1
    peak = np.max(np.abs(w), axis=-1)
    diag = 1 if importance is None else importance.reshape(-1, group)
    best = None
    for clip in ((1.0, .98, .94, .90, .85, .80, .74) if bits == 4 else (1.0,)):
        scales = np.maximum(1, np.rint(peak * (clip * SCALE / limit))).astype("<u4")
        step = scales.astype(np.float32) / SCALE
        quantized = np.rint(w / step[..., None]).clip(-limit, limit).astype(np.int8)
        if bits == 8:
            return quantized, scales
        loss = np.sum((w - quantized * step[..., None]) ** 2 * diag, axis=-1)
        if best is None:
            best = loss
            chosen = quantized
            multipliers = scales
        else:
            improve = loss < best
            best[improve] = loss[improve]
            chosen[improve] = quantized[improve]
            multipliers[improve] = scales[improve]
    return chosen, multipliers


def load_official(path, threads):
    import torch
    import transformers

    weights = path / "model.safetensors"
    if weights.stat().st_size != MODEL_BYTES or sha256(weights) != MODEL_SHA256:
        raise ValueError("Official LFS object differs from the pinned SHA-256")
    cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if any(cfg.get(name) != expected for name, expected in CONFIG.items()):
        raise ValueError("Unexpected official model configuration")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float32, local_files_only=True,
        attn_implementation="eager").eval()
    return torch, tokenizer, model


def chat_ids(tokenizer, text):
    return tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                         tokenize=True, add_generation_prompt=True)


def collect(torch, model, tokenizer):
    """Record per-input peaks and uncentered second moments from FP32 only."""
    peaks, squares, counts = {}, {}, defaultdict(int)
    hooks = []

    def hook(name):
        def add(_module, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1])
            p = x.abs().amax(dim=0).numpy()
            s = x.square().sum(dim=0).numpy()
            peaks[name] = p if name not in peaks else np.maximum(peaks[name], p)
            squares[name] = s if name not in squares else squares[name] + s
            counts[name] += len(x)
        return add

    for i, layer in enumerate(model.model.layers):
        for proj, _, _, _ in PROJECTS:
            name = f"{i}.{proj}"
            hooks.append(layer.get_submodule(proj).register_forward_pre_hook(hook(name)))
    hooks.append(model.lm_head.register_forward_pre_hook(hook("head")))
    try:
        with torch.inference_mode():
            for prompt in CALIBRATION:
                ids = chat_ids(tokenizer, prompt)
                answer = model(torch.tensor([ids]), use_cache=True)
                for _ in range(8):
                    token = answer.logits[:, -1:].argmax(dim=-1)
                    answer = model(token, past_key_values=answer.past_key_values,
                                   use_cache=True, logits_to_keep=1)
    finally:
        for handle in hooks:
            handle.remove()
    moments = {k: v / counts[k] for k, v in squares.items()}
    return peaks, moments, dict(counts)


def kv_exponents(torch, model, tokenizer):
    """Calibrate post-RoPE K and projected V on the same independent corpus."""
    from transformers.cache_utils import DynamicCache

    captures = [[] for _ in range(LAYERS)]

    class Observe(DynamicCache):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            sample = torch.stack((key_states[0], value_states[0]))
            captures[layer_idx].append(sample.permute(2, 0, 1, 3).numpy().copy())
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

    with torch.inference_mode():
        for prompt in CALIBRATION:
            ids = torch.tensor([chat_ids(tokenizer, prompt)])
            result = model(ids, past_key_values=Observe(), use_cache=True)
            for _ in range(8):
                token = result.logits[:, -1:].argmax(dim=-1)
                result = model(token, past_key_values=result.past_key_values,
                               use_cache=True, logits_to_keep=1)

    peak = np.stack([np.quantile(np.abs(np.concatenate(layer)), .975, axis=0)
                     for layer in captures])
    relative = np.maximum(peak, 1e-5) / np.maximum(np.median(peak, axis=-1, keepdims=True), 1e-5)
    return np.rint(np.log2(relative)).clip(-8, 8).astype(np.int8)

class Exporter:
    def __init__(self, file, group):
        self.file = file
        self.group = group
        self.parts = {}

    def align(self):
        self.file.write(bytes(-self.file.tell() % 64))
        return self.file.tell()

    def q16(self, vector):
        self.align()
        q = np.rint(vector * 65536.0)
        if not np.isfinite(q).all() or np.any(np.abs(q) > 0x7fffffff):
            raise ValueError("Q16 value cannot be represented")
        self.file.write(q.astype("<i4").tobytes())

    def matrix(self, name, w, bits, second_moment):
        start = self.align()
        rows, cols = w.shape
        stride = self.group * bits // 8
        groups = cols // self.group
        scales = (start + rows * cols * bits // 8 + 63) & ~63
        for r in range(0, rows, 1024):
            quant, scale = group_quantize(w[r:r + 1024], self.group, bits, second_moment)
            encoded = pack4(quant) if bits == 4 else quant
            for g in range(groups):
                self.file.seek(start + (g * rows + r) * stride)
                self.file.write(encoded[:, g].tobytes())
                self.file.seek(scales + (g * rows + r) * 4)
                self.file.write(scale[:, g].tobytes())
        self.file.seek(scales + groups * rows * 4)
        self.parts[name] = {"shape": [rows, cols], "bits": bits, "offset": start,
                            "scale_offset": scales, "extent_end": self.file.tell()}


def export(model, peaks, moments, out, alpha, group, context, protect, grouped_a8=False, kv4_outliers=False, smooth_w4_only=False, kv_scale=None):
    path = out.with_suffix(".part")
    alpha_value = 0 if alpha is None else round(100 * alpha)
    mask = sum(1 << bit for bit, name in enumerate(LAYOUT) if name in protect)
    if grouped_a8:
        mask |= 8
    if kv4_outliers:
        mask |= 16
    if kv_scale is not None:
        mask |= 32

    def f32(parameter):
        return parameter.detach().numpy()

    with path.open("wb") as file:
        file.write(struct.pack("<8s14I", MAGIC, 1, group, LAYERS, HIDDEN, INTER,
                               HEADS, KV_HEADS, HEAD_DIM, VOCAB, context, mask, 24,
                               EXP_POINTS, round(CONFIG["rms_norm_eps"] * (1 << 32))))
        writer = Exporter(file, group)
        writer.matrix("embedding/lm_head", f32(model.lm_head.weight),
                      8 if "head" in protect else 4, moments["head"])
        for l, layer in enumerate(model.model.layers):
            proj = {name: layer.get_submodule(name) for name, *_ in PROJECTS}
            qkv = tuple("self_attn." + name + "_proj" for name in ("q", "k", "v"))
            gate_up = ("mlp.gate_proj", "mlp.up_proj")
            shifts = {}

            def scale(names):
                if alpha is None:
                    return np.ones(proj[names[0]].in_features, dtype=np.float32)
                names = tuple(names)
                weight_max = np.maximum.reduce([np.max(np.abs(f32(proj[name].weight)), axis=0)
                                                for name in names])
                return smooth_scale(peaks[f"{l}.{names[0]}"], weight_max, alpha)

            sqkv = (np.ones(HIDDEN, dtype=np.float32)
                    if smooth_w4_only and "attention" in protect else scale(qkv))
            sffn = scale(gate_up)
            sdown = (np.ones(INTER, dtype=np.float32)
                     if smooth_w4_only and "down" in protect else scale(("mlp.down_proj",)))
            writer.q16(f32(layer.input_layernorm.weight) / sqkv)
            writer.q16(f32(layer.post_attention_layernorm.weight) / sffn)
            shifts.update((name, sqkv) for name in qkv)
            shifts.update((name, sffn) for name in gate_up)
            shifts["mlp.down_proj"] = sdown
            for name, module in proj.items():
                key = f"{l}.{name}"
                scale_in = shifts.get(name, 1)
                weight = f32(module.weight) * scale_in
                if name == "mlp.up_proj":
                    # SiLU(gate) * (up / s) is algebraically (SiLU(gate) * up) / s.
                    weight = weight / sdown[:, None]
                bits = 8 if (name.startswith("self_attn.") and "attention" in protect
                             or name == "mlp.down_proj" and "down" in protect) else 4
                writer.matrix(key, weight, bits, moments[key] / (scale_in * scale_in))
                if module.bias is not None:
                    writer.q16(f32(module.bias))
        writer.q16(f32(model.model.norm.weight))
        file.write(bytes(-file.tell() % 64))
        angle = np.arange(context, dtype=np.float64)[:, None] / (
            CONFIG["rope_theta"] ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM))
        writer.q16(np.stack((np.cos(angle), np.sin(angle)), axis=-1))
        writer.align()
        exp = np.rint(np.exp(-np.arange(EXP_POINTS, dtype=np.float64) / 256) * SCALE)
        file.write(exp.astype("<u4").tobytes())
        if kv_scale is not None:
            writer.align()
            file.write(kv_scale.tobytes())
        size = writer.align()

    # One Q16 scale per K/V token and 64-channel head, shared by both cache widths.
    kv4 = LAYERS * context * 2 * KV_HEADS * (HEAD_DIM // 2 + 4 + (20 if kv4_outliers else 0))
    kv8 = LAYERS * context * 2 * KV_HEADS * (HEAD_DIM + 4)
    if size + kv8 + WORKSPACE_BYTES > DDR_BYTES:
        path.unlink()
        raise ValueError("Model and cache exceed the Z15 DDR pool")
    path.replace(out)
    return {"revision": REVISION, "original_sha256": MODEL_SHA256, "alpha": alpha_value / 100,
            "group": group, "context": context, "W8": sorted(protect), "bytes": size,
            "kv4_outliers_per_head": 4 if kv4_outliers else 0,
            "activation_scale": "per_token_group" if grouped_a8 else "per_token",
            "smooth_w4_only": smooth_w4_only,
            "kv4_channel_equalization": kv_scale is not None,
            "kv4_bytes": kv4, "kv8_bytes": kv8, "workspace_reserved_bytes": WORKSPACE_BYTES,
            "kv4_estimated_ddr_bytes": size + kv4 + WORKSPACE_BYTES,
            "kv8_estimated_ddr_bytes": size + kv8 + WORKSPACE_BYTES,
            "tensors": writer.parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--context", type=int, choices=range(128, 2049), default=2048)
    parser.add_argument("--group", type=int, choices=(64, 128), default=64)
    parser.add_argument("--alphas", type=float, nargs="+", choices=ALPHAS, default=ALPHAS)
    parser.add_argument("--protect", nargs="*", choices=LAYOUT, default=LAYOUT)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--include-no-smooth", action="store_true")
    parser.add_argument("--grouped-a8", action="store_true")
    parser.add_argument("--kv4-outliers", action="store_true")
    parser.add_argument("--smooth-w4-only", action="store_true")
    parser.add_argument("--kv-equalization", action="store_true")
    args = parser.parse_args()
    torch, tokenizer, model = load_official(args.original, args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.calibration:
        with np.load(args.calibration) as records:
            peaks = {key[2:]: records[key] for key in records if key.startswith("p/")}
            moments = {key[2:]: records[key] for key in records if key.startswith("m/")}
    else:
        peaks, moments, counts = collect(torch, model, tokenizer)
        np.savez(args.out / "calibration.npz", **{f"p/{k}": v for k, v in peaks.items()},
                 **{f"m/{k}": v for k, v in moments.items()})
        (args.out / "calibration.json").write_text(json.dumps({
            "checkpoint": REVISION, "corpus": CALIBRATION, "projection_token_counts": counts,
            "decode_calibration_steps": 8,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    kv_scale = None
    if args.kv_equalization:
        print("calibrating post-RoPE KV channel exponents", flush=True)
        kv_scale = kv_exponents(torch, model, tokenizer)
        np.save(args.out / "kv-exponents.npy", kv_scale)
    alphas = list(args.alphas) + ([None] if args.include_no_smooth else [])
    for alpha in alphas:
        name = f"alpha-{alpha:.2f}" if alpha is not None else "no-smoothing"
        out = args.out / f"{name}.bin"
        print(f"exporting {name} (G{args.group}, W8={','.join(args.protect)})", flush=True)
        record = export(model, peaks, moments, out, alpha, args.group, args.context,
                        args.protect, args.grouped_a8, args.kv4_outliers, args.smooth_w4_only, kv_scale)
        out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
