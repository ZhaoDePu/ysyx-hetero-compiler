"""Train deterministic tiny vision and causal sequence classifiers with NumPy."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

SEED = 19027
SPLITS = {"train": (0, 1024), "validation": (1, 256),
          "calibration": (2, 256), "test": (3, 256)}
SHAPES = {"vision": (1, 4, 6, 6), "transformer": (4, 8)}
FMAX_MHZ = 164.555


def make_data(task, split):
    offset, count = SPLITS[split]
    rng = np.random.default_rng(SEED + 311 * offset + (503 if task == "transformer" else 0))
    if task == "transformer":
        tokens = rng.integers(0, 4, size=(count, 4))
        content = np.eye(4, dtype=np.float32)[tokens] * .85
        content += rng.normal(0, .045, content.shape).astype(np.float32)
        position = np.broadcast_to(np.eye(4, dtype=np.float32)[None], content.shape)
        position = position + rng.normal(0, .018, position.shape).astype(np.float32)
        x = np.concatenate((content, position), axis=-1)
        y = np.concatenate((tokens[:, :1], tokens[:, :-1]), axis=-1)
        return x, y

    h = w = 6
    rows, cols = np.indices((h, w))
    labels = rng.permutation(np.tile(np.arange(4), count // 4))
    labels = labels.astype(np.int64)
    images = np.zeros((count, h, w, 4), dtype=np.float32)
    for i, label in enumerate(labels):
        center = int(rng.integers(1, 5))
        delta = int(rng.integers(-1, 2))
        if label == 0:
            pattern = rows == center
        elif label == 1:
            pattern = cols == center
        elif label == 2:
            pattern = rows - cols == delta
        else:
            pattern = rows + cols == h - 1 + delta
        image = pattern.astype(np.float32) * rng.uniform(.72, 1.13)
        # Independent distractor spots and sensor gains yield real hold-out inputs.
        image += (rng.random((h, w)) < .065) * rng.uniform(.12, .30, (h, w))
        image[rng.integers(0, h), rng.integers(0, w)] += rng.uniform(.10, .24)
        gain = rng.uniform(.72, 1.12, size=(1, 1, 4))
        samples = image[:, :, None] * gain + rng.normal(0, .085, size=(h, w, 4))
        images[i] = samples.astype(np.float32)
    return images, labels


def _normal(rng, shape, fan):
    return rng.normal(0, np.sqrt(1.7 / fan), size=shape).astype(np.float32)


def _initial(task):
    rng = np.random.default_rng(SEED + (7 if task == "vision" else 11))
    if task == "vision":
        return {name: _normal(rng, (3, 3, 4, 4), 36)
                for name in ("c1", "c2", "c3")} | {
                    name: np.zeros(4, dtype=np.float32) for name in ("b1", "b2", "b3")}
    dims = {"query": (8, 4), "key": (8, 4), "value": (8, 4), "output": (4, 8),
            "ff_up": (8, 8), "ff_down": (8, 8), "classifier": (8, 4)}
    return {name: _normal(rng, shape, shape[0]) for name, shape in dims.items()}


def conv2d(x, weight, bias, stride, pad):
    n, height, width, channels = x.shape
    kh, kw, _, _ = weight.shape
    oh = (height+2*pad-kh)//stride+1
    ow = (width+2*pad-kw)//stride+1
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    patches = np.stack([padded[:, r:r+stride*oh:stride, c:c+stride*ow:stride, :]
                        for r in range(kh) for c in range(kw)], axis=-2)
    flat = patches.reshape(n, oh, ow, kh*kw*channels)
    return flat @ weight.reshape(flat.shape[-1], -1) + bias, (x.shape, flat, stride, pad)


def _conv_grad(cache, gradient, weights):
    (n, h, w, channels), flat, stride, pad = cache
    _, oh, ow, _ = gradient.shape
    kh, kw, _, out_ch = weights.shape
    dy = gradient.reshape(-1, out_ch)
    dw = (flat.reshape(-1, kh*kw*channels).T @ dy).reshape(weights.shape)
    db = dy.sum(axis=0)
    patch_grad = (dy @ weights.reshape(-1, out_ch).T).reshape(n, oh, ow, kh, kw, channels)
    dx = np.zeros((n, h+2*pad, w+2*pad, channels), dtype=np.float32)
    for r in range(kh):
        for c in range(kw):
            dx[:, r:r+stride*oh:stride, c:c+stride*ow:stride, :] += patch_grad[:, :, :, r, c, :]
    return dx[:, pad:pad+h, pad:pad+w, :], dw, db


def vision_forward(x, p, training=False):
    z1, st1 = conv2d(x, p["c1"], p["b1"], 1, 1)
    h1 = np.maximum(z1, 0)
    z2, st2 = conv2d(h1, p["c2"], p["b2"], 2, 1)
    h2 = np.maximum(z2, 0)
    z3, st3 = conv2d(h2, p["c3"], p["b3"], 1, 0)
    logits = z3.reshape(-1, 4)
    if training:
        return logits, (st1, h1, st2, h2, st3)
    return logits, {"input": x, "c1": z1, "c2": z2, "c3": logits}


def _vision_grad(cache, dlogits, p):
    st1, h1, st2, h2, st3 = cache
    gh2, dc3, db3 = _conv_grad(st3, dlogits[:, None, None, :], p["c3"])
    gh1, dc2, db2 = _conv_grad(st2, gh2*(h2 > 0), p["c2"])
    _, dc1, db1 = _conv_grad(st1, gh1*(h1 > 0), p["c1"])
    return {"c1": dc1, "b1": db1, "c2": dc2, "b2": db2, "c3": dc3, "b3": db3}


def _softmax(rows):
    exp = np.exp(rows - rows.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def transformer_forward(x, p, training=False):
    query = x @ p["query"]
    key = x @ p["key"]
    value = x @ p["value"]
    scores = (query @ key.swapaxes(-1, -2)) * np.float32(.5)
    allowed = np.tri(x.shape[-2], dtype=bool)
    prob = _softmax(np.where(allowed, scores, -np.inf))
    context = prob @ value
    output = context @ p["output"]
    attn_skip = x + output
    ffn_pre = attn_skip @ p["ff_up"]
    ffn_act = np.maximum(ffn_pre, 0)
    ffn = ffn_act @ p["ff_down"]
    skip = attn_skip + ffn
    logits = skip @ p["classifier"]
    features = {"input": x, "query": query, "key": key, "value": value, "scores": scores,
                "context": context, "output": output, "attn_skip": attn_skip, "ff_up": ffn_pre,
                "ff_down": ffn, "ff_skip": skip, "classifier": logits}
    if training:
        return logits, (features, prob, ffn_act)
    return logits, features


def _mat_grad(old, delta):
    return old.reshape(-1, old.shape[-1]).T @ delta.reshape(-1, delta.shape[-1])


def _transformer_grad(cache, dlogits, p):
    f, prob, ffn_act = cache
    grad = {"classifier": _mat_grad(f["ff_skip"], dlogits)}
    dskip = dlogits @ p["classifier"].T
    grad["ff_down"] = _mat_grad(ffn_act, dskip)
    dffn_pre = (dskip @ p["ff_down"].T) * (ffn_act > 0)
    grad["ff_up"] = _mat_grad(f["attn_skip"], dffn_pre)
    dout = dskip + dffn_pre @ p["ff_up"].T
    grad["output"] = _mat_grad(f["context"], dout)
    dc = dout @ p["output"].T
    dvalues = prob.swapaxes(-1, -2) @ dc
    dprobs = dc @ f["value"].swapaxes(-1, -2)
    dscores = prob * (dprobs - (dprobs*prob).sum(axis=-1, keepdims=True)) * np.float32(.5)
    grad["query"] = _mat_grad(f["input"], dscores @ f["key"])
    grad["key"] = _mat_grad(f["input"], dscores.swapaxes(-1, -2) @ f["query"])
    grad["value"] = _mat_grad(f["input"], dvalues)
    return grad


def _loss(logits, labels, task):
    prob = _softmax(logits)
    hot = np.eye(4, dtype=np.float32)[labels]
    actual = np.take_along_axis(prob, labels[..., None], axis=-1)[..., 0]
    if task == "transformer":
        grad = prob - hot
        grad[:, 0] = 0
        return float(-np.log(actual[:, 1:] + 1e-20).mean()), grad / labels[:, 1:].size
    return float(-np.log(actual + 1e-20).mean()), (prob - hot) / labels.size


def select_positions(values, task, *, logits=False):
    if task == "transformer":
        # The first position has no causal predecessor.
        return values[..., 1:, :] if logits else values[..., 1:]
    return values


def accuracy(logits, labels, task):
    target = select_positions(labels, task)
    pred = select_positions(logits.argmax(axis=-1), task)
    return float(np.mean(pred == target))


def train(task):
    p = _initial(task)
    forward, grad = ((vision_forward, _vision_grad) if task == "vision" else
                     (transformer_forward, _transformer_grad))
    x, y = make_data(task, "train")
    xv, yv = make_data(task, "validation")
    rng = np.random.default_rng(SEED + 173)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    best, best_acc, best_loss = {}, -1.0, float("inf")
    step, history = 0, []
    epochs = 22 if task == "vision" else 44
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        losses = []
        for begin in range(0, len(order), 64):
            idx = order[begin:begin+64]
            logits, cache = forward(x[idx], p, training=True)
            loss, dl = _loss(logits, y[idx], task)
            losses.append(loss)
            ds = grad(cache, dl, p)
            step += 1
            for name in p:
                g = ds[name]
                m[name] = .9*m[name] + .1*g
                v[name] = .999*v[name] + .001*g*g
                p[name] -= np.float32(.007 if task == "vision" else .010) * (
                    m[name]/(1-.9**step)) / (np.sqrt(v[name]/(1-.999**step)) + 1e-7)
        val_logits, _ = forward(xv, p)
        val_acc = accuracy(val_logits, yv, task)
        val_loss, _ = _loss(val_logits, yv, task)
        history.append({"epoch": epoch+1, "train_loss": float(np.mean(losses)),
                        "validation_loss": val_loss, "validation_top1": val_acc})
        if (val_acc, -val_loss) > (best_acc, -best_loss):
            best = {name: value.copy() for name, value in p.items()}
            best_acc, best_loss = val_acc, val_loss
    return best, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from .export import export_models
    args.out.mkdir(parents=True, exist_ok=True)
    for task in SHAPES:
        start = time.monotonic()
        weights, history = train(task)
        forward = vision_forward if task == "vision" else transformer_forward
        validation_x, validation_y = make_data(task, "validation")
        test_x, test_y = make_data(task, "test")
        calibration_x, _ = make_data(task, "calibration")
        calibration = forward(calibration_x, weights)[1]
        exported = export_models(task, weights, calibration, args.out)
        val_top1 = accuracy(forward(validation_x, weights)[0], validation_y, task)
        test_top1 = accuracy(forward(test_x, weights)[0], test_y, task)
        if test_top1 < .95:
            raise RuntimeError(f"{task} FP32 test accuracy {test_top1:.2%} below 95%")
        report = {"task": task, "seed": SEED, "splits": SPLITS, "history": history,
                  "validation_top1": val_top1, "test_top1": test_top1,
                  "elapsed_seconds": round(time.monotonic()-start, 3), **exported}
        (args.out / f"{task}.training.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{task} FP32 validation={val_top1:.2%}, test={test_top1:.2%}, "
              f"training+export {report['elapsed_seconds']:.3f} s", flush=True)


if __name__ == "__main__":
    main()
