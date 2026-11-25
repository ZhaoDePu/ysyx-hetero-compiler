"""Deterministic NumPy training for a small two-layer causal Transformer."""

import argparse
import json
from pathlib import Path

import numpy as np

WIDTH, HEADS, HEAD_WIDTH, FF, VOCAB = 8, 2, 4, 8, 64
LAYERS, PROMPT, DECODE, SEED = 2, 4, 8, 7126
EPSILON = 2**-10
CORPUS = Path(__file__).with_name("corpus.txt")


def load_corpus():
    lines = [line.split() for line in CORPUS.read_text().splitlines() if line.strip()]
    if not lines or any(len(line) != PROMPT + DECODE + 1 for line in lines):
        raise ValueError("each line must supply four prompt words and nine continuation tokens")
    vocab = sorted({word for line in lines for word in line})
    if len(vocab) >= VOCAB:
        raise ValueError("corpus exceeds vocabulary reservation")
    vocab += [f"<unused{i}>" for i in range(VOCAB - len(vocab))]
    ids = {word: i for i, word in enumerate(vocab)}
    return vocab, np.array([[ids[word] for word in line] for line in lines], dtype=np.int64)


def _matrix_grad(x, dy):
    return x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])


def _norm(x, gamma):
    inv = 1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPSILON)
    return x * inv * gamma, (x, inv, gamma)


def _norm_backward(dy, cache):
    x, inv, gamma = cache
    dscale = (dy * x * inv).sum((0, 1))
    scaled = dy * gamma
    dx = scaled * inv - x * inv**3 * np.mean(x * scaled, axis=-1, keepdims=True)
    return dx, dscale


def _softmax(x):
    y = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return y / y.sum(axis=-1, keepdims=True)


def fp32_forward(weights, tokens, *, save_for_grad=False):
    """Run the complete model in parallel over a causal prefix."""
    z = weights["embedding"][tokens]
    rows = z.shape[1]
    triangular = np.triu(np.ones((rows, rows), dtype=bool), 1)
    records = []
    for layer in range(LAYERS):
        base = f"l{layer}."
        rms, norm1 = _norm(z, weights[base + "gamma1"])
        heads = []
        context = []
        for head in range(HEADS):
            h = base + f"h{head}."
            q, k, v = (rms @ weights[h + key] for key in ("q", "k", "v"))
            score = (q @ k.swapaxes(1, 2)) / np.float32(2)
            p = _softmax(np.where(triangular, -np.inf, score))
            context.append(p @ v)
            heads.append((q, k, v, p))
        joined = np.concatenate(context, axis=-1)
        res = z + joined @ weights[base + "out"]
        rms2, norm2 = _norm(res, weights[base + "gamma2"])
        gate, up = (rms2 @ weights[base + key] for key in ("gate", "up"))
        sigmoid = 1 / (1 + np.exp(-np.clip(gate, -30, 30)))
        g = gate * sigmoid
        activated = g * up
        z = res + activated @ weights[base + "down"]
        records.append((rms, norm1, heads, joined, rms2, norm2, gate, up, sigmoid, activated))
    logits = z @ weights["lm"]
    if save_for_grad:
        return logits, (z, records)
    return logits


def _gradients(weights, tokens, target, logits, state):
    z, records = state
    probs = _softmax(logits)
    picked = np.take_along_axis(probs, target[..., None], axis=-1)
    loss = float(-np.log(np.maximum(picked, 1e-30)).mean())
    grad = probs.copy()
    np.put_along_axis(grad, target[..., None], picked - 1, axis=-1)
    grad /= np.float32(target.size)
    weights_grad = {"lm": _matrix_grad(z, grad)}
    dz = grad @ weights["lm"].T
    for layer in reversed(range(LAYERS)):
        base = f"l{layer}."
        rms, norm1, heads, joined, rms2, norm2, gate, up, sigmoid, activated = records[layer]
        weights_grad[base + "down"] = _matrix_grad(activated, dz)
        da = dz @ weights[base + "down"].T
        dgate = da * up * sigmoid * (1 + gate * (1 - sigmoid))
        dup = da * gate * sigmoid
        weights_grad[base + "gate"] = _matrix_grad(rms2, dgate)
        weights_grad[base + "up"] = _matrix_grad(rms2, dup)
        drms = dgate @ weights[base + "gate"].T + dup @ weights[base + "up"].T
        dres, weights_grad[base + "gamma2"] = _norm_backward(drms, norm2)
        dres += dz
        weights_grad[base + "out"] = _matrix_grad(joined, dres)
        dctx = dres @ weights[base + "out"].T
        drms = np.zeros_like(rms)
        for i, (q, k, v, prob) in enumerate(heads):
            key = base + f"h{i}."
            do = dctx[..., i*HEAD_WIDTH:(i+1)*HEAD_WIDTH]
            dp = do @ v.swapaxes(1, 2)
            dv = prob.swapaxes(1, 2) @ do
            dscore = prob * (dp - (dp * prob).sum(axis=-1, keepdims=True)) / np.float32(2)
            dq, dk = dscore @ k, dscore.swapaxes(1, 2) @ q
            for name, partial in (("q", dq), ("k", dk), ("v", dv)):
                weights_grad[key + name] = _matrix_grad(rms, partial)
                drms += partial @ weights[key + name].T
        dz, weights_grad[base + "gamma1"] = _norm_backward(drms, norm1)
        dz += dres
    d_embedding = np.zeros_like(weights["embedding"])
    np.add.at(d_embedding, tokens, dz)
    weights_grad["embedding"] = d_embedding
    return loss, weights_grad


def _init():
    rng = np.random.default_rng(SEED)

    def random(shape, std):
        return rng.normal(0, std, shape).astype(np.float32)

    weights = {"embedding": random((VOCAB, WIDTH), .50), "lm": random((WIDTH, VOCAB), .19)}
    for layer in range(LAYERS):
        base = f"l{layer}."
        for kind in ("gamma1", "gamma2"):
            weights[base + kind] = np.ones(WIDTH, dtype=np.float32)
        for head in range(HEADS):
            for kind in ("q", "k", "v"):
                weights[base + f"h{head}." + kind] = random((WIDTH, HEAD_WIDTH), .21)
        weights[base + "out"] = random((WIDTH, WIDTH), .18)
        weights[base + "gate"] = random((WIDTH, FF), .18)
        weights[base + "up"] = random((WIDTH, FF), .18)
        weights[base + "down"] = random((FF, WIDTH), .18)
    return weights


def generate_fp32(weights, prompt, steps=DECODE):
    ids, logits = list(prompt), []
    for _ in range(steps + 1):
        current = fp32_forward(weights, np.array([ids], dtype=np.int64))[0, -1]
        ids.append(int(np.argmax(current)))
        logits.append(current)
    return ids[len(prompt):], np.array(logits)


def train(iterations=240, learning_rate=.027):
    vocab, samples = load_corpus()
    tokens, target = samples[:, :-1], samples[:, 1:]
    weights = _init()
    adam_m = {k: np.zeros_like(v) for k, v in weights.items()}
    adam_v = {k: np.zeros_like(v) for k, v in weights.items()}
    trace = []
    for step in range(1, iterations + 1):
        logits, state = fp32_forward(weights,tokens,save_for_grad=True)
        loss, gradient = _gradients(weights,tokens,target,logits,state)
        for name, parameter in weights.items():
            g = gradient[name]
            adam_m[name] *= .9
            adam_m[name] += .1 * g
            adam_v[name] *= .99
            adam_v[name] += .01 * g * g
            m = adam_m[name] / (1 - .9**step)
            v = adam_v[name] / (1 - .99**step)
            parameter -= learning_rate * m / (np.sqrt(v) + 1e-8)
            # Bound residual branch and attention weight magnitudes during
            # training to keep the shared signed-byte activation scale viable.
            last = name.rsplit(".",1)[-1]
            bound = ({"out":.14,"down":.14,"gate":.45,"up":.45,"q":.42,
                      "k":.42,"v":.45}.get(last,2.2))
            np.clip(parameter,-bound,bound,out=parameter)
        if step == 1 or step % 20 == 0 or step == iterations:
            acc = float((np.argmax(logits, axis=-1) == target).mean())
            trace.append(dict(iteration=step, loss=loss, token_accuracy=acc))
    final = fp32_forward(weights, tokens)
    accuracy = float((np.argmax(final, axis=-1) == target).mean())
    generated = [generate_fp32(weights, row[:PROMPT])[0] for row in samples[:3]]
    if accuracy < .95 or any(ids != expected.tolist()
                             for ids, expected in zip(generated, samples[:3, PROMPT:], strict=True)):
        raise RuntimeError(f"training ceiling {accuracy:.3f} / generated {generated}")
    return weights, vocab, samples, dict(seed=SEED, iterations=iterations, token_accuracy=accuracy,
                                        quantization="symmetric dyadic PTQ",tokens_scored=target.size,
                                        trace=trace, fp32_continuations=generated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=240)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    w, words, data, report = train(args.iterations)
    np.savez(args.out / "weights.fp32.npz", **w)
    report.update(vocab=words, prompts=data[:3, :PROMPT].tolist(), corpus=data.tolist())
    (args.out / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"TINYLM_FP32_TRAIN_PASS token_accuracy={report['token_accuracy']:.4f} "
          f"scored={report['tokens_scored']}", flush=True)


if __name__ == "__main__":
    main()
