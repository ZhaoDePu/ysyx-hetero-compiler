"""Fixed-point reference for the frozen Z15 v1 vector arithmetic."""

from math import sqrt


def _s8(value):
    return max(-128, min(127, value))


def softmax(scores, rows, width, *, exp_shift, causal=False, position=0, fast=False):
    """Use the RTL base-two weights and normalize to unsigned probability bytes."""
    result = []
    for row in range(rows):
        active = min(width, position + row + 1) if causal else width
        weights = scores[row * width:row * width + active]
        peak = max(weights)
        weights = [256 >> ((peak - score) >> exp_shift) for score in weights]
        denom = sum(weights)
        if fast:
            # For nonzero power-of-two w, floor(floor(255*256/sum)*w/256)
            # equals floor(255*w/sum), including exact integer ties.
            reciprocal = 255 * 256 // denom
            result.extend((weight * reciprocal) >> 8 for weight in weights)
        else:
            result.extend(255 * weight // denom for weight in weights)
        result.extend([0] * (width - active))
    return result


def attention(q, k, v, qdims, kdims, vdims, attrs):
    """Model QK, native byte probabilities and the signed compact PV epilogue."""
    rows, dim = qdims
    length = kdims[0]
    packed_v = attrs.get("packed_v", False)
    columns = vdims[0] if packed_v else vdims[1]
    scores = [sum(q[row * dim + i] * k[col * dim + i] for i in range(dim))
              for row in range(rows) for col in range(length)]
    probabilities = softmax(scores, rows, length, exp_shift=attrs.get("exp_shift", 0),
                            causal=attrs.get("causal", False), position=attrs.get("position", 0),
                            fast=attrs.get("fast", False))
    if not packed_v:
        v = [v[i * columns + col] for col in range(columns) for i in range(length)]
    return [_s8(sum(probabilities[row * length + i] * v[col * length + i]
                    for i in range(length)) >> 8)
            for row in range(rows) for col in range(columns)]


# The RTL stores each segment's left endpoint and its drop to the next
# endpoint, rounded individually. The final endpoint is needed for bin 127.
_RSQRT = tuple(round(32768 * sqrt(32 / i)) for i in range(32, 129))


def rmsnorm(values, gamma, rows, width, *, gain, epsilon, shift):
    """Model the interpolated reciprocal square root, truncation and saturation."""
    result = []
    for row in range(rows):
        samples = values[row * width:(row + 1) * width]
        total = max(1, epsilon + sum(sample * sample for sample in samples))
        exponent = (total.bit_length() - 1) // 2
        normalize = 9 - 2 * exponent
        mantissa = (total << normalize if normalize >= 0 else total >> -normalize) & 0x7ff
        segment, fraction = divmod(mantissa, 16)
        left, right = _RSQRT[segment - 32:segment - 30]
        interpolated = left - (((left - right) * fraction + 8) >> 4)
        inv = ((interpolated * gain) >> (8 + exponent)) & 0x1fffff
        result.extend(_s8((sample * weight * inv) >> shift)
                      for sample, weight in zip(samples, gamma, strict=True))
    return result


def mul(a, b, *, scale, shift):
    """Model the EW signed product and the half-up requantization epilogue."""
    rounding = (1 << (shift - 1)) if shift else 0
    return [_s8((x * y * scale + rounding) >> shift) for x, y in zip(a, b, strict=True)]
