#include "c11b_qwen_ops.h"

#include "c11_qwen_ref.h"

static const uint16_t reciprocal_sqrt[] = {
    32768, 32268, 31790, 31332, 30894, 30474, 30070, 29682,
    29309, 28949, 28602, 28268, 27945, 27632, 27330, 27038,
    26755, 26481, 26214, 25956, 25705, 25462, 25225, 24994,
    24770, 24552, 24339, 24132, 23930, 23733, 23541, 23354,
    23170, 22992, 22817, 22646, 22479, 22315, 22155, 21999,
    21845, 21695, 21548, 21404, 21263, 21124, 20988, 20855,
    20724, 20596, 20470, 20346, 20225, 20106, 19988, 19873,
    19760, 19649, 19539, 19431, 19326, 19221, 19119, 19018,
    18919, 18821, 18725, 18630, 18536, 18444, 18354, 18264,
    18176, 18090, 18004, 17920, 17837, 17755, 17674, 17594,
    17515, 17438, 17361, 17285, 17211, 17137, 17064, 16992,
    16921, 16851, 16782, 16714, 16646, 16579, 16514, 16448,
    16384,
};

static int8_t clipped(int64_t value)
{
    return (int8_t)(value > 127 ? 127 : value < -128 ? -128 : value);
}

static int64_t floor_shift(int64_t value, unsigned shift)
{
    const int64_t divisor = INT64_C(1) << shift;
    return value >= 0 ? value / divisor : -((-value + divisor - 1) / divisor);
}

void c11b_cpu_rms(const int8_t *x, const int8_t *gamma, unsigned gain,
                  unsigned epsilon, unsigned shift, int8_t *out)
{
    uint64_t total = epsilon;
    for (unsigned i = 0; i < QWEN_HIDDEN; ++i)
        total += (unsigned)(x[i] * x[i]);
    if (!total)
        total = 1;
    unsigned high = 0;
    for (uint64_t v = total; v > 1; v >>= 1)
        ++high;
    const unsigned exp = high / 2;
    const uint64_t normal = exp <= 4 ? total << (9 - 2 * exp) : total >> (2 * exp - 9);
    const unsigned segment = ((unsigned)normal & 2047u) >> 4;
    const unsigned a = reciprocal_sqrt[segment - 32], b = reciprocal_sqrt[segment - 31];
    const unsigned inv = (unsigned)((uint64_t)(a - (((a - b) * (normal & 15u) + 8) >> 4)) *
                                    gain >> (8 + exp)) & 0x1fffffu;
    for (unsigned i = 0; i < QWEN_HIDDEN; ++i)
        out[i] = clipped(floor_shift((int64_t)x[i] * gamma[i] * inv, shift));
}

void c11b_cpu_attention(const int8_t *query, const int8_t *keys, const int8_t *values,
                        const unsigned *shift, unsigned position, int8_t *out)
{
    enum { Q_PER_KV = QWEN_HEADS / QWEN_KV_HEADS, D = QWEN_HEAD_DIM };
    for (unsigned h = 0; h < QWEN_HEADS; ++h) {
        const unsigned group = h / Q_PER_KV;
        const int8_t *q = query + h * D;
        int score[C11B_PITCH], peak = -2147483647;
        unsigned probs[C11B_PITCH], total = 0;
        for (unsigned i = 0; i <= position; ++i) {
            const int8_t *k = keys + i * C11B_KV_WIDTH + group * D;
            int dot = 0;
            for (unsigned d = 0; d < D; ++d)
                dot += q[d] * k[d];
            score[i] = dot;
            if (peak < dot)
                peak = dot;
        }
        for (unsigned i = 0; i <= position; ++i) {
            const unsigned dist = (unsigned)(peak - score[i]) >> shift[h];
            probs[i] = dist > 8 ? 0 : 256u >> dist;
            total += probs[i];
        }
        for (unsigned i = 0; i <= position; ++i)
            probs[i] = (255u * probs[i]) / total;
        for (unsigned d = 0; d < D; ++d) {
            int dot = 0;
            for (unsigned i = 0; i <= position; ++i)
                dot += (int)probs[i] * values[i * C11B_KV_WIDTH + group * D + d];
            out[h * D + d] = clipped(floor_shift(dot, 8));
        }
    }
}

void c11b_cpu_swiglu(const int8_t *gate, const int8_t *up, const int8_t *lut,
                      unsigned shift, int8_t *out)
{
    const int rounding = shift ? 1 << (shift - 1) : 0;
    for (unsigned i = 0; i < QWEN_INTER; ++i)
        out[i] = clipped(floor_shift(lut[(uint8_t)gate[i]] * up[i] + rounding, shift));
}
