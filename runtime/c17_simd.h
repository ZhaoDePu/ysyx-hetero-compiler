#ifndef YSYX_C17_SIMD_H
#define YSYX_C17_SIMD_H

#include <stddef.h>
#include <stdint.h>

#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(YSYX_SCALAR_C17)
#include <arm_neon.h>
#define C17_NEON 1
#else
#define C17_NEON 0
#endif

static inline int64_t c17_dot(const int32_t *a, const int32_t *b, unsigned width)
{
    unsigned i = 0;
    int64_t total = 0;
#if C17_NEON
    int64x2_t lo = vdupq_n_s64(0), hi = lo;
    for (; i + 4 <= width; i += 4) {
        const int32x4_t x = vld1q_s32(a + i), y = vld1q_s32(b + i);
        lo = vmlal_s32(lo, vget_low_s32(x), vget_low_s32(y));
        hi = vmlal_high_s32(hi, x, y);
    }
    total = vaddvq_s64(vaddq_s64(lo, hi));
#endif
    for (; i < width; ++i)
        total += (int64_t)a[i] * b[i];
    return total;
}

static inline void c17_accumulate(int64_t *out, const int32_t *in, int32_t weight, unsigned width)
{
    unsigned i = 0;
#if C17_NEON
    const int32x4_t w = vdupq_n_s32(weight);
    for (; i + 4 <= width; i += 4) {
        const int32x4_t x = vld1q_s32(in + i);
        vst1q_s64(out + i, vmlal_s32(vld1q_s64(out + i), vget_low_s32(x), vget_low_s32(w)));
        vst1q_s64(out + i + 2, vmlal_high_s32(vld1q_s64(out + i + 2), x, w));
    }
#endif
    for (; i < width; ++i)
        out[i] += (int64_t)weight * in[i];
}

static inline uint32_t c17_peak(const int32_t *values, unsigned width)
{
    unsigned i = 0;
    uint32_t peak = 0;
#if C17_NEON
    uint32x4_t maximum = vdupq_n_u32(0);
    for (; i + 4 <= width; i += 4)
        maximum = vmaxq_u32(maximum, vreinterpretq_u32_s32(vabsq_s32(vld1q_s32(values + i))));
    peak = vmaxvq_u32(maximum);
#endif
    for (; i < width; ++i) {
        const uint32_t magnitude = values[i] < 0 ? (uint32_t)-(int64_t)values[i] : (uint32_t)values[i];
        if (magnitude > peak)
            peak = magnitude;
    }
    return peak;
}

#if C17_NEON
/* The product with floor(2^32/scale) underestimates division by at most one.
 * Repair the remainder before restoring the sign. This preserves tie-away rounding. */
static inline uint32x4_t c17_divide4(uint32x4_t n, uint32_t scale, uint32_t reciprocal)
{
    const uint32x2_t mul = vdup_n_u32(reciprocal);
    const uint32x4_t q = vcombine_u32(vshrn_n_u64(vmull_u32(vget_low_u32(n), mul), 32),
                                    vshrn_n_u64(vmull_u32(vget_high_u32(n), mul), 32));
    const uint32x4_t rem = vmlsq_n_u32(n, q, scale);
    return vsubq_u32(q, vcgeq_u32(rem, vdupq_n_u32(scale)));
}

static inline int32x4_t c17_quant4(int32x4_t x, uint32_t scale, uint32_t reciprocal, unsigned limit)
{
    const uint32x4_t magnitude = vreinterpretq_u32_s32(vabsq_s32(x));
    uint32x4_t q = magnitude;
    if (scale > 1)
        q = c17_divide4(vaddq_u32(magnitude, vdupq_n_u32(scale / 2)), scale, reciprocal);
    const int32x4_t clipped = vreinterpretq_s32_u32(vminq_u32(q, vdupq_n_u32(limit)));
    return vbslq_s32(vcltq_s32(x, vdupq_n_s32(0)), vnegq_s32(clipped), clipped);
}

static inline int64x2_t c17_round_q16(int64x2_t x)
{
    const int64x2_t rounded = vshrq_n_s64(vaddq_s64(vabsq_s64(x), vdupq_n_s64(32768)), 16);
    return vbslq_s64(vcltq_s64(x, vdupq_n_s64(0)), vnegq_s64(rounded), rounded);
}

static inline uint8x16_t c17_lookup(const uint8x16x4_t *table, uint8x16_t x)
{
    uint8x16_t value = vqtbl4q_u8(table[0], x);
    for (unsigned i = 1; i < 4; ++i)
        value = vorrq_u8(value, vqtbl4q_u8(table[i], vsubq_u8(x, vdupq_n_u8((uint8_t)(i * 64)))));
    return value;
}
#endif

static inline void c17_rgb_quantize(uint8_t *out, const uint8_t *in, const uint8_t *table, size_t count)
{
    size_t i = 0;
#if C17_NEON
    uint8x16x4_t lut[4];
    for (unsigned j = 0; j < 4; ++j)
        for (unsigned k = 0; k < 4; ++k)
            lut[j].val[k] = vld1q_u8(table + (j * 4 + k) * 16);
    for (; i + 16 <= count; i += 16) {
        const uint8x16x3_t rgb = vld3q_u8(in + i * 3);
        const uint8x16x4_t rgba = {{c17_lookup(lut, rgb.val[0]), c17_lookup(lut, rgb.val[1]),
                                    c17_lookup(lut, rgb.val[2]), vdupq_n_u8(0)}};
        vst4q_u8(out + i * 4, rgba);
    }
#endif
    for (; i < count; ++i) {
        out[i * 4] = table[in[i * 3]];
        out[i * 4 + 1] = table[in[i * 3 + 1]];
        out[i * 4 + 2] = table[in[i * 3 + 2]];
        out[i * 4 + 3] = 0;
    }
}
#endif
