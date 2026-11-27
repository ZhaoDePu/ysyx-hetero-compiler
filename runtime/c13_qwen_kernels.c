/* Freestanding RV32E program for the frozen GPU. Scratch addresses are local. */
#include "c13_qwen_layout.h"

typedef int i32;
typedef unsigned u32;
typedef long long i64;
typedef unsigned long long u64;

#ifdef VALENCE_C13_EMULATED
extern unsigned char *c13_emulated_sram;
#define AT(type, address) ((volatile type *)(void *)(c13_emulated_sram + (address)))
#else
#define AT(type, address) ((volatile type *)(address))
#endif

/* All divisions have bounded quotients. Shift the divisor rather than
 * requiring a target-dependent 64-bit division helper or a wide divider. */
static u64 quotient(u64 numerator, u64 denominator)
{
    u64 bit = 1;
    while (denominator <= (numerator >> 1)) {
        denominator <<= 1;
        bit <<= 1;
    }
    u64 result = 0;
    do {
        if (numerator >= denominator) {
            numerator -= denominator;
            result |= bit;
        }
        denominator >>= 1;
        bit >>= 1;
    } while (bit);
    return result;
}

/* For a 32-bit divisor, feed the remainders through the native divider
 * in radix chunks sized so that every dividend fits uint32. */
static u64 quotient_word(u64 input, u32 d)
{
    const u32 high = (u32)(input >> 32);
    u32 lower = (u32)input;
    const u32 upper = high / d;
    u32 remainder = high - upper * d;
    if (d > (1u << 30))
        return quotient(input, d);
    const unsigned radix = d <= (1u << 16) ? 16 : d <= (1u << 24) ? 8 :
                           d <= (1u << 26) ? 6 : d <= (1u << 28) ? 4 : 2;
    u32 result = 0;
    unsigned remaining = 32;
    while (remaining) {
        const unsigned bits = remaining < radix ? remaining : radix;
        u32 dividend = (remainder << bits) | (lower >> (32 - bits));
        const u32 part = dividend / d;
        result = (result << bits) | part;
        remainder = dividend - part * d;
        lower <<= bits;
        remaining -= bits;
    }
    return ((u64)upper << 32) | result;
}

static i32 bounded(i64 n)
{
    if (n > C13_LIMIT)
        return C13_LIMIT;
    if (n < -C13_LIMIT)
        return -C13_LIMIT;
    return (i32)n;
}

static i32 rounded(i64 value, u64 divisor)
{
    const u64 numerator = (u64)(value < 0 ? -value : value) + (divisor >> 1);
    const u64 result = (divisor >> 32) ? quotient(numerator, divisor) :
                        quotient_word(numerator, (u32)divisor);
    return bounded(value < 0 ? -(i64)result : (i64)result);
}

static u32 reciprocal(u32 denominator, unsigned *high_shift)
{
    unsigned bits = 0;
    for (u32 v = denominator; v; v >>= 1)
        ++bits;
    const unsigned shift = 31 + bits;
    *high_shift = shift - 32;
    return (u32)quotient_word((1ull << shift) - 1u, denominator);
}

/* For |n| < d*2^26, reciprocal truncation reduces the exact quotient by
 * less than one. One residual check restores ties-away rounding. */
static i32 rounded_reciprocal(i64 input, u32 d, u32 multiplier, unsigned shift)
{
    const u64 n = (u64)(input < 0 ? -input : input) + d / 2u;
    u32 q = C13_LIMIT;
    if (n < (u64)d * C13_LIMIT) {
        const u64 low = (u64)(u32)n * multiplier;
        const u64 high = (u64)(u32)(n >> 32) * multiplier + (low >> 32);
        q = (u32)(high >> shift);
        if ((u64)(q + 1) * d <= n)
            ++q;
    }
    return input < 0 ? -(i32)q : (i32)q;
}

static u32 isqrt(u64 x)
{
    u64 result = 0;
    u64 bit = 1ull << 62;
    while (bit > x)
        bit >>= 2;
    while (bit) {
        if (x >= result + bit) {
            x -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    return (u32)result;
}

static u32 exp_negative(u32 magnitude)
{
    if (magnitude >= (C13_EXP_COUNT - 1u) * 256u)
        return 0;
    volatile const u32 *exp = AT(const u32, C13_EXP);
    u32 index = magnitude >> 8;
    u32 frac = magnitude & 255u;
    /* The weighted sum is at most 2^32. Use the complement formulation
     * so even the first entry (2^24) needs no 64-bit multiplication. */
    return exp[index] - (((exp[index] - exp[index + 1]) * frac + 127u) >> 8);
}

void c13_rms_reduce(u32 lane)
{
    (void)lane;
    volatile const i32 *input = AT(const i32, C13_LEFT);
    u64 sum = 0;
    for (u32 i = 0; i < C13_HIDDEN; ++i) {
        const i64 x = input[i];
        sum += (u64)(x * x);
    }
    /* Share a reciprocal across the output lanes. For n < root*2^26,
     * its truncation error is below one quotient unit. */
    const u32 root = isqrt(quotient_word(sum, C13_HIDDEN) + 4295u);
    unsigned high_shift;
    const u32 multiplier = reciprocal(root, &high_shift);
    volatile u32 *state = AT(u32, C13_OUT) + C13_HIDDEN;
    state[0] = root;
    state[1] = high_shift;
    state[2] = multiplier;
}

void c13_rms_scale(u32 lane)
{
    volatile const i32 *input = AT(const i32, C13_LEFT);
    volatile const i32 *gamma = AT(const i32, C13_RIGHT);
    volatile i32 *out = AT(i32, C13_OUT);
    volatile const u32 *state = AT(const u32, C13_OUT) + C13_HIDDEN;
    const u32 root = state[0], shift = state[1], reciprocal = state[2];
    for (u32 i = lane; i < C13_HIDDEN; i += 64)
        out[i] = rounded_reciprocal((i64)input[i] * gamma[i], root, reciprocal, shift);
}

/* With a nonzero LUT entry |gate| < 2^20, so n < 2^44 + 2^24.
 * d is Q24 plus the LUT value. Truncating d to multiples of 2^16
 * overestimates the quotient by at most 4096. Divide that residual at
 * 2^8 granularity and correct one remaining unit for an exact result. */
static u32 silu_magnitude(u64 n, u32 d)
{
    const u32 estimate = (u32)(n >> 16) / (d >> 16);
    const u64 product = (u64)estimate * d;
    if (product <= n)
        return estimate;
    const u64 overshoot = product - n;
    u32 cut = (u32)(overshoot >> 8) / (d >> 8);
    const u64 back = (u64)cut * d;
    if (back < overshoot)
        ++cut;
    else if (cut && back - d >= overshoot)
        --cut;
    return estimate - cut;
}

void c13_swiglu(u32 lane)
{
    volatile const i32 *gate = AT(const i32, C13_LEFT);
    volatile const i32 *up = AT(const i32, C13_RIGHT);
    volatile i32 *out = AT(i32, C13_OUT);
    for (u32 i = lane; i < C13_INTER; i += 64) {
        i32 x = gate[i];
        const u32 e = exp_negative(x < 0 ? (u32)-x : (u32)x);
        i32 silu;
        if (e) {
            const u32 d = C13_Q24 + e;
            const u32 p = x < 0 ? e : C13_Q24;
            u32 q = silu_magnitude((u64)p * (u32)(x < 0 ? -x : x) + d / 2u, d);
            silu = x < 0 ? -(i32)q : (i32)q;
        } else {
            silu = x < 0 ? 0 : x;
        }
        const i64 value = (i64)silu * up[i];
        const u64 magnitude = (u64)(value < 0 ? -value : value);
        const i64 shifted = (i64)((magnitude + C13_Q16 / 2) >> 16);
        out[i] = bounded(value < 0 ? -shifted : shifted);
    }
}

void c13_kv_append(u32 lane)
{
    const u32 position = AT(u32, C13_CONTROL)[0] % C13_TILE;
    volatile const i32 *input = AT(const i32, C13_RIGHT) + lane * C13_HEAD;
    volatile u32 *record = AT(u32, C13_CACHE) + position * (C13_KV_SLOT_BYTES / 4) +
                           lane * (C13_KV_HEAD_BYTES / 4);
    u32 peak = 0;
    for (u32 i = 0; i < C13_HEAD; ++i) {
        const i32 x = input[i];
        const u32 magnitude = x < 0 ? (u32)-x : (u32)x;
        if (magnitude > peak)
            peak = magnitude;
    }
    u32 scale = (peak + 63u) / 127u;
    if (!scale)
        scale = 1;
    for (u32 i = 0; i < C13_HEAD; i += 4) {
        u32 word = 0;
        for (u32 j = 0; j < 4; ++j) {
            i32 value = input[i + j];
            u32 magnitude = (value < 0 ? (u32)-value : (u32)value);
            u32 quant = (magnitude + scale / 2u) / scale;
            if (quant > 127u)
                quant = 127;
            u32 payload = (value < 0 ? 0u - quant : quant) & 255u;
            word |= payload << (8u * j);
        }
        record[i / 4] = word;
    }
    record[C13_HEAD / 4] = scale;
    if (lane == 0)
        for (u32 i = C13_KV_BYTES / 4; i < C13_KV_SLOT_BYTES / 4; ++i)
            AT(u32, C13_CACHE)[position * (C13_KV_SLOT_BYTES / 4) + i] = 0;
}

static i32 unpack(volatile const unsigned char *bytes, u32 d, u32 scale)
{
    return ((i32)(bytes[d] ^ 128u) - 128) * (i32)scale;
}

void c13_attention_scores(u32 lane)
{
    const u32 first = AT(u32, C13_CONTROL)[1];
    const u32 count = AT(u32, C13_CONTROL)[2];
    const u32 group = lane >= 7u;
    volatile const i32 *q = AT(const i32, C13_LEFT) + lane * C13_HEAD;
    volatile i32 *scores = AT(i32, C13_SCORES) + lane * C13_DEPTH + first;
    i32 maximum = first ? AT(i32, C13_MAX)[lane] : -C13_LIMIT;
    for (u32 p = 0; p < count; ++p) {
        volatile const unsigned char *key = AT(const unsigned char, C13_CACHE) +
                                           p * C13_KV_SLOT_BYTES + group * C13_KV_HEAD_BYTES;
        const u32 scale = *(volatile const u32 *)(key + C13_HEAD);
        i64 dot = 0;
        for (u32 d = 0; d < C13_HEAD; ++d)
            dot += (i64)q[d] * unpack(key, d, scale);
        u64 magnitude = (u64)(dot < 0 ? -dot : dot);
        i64 shift = (i64)((magnitude + 4u * C13_Q16) >> 19);
        i32 score = bounded(dot < 0 ? -shift : shift);
        scores[p] = score;
        if (score > maximum)
            maximum = score;
    }
    AT(i32, C13_MAX)[lane] = maximum;
}

void c13_attention_mix(u32 lane)
{
    const u32 position = AT(u32, C13_CONTROL)[0];
    const u32 first = AT(u32, C13_CONTROL)[1];
    const u32 count = AT(u32, C13_CONTROL)[2];
    const u32 group = lane >= 7u;
    const i32 maximum = AT(i32, C13_MAX)[lane];
    volatile i32 *scores = AT(i32, C13_SCORES) + lane * C13_DEPTH + first;
    volatile i64 *sums = AT(i64, C13_SUMS) + lane * C13_HEAD;
    volatile i32 *out = AT(i32, C13_OUT) + lane * C13_HEAD;
    u64 total = first ? AT(u64, C13_TOTAL)[lane] : 0;
    for (u32 p = 0; p < count; ++p) {
        u32 weight = exp_negative((u32)(maximum - scores[p]));
        total += weight;
        scores[p] = (i32)weight;
    }
    AT(u64, C13_TOTAL)[lane] = total;
    const int output_tile = first + count == position + 1;
    const int fast = output_tile && !(total >> 32);
    unsigned shift = 0;
    const u32 mult = fast ? reciprocal((u32)total, &shift) : 0;
    for (u32 d = 0; d < C13_HEAD; ++d) {
        i64 sum = first ? sums[d] : 0;
        for (u32 p = 0; p < count; ++p) {
            volatile const unsigned char *value = AT(const unsigned char, C13_CACHE) +
                p * C13_KV_SLOT_BYTES + (C13_KV_HEADS + group) * C13_KV_HEAD_BYTES;
            const u32 scale = *(volatile const u32 *)(value + C13_HEAD);
            sum += (i64)scores[p] * unpack(value, d, scale);
        }
        if (output_tile)
            out[d] = fast ? rounded_reciprocal(sum, (u32)total, mult, shift) : rounded(sum, total);
        else
            sums[d] = sum;
    }
}
