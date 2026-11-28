#ifndef YSYX_C12_QWEN_CACHE_H
#define YSYX_C12_QWEN_CACHE_H

#include "c17_simd.h"

#include <stddef.h>
#include <stdint.h>

/* Private C12 representation: G64 signed quants, followed by a LE Q16 scale. */
enum { C12_KV_GROUP = 64, C12_KV_OUTLIERS = 4, C12_KV_OUTLIER_BYTES = 5 };


static inline size_t c12_cache_record_size(unsigned bits, int exceptions)
{
    return C12_KV_GROUP * bits / 8 + 4 +
           (bits == 4 && exceptions ? C12_KV_OUTLIERS * C12_KV_OUTLIER_BYTES : 0);
}

static inline unsigned c12_cache_code(int32_t value, uint32_t scale, int limit)
{
    int64_t magnitude = value < 0 ? -(int64_t)value : value;
    int64_t rounded = (magnitude + scale / 2) / scale;
    int quant = rounded > limit ? limit : (int)rounded;
    return (unsigned)(value < 0 ? -quant : quant);
}

static inline void c12_cache_pack(uint8_t *record, const int32_t *values, unsigned bits)
{
    const uint32_t peak = c17_peak(values, C12_KV_GROUP);
    const unsigned limit = (1u << (bits - 1)) - 1;
    const unsigned payload = C12_KV_GROUP * bits / 8;
    uint32_t scale = (peak + limit / 2) / limit;
    if (!scale)
        scale = 1;
    if (bits == 4) {
        for (unsigned i = 0; i < C12_KV_GROUP; i += 2) {
            unsigned lo = c12_cache_code(values[i], scale, (int)limit) & 15u;
            unsigned hi = c12_cache_code(values[i + 1], scale, (int)limit) & 15u;
            record[i / 2] = (uint8_t)(lo | hi << 4);
        }
    } else {
#if C17_NEON
        const uint32_t reciprocal = scale > 1 ? (uint32_t)((UINT64_C(1) << 32) / scale) : 0;
        for (unsigned i = 0; i < C12_KV_GROUP; i += 8) {
            const int16x4_t lo = vmovn_s32(c17_quant4(vld1q_s32(values + i), scale, reciprocal, limit));
            const int16x4_t hi = vmovn_s32(c17_quant4(vld1q_s32(values + i + 4), scale, reciprocal, limit));
            vst1_s8((int8_t *)record + i, vmovn_s16(vcombine_s16(lo, hi)));
        }
#else
        for (unsigned i = 0; i < C12_KV_GROUP; ++i)
            record[i] = (uint8_t)c12_cache_code(values[i], scale, (int)limit);
#endif
    }
    for (unsigned i = 0; i < 4; ++i)
        record[payload + i] = (uint8_t)(scale >> (8 * i));
}

static inline void c12_cache_unpack(const uint8_t *record, int32_t *values, unsigned bits)
{
    const unsigned payload = C12_KV_GROUP * bits / 8;
    uint32_t scale = 0;
    for (unsigned i = 0; i < 4; ++i)
        scale |= (uint32_t)record[payload + i] << (8 * i);
#if C17_NEON
    if (bits == 8 && scale <= (UINT32_C(1) << 24)) {
        for (unsigned i = 0; i < C12_KV_GROUP; i += 8) {
            const int16x8_t q = vmovl_s8(vld1_s8((const int8_t *)record + i));
            vst1q_s32(values + i, vmulq_n_s32(vmovl_s16(vget_low_s16(q)), (int32_t)scale));
            vst1q_s32(values + i + 4, vmulq_n_s32(vmovl_high_s16(q), (int32_t)scale));
        }
        return;
    }
#endif
    for (unsigned i = 0; i < C12_KV_GROUP; ++i) {
        unsigned code = bits == 4 ? (record[i / 2] >> (4 * (i & 1u))) & 15u : record[i];
        unsigned sign = 1u << (bits - 1);
        int quant = (int)(code ^ sign) - (int)sign;
        int64_t value = (int64_t)quant * scale;
        values[i] = value > INT32_MAX ? INT32_MAX : value < INT32_MIN ? INT32_MIN : (int32_t)value;
    }
}

/* The compact base record is retained. Four channel indices and exact Q16
 * values follow the scale. Ties prefer lower channel indices. */
static inline void c12_cache_pack_outliers(uint8_t *record, const int32_t *values)
{
    int32_t filtered[C12_KV_GROUP];
    unsigned selected[C12_KV_OUTLIERS];
    uint32_t magnitudes[C12_KV_OUTLIERS] = {0};
    unsigned count = 0;
    for (unsigned i = 0; i < C12_KV_GROUP; ++i) {
        filtered[i] = values[i];
        uint32_t magnitude = (uint32_t)(values[i] < 0 ? -(int64_t)values[i] : values[i]);
        unsigned at = 0;
        while (at < count && magnitude <= magnitudes[at])
            ++at;
        if (at == C12_KV_OUTLIERS)
            continue;
        if (count < C12_KV_OUTLIERS)
            ++count;
        for (unsigned end = count - 1; end > at; --end) {
            magnitudes[end] = magnitudes[end - 1];
            selected[end] = selected[end - 1];
        }
        magnitudes[at] = magnitude;
        selected[at] = i;
    }
    uint8_t *sideband = record + c12_cache_record_size(4, 0);
    for (unsigned j = 0; j < C12_KV_OUTLIERS; ++j) {
        unsigned i = selected[j];
        filtered[i] = 0;
        sideband[0] = (uint8_t)i;
        for (unsigned k = 0; k < 4; ++k)
            sideband[k + 1] = (uint8_t)((uint32_t)values[i] >> (k * 8));
        sideband += C12_KV_OUTLIER_BYTES;
    }
    c12_cache_pack(record, filtered, 4);
}

static inline void c12_cache_unpack_outliers(const uint8_t *record, int32_t *values)
{
    c12_cache_unpack(record, values, 4);
    record += c12_cache_record_size(4, 0);
    for (unsigned j = 0; j < C12_KV_OUTLIERS; ++j) {
        uint32_t value = 0;
        for (unsigned k = 0; k < 4; ++k)
            value |= (uint32_t)record[k + 1] << (k * 8);
        values[record[0]] = value <= INT32_MAX ? (int32_t)value : -(int32_t)(UINT32_MAX - value) - 1;
        record += C12_KV_OUTLIER_BYTES;
    }
}
#endif
