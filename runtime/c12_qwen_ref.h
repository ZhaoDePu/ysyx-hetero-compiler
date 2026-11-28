#ifndef YSYX_C12_QWEN_REF_H
#define YSYX_C12_QWEN_REF_H

#include "c11_qwen_ref.h"
#include <stddef.h>

typedef struct {
    int32_t hidden[QWEN_HIDDEN];
    uint8_t cache[2 * QWEN_KV_HEADS * (QWEN_HEAD_DIM + 4)];
    size_t cache_bytes;
} c12_qwen_capture;

typedef struct {
    const uint8_t *packed;
    int32_t *partials;
    unsigned rows, bits;
} valence_raw_projection;

typedef struct {
    const uint8_t *packed, *fine;
    const uint16_t *coefficients;
    int32_t *partials;
    unsigned channels;
} valence_block16_projection;

enum { C16_HEAD_PANEL = 16384, C16_PREFILL_ROWS = 16, C16_GROUPED_COLS = 496 };

typedef struct {
    void *ctx;
    int (*project)(void *, const uint8_t *, unsigned, unsigned, unsigned, unsigned,
                   const int8_t *, int32_t *);
    int (*rope)(void *, int32_t *, int32_t *, const uint8_t *);
    int (*residual)(void *, int32_t *, const int32_t *);
    int (*project_block16)(void *, const uint8_t *, const uint8_t *, unsigned, unsigned, unsigned,
                           const int8_t *, const uint16_t *, int32_t *);
    int (*project_parallel)(void *, const valence_raw_projection *, unsigned, unsigned,
                            const int8_t *);
    int (*project_head)(void *, const uint8_t *, unsigned, unsigned, unsigned, unsigned,
                        unsigned, const int8_t *, int32_t *);
    int (*project_rows)(void *, const uint8_t *, unsigned, unsigned, unsigned, unsigned,
                        unsigned, const int8_t *, int32_t *);
    int (*project_block16_rows)(void *, const uint8_t *, const uint8_t *, unsigned, unsigned, unsigned,
                                const int8_t *, int32_t *);
    int (*project_block16_pair)(void *, const valence_block16_projection *, unsigned, const int8_t *);
} c12_qwen_device;

const int32_t *c16_qwen_prefill(qwen_reference *, const uint32_t *, unsigned);
int c14_qwen_block16(const qwen_reference *);
int c12_qwen_mixed(const qwen_reference *);
int c12_qwen_cache_bits(qwen_reference *, unsigned);
/* Bit 0..23 selects hardware layers, bit 24 selects the tied LM head. */
int c12_qwen_set_device(qwen_reference *, const c12_qwen_device *, uint32_t layers);
int c12_qwen_capture_layer(qwen_reference *, unsigned, c12_qwen_capture *);
int c16_qwen_capture_prefill(qwen_reference *, unsigned layer, c12_qwen_capture *, unsigned count);

#ifdef YSYX_C11B_NATIVE
int c12_z15_attach(qwen_reference *, c11b_z15 *, uint32_t layers);
/* Only after a trusted deployment profile has identified the Turbo image. */
int c16_z15_detect(const ysyx_hal_t *, valence_target *, int *turbo);
int c12_z15_enable_turbo(c11b_z15 *);
#endif
#endif
