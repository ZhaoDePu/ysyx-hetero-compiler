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
    void *ctx;
    int (*project)(void *, const uint8_t *, unsigned, unsigned, unsigned, unsigned,
                   const int8_t *, int32_t *);
    int (*rope)(void *, int32_t *, int32_t *, const uint8_t *);
    int (*residual)(void *, int32_t *, const int32_t *);
    int (*project_block16)(void *, const uint8_t *, const uint8_t *, unsigned, unsigned, unsigned,
                           const int8_t *, const uint16_t *, int32_t *);
} c12_qwen_device;

int c14_qwen_block16(const qwen_reference *);
int c12_qwen_mixed(const qwen_reference *);
int c12_qwen_cache_bits(qwen_reference *, unsigned);
/* Bit 0..23 selects hardware layers, bit 24 selects the tied LM head. */
int c12_qwen_set_device(qwen_reference *, const c12_qwen_device *, uint32_t layers);
int c12_qwen_capture_layer(qwen_reference *, unsigned, c12_qwen_capture *);

#ifdef YSYX_C11B_NATIVE
int c12_z15_attach(qwen_reference *, c11b_z15 *, uint32_t layers);
/* Only after a trusted deployment profile has identified the Turbo image. */
int c12_z15_enable_turbo(c11b_z15 *);
#endif
#endif
