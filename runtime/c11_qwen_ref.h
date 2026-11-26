#ifndef YSYX_C11_QWEN_REF_H
#define YSYX_C11_QWEN_REF_H

#include <stdint.h>

enum {
    QWEN_LAYERS = 24, QWEN_HIDDEN = 896, QWEN_INTER = 4864,
    QWEN_HEADS = 14, QWEN_KV_HEADS = 2, QWEN_HEAD_DIM = 64,
    QWEN_VOCAB = 151936, QWEN_GROUP = 128, QWEN_Q16 = 65536
};

typedef struct {
    int32_t before[QWEN_HEADS + QWEN_KV_HEADS][QWEN_HEAD_DIM];
    int32_t after[QWEN_HEADS + QWEN_KV_HEADS][QWEN_HEAD_DIM];
    int32_t values[QWEN_KV_HEADS][QWEN_HEAD_DIM];
} qwen_layer0_rope;

typedef struct {
    int32_t input[2][QWEN_HIDDEN], norm[2][QWEN_HIDDEN];
    int32_t gate[QWEN_INTER], up[QWEN_INTER], swiglu[QWEN_INTER];
} qwen_layer0_vectors;

typedef struct qwen_reference qwen_reference;

/* The reference owns an immutable mapped W4A8 image and mutable Q16 KV. */
qwen_reference *qwen_reference_open(const char *path);
unsigned qwen_reference_context(const qwen_reference *reference);
void qwen_reference_reset(qwen_reference *reference);
/* Returned logits remain valid until the next step. */
const int32_t *qwen_reference_step(qwen_reference *reference, unsigned token);
int qwen_reference_lm_input(const qwen_reference *reference, int8_t out[QWEN_HIDDEN],
                            uint32_t *scale_q16);
void qwen_reference_capture_rope(qwen_reference *model, qwen_layer0_rope *capture);
void qwen_reference_capture_vectors(qwen_reference *model, qwen_layer0_vectors *capture);
void qwen_reference_close(qwen_reference *reference);

#endif
