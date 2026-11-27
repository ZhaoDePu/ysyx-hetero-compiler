/* Algorithm-only oracle replay using the very same C as the GPU ELF. */
#include "c13_qwen.h"
#include "c13_qwen_layout.h"
#include <stdlib.h>
#include <string.h>

void c13_rms_reduce(unsigned);
void c13_rms_scale(unsigned);
void c13_swiglu(unsigned);
void c13_kv_append(unsigned);
void c13_attention_scores(unsigned);
void c13_attention_mix(unsigned);

unsigned char *c13_emulated_sram;

typedef struct {
    unsigned char *sram;
    unsigned char *cache;
} emulator;

static int rms(void *opaque, const int32_t *input, const uint8_t *gamma, int32_t *output)
{
    emulator *e = opaque;
    c13_emulated_sram = e->sram;
    memcpy(e->sram + C13_LEFT, input, C13_HIDDEN * 4u);
    memcpy(e->sram + C13_RIGHT, gamma, C13_HIDDEN * 4u);
    c13_rms_reduce(0);
    for (unsigned lane = 0; lane < 64; ++lane)
        c13_rms_scale(lane);
    memcpy(output, e->sram + C13_OUT, C13_HIDDEN * 4u);
    return 0;
}

static int swiglu(void *opaque, int32_t *gate, const int32_t *up, const uint8_t *exp)
{
    emulator *e = opaque;
    c13_emulated_sram = e->sram;
    memcpy(e->sram + C13_LEFT, gate, C13_INTER * 4u);
    memcpy(e->sram + C13_RIGHT, up, C13_INTER * 4u);
    memcpy(e->sram + C13_EXP, exp, C13_EXP_COUNT * 4u);
    for (unsigned lane = 0; lane < 64; ++lane)
        c13_swiglu(lane);
    memcpy(gate, e->sram + C13_OUT, C13_INTER * 4u);
    return 0;
}

static int attention(void *opaque, unsigned layer, unsigned pos, const int32_t *query,
                     const int32_t *key, const int32_t *value, const uint8_t *exp,
                     int32_t *output, uint8_t *capture)
{
    emulator *e = opaque;
    const size_t layer_start = (size_t)layer * C13_DEPTH * C13_KV_SLOT_BYTES;
    const size_t offset = (size_t)pos * C13_KV_SLOT_BYTES;
    c13_emulated_sram = e->sram;
    memcpy(e->sram + C13_CONTROL, &pos, sizeof(pos));
    memcpy(e->sram + C13_LEFT, query, C13_HIDDEN * 4u);
    memcpy(e->sram + C13_RIGHT, key, C13_KV_HEADS * C13_HEAD * 4u);
    memcpy(e->sram + C13_RIGHT + C13_KV_HEADS * C13_HEAD * 4u, value,
           C13_KV_HEADS * C13_HEAD * 4u);
    memcpy(e->sram + C13_EXP, exp, C13_EXP_COUNT * 4u);
    for (unsigned lane = 0; lane < 2 * C13_KV_HEADS; ++lane)
        c13_kv_append(lane);
    const size_t local = (size_t)(pos % C13_TILE) * C13_KV_SLOT_BYTES;
    memcpy(e->cache + layer_start + offset, e->sram + C13_CACHE + local, C13_KV_SLOT_BYTES);
    if (capture)
        memcpy(capture, e->sram + C13_CACHE + local, C13_KV_BYTES);
    for (unsigned pass = 0; pass < 2; ++pass) {
        for (unsigned first = 0; first <= pos; first += C13_TILE) {
            unsigned count = pos + 1 - first;
            if (count > C13_TILE)
                count = C13_TILE;
            const unsigned control[3] = {pos, first, count};
            memcpy(e->sram + C13_CONTROL, control, sizeof(control));
            memcpy(e->sram + C13_CACHE, e->cache + layer_start + (size_t)first * C13_KV_SLOT_BYTES,
                   (size_t)count * C13_KV_SLOT_BYTES);
            for (unsigned lane = 0; lane < C13_QUERIES; ++lane) {
                if (pass)
                    c13_attention_mix(lane);
                else
                    c13_attention_scores(lane);
            }
        }
    }
    memcpy(output, e->sram + C13_OUT, C13_HIDDEN * 4u);
    return 0;
}

void c13_emulator_close(void *opaque)
{
    emulator *e = opaque;
    if (!e)
        return;
    free(e->sram);
    free(e->cache);
    free(e);
    c13_emulated_sram = NULL;
}

void *c13_emulator_open(qwen_reference *model, unsigned mask)
{
    if (!mask || mask & ~C13_ACCUMULATE)
        return NULL;
    emulator *e = calloc(1, sizeof(*e));
    if (!e)
        return NULL;
    e->sram = calloc(1, 1u << 20);
    if (mask & C13_ATTENTION)
        e->cache = malloc((size_t)QWEN_LAYERS * C13_DEPTH * C13_KV_SLOT_BYTES);
    const c13_qwen_device ops = {
        e, mask & C13_RMS ? rms : NULL, mask & C13_SWIGLU ? swiglu : NULL,
        mask & C13_ATTENTION ? attention : NULL
    };
    uint32_t layers = (UINT32_C(1) << (QWEN_LAYERS + 1)) - 1;
    if (!e->sram || ((mask & C13_ATTENTION) && !e->cache) ||
        c13_qwen_set_device(model, &ops, layers)) {
        c13_emulator_close(e);
        return NULL;
    }
    return e;
}
