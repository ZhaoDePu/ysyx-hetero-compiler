/* Included only in the opt-in C13 driver build. */
#include "c13_qwen_layout.h"
#include "c13_gpu_words.h"

enum {
    C13_CODE = 0x1800,
    C13_DDR_IO = STAGE_DDR + 0x40000,
    C13_DDR_LEFT = C13_DDR_IO,
    C13_DDR_RIGHT = C13_DDR_IO + 0x5000,
    C13_DDR_OUT = C13_DDR_IO + 0xa000,
    C13_DDR_EXP = C13_DDR_IO + 0xf000,
    C13_DDR_CONTROL = C13_DDR_IO + 0x14000,
    C13_EXP_PAD = ((C13_EXP_COUNT * 4 + 31) / 32) * 32,
    C13_CACHE_LAYER = C13_DEPTH * C13_KV_SLOT_BYTES,
    C13_CACHE_POOL = QWEN_LAYERS * C13_CACHE_LAYER
};

static int c13_launch(c11b_z15 *d, const uint32_t *pcs, unsigned warps, unsigned threads)
{
    if (launch_gpu_jobs(d, C13_CODE, c13_gpu_words, sizeof(c13_gpu_words) / sizeof(uint32_t),
                        pcs, warps, threads, &d->c13_resident))
        return -1;
    d->io.gpu_launches += warps;
    return 0;
}

static int c13_load_exp(c11b_z15 *d, const uint8_t *exp)
{
    if (d->c13_exp_resident)
        return 0;
    uint8_t padded[C13_EXP_PAD] = {0};
    memcpy(padded, exp, C13_EXP_COUNT * 4u);
    if (upload(d, C13_EXP, d->ring + C13_DDR_EXP, padded, sizeof(padded)))
        return -1;
    d->c13_exp_resident = 1;
    return 0;
}

static int c13_rms_hw(void *ctx, const int32_t *input, const uint8_t *gamma, int32_t *output)
{
    c11b_z15 *d = ctx;
    const unsigned bytes = QWEN_HIDDEN * sizeof(*input);
    return upload(d, C13_LEFT, d->ring + C13_DDR_LEFT, input, bytes) ||
           upload(d, C13_RIGHT, d->ring + C13_DDR_RIGHT, gamma, bytes) ||
           c13_launch(d, c13_rms_reduce_pc, 1, 1) ||
           c13_launch(d, c13_rms_scale_pc, 4, 16) ||
           download(d, C13_OUT, d->ring + C13_DDR_OUT, output, bytes);
}

static int c13_swiglu_hw(void *ctx, int32_t *gate, const int32_t *up, const uint8_t *exp)
{
    c11b_z15 *d = ctx;
    const unsigned bytes = QWEN_INTER * sizeof(*gate);
    return c13_load_exp(d, exp) ||
           upload(d, C13_LEFT, d->ring + C13_DDR_LEFT, gate, bytes) ||
           upload(d, C13_RIGHT, d->ring + C13_DDR_RIGHT, up, bytes) ||
           c13_launch(d, c13_swiglu_pc, 4, 16) ||
           download(d, C13_OUT, d->ring + C13_DDR_OUT, gate, bytes);
}

static int c13_attn_control(c11b_z15 *d, unsigned pos, unsigned first, unsigned count)
{
    uint32_t control[8] = {pos, first, count};
    return upload(d, C13_CONTROL, d->ring + C13_DDR_CONTROL, control, sizeof(control));
}

static int c13_attn_sweep(c11b_z15 *d, unsigned base, unsigned pos, const uint32_t *kernel)
{
    if (pos < C13_TILE)
        return c13_launch(d, kernel, 1, QWEN_HEADS);
    for (unsigned first = 0; first <= pos; first += C13_TILE) {
        unsigned count = pos + 1 - first;
        if (count > C13_TILE)
            count = C13_TILE;
        if (c13_attn_control(d, pos, first, count) ||
            dma(d, base + first * C13_KV_SLOT_BYTES, C13_CACHE, count * C13_KV_SLOT_BYTES, 0) ||
            c13_launch(d, kernel, 1, QWEN_HEADS))
            return -1;
    }
    return 0;
}

static int c13_attention_hw(void *ctx, unsigned layer, unsigned pos, const int32_t *q,
                            const int32_t *k, const int32_t *v, const uint8_t *exp,
                            int32_t *output, uint8_t *capture)
{
    c11b_z15 *d = ctx;
    if (layer >= QWEN_LAYERS || pos >= C13_DEPTH)
        return -1;
    const unsigned first = d->ring + RING + layer * C13_CACHE_LAYER;
    const unsigned append = pos * C13_KV_SLOT_BYTES;
    const unsigned slot = (pos % C13_TILE) * C13_KV_SLOT_BYTES;
    int32_t input[4 * QWEN_HEAD_DIM];
    memcpy(input, k, sizeof(input) / 2);
    memcpy(input + 2 * QWEN_HEAD_DIM, v, sizeof(input) / 2);
    if (c13_load_exp(d, exp) ||
        c13_attn_control(d, pos, 0, pos + 1) ||
        upload(d, C13_LEFT, d->ring + C13_DDR_LEFT, q, QWEN_HIDDEN * 4u) ||
        upload(d, C13_RIGHT, d->ring + C13_DDR_RIGHT, input, sizeof(input)) ||
        (append && pos < C13_TILE && dma(d, first, C13_CACHE, append, 0)) ||
        c13_launch(d, c13_kv_append_pc, 1, 4) ||
        dma(d, first + append, C13_CACHE + slot, C13_KV_SLOT_BYTES, 1) ||
        c13_attn_sweep(d, first, pos, c13_attention_scores_pc) ||
        c13_attn_sweep(d, first, pos, c13_attention_mix_pc) ||
        download(d, C13_OUT, d->ring + C13_DDR_OUT, output, QWEN_HIDDEN * 4u))
        return -1;

    /* The application never reads cache records. Replay reads one appended
     * record to verify GPU encoding and actual DDR persistence. */
    return capture ? host_read(d, first + append, capture, C13_KV_BYTES) : 0;
}

int c13_z15_seed_kv(c11b_z15 *d, unsigned layer, unsigned pos, const uint8_t *record, size_t bytes)
{
    if (!d || !d->io.cache_reserved || layer >= QWEN_LAYERS || pos >= C13_DEPTH ||
        !record || bytes != C13_KV_BYTES)
        return -1;
    uint8_t slot[C13_KV_SLOT_BYTES] = {0};
    memcpy(slot, record, bytes);
    return host_write(d, d->ring + RING + layer * C13_CACHE_LAYER + pos * C13_KV_SLOT_BYTES,
                      slot, sizeof(slot));
}

int c13_z15_kernels(c11b_z15 *d, unsigned ops, c13_qwen_device *output)
{
    if (!d || !output || !ops || (ops & ~C13_ACCUMULATE) ||
        ((ops & C13_ATTENTION) && d->ring > Z15_POOL_SIZE - RING - C13_CACHE_POOL))
        return -1;
    *output = (c13_qwen_device){
        d, ops & C13_RMS ? c13_rms_hw : NULL,
        ops & C13_SWIGLU ? c13_swiglu_hw : NULL,
        ops & C13_ATTENTION ? c13_attention_hw : NULL
    };
    if (ops & C13_ATTENTION)
        d->io.cache_reserved = C13_CACHE_POOL;
    return 0;
}

int c13_z15_attach(qwen_reference *model, c11b_z15 *d, uint32_t layers, unsigned ops)
{
    c13_qwen_device device;
    if ((ops & C13_ATTENTION) && c13_qwen_kv_bits(model) != 8)
        return -1;
    if (c13_z15_kernels(d, ops, &device) || c12_z15_attach(model, d, layers))
        return -1;
    return c13_qwen_set_device(model, &device, layers);
}
