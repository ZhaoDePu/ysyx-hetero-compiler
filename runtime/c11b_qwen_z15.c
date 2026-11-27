#define _POSIX_C_SOURCE 200809L
#include "c11b_qwen_ops.h"

#include "c12_qwen_ref.h"
#include "z15_regs.h"

#include <sched.h>
#include <stdlib.h>
#include <string.h>

#include "c11b_rope_words.h"


enum {
    RING = 16 * 1024 * 1024, MAX_INPUT = QWEN_INTER, MAX_OUTPUT = 4864, PANEL = 496,
    EXPANDED = 8192, MAX_EXPANDED = QWEN_HIDDEN * QWEN_INTER,
    GEMM_OUTPUT = 8192 + MAX_EXPANDED, VECTOR_OUTPUT = 24576,
    STAGE_DDR = 8 * 1024 * 1024, CACHE_DDR = STAGE_DDR + 65536,
    A = 0, B = 0x2000, C = 0x6000, MID = 0x4000,
    NPU_A = 0x200, NPU_B = 0x1000, NPU_C = 0x11000,
    KEY = 0x20000, HEAD_CACHE = C11B_PITCH * QWEN_HEAD_DIM,
    AQ = 0x1000, AV = 0x1200, AS = 0xd0000, AP = 0xd0400, AO = 0xd0800,
    POLL = 16777216, V3_VERSION = 0x0120, V3_DMA_CYCLES = 0x0100,
    V3_DMA_COMPUTE = 0x0108, V3_DMA_TRANSACTIONS = 0x0110,
    V3_ACTIVATION_PACKED = 0x1800
};

typedef struct { uint32_t reg, value; } word;

struct c11b_z15 {
    const ysyx_hal_t *hal;
    c11b_ops ops;
    uint32_t ring;
    uint64_t vpu_ticks, dma_bytes, expanded_bytes, ddr_peak;
    uint32_t dma_moves;
    int failed, rope_resident, residual_resident, pipeline, turbo;
    int8_t *expanded;
};

static int bad(void *ctx)
{
    return ((c11b_z15 *)ctx)->failed;
}

static int rd(c11b_z15 *d, uint32_t reg, uint32_t *x)
{
    if (d->failed || d->hal->mmio_read32(d->hal->ctx, reg, x)) {
        d->failed = 1;
        return -1;
    }
    return 0;
}

static int wr(c11b_z15 *d, uint32_t reg, uint32_t x)
{
    if (d->failed || d->hal->mmio_write32(d->hal->ctx, reg, x)) {
        d->failed = 1;
        return -1;
    }
    return 0;
}

static int error_reg(c11b_z15 *d)
{
    uint32_t err;
    if (rd(d, Z15_ERROR, &err) || err) {
        d->failed = 1;
        return -1;
    }
    return 0;
}

static int action(c11b_z15 *d, unsigned bit)
{
    return wr(d, Z15_CTRL, bit) || error_reg(d);
}

static int ready(c11b_z15 *d, unsigned require, unsigned busy)
{
    for (unsigned i = 0; i < POLL; ++i) {
        uint32_t v;
        if (rd(d, Z15_STATUS, &v) || (v & Z15_HAS_ERROR) || !(v & Z15_LOCKED))
            break;
        if ((v & require) == require && !(v & busy))
            return error_reg(d);
        if (!d->hal->observe && (i & 1023u) == 1023u)
            (void)sched_yield();
    }
    d->failed = 1;
    return -1;
}

static int ticks(c11b_z15 *d, unsigned reg, uint64_t *value)
{
    for (unsigned i = 0; i < 8; ++i) {
        uint32_t a, lo, b;
        if (rd(d, reg + 4, &a) || rd(d, reg, &lo) || rd(d, reg + 4, &b))
            return -1;
        if (a == b) {
            *value = (uint64_t)a << 32 | lo;
            return 0;
        }
    }
    d->failed = 1;
    return -1;
}

static int host_write(c11b_z15 *d, unsigned where, const void *data, size_t bytes)
{
    if (d->failed || where > Z15_POOL_SIZE || bytes > Z15_POOL_SIZE - where ||
        d->hal->ddr_write(d->hal->ctx, where, data, bytes) ||
        d->hal->cache_sync(d->hal->ctx, where, bytes, YSYX_SYNC_FOR_DEVICE)) {
        d->failed = 1;
        return -1;
    }
    if ((uint64_t)where + bytes > d->ddr_peak)
        d->ddr_peak = (uint64_t)where + bytes;
    return 0;
}

static int host_read(c11b_z15 *d, unsigned where, void *data, size_t bytes)
{
    if (d->failed || where > Z15_POOL_SIZE || bytes > Z15_POOL_SIZE - where ||
        d->hal->cache_sync(d->hal->ctx, where, bytes, YSYX_SYNC_FOR_HOST) ||
        d->hal->ddr_read(d->hal->ctx, where, data, bytes)) {
        d->failed = 1;
        return -1;
    }
    return 0;
}

static int dma_begin(c11b_z15 *d, unsigned offset, unsigned sram, unsigned bytes, int store)
{
    if (!bytes || ((offset | sram | bytes) & 31u) || offset >= Z15_POOL_SIZE ||
        bytes > Z15_POOL_SIZE - offset || sram >= Z15_SCRATCH_SIZE ||
        bytes > Z15_SCRATCH_SIZE - sram || (sram >> 20) != ((sram + bytes - 1) >> 20)) {
        d->failed = 1;
        return -1;
    }
    if (ready(d, 0, Z15_DMA_BUSY) ||
        d->hal->cache_sync(d->hal->ctx, offset, bytes, YSYX_SYNC_FOR_DEVICE) ||
        wr(d, Z15_DMA_BASE, Z15_POOL_BASE) ||
        wr(d, Z15_DMA_OFFSET, offset) || wr(d, Z15_DMA_SCRATCH, sram) ||
        wr(d, Z15_DMA_BYTES, bytes) || wr(d, Z15_DMA_FLAGS, 2u | (unsigned)store) ||
        action(d, Z15_START_DMA))
        return -1;
    d->dma_bytes += bytes;
    ++d->dma_moves;
    if (store && (uint64_t)offset + bytes > d->ddr_peak)
        d->ddr_peak = (uint64_t)offset + bytes;
    return 0;
}

static int dma(c11b_z15 *d, unsigned offset, unsigned sram, unsigned bytes, int store)
{
    return dma_begin(d, offset, sram, bytes, store) || ready(d, Z15_DMA_DONE, Z15_DMA_BUSY);
}

static int upload(c11b_z15 *d, unsigned sram, unsigned ddr, const void *data, unsigned bytes)
{
    return host_write(d, ddr, data, bytes) || dma(d, ddr, sram, bytes, 0);
}

static int download(c11b_z15 *d, unsigned sram, unsigned ddr, void *data, unsigned bytes)
{
    return dma(d, ddr, sram, bytes, 1) || host_read(d, ddr, data, bytes);
}

static int config(c11b_z15 *d, const word *words, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        if (wr(d, words[i].reg, words[i].value))
            return -1;
    return 0;
}

static int gemm(c11b_z15 *d, unsigned base, unsigned n, unsigned group)
{
    const word desc[] = {
        {Z15_NPU_DIMS, 1 | n << 16}, {Z15_NPU_K, group | group << 16},
        {Z15_NPU_A, base + NPU_A}, {Z15_NPU_B, base + NPU_B}, {Z15_NPU_C, base + NPU_C},
        {Z15_NPU_BIAS, 0}, {Z15_NPU_FLAGS, 0}, {Z15_NPU_SHAPE, 0},
        {Z15_NPU_SLICE, 0}, {Z15_NPU_PARTIAL, 0}
    };
    return ready(d, 0, Z15_NPU_BUSY) || config(d, desc, sizeof(desc) / sizeof(desc[0])) ||
           action(d, Z15_START_NPU);
}

static void widen(int8_t *out, const uint8_t *packed, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        unsigned v = packed[i];
        out[2 * i] = (int8_t)((int)((v & 15u) ^ 8u) - 8);
        out[2 * i + 1] = (int8_t)((int)((v >> 4) ^ 8u) - 8);
    }
}

typedef struct { unsigned block, group, width, sram; } tile;

static tile tensor_tile(unsigned ordinal, unsigned channels)
{
    const unsigned tiles = (channels + PANEL - 1) / PANEL;
    const unsigned column = (ordinal % tiles) * PANEL;
    const unsigned group = ordinal / tiles;
    unsigned width = channels - column;
    if (width > PANEL)
        width = PANEL;
    return (tile){group * channels + column, group, width, (ordinal & 1u) * Z15_HALF_BYTES};
}

static int prepare_tile(c11b_z15 *d, unsigned a, unsigned b, tile t, unsigned group)
{
    return dma(d, a + t.group * group, t.sram + NPU_A, group, 0) ||
           dma(d, b + t.block * group, t.sram + NPU_B, t.width * group, 0);
}

static int store_tile(c11b_z15 *d, unsigned output, tile t)
{
    return dma(d, output + t.block * 4u, t.sram + NPU_C, t.width * 4u, 1);
}

static int project_grouped(void *ctx, const uint8_t *packed, unsigned n, unsigned k,
                           unsigned group, unsigned bits, const int8_t *a8, int32_t *result)
{
    c11b_z15 *d = ctx;
    if (!n || n > MAX_OUTPUT || (n & 15u) || (k != QWEN_HIDDEN && k != QWEN_INTER) ||
        (size_t)n * k > MAX_EXPANDED || (group != 64 && group != 128) ||
        (bits != 4 && bits != 8))
        return -1;
    const unsigned stage = d->ring;
    const unsigned base_weight = stage + EXPANDED;
    const unsigned base_result = stage + GEMM_OUTPUT;
    const void *weights = packed;
    if (bits == 4) {
        widen(d->expanded, packed, (size_t)n * k / 2);
        weights = d->expanded;
        d->expanded_bytes += (size_t)n * k;
    }
    if (host_write(d, base_weight, weights, (size_t)n * k) ||
        host_write(d, stage, a8, k))
        return -1;
    const unsigned count = (k / group) * ((n + PANEL - 1) / PANEL);
    tile previous = tensor_tile(0, n);
    if (prepare_tile(d, stage, base_weight, previous, group) || gemm(d, previous.sram, previous.width, group))
        return -1;
    for (unsigned ordinal = 1; ordinal < count; ++ordinal) {
        const tile next = tensor_tile(ordinal, n);
        if (d->pipeline) {
            if (prepare_tile(d, stage, base_weight, next, group) || ready(d, Z15_NPU_DONE, Z15_NPU_BUSY) ||
                gemm(d, next.sram, next.width, group) || store_tile(d, base_result, previous))
                return -1;
        } else {
            if (ready(d, Z15_NPU_DONE, Z15_NPU_BUSY) || store_tile(d, base_result, previous) ||
                prepare_tile(d, stage, base_weight, next, group) || gemm(d, next.sram, next.width, group))
                return -1;
        }
        previous = next;
    }
    if (ready(d, Z15_NPU_DONE, Z15_NPU_BUSY) || store_tile(d, base_result, previous))
        return -1;
    return host_read(d, base_result, result, (size_t)(k / group) * n * sizeof(*result));
}

static int project(void *ctx, const uint8_t *packed, unsigned n, unsigned k,
                   const int8_t *a8, int32_t *result)
{
    return project_grouped(ctx, packed, n, k, QWEN_GROUP, 4, a8, result);
}

static int ew(c11b_z15 *d, unsigned a, unsigned b, unsigned c, unsigned length,
              unsigned h, unsigned channels, unsigned op, unsigned scale, unsigned shift,
              unsigned eps)
{
    const word desc[] = {
        {Z15_EW_A, a}, {Z15_EW_B, b}, {Z15_EW_C, c}, {Z15_EW_D, 0},
        {Z15_EW_LENGTH, length}, {Z15_EW_SHAPE, h}, {Z15_EW_CHANNELS, channels},
        {Z15_EW_FLAGS, op | scale << 8 | shift << 24}, {Z15_EW_EPS, eps}
    };
    uint64_t first, last;
    if (ready(d, 0, Z15_NPU_BUSY) || config(d, desc, sizeof(desc) / sizeof(desc[0])) ||
        ticks(d, Z15_NPU_CYCLES, &first) || action(d, Z15_START_EW) ||
        ready(d, Z15_NPU_DONE, Z15_NPU_BUSY) || ticks(d, Z15_NPU_CYCLES, &last) || last < first)
        return -1;
    d->vpu_ticks += last - first;
    return 0;
}

static int rms_hw(void *ctx, const int8_t *a, const int8_t *g, unsigned gain,
                   unsigned eps, unsigned shift, int8_t *result)
{
    c11b_z15 *d = ctx;
    const unsigned buf = d->ring + STAGE_DDR;
    return upload(d, A, buf, a, QWEN_HIDDEN) ||
           upload(d, B, buf + 8192, g, QWEN_HIDDEN) ||
           ew(d, A, B, C, QWEN_HIDDEN, 1, QWEN_HIDDEN, 7, gain, shift, eps) ||
           download(d, C, buf + VECTOR_OUTPUT, result, QWEN_HIDDEN);
}

static int swiglu_hw(void *ctx, const int8_t *gate, const int8_t *up, const int8_t *lut,
                      unsigned shift, int8_t *result)
{
    c11b_z15 *d = ctx;
    const unsigned buf = d->ring + STAGE_DDR;
    if (upload(d, A, buf, gate, QWEN_INTER) ||
        upload(d, B, buf + 8192, up, QWEN_INTER) || ready(d, 0, Z15_NPU_BUSY))
        return -1;
    if (d->turbo) {
        for (unsigned i = 0; i < 256; i += 4) {
            uint32_t packed = (uint8_t)lut[i] | (uint32_t)(uint8_t)lut[i + 1] << 8 |
                              (uint32_t)(uint8_t)lut[i + 2] << 16 |
                              (uint32_t)(uint8_t)lut[i + 3] << 24;
            if (wr(d, V3_ACTIVATION_PACKED + i, packed))
                return -1;
        }
        if (error_reg(d))
            return -1;
    } else {
        for (unsigned i = 0; i < 256; ++i)
            if (wr(d, Z15_ACTIVATION, i | (unsigned)(uint8_t)lut[i] << 8) ||
                action(d, Z15_WRITE_ACT))
                return -1;
    }
    return ew(d, A, 0, MID, QWEN_INTER, 0, 0, 11, 1, 0, 0) ||
           ew(d, MID, B, C, QWEN_INTER, 0, 0, 6, 1, shift, 0) ||
           download(d, C, buf + VECTOR_OUTPUT, result, QWEN_INTER);
}

static int launch_warps(c11b_z15 *d, unsigned offset, const uint32_t *code, size_t length,
                        const uint32_t pc[4], int *resident)
{
    if (ready(d, 0, Z15_GPU_BUSY))
        return -1;
    if (!*resident) {
        for (size_t i = 0; i < length; ++i)
            if (wr(d, Z15_PROGRAM + offset + (unsigned)i * 4, code[i]))
                return -1;
        *resident = 1;
    }
    for (unsigned warp = 0; warp < 4; ++warp) {
        unsigned i = 0;
        uint32_t slots;
        do {
            if (rd(d, Z15_GPU_SLOTS, &slots) || (slots & Z15_GPU_SLOT_FAULTS))
                return -1;
            if (++i == POLL)
                return -1;
        } while (slots & (Z15_GPU_PENDING | 1u << warp));
        if (wr(d, Z15_GPU_PC, offset + pc[warp]) ||
            wr(d, Z15_GPU_JOB, warp << 24 | 16u << 16) ||
            action(d, Z15_START_GPU))
            return -1;
    }
    return ready(d, Z15_GPU_DONE, Z15_GPU_BUSY);
}

static int rope_hw(void *ctx, int32_t *q, int32_t *k, const uint8_t *trig)
{
    c11b_z15 *d = ctx;
    const unsigned buf = d->ring + STAGE_DDR;
    const size_t q_bytes = 7 * QWEN_HEAD_DIM * sizeof(*q);
    const size_t k_bytes = QWEN_HEAD_DIM * sizeof(*k);
    int32_t staged[8 * QWEN_HEAD_DIM];
    if (upload(d, 0x900, buf + 8192, trig, C11B_TRIG_BYTES))
        return -1;
    for (unsigned group = 0; group < 2; ++group) {
        int32_t *qi = q + group * 7 * QWEN_HEAD_DIM;
        int32_t *ki = k + group * QWEN_HEAD_DIM;
        memcpy(staged, qi, q_bytes);
        memcpy(staged + 7 * QWEN_HEAD_DIM, ki, k_bytes);
        if (upload(d, A, buf, staged, sizeof(staged)) ||
            launch_warps(d, 0, c11b_rope_words, sizeof(c11b_rope_words) / sizeof(uint32_t),
                         c11b_rope_pc, &d->rope_resident) ||
            download(d, A, buf + VECTOR_OUTPUT, staged, sizeof(staged)))
            return -1;
        memcpy(qi, staged, q_bytes);
        memcpy(ki, staged + 7 * QWEN_HEAD_DIM, k_bytes);
    }
    return 0;
}

static int attention_one(c11b_z15 *d, unsigned half, unsigned key, unsigned q,
                         unsigned out, unsigned pos, unsigned shift)
{
    const word words[] = {
        {Z15_NPU_DIMS, C11B_PITCH | QWEN_HEAD_DIM << 16},
        {Z15_NPU_K, QWEN_HEAD_DIM | C11B_PITCH << 16},
        {Z15_NPU_A, half + q}, {Z15_NPU_B, half + key}, {Z15_NPU_C, half + out},
        {Z15_ATTN_CONTROL, 4u | 2u | 1u << 8 | shift << 20},
        {Z15_ATTN_V, half + key + HEAD_CACHE}, {Z15_ATTN_SCORES, half + AS},
        {Z15_ATTN_P, half + AP}, {Z15_ATTN_POSITION, pos}
    };
    return ready(d, 0, Z15_NPU_BUSY) || config(d, words, sizeof(words) / sizeof(words[0])) ||
           action(d, Z15_START_ATTN) || ready(d, Z15_NPU_DONE, Z15_NPU_BUSY);
}

static int cache_view(c11b_z15 *d, unsigned half, unsigned key, unsigned group,
                      int8_t *keys, int8_t *values)
{
    int8_t page[HEAD_CACHE];
    const unsigned buf = d->ring + CACHE_DDR;
    if (download(d, half + key, buf, page, HEAD_CACHE))
        return -1;
    for (unsigned i = 0; i < C11B_PITCH; ++i)
        memcpy(keys + i * C11B_KV_WIDTH + group * QWEN_HEAD_DIM,
               page + i * QWEN_HEAD_DIM, QWEN_HEAD_DIM);
    if (download(d, half + key + HEAD_CACHE, buf, page, HEAD_CACHE))
        return -1;
    for (unsigned i = 0; i < C11B_PITCH; ++i)
        for (unsigned c = 0; c < QWEN_HEAD_DIM; ++c)
            values[i * C11B_KV_WIDTH + group * QWEN_HEAD_DIM + c] = page[c * C11B_PITCH + i];
    return 0;
}

static int attention_hw(void *ctx, unsigned layer, unsigned pos, const int8_t *q, const int8_t *k,
                        const int8_t *v, const unsigned *shift, int8_t *out,
                        int8_t *snapshot_k, int8_t *snapshot_v)
{
    c11b_z15 *d = ctx;
    const unsigned buf = d->ring + STAGE_DDR;
    const unsigned key = KEY + layer * 2 * HEAD_CACHE;
    static const uint8_t zero[2 * HEAD_CACHE] = {0};
    for (unsigned group = 0; group < QWEN_KV_HEADS; ++group) {
        const unsigned half = group * Z15_HALF_BYTES, width = 7 * QWEN_HEAD_DIM;
        if ((!pos && upload(d, half + key, buf + 32768, zero, sizeof(zero))) ||
            upload(d, half + AQ, buf, q + group * width, width) ||
            upload(d, half + key + pos * QWEN_HEAD_DIM, buf + 8192, k + group * QWEN_HEAD_DIM, QWEN_HEAD_DIM) ||
            upload(d, half + AV, buf + 16384, v + group * QWEN_HEAD_DIM, QWEN_HEAD_DIM) ||
            ew(d, half + AV, 0, half + key + HEAD_CACHE + pos, 1 + 63 * C11B_PITCH,
               1, QWEN_HEAD_DIM | C11B_PITCH << 16, 9, 1, 0, 0))
            return -1;
        for (unsigned h = 0; h < 7; ++h) {
            unsigned col = h * QWEN_HEAD_DIM;
            if (attention_one(d, half, key, AQ + col, AO + col, pos, shift[group * 7 + h]))
                return -1;
        }
        if (download(d, half + AO, buf + VECTOR_OUTPUT, out + group * width, width) ||
            (snapshot_k && snapshot_v && cache_view(d, half, key, group, snapshot_k, snapshot_v)))
            return -1;
    }
    return 0;
}

c11b_z15 *c11b_z15_open(const ysyx_hal_t *hal, const uint8_t *image, size_t count,
                       const char **error)
{
    if (!hal || !image || !count || count > Z15_POOL_SIZE - RING - 32 || !hal->mmio_read32 ||
        !hal->mmio_write32 || !hal->ddr_write || !hal->ddr_read || !hal->cache_sync)
        return NULL;
    c11b_z15 *d = calloc(1, sizeof(*d));
    if (!d)
        return NULL;
    d->expanded = malloc(MAX_EXPANDED);
    d->hal = hal;
    d->pipeline = 1;
    d->ring = ((unsigned)count + 31u) & ~31u;
    d->ops = (c11b_ops){d, project, rms_hw, rope_hw, attention_hw, swiglu_hw, bad};
    uint32_t magic, version;
    if (!d->expanded || rd(d, Z15_MAGIC, &magic) || magic != Z15_MAGIC_VALUE ||
        rd(d, Z15_VERSION, &version) || (version & Z15_VERSION_COMPAT_MASK) != Z15_VERSION_VALUE ||
        ready(d, 0, Z15_GPU_BUSY | Z15_DMA_BUSY | Z15_NPU_BUSY) ||
        (!hal->observe && host_write(d, 0, image, count))) {
        if (error)
            *error = "Qwen allocation, Z15 ABI, or DDR initialization failed";
        c11b_z15_close(d);
        return NULL;
    }
    return d;
}

const c11b_ops *c11b_z15_ops(c11b_z15 *d)
{
    return d ? &d->ops : NULL;
}

void c11b_z15_close(c11b_z15 *d)
{
    if (d) {
        free(d->expanded);
        free(d);
    }
}

void c11b_z15_pipeline(c11b_z15 *d, int enable)
{
    if (d)
        d->pipeline = !!enable;
}

int c11b_z15_stats(c11b_z15 *d, ysyx_stats_t *s)
{
    if (!d || !s)
        return -1;
    memset(s, 0, sizeof(*s));
    s->dma_bytes = d->dma_bytes;
    s->dma_operations = d->dma_moves;
    if (ticks(d, Z15_NPU_CYCLES, &s->npu_ticks) || ticks(d, Z15_GPU_CYCLES, &s->gpu_ticks) ||
        ticks(d, Z15_WALL_CYCLES, &s->wall_ticks) || ticks(d, Z15_OVERLAP_CYCLES, &s->npu_gpu_overlap_ticks) ||
        rd(d, Z15_ERROR, &s->abi_error))
        return -1;
    s->invalid_dma_descriptors = !!(s->abi_error & Z15_ERR_DMA_DESC);
    if (d->hal->observe) {
        s->has_observations = 1;
        d->hal->observe(d->hal->ctx, &s->observed);
    } else if (d->turbo) {
        if (ticks(d, V3_DMA_CYCLES, &s->observed.dma_busy_ticks) ||
            ticks(d, V3_DMA_COMPUTE, &s->observed.dma_compute_overlap_ticks) ||
            ticks(d, V3_DMA_TRANSACTIONS, &s->observed.axi_transactions))
            return -1;
        s->has_observations = 1;
    }
    return error_reg(d);
}

uint64_t c11b_z15_vpu_ticks(const c11b_z15 *d) { return d->vpu_ticks; }
uint64_t c11b_z15_peak_bytes(const c11b_z15 *d) { return d->ddr_peak; }
uint64_t c11b_z15_expanded_bytes(const c11b_z15 *d) { return d->expanded_bytes; }

static int residual_q16(void *ctx, int32_t *hidden, const int32_t *delta)
{
    c11b_z15 *d = ctx;
    const unsigned ddr = d->ring + STAGE_DDR;
    const unsigned bytes = QWEN_HIDDEN * sizeof(*hidden);
    return upload(d, A, ddr, hidden, bytes) ||
           upload(d, B, ddr + 8192, delta, bytes) ||
           launch_warps(d, 0x1000, c12_residual_words, sizeof(c12_residual_words) / sizeof(uint32_t),
                        c12_residual_pc, &d->residual_resident) ||
           download(d, A, ddr + VECTOR_OUTPUT, hidden, bytes);
}

int c12_z15_attach(qwen_reference *m, c11b_z15 *d, uint32_t layers)
{
    if (!d)
        return -1;
    c12_qwen_device ops = {d, project_grouped, rope_hw, residual_q16};
    return c12_qwen_set_device(m, &ops, layers);
}

int c12_z15_enable_turbo(c11b_z15 *d)
{
    uint32_t version;
    if (!d || ready(d, 0, Z15_GPU_BUSY | Z15_DMA_BUSY | Z15_NPU_BUSY) ||
        rd(d, V3_VERSION, &version) || version != UINT32_C(0x0c0c0101) || error_reg(d))
        return -1;
    d->turbo = 1;
    return 0;
}
