#define _POSIX_C_SOURCE 200809L
#include "c11b_qwen_z15_cmdq.h"

#include "c12_qwen_ref.h"
#include "z15_regs.h"

#include <inttypes.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>

#include "c11b_rope_words.h"


enum {
    RING = 16 * 1024 * 1024, RESIDUAL_RING = 10 * 1024 * 1024, MAX_INPUT = QWEN_INTER, MAX_OUTPUT = 4864, PANEL = C16_GROUPED_COLS,
    EXPANDED = 8192, MAX_EXPANDED = QWEN_HIDDEN * QWEN_INTER,
    GEMM_OUTPUT = 8192 + MAX_EXPANDED, VECTOR_OUTPUT = 24576,
    STAGE_DDR = 8 * 1024 * 1024, CACHE_DDR = STAGE_DDR + 65536,
    A = 0, B = 0x2000, C = 0x6000, MID = 0x4000,
    NPU_A = 0x200, NPU_B = 0x1000, NPU_C = 0x11000,
    KEY = 0x20000, HEAD_CACHE = C11B_PITCH * QWEN_HEAD_DIM,
    AQ = 0x1000, AV = 0x1200, AS = 0xd0000, AP = 0xd0400, AO = 0xd0800,
    POLL = 16777216, V3_VERSION = 0x0120, V3_DMA_CYCLES = 0x0100,
    V3_DMA_COMPUTE = 0x0108, V3_DMA_TRANSACTIONS = 0x0110,
    V3_ACTIVATION_PACKED = 0x1800,
    P17_WEIGHT_CONFIG = 0xd0, P17_KV_CONFIG = 0xd4, P17_SIGNED_W4 = 1
};

typedef struct { uint32_t reg, value; } word;

typedef struct {
    const word *words;
    size_t count;
    uint32_t action;
} valence_cmd_batch_t;

enum {
    C18_CACHE_ENTRIES = QWEN_LAYERS * 5 + 10,
    C18_REPACK_SLOTS = QWEN_LAYERS * 4,
    C18_QUEUE_GUARD = 512 * 1024,
    C18_TRANSIENT_RESERVE = CMDQ_LIMIT * 8
};

typedef struct {
    unsigned kind, k, group, n, weights[4], shape[4];
} static_key;

typedef struct {
    static_key key;
    z15_cmdq_plan *plan;
    uint64_t dma_bytes;
    uint32_t dma_moves;
} static_entry;

struct c11b_z15 {
    const ysyx_hal_t *hal;
    c11b_ops ops;
    uint32_t ring;
    uint64_t vpu_ticks, dma_bytes, expanded_bytes, ddr_peak;
    uint32_t dma_moves;
    int failed, rope_resident, residual_resident, pipeline, turbo;
    int8_t *expanded;
    valence_target target;
    const uint8_t *resident_image;
    size_t resident_bytes;
    valence_submission_stats submissions;
    int cmdq, c17_head, queue_pending, c18_static, recording;
    uint32_t cache_floor;
    unsigned cache_count, repack_count, repacked[C18_REPACK_SLOTS];
    static_entry plans[C18_CACHE_ENTRIES];
    uint64_t record_dma_bytes;
    uint32_t record_dma_moves;
    uint8_t *repack_buffer;
    uint32_t queue_base;
    z15_cmdq_plan *queue;
    z15_cmdq_stats queued;
    uint32_t config_values[P17_WEIGHT_CONFIG / 4 + 1];
    uint8_t config_known[P17_WEIGHT_CONFIG / 4 + 1];
};

static int queue_add(c11b_z15 *d, z15_cmdq_op op, uint32_t reg, uint32_t value)
{
    if (!d->queue)
        d->queue = z15_cmdq_create();
    if (!d->queue || z15_cmdq_add(d->queue, op, reg, value)) {
        d->failed = 1;
        return -1;
    }
    d->queue_pending = 1;
    return 0;
}

static int run_queue(c11b_z15 *d, z15_cmdq_plan *plan)
{
    z15_cmdq_stats stats = {0};
    const int rc = z15_cmdq_run(plan, d->hal, &stats);
    d->queued.starts += stats.starts;
    d->queued.commands += stats.commands;
    d->queued.cycles += stats.cycles;
    d->queued.upload_bytes += stats.upload_bytes;
    d->queued.reads += stats.reads;
    d->queued.writes += stats.writes;
    d->submissions.mmio_writes += stats.writes;
    d->submissions.doorbells += stats.starts;
    if (d->ddr_peak < stats.ddr_end)
        d->ddr_peak = stats.ddr_end;
    if (rc)
        z15_cmdq_report(stderr, &stats);
    d->failed = rc;
    return rc ? -1 : 0;
}

static int queue_flush(c11b_z15 *d)
{
    if (d->failed)
        return -1;
    if (!d->queue_pending)
        return 0;
    const int keep = d->recording &&
        !z15_cmdq_place_tail(d->queue, d->queue_base + C18_TRANSIENT_RESERVE, &d->cache_floor);
    if (!keep && z15_cmdq_place_bounded(d->queue, d->queue_base, d->cache_floor)) {
        d->failed = 1;
        return -1;
    }
    const int rc = run_queue(d, d->queue);
    if (keep && !rc) {
        static_entry *entry = d->plans + d->cache_count++;
        entry->plan = d->queue;
        entry->dma_bytes = d->dma_bytes - d->record_dma_bytes;
        entry->dma_moves = d->dma_moves - d->record_dma_moves;
        z15_cmdq_discard_uploaded_bytes(d->queue);
        d->queue = NULL;
    } else {
        z15_cmdq_reset(d->queue);
    }
    if (d->recording)
        memset(d->config_known, 0, sizeof(d->config_known));
    d->recording = d->queue_pending = 0;
    return rc;
}

static int cached_start(c11b_z15 *d, const static_key *key)
{
    if (!d->c18_static)
        return 0;
    for (unsigned i = 0; i < d->cache_count; ++i) {
        static_entry *entry = d->plans + i;
        if (memcmp(&entry->key, key, sizeof(*key)))
            continue;
        memset(d->config_known, 0, sizeof(d->config_known));
        if (run_queue(d, entry->plan))
            return -1;
        d->dma_bytes += entry->dma_bytes;
        d->dma_moves += entry->dma_moves;
        return 1;
    }
    if (d->cache_count < C18_CACHE_ENTRIES) {
        memset(d->config_known, 0, sizeof(d->config_known));
        d->plans[d->cache_count].key = *key;
        d->record_dma_bytes = d->dma_bytes;
        d->record_dma_moves = d->dma_moves;
        d->recording = 1;
    }
    return 0;
}

static int bad(void *ctx)
{
    return ((c11b_z15 *)ctx)->failed;
}

static int rd(c11b_z15 *d, uint32_t reg, uint32_t *x)
{
    ++d->queued.reads;
    if (d->failed || d->hal->mmio_read32(d->hal->ctx, reg, x)) {
        d->failed = 1;
        return -1;
    }
    return 0;
}

static int wr(c11b_z15 *d, uint32_t reg, uint32_t x)
{
    const int record = d->cmdq && reg < Z15_PROGRAM;
    if (d->failed ||
        (record ? queue_add(d, CMDQ_WRITE, reg, x) :
                  queue_flush(d) || d->hal->mmio_write32(d->hal->ctx, reg, x))) {
        d->failed = 1;
        return -1;
    }
    if (d->target == VALENCE_P17 && reg <= P17_WEIGHT_CONFIG) {
        d->config_known[reg / 4] = 1;
        d->config_values[reg / 4] = x;
    }
    d->submissions.mmio_writes += !record;
    d->queued.writes += !record;
    return 0;
}

static int error_reg(c11b_z15 *d)
{
    if (d->queue_pending)
        return 0;
    uint32_t err;
    if (rd(d, Z15_ERROR, &err) || err) {
        d->failed = 1;
        return -1;
    }
    return 0;
}

static int action(c11b_z15 *d, unsigned bit)
{
    if (wr(d, Z15_CTRL, bit))
        return -1;
    d->submissions.doorbells += !d->cmdq;
    return error_reg(d);
}

static int ready(c11b_z15 *d, unsigned require, unsigned busy)
{
    if (d->cmdq)
        return (require && queue_add(d, CMDQ_ONE, Z15_STATUS, require)) ||
               (busy && queue_add(d, CMDQ_ZERO, Z15_STATUS, busy));
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
    if (queue_flush(d))
        return -1;
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
    if (queue_flush(d) || where > Z15_POOL_SIZE || bytes > Z15_POOL_SIZE - where ||
        d->hal->ddr_write(d->hal->ctx, where, data, bytes) ||
        d->hal->cache_sync(d->hal->ctx, where, bytes, YSYX_SYNC_FOR_DEVICE)) {
        d->failed = 1;
        return -1;
    }
    ++d->submissions.host_writes;
    if ((uint64_t)where + bytes > d->ddr_peak)
        d->ddr_peak = (uint64_t)where + bytes;
    return 0;
}

static int host_read(c11b_z15 *d, unsigned where, void *data, size_t bytes)
{
    if (queue_flush(d) || where > Z15_POOL_SIZE || bytes > Z15_POOL_SIZE - where ||
        d->hal->cache_sync(d->hal->ctx, where, bytes, YSYX_SYNC_FOR_HOST) ||
        d->hal->ddr_read(d->hal->ctx, where, data, bytes)) {
        d->failed = 1;
        return -1;
    }
    ++d->submissions.host_reads;
    return 0;
}

static int submit_batch(c11b_z15 *d, const valence_cmd_batch_t *batch)
{
    /* Callers pass only passive DMA and GEMM descriptor registers.
       Table writes, program writes and doorbells always reach MMIO. */
    ++d->submissions.batches;
    d->submissions.descriptor_words += batch->count;
    for (size_t i = 0; i < batch->count; ++i) {
        const word w = batch->words[i];
        const unsigned slot = w.reg / 4;
        if (d->target == VALENCE_P17 && d->config_known[slot] && d->config_values[slot] == w.value)
            continue;
        if (wr(d, w.reg, w.value))
            return -1;
    }
    return action(d, batch->action);
}

static int dma_begin(c11b_z15 *d, unsigned offset, unsigned sram, unsigned bytes, int store)
{
    if (!bytes || ((offset | sram | bytes) & 31u) || offset >= Z15_POOL_SIZE ||
        bytes > Z15_POOL_SIZE - offset || sram >= Z15_SCRATCH_SIZE ||
        bytes > Z15_SCRATCH_SIZE - sram || (sram >> 20) != ((sram + bytes - 1) >> 20)) {
        d->failed = 1;
        return -1;
    }
    const word desc[] = {
        {Z15_DMA_BASE, Z15_POOL_BASE}, {Z15_DMA_OFFSET, offset}, {Z15_DMA_SCRATCH, sram},
        {Z15_DMA_BYTES, bytes}, {Z15_DMA_FLAGS, 2u | (unsigned)store}
    };
    const valence_cmd_batch_t batch = {desc, sizeof(desc) / sizeof(desc[0]), Z15_START_DMA};
    if (ready(d, 0, Z15_DMA_BUSY) ||
        d->hal->cache_sync(d->hal->ctx, offset, bytes, YSYX_SYNC_FOR_DEVICE) ||
        submit_batch(d, &batch))
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

static int gemm_at(c11b_z15 *d, unsigned a, unsigned b, unsigned c,
                   unsigned rows, unsigned n, unsigned group, unsigned packed_w4)
{
    const word desc[] = {
        {Z15_NPU_DIMS, rows | n << 16}, {Z15_NPU_K, group | group << 16},
        {Z15_NPU_A, a}, {Z15_NPU_B, b}, {Z15_NPU_C, c},
        {Z15_NPU_BIAS, 0}, {Z15_NPU_FLAGS, 0}, {Z15_NPU_SHAPE, 0},
        {Z15_NPU_SLICE, 0}, {Z15_NPU_PARTIAL, 0}, {P17_WEIGHT_CONFIG, packed_w4}
    };
    const valence_cmd_batch_t batch = {desc, sizeof(desc) / sizeof(desc[0]) -
                                      (d->target == VALENCE_STEP2), Z15_START_NPU};
    return ready(d, 0, Z15_NPU_BUSY) || submit_batch(d, &batch);
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

static int prepare_tile(c11b_z15 *d, unsigned a, unsigned b, tile t, unsigned group,
                        unsigned row_bytes, unsigned channels, unsigned weight_stride, unsigned rows,
                        int resident_a)
{
    return (!resident_a && dma(d, a + t.group * group * rows, t.sram + NPU_A, group * rows, 0)) ||
           dma(d, b + (t.group * weight_stride + t.block % channels) * row_bytes, t.sram + NPU_B, t.width * row_bytes, 0);
}

static int store_tile(c11b_z15 *d, unsigned output, tile t, unsigned rows)
{
    return dma(d, output + t.block * 4u * rows, t.sram + NPU_C, t.width * 4u * rows, 1);
}

static int launch_tile(c11b_z15 *d, tile t, unsigned rows, unsigned group,
                       int resident_a, unsigned packed_w4)
{
    const unsigned a = t.sram + NPU_A + (resident_a ? t.group * group * rows : 0);
    return gemm_at(d, a, t.sram + NPU_B, t.sram + NPU_C, rows, t.width, group, packed_w4);
}

static int schedule_grouped(c11b_z15 *d, unsigned stage, unsigned base_weight, unsigned base_result,
                            unsigned n, unsigned k, unsigned group, unsigned packed_w4,
                            unsigned weight_stride, unsigned rows, int resident_a)
{
    const unsigned row_bytes = packed_w4 ? group / 2 : group;
    const unsigned count = (k / group) * ((n + PANEL - 1) / PANEL);
    if (resident_a && (k * rows > NPU_B - NPU_A ||
        dma(d, stage, NPU_A, k * rows, 0) ||
        (count > 1 && dma(d, stage, Z15_HALF_BYTES + NPU_A, k * rows, 0))))
        return -1;
    tile previous = tensor_tile(0, n);
    if (prepare_tile(d, stage, base_weight, previous, group, row_bytes, n, weight_stride, rows, resident_a) ||
        launch_tile(d, previous, rows, group, resident_a, packed_w4))
        return -1;
    for (unsigned ordinal = 1; ordinal < count; ++ordinal) {
        const tile next = tensor_tile(ordinal, n);
        if (d->pipeline) {
            if (prepare_tile(d, stage, base_weight, next, group, row_bytes, n, weight_stride, rows, resident_a) ||
                ready(d, Z15_NPU_DONE, Z15_NPU_BUSY) || launch_tile(d, next, rows, group, resident_a, packed_w4) ||
                store_tile(d, base_result, previous, rows))
                return -1;
        } else {
            if (ready(d, Z15_NPU_DONE, Z15_NPU_BUSY) || store_tile(d, base_result, previous, rows) ||
                prepare_tile(d, stage, base_weight, next, group, row_bytes, n, weight_stride, rows, resident_a) ||
                launch_tile(d, next, rows, group, resident_a, packed_w4))
                return -1;
        }
        previous = next;
    }
    if (ready(d, Z15_NPU_DONE, Z15_NPU_BUSY) || store_tile(d, base_result, previous, rows))
        return -1;
    return 0;
}

int c11b_z15_project_grouped(void *ctx, const uint8_t *packed, unsigned n, unsigned k,
                           unsigned group, unsigned bits, const int8_t *a8, int32_t *result)
{
    c11b_z15 *d = ctx;
    if (!d || !packed || !a8 || !result || !n || n > MAX_OUTPUT || (n & 15u) || (k != QWEN_HIDDEN && k != QWEN_INTER) ||
        (size_t)n * k > MAX_EXPANDED || (group != 64 && group != 128) ||
        k % group || (bits != 4 && bits != 8))
        return -1;
    const unsigned stage = d->ring;
    unsigned base_weight = stage + EXPANDED;
    const unsigned base_result = stage + GEMM_OUTPUT;
    const void *weights = packed;
    const unsigned packed_w4 = d->target == VALENCE_P17 && bits == 4 ? P17_SIGNED_W4 : 0;
    const unsigned row_bytes = packed_w4 ? group / 2 : group;
    if (bits == 4 && !packed_w4) {
        widen(d->expanded, packed, (size_t)n * k / 2);
        weights = d->expanded;
        d->expanded_bytes += (size_t)n * k;
    }
    const size_t weight_bytes = (size_t)n * (k / group) * row_bytes;
    const uintptr_t image_base = (uintptr_t)d->resident_image;
    const uintptr_t address = (uintptr_t)packed;
    const uintptr_t offset = address - image_base;
    if (d->target == VALENCE_P17 && image_base && address >= image_base && !(offset & 31u) &&
        offset <= d->resident_bytes && weight_bytes <= d->resident_bytes - offset)
        base_weight = (unsigned)offset;
    else if (host_write(d, base_weight, weights, weight_bytes))
        return -1;
    if (host_write(d, stage, a8, k))
        return -1;
    const static_key key = {.kind = 1, .k = k, .group = group, .n = n,
                            .weights = {base_weight}, .shape = {bits}};
    const int hit = cached_start(d, &key);
    if (hit < 0 || (!hit && schedule_grouped(d, stage, base_weight, base_result, n, k, group, packed_w4, n, 1, 0)))
        return -1;
    return host_read(d, base_result, result, (size_t)(k / group) * n * sizeof(*result));
}

static int resident_offset(const c11b_z15 *d, const uint8_t *weights, size_t bytes, unsigned *offset)
{
    const uintptr_t base = (uintptr_t)d->resident_image, address = (uintptr_t)weights;
    const uintptr_t at = address - base;
    if (!base || address < base || (at & 31u) || at > d->resident_bytes ||
        bytes > d->resident_bytes - at)
        return -1;
    *offset = (unsigned)at;
    return 0;
}

static int project_parallel(void *ctx, const valence_raw_projection *jobs, unsigned count,
                            unsigned group, const int8_t *input)
{
    c11b_z15 *d = ctx;
    const unsigned stage = d->ring;
    unsigned weights[3], offsets[3];
    unsigned next = stage + GEMM_OUTPUT;
    if (!count || count > 3 || (group != 64 && group != 128))
        return -1;
    for (unsigned i = 0; i < count; ++i) {
        const unsigned n = jobs[i].rows, bits = jobs[i].bits;
        if (!n || n > MAX_OUTPUT || n % 16 || (bits != 4 && bits != 8) ||
            resident_offset(d, jobs[i].packed, (size_t)n * QWEN_HIDDEN * bits / 8, weights + i))
            return -1;
        offsets[i] = next;
        next += n * (QWEN_HIDDEN / group) * 4;
    }
    if (next > stage + STAGE_DDR || host_write(d, stage, input, QWEN_HIDDEN))
        return -1;
    static_key key = {.kind = 2, .group = group, .n = count};
    for (unsigned i = 0; i < count; ++i) {
        key.weights[i] = weights[i];
        key.shape[i] = jobs[i].rows * 16u + jobs[i].bits;
    }
    const int hit = cached_start(d, &key);
    if (hit < 0)
        return -1;
    for (unsigned i = 0; !hit && i < count; ++i)
        if (schedule_grouped(d, stage, weights[i], offsets[i], jobs[i].rows,
                             QWEN_HIDDEN, group, jobs[i].bits == 4, jobs[i].rows, 1, 0))
            return -1;
    for (unsigned i = 0; i < count; ++i)
        if (host_read(d, offsets[i], jobs[i].partials, jobs[i].rows * (QWEN_HIDDEN / group) * 4u))
            return -1;
    return 0;
}

static int project_head(void *ctx, const uint8_t *weights, unsigned first, unsigned rows, unsigned vocab,
                        unsigned group, unsigned bits, const int8_t *input, int32_t *partials)
{
    c11b_z15 *d = ctx;
    unsigned offset;
    const unsigned stage = d->ring;
    /* The full Q16-epilogue input is 8,508,416 bytes at G=64.
       It may occupy the ring up to the independently placed CMDQ buffer. */
    const unsigned output = stage + (d->c17_head ? 4096u : GEMM_OUTPUT);
    const unsigned limit = d->c17_head ? d->queue_base : stage + STAGE_DDR;
    const unsigned panel = d->c17_head ? QWEN_VOCAB : C16_HEAD_PANEL;
    if (group != 64 && group != 128)
        return -1;
    const unsigned groups = QWEN_HIDDEN / group;
    const unsigned bytes = rows * groups * 4;
    if (!rows || rows > panel || ((rows | first) & 15u) || first > vocab || rows > vocab - first ||
        (bits != 4 && bits != 8) || (output > limit || bytes > limit - output) ||
        resident_offset(d, weights, (size_t)vocab * QWEN_HIDDEN * bits / 8, &offset))
        return -1;
    offset += first * (group * bits / 8);
    if (host_write(d, stage, input, QWEN_HIDDEN))
        return -1;
    const static_key key = {.kind = 3, .k = vocab, .group = group, .n = rows,
                            .weights = {offset}, .shape = {bits, first}};
    const int hit = cached_start(d, &key);
    if (hit < 0 || (!hit && schedule_grouped(d, stage, offset, output, rows, QWEN_HIDDEN,
                                             group, bits == 4, vocab, 1, d->c17_head)))
        return -1;
    return host_read(d, output, partials, bytes);
}

static int project_rows(void *ctx, const uint8_t *weights, unsigned n, unsigned k,
                        unsigned group, unsigned bits, unsigned rows,
                        const int8_t *activation, int32_t *partials)
{
    c11b_z15 *d = ctx;
    enum { OUTPUT_OFFSET = 128 * 1024 };
    unsigned offset;
    if (!rows || rows > C16_PREFILL_ROWS || !n || n > MAX_OUTPUT || n % 16 ||
        (k != QWEN_HIDDEN && k != QWEN_INTER) || (group != 64 && group != 128) ||
        k % group || (bits != 4 && bits != 8) ||
        resident_offset(d, weights, (size_t)n * k * bits / 8, &offset))
        return -1;
    const unsigned out_bytes = (k / group) * n * rows * 4u;
    const unsigned stage = d->ring;
    const unsigned output = stage + OUTPUT_OFFSET;
    if (k * rows > OUTPUT_OFFSET || out_bytes > STAGE_DDR - OUTPUT_OFFSET ||
        host_write(d, stage, activation, k * rows) ||
        schedule_grouped(d, stage, offset, output, n, k, group, bits == 4, n, rows, 0))
        return -1;
    return host_read(d, output, partials, out_bytes);
}

/* Raw products are independent across batch rows. The host applies each
   token's own dynamic Q0.8 coefficient with the same rounding as C15. */
/* The weight image remains in its original public format on the host.
   The device copy is reordered once from [OC/16][K group] to [K group][OC]. */
static int repack_w4(c11b_z15 *d, const uint8_t *packed, unsigned n, unsigned group, unsigned *offset)
{
    const unsigned chunk = 16 * (group / 2), groups = QWEN_HIDDEN / group;
    const unsigned bytes = n * QWEN_HIDDEN / 2;
    if (resident_offset(d, packed, bytes, offset))
        return -1;
    for (unsigned i = 0; i < d->repack_count; ++i)
        if (d->repacked[i] == *offset)
            return 0;
    if (d->repack_count == C18_REPACK_SLOTS)
        return -1;
    if (!d->repack_buffer)
        d->repack_buffer = malloc(QWEN_HIDDEN * QWEN_INTER / 2);
    if (!d->repack_buffer)
        return -1;
    const unsigned panels = n / 16;
    for (unsigned g = 0; g < groups; ++g)
        for (unsigned p = 0; p < panels; ++p)
            memcpy(d->repack_buffer + (g * panels + p) * chunk,
                   packed + (p * groups + g) * chunk, chunk);
    if (host_write(d, *offset, d->repack_buffer, bytes))
        return -1;
    d->repacked[d->repack_count++] = *offset;
    if (d->repack_count == C18_REPACK_SLOTS) {
        free(d->repack_buffer);
        d->repack_buffer = NULL;
    }
    return 0;
}

static int project_block16_wide(void *ctx, const valence_block16_projection *jobs, unsigned count,
                                unsigned group, unsigned rows, const int8_t *a8)
{
    c11b_z15 *d = ctx;
    enum { OUTPUT_OFFSET = 128 * 1024 };
    if (!count || count > 2 || !rows || rows > C16_PREFILL_ROWS || (group != 64 && group != 128))
        return -1;
    const unsigned stage = d->ring;
    unsigned output = stage + OUTPUT_OFFSET, lengths[2], offsets[2];
    static_key key = {.kind = 4, .group = group, .n = count};
    for (unsigned i = 0; i < count; ++i) {
        const unsigned n = jobs[i].channels;
        if (!n || n > MAX_OUTPUT || n % 16 ||
            repack_w4(d, jobs[i].packed, n, group, key.weights + i * 2) ||
            (jobs[i].fine && repack_w4(d, jobs[i].fine, n, group, key.weights + i * 2 + 1)))
            return -1;
        key.shape[i] = n;
        lengths[i] = n * (QWEN_HIDDEN / group) * 4u * rows;
        offsets[i] = output;
        output += lengths[i] * (jobs[i].fine ? 2u : 1u);
    }
    if (output > d->queue_base || host_write(d, stage, a8, QWEN_HIDDEN * rows))
        return -1;
    const int hit = rows == 1 ? cached_start(d, &key) : 0;
    if (hit < 0)
        return -1;
    for (unsigned i = 0; !hit && i < count; ++i)
        for (unsigned j = 0; j < (jobs[i].fine ? 2u : 1u); ++j)
            if (schedule_grouped(d, stage, key.weights[i * 2 + j], offsets[i] + j * lengths[i],
                                 jobs[i].channels, QWEN_HIDDEN, group, 1, jobs[i].channels, rows,
                                 rows == 1))
                return -1;
    for (unsigned i = 0; i < count; ++i)
        if (host_read(d, offsets[i], jobs[i].partials, lengths[i] * (jobs[i].fine ? 2u : 1u)))
            return -1;
    return 0;
}

static int project_block16_rows(void *ctx, const uint8_t *packed, const uint8_t *fine,
                                unsigned n, unsigned group, unsigned rows,
                                const int8_t *activation, int32_t *partials)
{
    c11b_z15 *d = ctx;
    enum { WIDTH = 16, B_BASE = 0x5000, C_BASE = 0xf6000, OUTPUT_OFFSET = 128 * 1024 };
    if (!rows || rows > C16_PREFILL_ROWS || !n || n > MAX_OUTPUT || n % WIDTH ||
        (group != 64 && group != 128))
        return -1;
    const unsigned stage = d->ring, output = stage + OUTPUT_OFFSET;
    const unsigned passes = fine ? 2 : 1, chunks = (QWEN_HIDDEN / group) * passes;
    const unsigned panel_bytes = WIDTH * QWEN_HIDDEN / 2;
    const unsigned max_panels = (C_BASE - B_BASE) / (panel_bytes * passes);
    const unsigned chunk_result = rows * WIDTH * 4u, panel_result = chunks * chunk_result;
    const unsigned count = n / WIDTH, bytes = count * panel_result;
    unsigned weights[2];
    if (NPU_A + QWEN_HIDDEN * rows > B_BASE || C_BASE + panel_result > Z15_HALF_BYTES ||
        bytes > RESIDUAL_RING - OUTPUT_OFFSET ||
        resident_offset(d, packed, n * QWEN_HIDDEN / 2, weights) ||
        (fine && resident_offset(d, fine, n * QWEN_HIDDEN / 2, weights + 1)) ||
        host_write(d, stage, activation, QWEN_HIDDEN * rows))
        return -1;
    for (unsigned first = 0, half = 0; first < count; first += max_panels, half ^= Z15_HALF_BYTES) {
        const unsigned panels = count - first < max_panels ? count - first : max_panels;
        if (dma(d, stage, half + NPU_A, QWEN_HIDDEN * rows, 0))
            return -1;
        for (unsigned pass = 0; pass < passes; ++pass)
            if (dma(d, weights[pass] + first * panel_bytes,
                    half + B_BASE + pass * panels * panel_bytes, panels * panel_bytes, 0))
                return -1;
        for (unsigned p = 0; p < panels; ++p) {
            for (unsigned chunk = 0; chunk < chunks; ++chunk) {
                const unsigned g = chunk / passes, pass = chunk % passes;
                const unsigned a = half + NPU_A + g * group * rows;
                const unsigned b = half + B_BASE + (pass * panels + p) * panel_bytes + g * WIDTH * (group / 2);
                const unsigned c = half + C_BASE + chunk * chunk_result;
                if (gemm_at(d, a, b, c, rows, WIDTH, group, P17_SIGNED_W4) ||
                    ready(d, Z15_NPU_DONE, Z15_NPU_BUSY))
                    return -1;
            }
            if (dma(d, output + (first + p) * panel_result, half + C_BASE, panel_result, 1))
                return -1;
        }
    }
    return host_read(d, output, partials, bytes);
}

/* One activation and one contiguous W4 window per scratch half. Only the
   completed INT32 projection travels back to the host. */
static int schedule_block16(c11b_z15 *d, const uint8_t *packed, const uint8_t *fine,
                            unsigned n, unsigned k,
                           unsigned group, const uint16_t *coeff, unsigned output, int staged)
{
    enum { WIDTH = 16, C_BASE = 0xf8000, CONTINUE = 1u << 18, DEFER = 1u << 19,
           GROUP_SCALE = 4u };
    if (!d || !packed || !coeff || d->target != VALENCE_P17 || !n || n > MAX_OUTPUT || n % WIDTH ||
        k != QWEN_HIDDEN || (group != 64 && group != 128) || k % group)
        return -1;
    const unsigned group_count = k / group;
    const unsigned panel_bytes = WIDTH * k / 2;
    const unsigned passes = fine ? 2 : 1;
    const unsigned max_panels = (C_BASE - NPU_B) / (panel_bytes * passes);
    const unsigned count = n / WIDTH;
    const unsigned stage = d->ring;
    const uintptr_t image = (uintptr_t)d->resident_image;
    const size_t bytes = (size_t)n * k / 2;
    unsigned weights[2];
    for (unsigned pass = 0; pass < passes; ++pass) {
        const uint8_t *base = pass ? fine : packed;
        const uintptr_t address = (uintptr_t)base, offset = address - image;
        weights[pass] = stage + EXPANDED + pass * (unsigned)bytes;
        if (image && address >= image && !(offset & 31u) &&
            offset <= d->resident_bytes && bytes <= d->resident_bytes - offset)
            weights[pass] = (unsigned)offset;
        else if (staged || host_write(d, weights[pass], base, bytes))
            return -1;
    }

    unsigned half = 0;
    for (unsigned first = 0; first < count; first += max_panels, half ^= Z15_HALF_BYTES) {
        unsigned panels = count - first < max_panels ? count - first : max_panels;
        if (dma(d, stage, half + NPU_A, k, 0))
            return -1;
        for (unsigned pass = 0; pass < passes; ++pass)
            if (dma(d, weights[pass] + first * panel_bytes,
                    half + NPU_B + pass * panels * panel_bytes, panels * panel_bytes, 0))
                return -1;
        const unsigned chunks = group_count * passes;
        for (unsigned p = 0; p < panels; ++p)
            for (unsigned chunk = 0; chunk < chunks; ++chunk) {
                const unsigned g = chunk / passes, pass = chunk % passes;
                const unsigned coefficient = coeff[(first + p) * chunks + chunk];
                if (coefficient > 256)
                    return -1;
                const unsigned flags = (chunk ? CONTINUE : 0) | (chunk + 1 < chunks ? DEFER : 0);
                const word words[] = {
                    {Z15_NPU_DIMS, 1 | WIDTH << 16}, {Z15_NPU_K, group | group << 16},
                    {Z15_NPU_A, half + NPU_A + g * group},
                    {Z15_NPU_B, half + NPU_B + (pass * panels + p) * panel_bytes + g * WIDTH * (group / 2)},
                    {Z15_NPU_C, half + C_BASE + p * WIDTH * 4}, {Z15_NPU_BIAS, 0},
                    {Z15_NPU_FLAGS, flags}, {Z15_NPU_SHAPE, 0}, {Z15_NPU_SLICE, 0}, {Z15_NPU_PARTIAL, 0},
                    {P17_WEIGHT_CONFIG, P17_SIGNED_W4 | GROUP_SCALE | coefficient << 7}
                };
                const valence_cmd_batch_t batch = {words, sizeof(words) / sizeof(words[0]), Z15_START_NPU};
                if (ready(d, 0, Z15_NPU_BUSY) || submit_batch(d, &batch) ||
                    ready(d, Z15_NPU_DONE, Z15_NPU_BUSY))
                    return -1;
            }
        if (dma(d, output + first * WIDTH * 4, half + C_BASE, panels * WIDTH * 4, 1))
            return -1;
    }
    return 0;
}

int c11b_z15_project_block16(void *ctx, const uint8_t *packed, const uint8_t *fine,
                            unsigned n, unsigned k, unsigned group, const int8_t *a8,
                            const uint16_t *coeff, int32_t *out)
{
    c11b_z15 *d = ctx;
    if (!d || !a8 || !out || k != QWEN_HIDDEN || d->target != VALENCE_P17)
        return -1;
    const unsigned output = d->ring + GEMM_OUTPUT;
    return host_write(d, d->ring, a8, k) ||
           schedule_block16(d, packed, fine, n, k, group, coeff, output, 0) ||
           host_read(d, output, out, n * sizeof(*out)) ? -1 : 0;
}

static int project_block16_pair(void *ctx, const valence_block16_projection *jobs,
                                unsigned group, const int8_t *a8)
{
    c11b_z15 *d = ctx;
    const unsigned base = d->ring + GEMM_OUTPUT;
    unsigned output = base;
    if (host_write(d, d->ring, a8, QWEN_HIDDEN))
        return -1;
    for (unsigned i = 0; i < 2; ++i) {
        const valence_block16_projection *job = jobs + i;
        if (schedule_block16(d, job->packed, job->fine, job->channels, QWEN_HIDDEN,
                             group, job->coefficients, output, 1))
            return -1;
        output += job->channels * 4u;
    }
    output = base;
    for (unsigned i = 0; i < 2; ++i) {
        const unsigned bytes = jobs[i].channels * 4u;
        if (host_read(d, output, jobs[i].partials, bytes))
            return -1;
        output += bytes;
    }
    return 0;
}

static int project(void *ctx, const uint8_t *packed, unsigned n, unsigned k,
                   const int8_t *a8, int32_t *result)
{
    return c11b_z15_project_grouped(ctx, packed, n, k, QWEN_GROUP, 4, a8, result);
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
    if (d->turbo || d->cmdq) {
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
        const uint32_t wait = Z15_GPU_PENDING | (1u << warp);
        if (d->cmdq) {
            if (queue_add(d, CMDQ_ZERO, Z15_GPU_SLOTS, wait))
                return -1;
        } else {
            unsigned i = 0;
            uint32_t slots;
            do {
                if (rd(d, Z15_GPU_SLOTS, &slots) || (slots & Z15_GPU_SLOT_FAULTS) || ++i == POLL)
                    return -1;
            } while (slots & wait);
        }
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

static c11b_z15 *open_device(const ysyx_hal_t *hal, const uint8_t *image, size_t count,
                            valence_target target, const char **error, int probe)
{
    const unsigned ring_bytes = target == VALENCE_P17 && image && count >= 8 &&
        (!memcmp(image, "YSQC14D", 8) || !memcmp(image, "YSQC14H", 8)) ? RESIDUAL_RING : RING;
    if ((target != VALENCE_STEP2 && target != VALENCE_P17) || !image || !count ||
        count > Z15_POOL_SIZE - ring_bytes - 32 || !hal || !hal->mmio_read32 ||
        !hal->mmio_write32 || !hal->ddr_write || !hal->ddr_read || !hal->cache_sync)
        return NULL;
    c11b_z15 *d = calloc(1, sizeof(*d));
    if (!d)
        return NULL;
    d->target = target;
    if (target == VALENCE_STEP2)
        d->expanded = malloc(MAX_EXPANDED);
    d->hal = hal;
    d->pipeline = 1;
    d->ring = ((unsigned)count + 31u) & ~31u;
    d->ops = (c11b_ops){d, project, rms_hw, rope_hw, attention_hw, swiglu_hw, bad};
    uint32_t magic, version;
    if ((target == VALENCE_STEP2 && !d->expanded) ||
        rd(d, Z15_MAGIC, &magic) || magic != Z15_MAGIC_VALUE ||
        rd(d, Z15_VERSION, &version) || (version & Z15_VERSION_COMPAT_MASK) != Z15_VERSION_VALUE ||
        ready(d, 0, Z15_GPU_BUSY | Z15_DMA_BUSY | Z15_NPU_BUSY)) {
        if (error)
            *error = "Qwen allocation, Z15 ABI, or DDR initialization failed";
        c11b_z15_close(d);
        return NULL;
    }
    const int mode = probe ? z15_cmdq_select(hal) : 0;
    d->queue_base = d->ring + ring_bytes;
    d->cache_floor = Z15_POOL_SIZE;
    if (mode < 0 || (mode && (d->queue_base > Z15_POOL_SIZE - 32u || !z15_cmdq_idle(hal)))) {
        if (error)
            *error = "Qwen requires Phase 18 CMDQ capability and reserved queue DDR";
        c11b_z15_close(d);
        return NULL;
    }
    if ((target == VALENCE_P17 && (wr(d, P17_WEIGHT_CONFIG, 0) ||
                                  wr(d, P17_KV_CONFIG, 0) || error_reg(d))) ||
        ((target == VALENCE_P17 || !hal->observe) && host_write(d, 0, image, count))) {
        if (error)
            *error = "Qwen Z15 register or DDR initialization failed";
        c11b_z15_close(d);
        return NULL;
    }
    d->cmdq = mode;
    if (target == VALENCE_P17) {
        d->resident_image = image;
        d->resident_bytes = count;
    }
    return d;
}

c11b_z15 *c11b_z15_open_target(const ysyx_hal_t *hal, const uint8_t *image, size_t count,
                              valence_target target, const char **error)
{
    return open_device(hal, image, count, target, error, 1);
}

c11b_z15 *c11b_z15_open(const ysyx_hal_t *hal, const uint8_t *image, size_t count,
                       const char **error)
{
    return open_device(hal, image, count, VALENCE_STEP2, error, 0);
}

valence_submission_stats c11b_z15_submission_stats(const c11b_z15 *d)
{
    return d ? d->submissions : (valence_submission_stats){0};
}

z15_cmdq_stats c11b_z15_queue_stats(const c11b_z15 *d)
{
    return d ? d->queued : (z15_cmdq_stats){0};
}

const c11b_ops *c11b_z15_ops(c11b_z15 *d)
{
    return d ? &d->ops : NULL;
}

void c11b_z15_close(c11b_z15 *d)
{
    if (d) {
        z15_cmdq_destroy(d->queue);
        for (unsigned i = 0; i < d->cache_count; ++i)
            z15_cmdq_destroy(d->plans[i].plan);
        free(d->repack_buffer);
        if (d->cmdq && getenv("YSYX_CMDQ_STATS")) {
            z15_cmdq_report(stderr, &d->queued);
            if (d->c18_static)
                fprintf(stderr, "C18 resident_plans=%u resident_cmdq_bytes=%u repacked_planes=%u\n",
                        d->cache_count, Z15_POOL_SIZE - C18_QUEUE_GUARD - d->cache_floor,
                        d->repack_count);
        }
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
    if (!d || (c14_qwen_block16(m) && (d->target != VALENCE_P17 || c14_qwen_block16(m) < 2)))
        return -1;
    c12_qwen_device ops = {.ctx = d, .project = c11b_z15_project_grouped, .rope = rope_hw,
                          .residual = residual_q16, .project_block16 = c11b_z15_project_block16};
    const char *turbo = getenv("YSYX_QWEN_C16");
    if (d->target == VALENCE_P17 && d->cmdq && c14_qwen_block16(m) == 4 &&
        (!turbo || !strcmp(turbo, "1"))) {
        ops.project_parallel = project_parallel;
        const char *c17 = getenv("YSYX_QWEN_C17");
        d->c17_head = !c17 || !strcmp(c17, "1");
        const char *c18 = getenv("YSYX_QWEN_C18");
        d->c18_static = d->c17_head && (!c18 || !strcmp(c18, "1")) &&
                        d->queue_base <= Z15_POOL_SIZE - C18_QUEUE_GUARD - C18_TRANSIENT_RESERVE;
        if (d->c18_static) {
            d->cache_floor = Z15_POOL_SIZE - C18_QUEUE_GUARD;
            ops.project_block16_wide = project_block16_wide;
        }
        ops.project_head = project_head;
        ops.head_panel = d->c17_head ? QWEN_VOCAB : C16_HEAD_PANEL;
        ops.project_rows = project_rows;
        ops.project_block16_rows = project_block16_rows;
        ops.project_block16_pair = project_block16_pair;
        ops.rope = NULL;
        ops.residual = NULL;
    }
    return c12_qwen_set_device(m, &ops, layers);
}

int c16_z15_detect(const ysyx_hal_t *hal, valence_target *target, int *turbo)
{
    uint32_t version, capabilities;
    *target = VALENCE_STEP2;
    *turbo = 0;
    if (hal->mmio_read32(hal->ctx, V3_VERSION, &version))
        return 0;
    if (version == Z15_P18_VERSION) {
        if (hal->mmio_read32(hal->ctx, Z15_CAP_FLAGS, &capabilities) ||
            !(capabilities & Z15_CAP_CMDQ))
            return -1;
        *target = VALENCE_P17;
    } else if (version == UINT32_C(0x0c0c0101)) {
        *turbo = 1;
    } else if (version) {
        return -1;
    }
    return 0;
}

int c12_z15_enable_turbo(c11b_z15 *d)
{
    uint32_t version;
    if (!d || d->target != VALENCE_STEP2 || ready(d, 0, Z15_GPU_BUSY | Z15_DMA_BUSY | Z15_NPU_BUSY) ||
        rd(d, V3_VERSION, &version) || version != UINT32_C(0x0c0c0101) || error_reg(d))
        return -1;
    d->turbo = 1;
    return 0;
}
