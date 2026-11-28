#define _POSIX_C_SOURCE 200809L
#include "z15_cmdq.h"
#include "z15_regs.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    size_t first, count;
    uint32_t offset;
    const uint8_t *image;
    size_t image_bytes;
} batch_t;

struct z15_cmdq_plan {
    uint8_t *bytes;
    size_t count, capacity;
    batch_t *batches;
    size_t batch_count, batch_capacity;
    const uint8_t *image, *loaded_image;
    size_t image_bytes, loaded_bytes, first;
    uint32_t ddr_end;
    int placed, uploaded, p19;
};

static int writable(uint32_t reg, uint32_t value)
{
    return reg == Z15_SCRATCH || (reg == Z15_CTRL && !(value & 1u)) ||
           (reg >= 0x20 && reg <= 0x30) || (reg >= 0x40 && reg <= 0x44) ||
           (reg >= 0x60 && reg <= 0x98) || (reg >= 0xa0 && reg <= 0xd8) ||
           (reg >= 0x1000 && reg <= 0x18fc) || (reg >= 0x1c00 && reg <= 0x1ffc);
}

int z15_cmdq_validate(uint64_t word)
{
    const uint32_t value = (uint32_t)word;
    const uint32_t reg = (uint16_t)(word >> 32);
    switch (word >> 60) {
    case CMDQ_NOP:
    case CMDQ_IDLE:
    case CMDQ_END:
        return (word << 4) ? 4 : 0;
    case CMDQ_WRITE:
    case CMDQ_ZERO:
    case CMDQ_ONE:
        if ((word >> 48) & 0xfffu)
            return 4;
        if (reg & 3u)
            return 5;
        if ((word >> 60) == CMDQ_WRITE)
            return writable(reg, value) ? 0 : 5;
        return (reg == Z15_STATUS || reg == Z15_ERROR || reg == Z15_GPU_SLOTS) ? 0 : 5;
    default:
        return 4;
    }
}

static unsigned hardware_capabilities(const ysyx_hal_t *hal)
{
    uint32_t ext, cap;
    if (hal->mmio_read32(hal->ctx, Z15_EXT_VERSION, &ext) || ext != Z15_P18_VERSION ||
        hal->mmio_read32(hal->ctx, Z15_CAP_FLAGS, &cap))
        return 0;
    return cap;
}

int z15_cmdq_select(const ysyx_hal_t *hal)
{
    const char *mode = getenv("YSYX_EXEC_MODE");
    if (mode && !strcmp(mode, "mmio"))
        return 0;
    const unsigned cap = hardware_capabilities(hal);
    if (mode && !strcmp(mode, "p19a"))
        return (cap & (Z15_CAP_P19A | Z15_CAP_CMDQ)) == (Z15_CAP_P19A | Z15_CAP_CMDQ) ? 1 : -1;
    const int forced = mode && (!strcmp(mode, "cmdq") || !strcmp(mode, "p18b"));
    return cap & Z15_CAP_CMDQ ? 1 : forced ? -1 : 0;
}

int z15_cmdq_p19(const ysyx_hal_t *hal)
{
    const char *mode = getenv("YSYX_EXEC_MODE");
    /* TMA is an explicit comparison path until the frozen wall gate passes. */
    if (!mode || strcmp(mode, "p19a"))
        return 0;
    return (hardware_capabilities(hal) & Z15_CAP_P19A) == Z15_CAP_P19A;
}

void z15_cmdq_allow_p19(z15_cmdq_plan *plan)
{
    plan->p19 = 1;
}

static int p19_word(z15_cmdq_op op, unsigned reg, unsigned value)
{
    if (op == CMDQ_SIGNAL || op == CMDQ_CONSUME || op == CMDQ_RESET)
        return reg < 8 && !value;
    if (reg & 3u)
        return 0;
    if (op == CMDQ_WRITE)
        return (reg >= Z15_TMA_DDR && reg <= Z15_TMA_CTRL) || reg == Z15_NPU_BIND || reg == Z15_VPU_BIND;
    return (op == CMDQ_ONE || op == CMDQ_ZERO) &&
           (reg == Z15_TMA_STATUS || reg == Z15_TMA_ERR || reg == 0x1c8 || reg == 0x1cc);
}

int z15_cmdq_idle(const ysyx_hal_t *hal)
{
    uint32_t status;
    return !hal->mmio_read32(hal->ctx, CMDQ_STATUS, &status) &&
           !(status & (CMDQ_BUSY | CMDQ_ERROR));
}

z15_cmdq_plan *z15_cmdq_create(void)
{
    return calloc(1, sizeof(z15_cmdq_plan));
}

void z15_cmdq_reset(z15_cmdq_plan *plan)
{
    const z15_cmdq_plan storage = {
        .bytes = plan->bytes, .capacity = plan->capacity,
        .batches = plan->batches, .batch_capacity = plan->batch_capacity, .p19 = plan->p19
    };
    *plan = storage;
}

void z15_cmdq_destroy(z15_cmdq_plan *plan)
{
    if (plan) {
        free(plan->bytes);
        free(plan->batches);
        free(plan);
    }
}

static int append(z15_cmdq_plan *plan, uint64_t word)
{
    if (plan->count == plan->capacity) {
        const size_t max_words = Z15_POOL_SIZE / 8u;
        size_t capacity = plan->capacity ? plan->capacity * 2u : 1024u;
        if (capacity > max_words)
            return -1;
        uint8_t *bytes = realloc(plan->bytes, capacity * 8u);
        if (!bytes)
            return -1;
        plan->bytes = bytes;
        plan->capacity = capacity;
    }
    uint8_t *dst = plan->bytes + plan->count++ * 8u;
    for (unsigned i = 0; i < 8; ++i)
        dst[i] = (uint8_t)(word >> (i * 8u));
    return 0;
}

static int seal(z15_cmdq_plan *plan)
{
    if (plan->count == plan->first)
        return 0;
    if (plan->batch_count == plan->batch_capacity) {
        size_t capacity = plan->batch_capacity ? plan->batch_capacity * 2u : 8u;
        batch_t *batches = realloc(plan->batches, capacity * sizeof(*batches));
        if (!batches)
            return -1;
        plan->batches = batches;
        plan->batch_capacity = capacity;
    }
    if (append(plan, (uint64_t)CMDQ_IDLE << 60) || append(plan, (uint64_t)CMDQ_END << 60))
        return -1;
    plan->batches[plan->batch_count++] = (batch_t){
        .first = plan->first, .count = plan->count - plan->first,
        .image = plan->image, .image_bytes = plan->image_bytes
    };
    plan->first = plan->count;
    return 0;
}

int z15_cmdq_add(z15_cmdq_plan *plan, z15_cmdq_op op, uint32_t reg, uint32_t value)
{
    if (plan->placed || reg > UINT16_MAX || (unsigned)op >= CMDQ_END)
        return -1;
    const uint64_t word = (uint64_t)op << 60 | (uint64_t)reg << 32 | value;
    if ((z15_cmdq_validate(word) && !(plan->p19 && p19_word(op, reg, value))) ||
        (plan->count - plan->first == CMDQ_LIMIT - 2u && seal(plan)))
        return -1;
    return append(plan, word);
}

int z15_cmdq_image(z15_cmdq_plan *plan, const uint8_t *image, size_t size)
{
    if (plan->placed || !size || size > Z15_PROGRAM_BYTES || (size & 3u))
        return -1;
    if (plan->image && seal(plan))
        return -1;
    plan->image = image;
    plan->image_bytes = size;
    return 0;
}

size_t z15_cmdq_batches(const z15_cmdq_plan *plan)
{
    return plan->batch_count;
}

unsigned z15_cmdq_count(const z15_cmdq_plan *plan)
{
    return (unsigned)plan->count;
}

int z15_cmdq_place_bounded(z15_cmdq_plan *plan, uint32_t base, uint32_t limit)
{
    if (plan->placed || base > limit || limit > Z15_POOL_SIZE || seal(plan))
        return -1;
    uint64_t cursor = ((uint64_t)base + 31u) & ~UINT64_C(31);
    if (cursor > limit)
        return -1;
    for (size_t i = 0; i < plan->batch_count; ++i) {
        batch_t *batch = plan->batches + i;
        uint64_t end = (cursor + batch->count * 8u + 31u) & ~UINT64_C(31);
        if (end > limit)
            return -1;
        batch->offset = (uint32_t)cursor;
        cursor = end;
    }
    plan->ddr_end = (uint32_t)cursor;
    plan->placed = 1;
    return 0;
}

int z15_cmdq_place(z15_cmdq_plan *plan, uint32_t base)
{
    return z15_cmdq_place_bounded(plan, base, Z15_POOL_SIZE);
}

int z15_cmdq_place_tail(z15_cmdq_plan *plan, uint32_t low, uint32_t *upper)
{
    if (seal(plan) || low > *upper || (*upper & 31u))
        return -1;
    size_t bytes = 0;
    for (size_t i = 0; i < plan->batch_count; ++i)
        bytes += (plan->batches[i].count * 8u + 31u) & ~(size_t)31;
    if (bytes > *upper - low)
        return -1;
    const uint32_t base = *upper - (uint32_t)bytes;
    if (z15_cmdq_place_bounded(plan, base, *upper))
        return -1;
    *upper = base;
    return 0;
}

void z15_cmdq_discard_uploaded_bytes(z15_cmdq_plan *plan)
{
    if (plan->uploaded) {
        free(plan->bytes);
        plan->bytes = NULL;
        plan->capacity = 0;
    }
}

static int rd(const ysyx_hal_t *hal, z15_cmdq_stats *s, uint32_t reg, uint32_t *value)
{
    ++s->reads;
    return hal->mmio_read32(hal->ctx, reg, value);
}

static int wr(const ysyx_hal_t *hal, z15_cmdq_stats *s, uint32_t reg, uint32_t value)
{
    ++s->writes;
    return hal->mmio_write32(hal->ctx, reg, value);
}

static int wait_clear(const ysyx_hal_t *hal, z15_cmdq_stats *s, uint32_t reg,
                      uint32_t busy, uint32_t *value)
{
    struct timespec start, now;
    if (clock_gettime(CLOCK_MONOTONIC, &start))
        return -1;
    for (unsigned i = 0; i < 50000000u; ++i) {
        if (rd(hal, s, reg, value))
            return -1;
        if (!(*value & busy))
            return 0;
        if (!hal->observe && i >= 2u) {
            const struct timespec pause = {0, 50000};
            if (clock_gettime(CLOCK_MONOTONIC, &now) ||
                now.tv_sec - start.tv_sec > 30 ||
                (now.tv_sec - start.tv_sec == 30 && now.tv_nsec >= start.tv_nsec))
                return -1;
            (void)nanosleep(&pause, NULL);
        }
    }
    return -1;
}

static int upload_image(z15_cmdq_plan *plan, const batch_t *b, const ysyx_hal_t *hal,
                        z15_cmdq_stats *s)
{
    if (!b->image || (b->image_bytes == plan->loaded_bytes &&
                      !memcmp(b->image, plan->loaded_image, b->image_bytes)))
        return 0;
    uint32_t status;
    if (wait_clear(hal, s, Z15_STATUS, Z15_GPU_BUSY, &status) ||
        !(status & Z15_LOCKED) || (status & Z15_HAS_ERROR))
        return -1;
    for (size_t i = 0; i < b->image_bytes; i += 4u) {
        const uint8_t *p = b->image + i;
        uint32_t word = (uint32_t)p[0] | (uint32_t)p[1] << 8 |
                        (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
        if (wr(hal, s, Z15_PROGRAM + (uint32_t)i, word))
            return -1;
    }
    plan->loaded_image = b->image;
    plan->loaded_bytes = b->image_bytes;
    return 0;
}

static int submit(const ysyx_hal_t *hal, const batch_t *b, z15_cmdq_stats *s)
{
    uint32_t lo, hi;
    if (wr(hal, s, CMDQ_BASE_LO, Z15_POOL_BASE + b->offset) ||
        wr(hal, s, CMDQ_BASE_HI, 0) || wr(hal, s, CMDQ_COUNT, (uint32_t)b->count))
        return -1;
    __sync_synchronize();
    ++s->starts;
    const int start = wr(hal, s, CMDQ_CTRL, CMDQ_START);
    const int wait = wait_clear(hal, s, CMDQ_STATUS, CMDQ_BUSY, &s->status);
    const int queue_error = rd(hal, s, CMDQ_ERR_CODE, &s->code);
    const int legacy_error = rd(hal, s, Z15_ERROR, &s->legacy_error);
    const int completed = rd(hal, s, CMDQ_COMPLETED, &s->completed);
    if (start || wait || queue_error || legacy_error || completed ||
        !(s->status & CMDQ_DONE) || (s->status & CMDQ_ERROR) ||
        s->code || s->legacy_error || s->completed != b->count) {
        uint32_t ignored;
        if (!wr(hal, s, CMDQ_CTRL, CMDQ_CLEAR)) {
            (void)wait_clear(hal, s, CMDQ_STATUS, CMDQ_BUSY, &ignored);
            (void)wait_clear(hal, s, Z15_STATUS,
                             Z15_NPU_BUSY | Z15_GPU_BUSY | Z15_DMA_BUSY, &ignored);
        }
        return -1;
    }
    /* The ABI freezes both halves when BUSY falls. */
    if (rd(hal, s, CMDQ_CYCLES + 4, &hi) || rd(hal, s, CMDQ_CYCLES, &lo))
        return -1;
    s->cycles += (uint64_t)hi << 32 | lo;
    s->commands += s->completed;
    return 0;
}

int z15_cmdq_run(z15_cmdq_plan *plan, const ysyx_hal_t *hal, z15_cmdq_stats *s)
{
    memset(s, 0, sizeof(*s));
    s->ddr_end = plan->ddr_end;
    if (!plan->placed || rd(hal, s, CMDQ_STATUS, &s->status) ||
        (s->status & (CMDQ_BUSY | CMDQ_ERROR)))
        return -1;
    if (!plan->uploaded) {
        for (size_t i = 0; i < plan->batch_count; ++i) {
            const batch_t *b = plan->batches + i;
            const size_t length = b->count * 8u;
            if (hal->ddr_write(hal->ctx, b->offset, plan->bytes + b->first * 8u, length) ||
                hal->cache_sync(hal->ctx, b->offset, length, YSYX_SYNC_FOR_DEVICE))
                return -1;
            s->upload_bytes += length;
        }
        plan->uploaded = 1;
    }
    for (size_t i = 0; i < plan->batch_count; ++i) {
        const batch_t *b = plan->batches + i;
        if (upload_image(plan, b, hal, s) || submit(hal, b, s))
            return -1;
    }
    return 0;
}

void z15_cmdq_report(FILE *out, const z15_cmdq_stats *s)
{
    fprintf(out, "CMDQ starts=%" PRIu64 " commands=%" PRIu64 " cycles=%" PRIu64
            " arm_reads=%" PRIu64 " arm_writes=%" PRIu64 " upload_bytes=%" PRIu64
            " status=0x%x completed=%u code=%u abi_error=0x%x\n",
            s->starts, s->commands, s->cycles, s->reads, s->writes,
            s->upload_bytes, s->status, s->completed, s->code, s->legacy_error);
}
