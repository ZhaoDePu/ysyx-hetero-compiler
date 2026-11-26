#include "ysyx_rt.h"
#include "z15_regs.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    uint32_t version, status, error;
    uint32_t counter_reg, phase, ticks, rollovers;
    int seen_status, seen_slot, dma_sync, due_error, inject;
    unsigned initial_writes, starts, statuses, gpu_writes, gpu_starts;
} model_t;

static int read_reg(void *opaque, uint32_t reg, uint32_t *value)
{
    model_t *m = opaque;
    if (m->due_error && reg != Z15_ERROR)
        return -1;
    switch (reg) {
    case Z15_MAGIC: *value = Z15_MAGIC_VALUE; return 0;
    case Z15_VERSION: *value = m->version; return 0;
    case Z15_ERROR: m->due_error = 0; *value = m->error; return 0;
    case Z15_STATUS:
        m->seen_status = 1;
        ++m->statuses;
        *value = m->status;
        return 0;
    case Z15_GPU_SLOTS: m->seen_slot = 1; *value = 0; return 0;
    default: break;
    }
    uint32_t low = reg & ~7u;
    if (low < Z15_NPU_CYCLES || low > Z15_OVERLAP_CYCLES ||
        (m->phase && m->counter_reg != low))
        return -1;
    if (!m->phase || m->phase == 2u) {
        if (reg != low + 4u)
            return -1;
    } else if (reg != low) {
        return -1;
    }
    m->counter_reg = low;
    m->phase = (m->phase + 1u) % 3u;
    if (low == Z15_WALL_CYCLES && !m->rollovers) {
        if (!m->phase)
            ++m->rollovers;
        *value = m->phase == 1u ? 0u : m->phase == 2u ? UINT32_MAX : 1u;
    } else {
        *value = reg == low + 4u ? 1u : m->ticks;
    }
    return 0;
}

static int write_reg(void *opaque, uint32_t reg, uint32_t value)
{
    model_t *m = opaque;
    if (m->due_error)
        return -1;
    if (reg == Z15_DMA_BASE && (!m->seen_status || !m->dma_sync || value != Z15_POOL_BASE))
        return -1;
    if ((reg == Z15_NPU_DIMS || (reg >= Z15_PROGRAM && reg < Z15_PROGRAM + Z15_PROGRAM_BYTES)) &&
        !m->seen_status)
        return -1;
    if (reg >= Z15_PROGRAM && reg < Z15_PROGRAM + Z15_PROGRAM_BYTES)
        ++m->gpu_writes;
    if (reg == Z15_CTRL) {
        if ((value & ~Z15_ACTIONS) || !value || (value & (value - 1u)))
            return -1;
        if (value == Z15_START_GPU && !m->seen_slot)
            return -1;
        if (value == Z15_START_GPU)
            ++m->gpu_starts;
        m->seen_status = m->seen_slot = m->dma_sync = 0;
        m->due_error = 1;
        ++m->starts;
        m->ticks += 11;
        if (m->inject)
            m->error = Z15_ERR_DMA_DESC; /* Deliberately omit STATUS[7]. */
        switch (value) {
        case Z15_START_DMA: m->status |= Z15_DMA_DONE; break;
        case Z15_START_GPU: m->status |= Z15_GPU_DONE; break;
        case Z15_START_NPU:
        case Z15_START_ATTN:
        case Z15_START_EW: m->status |= Z15_NPU_DONE; break;
        default: break;
        }
    }
    return 0;
}

static int put_ddr(void *opaque, uint32_t offset, const void *data, size_t size)
{
    model_t *m = opaque;
    if (!m->seen_status || m->error || !data || !size || offset >= Z15_POOL_SIZE)
        return -1;
    ++m->initial_writes;
    return 0;
}

static int get_ddr(void *opaque, uint32_t offset, void *data, size_t size)
{
    (void)opaque;
    (void)offset;
    (void)data;
    (void)size;
    return 0;
}

static int synchronize(void *opaque, uint32_t offset, size_t size, int dir)
{
    model_t *m = opaque;
    if (!size || offset >= Z15_POOL_SIZE)
        return -1;
    m->dma_sync = dir == YSYX_SYNC_FOR_DEVICE;
    return 0;
}

static int check(ysyx_program_t *p, const ysyx_input_t *input)
{
    const char *error = NULL;
    model_t m = {.version = Z15_VERSION_VALUE | 7u, .status = Z15_LOCKED};
    const ysyx_hal_t h = {&m, read_reg, write_reg, put_ddr, get_ddr, synchronize, NULL, NULL};
    ysyx_stats_t stats;
    if (ysyx_execute(p, &h, input, 1, &stats, NULL, &error) ||
        m.due_error || m.rollovers != 1 || stats.abi_error ||
        stats.invalid_dma_descriptors || stats.dma_operations != 3u ||
        m.starts < 5u || stats.wall_ticks != m.ticks) {
        fprintf(stderr, "baseline ABI sequencing test failed: %s\n", error ? error : "bad counters/order");
        return 1;
    }
    m = (model_t){.version = Z15_VERSION_VALUE ^ 0x100u, .status = Z15_LOCKED};
    if (!ysyx_execute(p, &h, input, 1, &stats, NULL, &error) || m.initial_writes || m.starts)
        return 2;
    m = (model_t){.version = Z15_VERSION_VALUE};
    if (!ysyx_execute(p, &h, input, 1, &stats, NULL, &error) || m.initial_writes || m.starts)
        return 3;
    m = (model_t){.version = Z15_VERSION_VALUE, .status = Z15_LOCKED, .error = Z15_ERR_DMA_DESC};
    if (!ysyx_execute(p, &h, input, 1, &stats, NULL, &error) || m.initial_writes || m.starts)
        return 4;
    m = (model_t){.version = Z15_VERSION_VALUE, .status = Z15_LOCKED, .inject = 1};
    if (!ysyx_execute(p, &h, input, 1, &stats, NULL, &error) || m.starts != 1 || m.due_error)
        return 5;
    puts("STEP2_MOCK_HAL_PASS version ready sticky-error post-action-fault ordering counter-rollover");
    return 0;
}

static void *open_fixture(const char *path, ysyx_program_t **p)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    long length = fseek(f, 0, SEEK_END) ? -1 : ftell(f);
    void *bytes = length > 0 && !fseek(f, 0, SEEK_SET) ? malloc((size_t)length) : NULL;
    int read_ok = bytes && fread(bytes, 1, (size_t)length, f) == (size_t)length;
    if (fclose(f) || !read_ok || ysyx_program_open(bytes, (size_t)length, p, NULL)) {
        free(bytes);
        return NULL;
    }
    return bytes;
}

static int check_gpu_upload(const char *path, unsigned expected_writes)
{
    ysyx_program_t *p = NULL;
    void *bytes = open_fixture(path, &p);
    if (!bytes)
        return 1;
    model_t m = {.version = Z15_VERSION_VALUE, .status = Z15_LOCKED};
    const ysyx_hal_t h = {&m, read_reg, write_reg, put_ddr, get_ddr, synchronize, NULL, NULL};
    char names[24][16];
    const uint8_t tokens[192] = {0};
    ysyx_input_t inputs[24];
    for (unsigned i = 0; i < 24; ++i) {
        snprintf(names[i], sizeof(names[i]), "c%02u_tokens", i);
        inputs[i] = (ysyx_input_t){names[i], tokens, sizeof(tokens)};
    }
    ysyx_stats_t stats;
    const char *error = NULL;
    int rc = ysyx_execute(p, &h, inputs, 24, &stats, NULL, &error);
    ysyx_program_close(p);
    free(bytes);
    if (rc || m.gpu_writes != expected_writes || m.gpu_starts != 384) {
        fprintf(stderr, "GPU image upload test failed: %u writes, %u launches (%s)\n",
                m.gpu_writes, m.gpu_starts, error ? error : "bad upload count");
        return 1;
    }
    printf("GPU_IMAGE_UPLOAD_PASS writes=%u launches=%u\n", m.gpu_writes, m.gpu_starts);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 2 && argc != 4)
        return 1;
    ysyx_program_t *p = NULL;
    void *bytes = open_fixture(argv[1], &p);
    if (!bytes)
        return 1;
    ysyx_symbol_t symbol;
    const uint8_t data[32] = {0};
    if (ysyx_find_symbol(p, "x", YSYX_INPUT, &symbol) || symbol.bytes > sizeof(data))
        return 1;
    const ysyx_input_t input = {"x", data, symbol.bytes};
    int rc = check(p, &input);
    ysyx_program_close(p);
    free(bytes);
    if (rc || argc == 2)
        return rc;
    return check_gpu_upload(argv[2], (unsigned)strtoul(argv[3], NULL, 10));
}
