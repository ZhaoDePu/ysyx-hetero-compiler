/* Exercise extension discovery and packed table writes with no physical I/O. */
#include "c12_qwen_ref.h"
#include "z15_regs.h"

#include <stdio.h>
#include <string.h>

typedef struct {
    int is_turbo;
    unsigned capability_reads, packed_writes, classic_writes;
    unsigned dma_high_reads;
    uint8_t activations[256];
} bus;

static int read_reg(void *ctx, uint32_t reg, uint32_t *value)
{
    bus *state = ctx;
    *value = 0;
    if (reg == Z15_MAGIC)
        *value = Z15_MAGIC_VALUE;
    else if (reg == Z15_VERSION)
        *value = Z15_VERSION_VALUE;
    else if (reg == Z15_STATUS)
        *value = Z15_LOCKED | Z15_DMA_DONE | Z15_NPU_DONE | Z15_GPU_DONE;
    else if (reg == 0x0120) {
        ++state->capability_reads;
        if (!state->is_turbo)
            return -1;
        *value = UINT32_C(0x0c0c0101);
    } else if (reg == 0x0100) {
        *value = 3;
    } else if (reg == 0x0104) {
        *value = state->dma_high_reads++ == 0 ? 1 : 2;
    }
    return 0;
}

static int write_reg(void *ctx, uint32_t reg, uint32_t value)
{
    bus *state = ctx;
    if (reg >= 0x1800 && reg < 0x1900) {
        if (!state->is_turbo || (reg & 3))
            return -1;
        for (unsigned lane = 0; lane < 4; ++lane)
            state->activations[reg - 0x1800 + lane] = (uint8_t)(value >> (lane * 8));
        ++state->packed_writes;
    } else if (reg == Z15_ACTIVATION) {
        state->activations[value & 255] = (uint8_t)(value >> 8);
        ++state->classic_writes;
    }
    return 0;
}

static int write_data(void *ctx, uint32_t offset, const void *data, size_t size)
{
    (void)ctx;
    (void)offset;
    (void)data;
    (void)size;
    return 0;
}

static int read_data(void *ctx, uint32_t offset, void *data, size_t size)
{
    (void)ctx;
    (void)offset;
    memset(data, 0, size);
    return 0;
}

static int sync_data(void *ctx, uint32_t offset, size_t size, int direction)
{
    (void)direction;
    return write_data(ctx, offset, NULL, size);
}

static int exercise(int turbo)
{
    bus state = {.is_turbo = turbo};
    const ysyx_hal_t hal = {&state, read_reg, write_reg, write_data, read_data,
                            sync_data, NULL, NULL};
    const uint8_t dummy[] = {0};
    int8_t zero[QWEN_INTER] = {0}, lut[256], result[QWEN_INTER];
    for (unsigned i = 0; i < 256; ++i)
        lut[i] = (int8_t)((int)i - 128);
    c11b_z15 *device = c11b_z15_open(&hal, dummy, sizeof(dummy), NULL);
    if (!device)
        return 1;
    const c11b_ops *ops = c11b_z15_ops(device);
    ysyx_stats_t stats;
    int fail = state.capability_reads ||
               (turbo && c12_z15_enable_turbo(device)) ||
               ops->swiglu(ops->ctx, zero, zero, lut, 7, result) ||
               c11b_z15_stats(device, &stats);
    if (!fail)
        fail = state.capability_reads != (unsigned)turbo ||
               state.packed_writes != (turbo ? 64u : 0u) ||
               state.classic_writes != (turbo ? 0u : 256u) ||
               memcmp(lut, state.activations, sizeof(lut)) ||
               stats.has_observations != turbo ||
               (turbo && (stats.observed.dma_busy_ticks != (UINT64_C(2) << 32 | 3u) ||
                          state.dma_high_reads != 4));
    c11b_z15_close(device);
    return fail;
}

int main(void)
{
    if (exercise(0) || exercise(1))
        return 1;
    puts("C12 v1 no extension probe, Turbo capability, packed LUT and 64-bit retry PASS");
    return 0;
}
