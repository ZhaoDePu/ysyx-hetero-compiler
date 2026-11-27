#include "c11b_qwen_ops.h"
#include "z15_regs.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WEIGHT_CONFIG = 0xd0, KV_CONFIG = 0xd4, CHANNELS = 32, GROUP = 64,
       WEIGHTS = CHANNELS * 896, DDR_BYTES = 16 * 1024 * 1024 };

typedef struct {
    uint8_t *ddr, *scratch;
    uint32_t reg[128];
    unsigned p17, words, launches, reads, writes, w4_launches, w8_launches;
    unsigned weight_dma_bytes, total_dma_bytes, group_launches;
    unsigned held_address;
    int32_t accumulator[16];
    int held;
} bus;

static int read_reg(void *ctx, uint32_t reg, uint32_t *value)
{
    (void)ctx;
    switch (reg) {
        case Z15_MAGIC: *value = Z15_MAGIC_VALUE; break;
        case Z15_VERSION: *value = Z15_VERSION_VALUE; break;
        case Z15_STATUS: *value = Z15_LOCKED | Z15_NPU_DONE | Z15_DMA_DONE; break;
        default: *value = 0; break;
    }
    return 0;
}

static int nibble(uint8_t byte, unsigned index)
{
    return (int)(((byte >> (4 * (index & 1u))) & 15u) ^ 8u) - 8;
}

static int write_reg(void *ctx, uint32_t reg, uint32_t value)
{
    bus *s = ctx;
    if ((reg & 3) || reg / 4 >= sizeof(s->reg) / sizeof(s->reg[0]) ||
        (!s->p17 && (reg == WEIGHT_CONFIG || reg == KV_CONFIG)))
        return -1;
    ++s->words;
    s->reg[reg / 4] = value;
    if (reg != Z15_CTRL)
        return 0;
    ++s->launches;
    if (value == Z15_START_DMA) {
        unsigned offset = s->reg[Z15_DMA_OFFSET / 4];
        unsigned sp = s->reg[Z15_DMA_SCRATCH / 4];
        unsigned length = s->reg[Z15_DMA_BYTES / 4];
        if (s->reg[Z15_DMA_BASE / 4] != Z15_POOL_BASE || !length ||
            ((offset | sp | length) & 31) || offset > DDR_BYTES ||
            length > DDR_BYTES - offset || sp > Z15_SCRATCH_SIZE ||
            length > Z15_SCRATCH_SIZE - sp || (sp >> 20) != ((sp + length - 1) >> 20))
            return -1;
        s->total_dma_bytes += length;
        if (s->reg[Z15_DMA_FLAGS / 4] & 1u)
            memcpy(s->ddr + offset, s->scratch + sp, length);
        else {
            memcpy(s->scratch + sp, s->ddr + offset, length);
            if ((sp & (Z15_HALF_BYTES - 1)) >= 0x1000)
                s->weight_dma_bytes += length;
        }
        return 0;
    }
    unsigned quant = s->reg[WEIGHT_CONFIG / 4], group_mode = !!(quant & 4u);
    unsigned flags = s->reg[Z15_NPU_FLAGS / 4];
    unsigned width = s->reg[Z15_NPU_DIMS / 4] >> 16;
    int keep = !!(flags & (1u << 18)), hold = !!(flags & (1u << 19));
    unsigned c0 = s->reg[Z15_NPU_C / 4];
    if (value != Z15_START_NPU || !width || width > 496 ||
        (s->reg[Z15_NPU_DIMS / 4] & 65535u) != 1u ||
        s->reg[Z15_NPU_K / 4] != (GROUP | GROUP << 16) || s->reg[KV_CONFIG / 4])
        return -1;
    if (group_mode) {
        if (!s->p17 || width != 16 || (quant & 0x7fu) != 5u || (quant >> 7) > 256 ||
            flags & ~(3u << 18) || keep != s->held || (keep && s->held_address != c0))
            return -1;
        ++s->group_launches;
        s->held = hold;
        s->held_address = c0;
    } else if (quant > 1 || flags) {
        return -1;
    }
    unsigned w4 = quant & 1u;
    s->w4_launches += w4;
    s->w8_launches += !w4;
    unsigned a0 = s->reg[Z15_NPU_A / 4], b0 = s->reg[Z15_NPU_B / 4];
    unsigned row_bytes = w4 ? GROUP / 2 : GROUP;
    if (a0 > Z15_SCRATCH_SIZE - GROUP || b0 > Z15_SCRATCH_SIZE - width * row_bytes ||
        c0 > Z15_SCRATCH_SIZE - width * sizeof(int32_t))
        return -1;
    const uint8_t *a = s->scratch + a0, *b = s->scratch + b0;
    for (unsigned c = 0; c < width; ++c) {
        int32_t acc = 0;
        for (unsigned k = 0; k < GROUP; ++k) {
            const uint8_t *row = b + c * row_bytes;
            const int weight = w4 ? nibble(row[k / 2], k) : (int8_t)row[k];
            acc += (int8_t)a[k] * weight;
        }
        if (group_mode) {
            const int scaled = acc * (int)(quant >> 7);
            const unsigned magnitude = (unsigned)abs(scaled);
            const unsigned base = magnitude >> 8, rem = magnitude & 255u;
            const int rounded = (int)(base + (rem > 128 || (rem == 128 && (base & 1))));
            acc = (keep ? s->accumulator[c] : 0) + (scaled < 0 ? -rounded : rounded);
            s->accumulator[c] = acc;
        }
        if (!hold)
            memcpy(s->scratch + c0 + sizeof(acc) * c, &acc, sizeof(acc));
    }
    return 0;
}

static int write_ddr(void *ctx, uint32_t off, const void *buf, size_t bytes)
{
    bus *s = ctx;
    if (off > DDR_BYTES || bytes > DDR_BYTES - off)
        return -1;
    memcpy(s->ddr + off, buf, bytes);
    ++s->writes;
    return 0;
}

static int read_ddr(void *ctx, uint32_t off, void *buf, size_t bytes)
{
    bus *s = ctx;
    if (off > DDR_BYTES || bytes > DDR_BYTES - off)
        return -1;
    memcpy(buf, s->ddr + off, bytes);
    ++s->reads;
    return 0;
}

static int sync_ddr(void *ctx, uint32_t off, size_t bytes, int direction)
{
    (void)ctx;
    (void)direction;
    return off > DDR_BYTES || bytes > DDR_BYTES - off;
}

static int exercise(valence_target target, const uint8_t *image, const int8_t *input,
                    unsigned *mmio_writes)
{
    enum { GROUPS = 896 / GROUP, RESULTS = GROUPS * CHANNELS };
    bus s = {.p17 = target == VALENCE_P17, .ddr = calloc(DDR_BYTES, 1),
             .scratch = calloc(Z15_SCRATCH_SIZE, 1)};
    const ysyx_hal_t hal = {&s, read_reg, write_reg, write_ddr, read_ddr, sync_ddr, NULL, NULL};
    c11b_z15 *device = NULL;
    int32_t raw[RESULTS];
    int rc = 1;
    if (!s.ddr || !s.scratch)
        goto done;
    device = c11b_z15_open_target(&hal, image, WEIGHTS / 2 + WEIGHTS, target, NULL);
    if (!device)
        goto done;
    for (unsigned bits = 4; bits <= 8; bits += 4) {
        const uint8_t *weights = image + (bits == 4 ? 0 : WEIGHTS / 2);
        if (c11b_z15_project_grouped(device, weights, CHANNELS, 896, GROUP, bits, input, raw))
            goto done;
        for (unsigned i = 0; i < RESULTS; ++i) {
            int32_t expected = 0;
            for (unsigned k = 0; k < GROUP; ++k) {
                const unsigned at = i * GROUP + k;
                const int w = bits == 4 ? nibble(weights[at / 2], at) : (int8_t)weights[at];
                expected += w * input[(i / CHANNELS) * GROUP + k];
            }
            if (raw[i] != expected)
                goto done;
        }
    }
    const valence_submission_stats stats = c11b_z15_submission_stats(device);
    if (s.w4_launches != (s.p17 ? GROUPS : 0) || s.w8_launches != (s.p17 ? GROUPS : GROUPS * 2) ||
        s.weight_dma_bytes != (s.p17 ? WEIGHTS * 3 / 2 : WEIGHTS * 2) ||
        c11b_z15_expanded_bytes(device) != (s.p17 ? 0 : WEIGHTS) ||
        stats.batches != GROUPS * 8 || stats.doorbells != stats.batches ||
        stats.doorbells != s.launches || stats.mmio_writes != s.words ||
        stats.host_writes != s.writes || stats.host_reads != s.reads ||
        stats.host_reads != 2 || stats.host_writes != (s.p17 ? 3u : 5u))
        goto done;
    *mmio_writes = s.words;
    printf("target=%s W4_bytes=%u all_weight_DMA_B=%u expanded_B=%" PRIu64
           " MMIO_writes=%u doorbells=%u host_writes=%u host_reads=%u\n",
           s.p17 ? "p17" : "step2", s.p17 ? WEIGHTS / 2 : WEIGHTS, s.weight_dma_bytes,
           c11b_z15_expanded_bytes(device), s.words, s.launches, s.writes, s.reads);
    rc = 0;
done:
    c11b_z15_close(device);
    free(s.scratch);
    free(s.ddr);
    return rc;
}

static int block16(const uint8_t *image, const int8_t *input)
{
    enum { GROUPS = 896 / GROUP };
    bus s = {.p17 = 1, .ddr = calloc(DDR_BYTES, 1), .scratch = calloc(Z15_SCRATCH_SIZE, 1)};
    const ysyx_hal_t hal = {&s, read_reg, write_reg, write_ddr, read_ddr, sync_ddr, NULL, NULL};
    uint16_t coeffs[2 * GROUPS];
    int32_t results[CHANNELS];
    const uint16_t scales[] = {0, 3, 127, 128, 200, 255, 256};
    for (unsigned i = 0; i < 2 * GROUPS; ++i)
        coeffs[i] = scales[i % (sizeof(scales) / sizeof(scales[0]))];
    c11b_z15 *d = s.ddr && s.scratch ? c11b_z15_open_target(&hal, image, WEIGHTS / 2, VALENCE_P17, NULL) : NULL;
    int bad = !d || c11b_z15_project_block16(d, image, NULL, CHANNELS, 896, GROUP, input, coeffs, results);
    for (unsigned col = 0; !bad && col < CHANNELS; ++col) {
        int result = 0;
        for (unsigned g = 0; g < GROUPS; ++g) {
            unsigned row = (col / 16 * GROUPS + g) * 16 + col % 16;
            int dot = 0;
            for (unsigned k = 0; k < GROUP; ++k)
                dot += input[g * GROUP + k] * nibble(image[row * GROUP / 2 + k / 2], k);
            const int scaled = dot * coeffs[col / 16 * GROUPS + g];
            int rounded = scaled / 256, rem = abs(scaled % 256);
            if (rem > 128 || (rem == 128 && (rounded % 2)))
                rounded += scaled < 0 ? -1 : 1;
            result += rounded;
        }
        bad = results[col] != result;
    }
    const valence_submission_stats stats = c11b_z15_submission_stats(d);
    bad |= s.held || s.group_launches != 2 * GROUPS ||
           stats.doorbells != 2 * GROUPS + 3 || stats.host_writes != 2 ||
           stats.host_reads != 1 || s.weight_dma_bytes != WEIGHTS / 2;
    if (!bad)
        printf("P17_BLOCK16_MOCK_PASS groups=%u final_bytes=%zu W4_DMA_B=%u MMIO=%" PRIu64
               " doorbells=%" PRIu64 " host_writes=%" PRIu64 " host_reads=%" PRIu64 "\n",
               s.group_launches, sizeof(results), s.weight_dma_bytes, stats.mmio_writes,
               stats.doorbells, stats.host_writes, stats.host_reads);
    c11b_z15_close(d);
    free(s.ddr);
    free(s.scratch);
    return bad;
}

static int full_projection(const int8_t *input, unsigned variant)
{
    enum { K = 896, N = 4864, GROUPS = K / GROUP, PANELS = N / 16, PACKED = K * N / 2 };
    const unsigned passes = variant == 2 ? 2 : 1;
    bus s = {.p17 = variant != 0, .ddr = calloc(DDR_BYTES, 1), .scratch = calloc(Z15_SCRATCH_SIZE, 1)};
    const ysyx_hal_t hal = {&s, read_reg, write_reg, write_ddr, read_ddr, sync_ddr, NULL, NULL};
    uint8_t *weights = malloc(PACKED * passes);
    int32_t *result = malloc(N * (variant < 2 ? GROUPS : 1) * sizeof(*result));
    uint16_t coeffs[2 * GROUPS * PANELS];
    c11b_z15 *device = NULL;
    int rc = 1;
    if (!s.ddr || !s.scratch || !weights || !result)
        goto done;
    for (size_t i = 0; i < PACKED * passes; ++i)
        weights[i] = (uint8_t)((31u * i + 57u) ^ (i / 5u));
    for (unsigned i = 0; i < 2 * GROUPS * PANELS; ++i)
        coeffs[i] = (uint16_t)(i & 1 ? 4u + i % 13u : 64u + i % 193u);
    device = c11b_z15_open_target(&hal, weights, PACKED * passes, s.p17 ? VALENCE_P17 : VALENCE_STEP2, NULL);
    if (!device)
        goto done;
    const valence_submission_stats start = c11b_z15_submission_stats(device);
    if (variant < 2) {
        if (c11b_z15_project_grouped(device, weights, N, K, GROUP, 4, input, result))
            goto done;
    } else {
        if (c11b_z15_project_block16(device, weights, weights + PACKED, N, K, GROUP, input, coeffs, result))
            goto done;
        for (unsigned c = 0; c < N; ++c) {
            int expected = 0;
            for (unsigned g = 0; g < GROUPS; ++g)
                for (unsigned pass = 0; pass < 2; ++pass) {
                    const unsigned offset = ((c/16 * GROUPS + g) * 16 + c % 16) * (GROUP/2);
                    const uint8_t *w = weights + pass * PACKED + offset;
                    int dot = 0;
                    for (unsigned k = 0; k < GROUP; ++k)
                        dot += input[g * GROUP + k] * nibble(w[k/2], k);
                    const int scaled = dot * coeffs[2*(c/16*GROUPS+g)+pass];
                    int nearest = scaled / 256, fraction = abs(scaled % 256);
                    if (fraction > 128 || (fraction == 128 && (nearest & 1)))
                        nearest += scaled < 0 ? -1 : 1;
                    expected += nearest;
                }
            if (result[c] != expected)
                goto done;
        }
    }
    const valence_submission_stats end = c11b_z15_submission_stats(device);
    printf("QWEN_SHAPE_MOCK variant=%u dma_B=%u MMIO=%" PRIu64 " doorbells=%" PRIu64
           " host_writes=%" PRIu64 " host_reads=%" PRIu64 " returned_B=%zu\n",
           variant, s.total_dma_bytes, end.mmio_writes - start.mmio_writes, end.doorbells - start.doorbells,
           end.host_writes - start.host_writes, end.host_reads - start.host_reads,
           (size_t)N * (variant < 2 ? GROUPS : 1) * sizeof(*result));
    rc = 0;
done:
    c11b_z15_close(device);
    free(s.scratch);
    free(s.ddr);
    free(weights);
    free(result);
    return rc;
}

int main(void)
{
    uint8_t image[WEIGHTS / 2 + WEIGHTS];
    int8_t input[896];
    unsigned step2, p17;
    for (size_t i = 0; i < sizeof(image); ++i)
        image[i] = (uint8_t)((i * 71 + 123) ^ (i >> 4));
    for (unsigned i = 0; i < sizeof(input); ++i)
        input[i] = (int8_t)((i * 103 + 93) & 255);
    if (exercise(VALENCE_STEP2, image, input, &step2) ||
        exercise(VALENCE_P17, image, input, &p17) || p17 >= step2 || block16(image, input) ||
        full_projection(input, 0) || full_projection(input, 1) || full_projection(input, 2))
        return 1;
    puts("P17_PROJECTION_MOCK_PASS signed low/high nibble, persistent DDR, W4->W8, ordered doorbells");
    return 0;
}
