/* Small frozen-RTL gates. Each stage is checked before full-layer replay. */
#include "c13_qwen.h"
#include "c13_qwen_layout.h"
#include "c12_qwen_cache.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

static int32_t rnd(int64_t n, int64_t den)
{
    int64_t value = n < 0 ? -((-n + den / 2) / den) : (n + den / 2) / den;
    if (value > C13_LIMIT)
        return C13_LIMIT;
    if (value < -C13_LIMIT)
        return -C13_LIMIT;
    return (int32_t)value;
}

static uint32_t root(uint64_t value)
{
    unsigned low = 0, high = C13_LIMIT + 1;
    while (low + 1 < high) {
        unsigned middle = (low + high) / 2;
        if ((uint64_t)middle * middle <= value)
            low = middle;
        else
            high = middle;
    }
    return low;
}

static int compare(const char *label, const int32_t *expected, const int32_t *actual, unsigned count)
{
    for (unsigned i = 0; i < count; ++i) {
        if (expected[i] != actual[i]) {
            fprintf(stderr, "%s mismatch at %u expected %" PRId32 " observed %" PRId32 "\n",
                    label, i, expected[i], actual[i]);
            return 1;
        }
    }
    printf("C13_OPERATOR_PASS %s outputs=%u\n", label, count);
    return 0;
}

static int32_t fixture(unsigned i, unsigned salt, unsigned width)
{
    return (int32_t)(((i + 5u) * (i + salt + 1u) * 1237u) % (2 * width + 1)) - (int32_t)width;
}

static int check_rms(const c13_qwen_device *op)
{
    int32_t input[C13_HIDDEN], gamma[C13_HIDDEN], expected[C13_HIDDEN], actual[C13_HIDDEN];
    uint64_t sum = 0;
    for (unsigned i = 0; i < C13_HIDDEN; ++i) {
        input[i] = fixture(i, 7, 1000000);
        gamma[i] = fixture(i, 23, 7000000);
        sum += (int64_t)input[i] * input[i];
    }
    uint32_t den = root(sum / C13_HIDDEN + 4295);
    for (unsigned i = 0; i < C13_HIDDEN; ++i)
        expected[i] = rnd((int64_t)input[i] * gamma[i], den);
    return op->rms(op->ctx, input, (const uint8_t *)gamma, actual) ||
           compare("rms-q16", expected, actual, C13_HIDDEN);
}

static unsigned exp_lookup(const uint32_t *table, unsigned value)
{
    if (value >= (C13_EXP_COUNT - 1u) * 256u)
        return 0;
    unsigned i = value >> 8, f = value & 255u;
    return (unsigned)(((uint64_t)table[i] * (256u - f) +
                       (uint64_t)table[i + 1] * f + 128u) >> 8);
}

static int check_swiglu(const c13_qwen_device *op, const uint32_t *exp)
{
    int32_t gate[C13_INTER], up[C13_INTER], expected[C13_INTER];
    for (unsigned i = 0; i < C13_INTER; ++i) {
        int32_t x = fixture(i, 19, 1400000);
        gate[i] = x;
        up[i] = fixture(i, 29, 2100000);
        unsigned e = exp_lookup(exp, (unsigned)(x < 0 ? -x : x));
        uint32_t p = x < 0 ? e : C13_Q24;
        expected[i] = rnd((int64_t)rnd((int64_t)p * x, C13_Q24 + e) * up[i], C13_Q16);
    }
    return op->swiglu(op->ctx, gate, up, (const uint8_t *)exp) ||
           compare("swiglu-q16", expected, gate, C13_INTER);
}

static int check_attention(const c13_qwen_device *op, const uint32_t *exp)
{
    enum { POSITIONS = 3 };
    int32_t query[C13_HIDDEN], out[C13_HIDDEN], expected[C13_HIDDEN];
    int32_t keys[POSITIONS][2 * C13_HEAD], values[POSITIONS][2 * C13_HEAD];
    uint8_t cache[C13_KV_BYTES], gold[C13_KV_BYTES];
    for (unsigned pos = 0; pos < POSITIONS; ++pos) {
        for (unsigned i = 0; i < C13_HIDDEN; ++i)
            query[i] = fixture(i, 4 + pos, 100000);
        for (unsigned i = 0; i < 2 * C13_HEAD; ++i) {
            keys[pos][i] = fixture(i, 11 + pos, 180000);
            values[pos][i] = fixture(i, 21 + pos, 120000);
        }
        if (op->attention(op->ctx, 0, pos, query, keys[pos], values[pos],
                          (const uint8_t *)exp, out, cache))
            return 1;
        for (unsigned head = 0; head < 2; ++head) {
            c12_cache_pack(gold + head * C13_KV_HEAD_BYTES, keys[pos] + head * C13_HEAD, 8);
            c12_cache_pack(gold + (head + 2u) * C13_KV_HEAD_BYTES, values[pos] + head * C13_HEAD, 8);
            c12_cache_unpack(gold + head * C13_KV_HEAD_BYTES, keys[pos] + head * C13_HEAD, 8);
            c12_cache_unpack(gold + (head + 2u) * C13_KV_HEAD_BYTES, values[pos] + head * C13_HEAD, 8);
        }
        if (memcmp(cache, gold, sizeof(cache))) {
            fprintf(stderr, "C13 DDR KV encoding differs at position %u\n", pos);
            return 1;
        }
        for (unsigned head = 0; head < C13_QUERIES; ++head) {
            unsigned group = head / 7;
            int32_t score[POSITIONS], maximum = -C13_LIMIT;
            uint64_t total = 0;
            int64_t numerator[C13_HEAD] = {0};
            for (unsigned p = 0; p <= pos; ++p) {
                int64_t dot = 0;
                for (unsigned d = 0; d < C13_HEAD; ++d)
                    dot += (int64_t)query[head * C13_HEAD + d] *
                           keys[p][group * C13_HEAD + d];
                score[p] = rnd(dot, 8 * C13_Q16);
                if (score[p] > maximum)
                    maximum = score[p];
            }
            for (unsigned p = 0; p <= pos; ++p) {
                unsigned weight = exp_lookup(exp, (unsigned)(maximum - score[p]));
                for (unsigned d = 0; d < C13_HEAD; ++d)
                    numerator[d] += (int64_t)weight * values[p][group * C13_HEAD + d];
                total += weight;
            }
            for (unsigned d = 0; d < C13_HEAD; ++d)
                expected[head * C13_HEAD + d] = rnd(numerator[d], (int64_t)total);
        }
        if (compare("gqa-kv8-q16", expected, out, C13_HIDDEN))
            return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 2 || strlen(argv[1]) != 1 || argv[1][0] < '1' || argv[1][0] > '7')
        return 2;
    unsigned mask = (unsigned)(argv[1][0] - '0');
    const uint8_t dummy[64] = {0};
    uint32_t exp[C13_EXP_COUNT];
    exp[0] = C13_Q24;
    for (unsigned i = 1; i < C13_EXP_COUNT; ++i)
        exp[i] = (uint32_t)(((uint64_t)exp[i - 1] * 65281u + 32768u) >> 16);
    ysyx_hal_t hal = {0};
    const char *why = NULL;
    if (ysyx_hal_verilator_open(&hal, &why))
        return 1;
    c11b_z15 *device = c11b_z15_open(&hal, dummy, sizeof(dummy), &why);
    c13_qwen_device ops;
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("C13_OPERATOR_START mask=%u\n", mask);
    int failed = !device || c13_z15_kernels(device, mask, &ops) ||
                 ((mask & C13_RMS) && check_rms(&ops)) ||
                 ((mask & C13_SWIGLU) && check_swiglu(&ops, exp)) ||
                 ((mask & C13_ATTENTION) && check_attention(&ops, exp));
    ysyx_stats_t stats;
    if (!failed && !c11b_z15_stats(device, &stats)) {
        printf("C13_OPERATOR_TOTAL wall=%" PRIu64 " gpu=%" PRIu64 " dma_bytes=%" PRIu64
               " ERROR=%u SLVERR=0 illegal=%u\n", stats.wall_ticks, stats.gpu_ticks,
               stats.dma_bytes, stats.abi_error, stats.invalid_dma_descriptors);
    } else {
        failed = 1;
    }
    c11b_z15_close(device);
    hal.close(hal.ctx);
    return failed;
}
