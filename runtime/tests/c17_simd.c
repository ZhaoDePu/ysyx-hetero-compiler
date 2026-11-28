/* Exercise the exact production fixed-point operations in both ARM builds. */
#include "../c11_qwen_ref.c"
#include "../c10_pixels.h"
#include <inttypes.h>
#include <stdio.h>
#include <time.h>

static uint32_t random_word(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

static uint64_t digest(uint64_t hash, const void *data, size_t bytes)
{
    const uint8_t *p = data;
    for (size_t i = 0; i < bytes; ++i)
        hash = (hash ^ p[i]) * UINT64_C(1099511628211);
    return hash;
}

int main(void)
{
    int32_t a[QWEN_HIDDEN], b[QWEN_HIDDEN], work[QWEN_HIDDEN];
    uint8_t rope[QWEN_HEAD_DIM * 4], gamma[QWEN_HIDDEN * 4];
    int64_t mix[QWEN_HEAD_DIM];
    uint8_t record[C12_KV_GROUP + 4];
    int8_t quant[QWEN_HIDDEN];
    uint8_t input[3 * 1027], packed[4 * 1027], table[256];
    int32_t gate[QWEN_INTER], up[QWEN_INTER];
    uint8_t exp[EXP_POINTS * 4] = {0};
    static uint8_t letterbox[YSYX_YOLO26N_RGB_BYTES];
    uint32_t state = 0xC170053;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned i = 0; i < sizeof(input); ++i)
        input[i] = (uint8_t)(random_word(&state) >> 22);
    for (unsigned i = 0; i < sizeof(table); ++i)
        table[i] = (uint8_t)(random_word(&state) >> 22);
    for (unsigned i = 0; i < QWEN_INTER; ++i)
        up[i] = (int32_t)(random_word(&state) >> 8) - (1 << 23);
    for (unsigned i = 0; i < EXP_POINTS; ++i)
        exp[i * 4 + 1] = (uint8_t)(255 - (i >> 4));
    const c10_rgb_frame frame = {.pixels = input, .width = 31, .height = 29};
    uint8_t small_frame[31 * 29 * 3];
    memcpy(small_frame, input, sizeof(small_frame));
    const c10_rgb_frame source = {.pixels = small_frame, .width = frame.width, .height = frame.height};
    c10_letterbox_geometry geometry;
    c10_letterbox_rgb320(&source, letterbox, &geometry);
    hash = digest(hash, letterbox, sizeof(letterbox));
    memset(gamma, 1, sizeof(gamma));
    memset(rope, 1, sizeof(rope));

    const clock_t start = clock();
    for (unsigned round = 0; round < 1500; ++round) {
        for (unsigned i = 0; i < QWEN_HIDDEN; ++i) {
            a[i] = (int32_t)(random_word(&state) >> 8) - (1 << 23);
            b[i] = (int32_t)(random_word(&state) >> 8) - (1 << 23);
        }
        a[0] = round & 1 ? 63 : -63;
        rms(a, gamma, work);
        rotate(work, QWEN_HEADS, rope);
        const int64_t score = c17_dot(work, b, QWEN_HEAD_DIM - (round & 3));
        memset(mix, 0, sizeof(mix));
        c17_accumulate(mix, work, (int32_t)(round % 123), QWEN_HEAD_DIM - (round & 3));
        const int32_t scale = quantize_block(quant, work, QWEN_HIDDEN - (round & 7));
        c12_cache_pack(record, work, 8);
        c12_cache_unpack(record, a, 8);
        memcpy(gate, up, sizeof(gate));
        swiglu(gate, up, exp);
        c17_rgb_quantize(packed, input, table, 1027);

        hash = digest(hash, work, sizeof(work));
        hash = digest(hash, &score, sizeof(score));
        hash = digest(hash, mix, sizeof(mix));
        hash = digest(hash, &scale, sizeof(scale));
        hash = digest(hash, quant, QWEN_HIDDEN - (round & 7));
        hash = digest(hash, record, sizeof(record));
        hash = digest(hash, a, C12_KV_GROUP * sizeof(*a));
        hash = digest(hash, gate, sizeof(gate));
        hash = digest(hash, packed, sizeof(packed));
    }
    printf("%016" PRIx64 "\n", hash);
    fprintf(stderr, "iterations=1500 cpu_ms=%.3f\n",
            (clock() - start) * 1000.0 / CLOCKS_PER_SEC);
    return 0;
}
