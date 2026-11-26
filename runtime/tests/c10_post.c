/* Compare allocation-free Top-K with the C9 stable ranking on dense ties. */
#define _POSIX_C_SOURCE 200809L
#include "yolo26n_post.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int ysyx_c9_yolo_decode(const ysyx_yolo_heads_t *, int, float, ysyx_det_t *, int);
void *__real_malloc(size_t);
static int allocations_blocked;

void *__wrap_malloc(size_t size)
{
    return allocations_blocked ? NULL : __real_malloc(size);
}

static double microseconds(void)
{
    struct timespec t;
    return clock_gettime(CLOCK_MONOTONIC, &t) ? 0.0 :
        1e6 * (double)t.tv_sec + 1e-3 * (double)t.tv_nsec;
}

int main(void)
{
    static int8_t boxes[2100 * 4];
    static uint8_t scores[2100 * 80];
    ysyx_yolo_heads_t heads = {0};
    size_t start = 0;
    for (unsigned level = 0; level < 3; ++level) {
        const unsigned side = 40u >> level;
        heads.boxes[level] = boxes + start * 4;
        heads.scores[level] = scores + start * 80;
        heads.box_scale[level] = 0.0625f;
        for (unsigned j = 0; j < 256; ++j)
            heads.probabilities[level][j] = (float)(j % 17u) / 16.0f;
        start += side * side;
    }
    uint32_t state = UINT32_C(912833);
    for (size_t i = 0; i < sizeof(scores); ++i) {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        scores[i] = (uint8_t)(state >> 24);
    }
    for (size_t i = 0; i < sizeof(boxes); ++i)
        boxes[i] = (int8_t)((i + i / 5) % 15 - 7);

    ysyx_det_t expected[300], output[300];
    const float thresholds[] = {0.0f, 0.4f, 1.0f};
    const int capacities[] = {0, 1, 17, 300};
    for (unsigned i = 0; i < sizeof(thresholds) / sizeof(*thresholds); ++i) {
        for (unsigned j = 0; j < sizeof(capacities) / sizeof(*capacities); ++j) {
            int n = ysyx_c9_yolo_decode(&heads, 320, thresholds[i], expected, capacities[j]);
            allocations_blocked = 1;
            int m = ysyx_yolo_decode(&heads, 320, thresholds[i], output, capacities[j]);
            allocations_blocked = 0;
            if (n < 0 || n != m || memcmp(expected, output, (size_t)n * sizeof(*expected))) {
                fprintf(stderr, "Top-K mismatch at threshold %.3f, capacity %d, expected %d, got %d\n",
                        thresholds[i], capacities[j], n, m);
                return 1;
            }
        }
    }
    enum { ITERATIONS = 40 };
    double begin = microseconds();
    for (unsigned i = 0; i < ITERATIONS; ++i)
        if (ysyx_c9_yolo_decode(&heads, 320, 0.4f, expected, 300) < 0)
            return 1;
    const double old_time = (microseconds() - begin) / ITERATIONS;
    begin = microseconds();
    for (unsigned i = 0; i < ITERATIONS; ++i)
        if (ysyx_yolo_decode(&heads, 320, 0.4f, output, 300) < 0)
            return 1;
    const double new_time = (microseconds() - begin) / ITERATIONS;
    printf("C10_TOPK_PASS tie_cases=12 mean_host_us_c9=%.3f mean_host_us_c10=%.3f\n", old_time, new_time);
    return 0;
}
