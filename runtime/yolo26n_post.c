#include "yolo26n_post.h"

#include <errno.h>
#include <stdlib.h>

typedef struct {
    float score;
    int index;
} item_t;

static int ranked(const void *aa, const void *bb)
{
    const item_t *a = aa, *b = bb;
    if (a->score != b->score)
        return a->score > b->score ? -1 : 1;
    return (a->index > b->index) - (a->index < b->index);
}

static int shape(int size, int *cells, int *side)
{
    if (size <= 0 || size > YSYX_YOLO26N_SIZE || size % 32)
        return -1;
    int sum = 0;
    for (int i = 0; i < YSYX_YOLO_LEVELS; ++i) {
        side[i] = size / (8 << i);
        cells[i] = side[i] * side[i];
        sum += cells[i];
    }
    return sum;
}

static int level_of(int *cell, const int *cells)
{
    int level = 0;
    while (*cell >= cells[level])
        *cell -= cells[level++];
    return level;
}

int ysyx_yolo_decode(const ysyx_yolo_heads_t *heads, int size, float threshold,
                    ysyx_det_t *out, int max_dets)
{
    int cells[YSYX_YOLO_LEVELS], side[YSYX_YOLO_LEVELS];
    const int total = shape(size, cells, side);
    if (!heads || total < 1 || !(threshold >= 0.0f && threshold <= 1.0f)
            || max_dets < 0 || (max_dets && !out)) {
        errno = EINVAL;
        return -1;
    }
    if (!max_dets)
        return 0;
    const int k = total < YSYX_YOLO26N_MAX_DETECTIONS ? total : YSYX_YOLO26N_MAX_DETECTIONS;
    item_t *anchors = malloc((size_t)total * sizeof(*anchors));
    item_t *pairs = malloc((size_t)k * YSYX_YOLO_CLASSES * sizeof(*pairs));
    if (!anchors || !pairs) {
        free(anchors);
        free(pairs);
        errno = ENOMEM;
        return -1;
    }
    int at = 0;
    for (int level = 0; level < YSYX_YOLO_LEVELS; ++level) {
        for (int cell = 0; cell < cells[level]; ++cell) {
            const uint8_t *scores = heads->scores[level] + cell * YSYX_YOLO_CLASSES;
            const float *table = heads->probabilities[level];
            float best = 0.0f;
            for (int cls = 0; cls < YSYX_YOLO_CLASSES; ++cls)
                if (table[scores[cls]] > best)
                    best = table[scores[cls]];
            anchors[at] = (item_t){best, at};
            ++at;
        }
    }
    qsort(anchors, (size_t)total, sizeof(*anchors), ranked);
    at = 0;
    for (int order = 0; order < k; ++order) {
        int cell = anchors[order].index;
        const int level = level_of(&cell, cells);
        const uint8_t *scores = heads->scores[level] + cell * YSYX_YOLO_CLASSES;
        for (int cls = 0; cls < YSYX_YOLO_CLASSES; ++cls) {
            pairs[at] = (item_t){heads->probabilities[level][scores[cls]], at};
            ++at;
        }
    }
    qsort(pairs, (size_t)at, sizeof(*pairs), ranked);
    int count = 0;
    while (count < k && count < max_dets && pairs[count].score >= threshold) {
        const item_t pair = pairs[count];
        int cell = anchors[pair.index / YSYX_YOLO_CLASSES].index;
        const int level = level_of(&cell, cells);
        const float stride = (float)(8 << level);
        const float x = (float)(cell % side[level]) + 0.5f;
        const float y = (float)(cell / side[level]) + 0.5f;
        const float scale = heads->box_scale[level];
        const int8_t *box = heads->boxes[level] + cell * 4;
        out[count++] = (ysyx_det_t){
            (x - box[0] * scale) * stride, (y - box[1] * scale) * stride,
            (x + box[2] * scale) * stride, (y + box[3] * scale) * stride,
            pair.score, pair.index % YSYX_YOLO_CLASSES
        };
    }
    free(pairs);
    free(anchors);
    return count;
}
