#include "yolo26n_post.h"

#include <errno.h>

typedef struct {
    float score;
    int index;
} item_t;

/* Keep the worst retained entry at the root, with the original ID breaking ties. */
static int better(item_t a, item_t b)
{
    return a.score > b.score || (a.score == b.score && a.index < b.index);
}

static void sift_down(item_t *items, int count, item_t value)
{
    int at = 0;
    for (int child = 1; child < count; child = at * 2 + 1) {
        if (child + 1 < count && better(items[child], items[child + 1]))
            ++child;
        if (!better(value, items[child]))
            break;
        items[at] = items[child];
        at = child;
    }
    items[at] = value;
}

static void retain(item_t *items, int *count, int capacity, item_t value)
{
    if (*count == capacity) {
        if (better(value, items[0]))
            sift_down(items, *count, value);
        return;
    }
    int at = (*count)++;
    while (at > 0) {
        const int parent = (at - 1) / 2;
        if (!better(items[parent], value))
            break;
        items[at] = items[parent];
        at = parent;
    }
    items[at] = value;
}

static void sort_retained(item_t *items, int count)
{
    while (count > 1) {
        const item_t worst = items[0];
        --count;
        sift_down(items, count, items[count]);
        items[count] = worst;
    }
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
    const int limit = k < max_dets ? k : max_dets;
    item_t anchors[YSYX_YOLO26N_MAX_DETECTIONS];
    item_t pairs[YSYX_YOLO26N_MAX_DETECTIONS];
    int anchors_used = 0, pairs_used = 0;
    int at = 0;
    for (int level = 0; level < YSYX_YOLO_LEVELS; ++level) {
        for (int cell = 0; cell < cells[level]; ++cell) {
            const uint8_t *scores = heads->scores[level] + cell * YSYX_YOLO_CLASSES;
            const float *table = heads->probabilities[level];
            float best = 0.0f;
            for (int cls = 0; cls < YSYX_YOLO_CLASSES; ++cls)
                if (table[scores[cls]] > best)
                    best = table[scores[cls]];
            retain(anchors, &anchors_used, k, (item_t){best, at});
            ++at;
        }
    }
    sort_retained(anchors, anchors_used);
    for (int order = 0; order < anchors_used; ++order) {
        int cell = anchors[order].index;
        const int level = level_of(&cell, cells);
        const uint8_t *scores = heads->scores[level] + cell * YSYX_YOLO_CLASSES;
        const float *table = heads->probabilities[level];
        for (int cls = 0; cls < YSYX_YOLO_CLASSES; ++cls) {
            const float score = table[scores[cls]];
            if (score >= threshold)
                retain(pairs, &pairs_used, limit,
                       (item_t){score, order * YSYX_YOLO_CLASSES + cls});
        }
    }
    sort_retained(pairs, pairs_used);
    int count = 0;
    while (count < pairs_used) {
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
    return count;
}
