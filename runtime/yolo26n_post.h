#ifndef YSYX_YOLO26N_POST_H
#define YSYX_YOLO26N_POST_H

#include "../include/ysyx_yolo26n_api.h"

enum { YSYX_YOLO_LEVELS = 3, YSYX_YOLO_CLASSES = 80 };

typedef struct {
    const int8_t *boxes[YSYX_YOLO_LEVELS];
    const uint8_t *scores[YSYX_YOLO_LEVELS];
    float box_scale[YSYX_YOLO_LEVELS];
    float probabilities[YSYX_YOLO_LEVELS][256];
} ysyx_yolo_heads_t;

/* Scratch-free for the caller. Selects the two ONNX TopK results, with source
 * index as the ascending tie-breaker of each TopK. */
int ysyx_yolo_decode(const ysyx_yolo_heads_t *heads, int size, float threshold,
                    ysyx_det_t *out, int max_dets);

#endif
