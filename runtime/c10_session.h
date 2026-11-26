#ifndef YSYX_C10_SESSION_H
#define YSYX_C10_SESSION_H

#include "c10_session_core.h"
#include "../include/ysyx_yolo26n_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Same RGB, detection, and count/error contract as the original YOLO API.
 * A positive finite npu_mhz converts the available WALL cycles; the timing
 * struct contains raw counters regardless of whether MHz is supplied.
 */
int ysyx_c10_yolo_infer_rgb320(ysyx_yolo26n_ctx_t *ctx, const uint8_t *rgb,
                              float threshold, ysyx_det_t *dets, int max_dets,
                              ysyx_perf_t *perf, ysyx_c10_timing_t *timing,
                              double npu_mhz);

enum { YSYX_C10_YOLO_HEAD_BYTES = 176400 };

/* Packed stride-8, -16, -32 boxes/scores, after a successful C10 inference. */
int ysyx_c10_yolo_read_heads(const ysyx_yolo26n_ctx_t *ctx, void *data, size_t size);

#ifdef __cplusplus
}
#endif
#endif
