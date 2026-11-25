#ifndef YSYX_YOLO26N_API_H
#define YSYX_YOLO26N_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    YSYX_YOLO26N_SIZE = 320,
    YSYX_YOLO26N_RGB_BYTES = 320 * 320 * 3,
    YSYX_YOLO26N_MAX_DETECTIONS = 300
};

typedef struct ysyx_yolo26n_ctx ysyx_yolo26n_ctx_t;

typedef struct {
    float x1, y1, x2, y2, score;
    int class_id;
} ysyx_det_t;

typedef struct {
    uint64_t npu_ticks, gpu_ticks, wall_ticks, overlap_ticks, dma_bytes;
    uint32_t dma_operations, abi_error, invalid_dma_descriptors;
    int hardware_counters_valid;
    double io_and_compute_ms;
    double estimated_ms_at_125_mhz;
    double estimated_ms_at_164_555_mhz;
} ysyx_perf_t;

/* Open one exclusive context. A nonzero use_mock_backend selects the frozen
 * Verilator HAL when linked, or the portable C99 reference otherwise.
 * Failure returns NULL and errno. */
ysyx_yolo26n_ctx_t *ysyx_yolo26n_open(const char *yhb_path, int use_mock_backend);

/* Input is packed 320x320 RGB uint8, with resize/letterbox done by the caller.
 * Output coordinates are in the 320x320 network frame, in descending score
 * order. Returns the number written, or -1 and errno on failure.
 * One context may be called by one thread at a time. */
int ysyx_yolo26n_infer_rgb320(ysyx_yolo26n_ctx_t *ctx, const uint8_t *rgb320_hwc,
                            float conf_thresh, ysyx_det_t *out_dets, int max_dets,
                            ysyx_perf_t *perf);

void ysyx_yolo26n_close(ysyx_yolo26n_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif
