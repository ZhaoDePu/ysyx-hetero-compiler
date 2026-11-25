#define _POSIX_C_SOURCE 200809L
#include "yolo26n_post.h"
#include "hal_reference_c6.h"
#include "ysyx_rt.h"
#include "z15_regs.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef char ieee_float32_required[(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && sizeof(float) == 4) ? 1 : -1];

struct ysyx_yolo26n_ctx {
    uint8_t *image;
    uint8_t *input;
    ysyx_program_t *program;
    ysyx_hal_t hal;
    int software_reference;
    ysyx_yolo_heads_t heads;
    uint8_t *boxes[YSYX_YOLO_LEVELS];
    uint8_t *scores[YSYX_YOLO_LEVELS];
};

static const char *const box_names[YSYX_YOLO_LEVELS] = {
    "boxes_stride8", "boxes_stride16", "boxes_stride32"
};
static const char *const score_names[YSYX_YOLO_LEVELS] = {
    "scores_stride8", "scores_stride16", "scores_stride32"
};

static uint8_t *load_program(const char *path, size_t *size)
{
    FILE *stream = fopen(path, "rb");
    if (!stream)
        return NULL;
    uint8_t *data = NULL;
    if (!fseek(stream, 0, SEEK_END)) {
        const long bytes = ftell(stream);
        if (bytes > 0 && (unsigned long)bytes <= Z15_POOL_SIZE && !fseek(stream, 0, SEEK_SET)) {
            data = malloc((size_t)bytes);
            if (data && fread(data, 1, (size_t)bytes, stream) == (size_t)bytes)
                *size = (size_t)bytes;
            else {
                free(data);
                data = NULL;
            }
        }
    }
    fclose(stream);
    return data;
}

static int symbol(const ysyx_program_t *p, const char *name, unsigned flags,
                  unsigned dtype, unsigned rank, const uint32_t *shape)
{
    ysyx_symbol_t out;
    return !ysyx_find_symbol(p, name, flags, &out) && out.dtype == dtype && out.rank == rank
        && !memcmp(out.dims, shape, rank * sizeof(*shape));
}

static int open_buffers(ysyx_yolo26n_ctx_t *ctx)
{
    if (!symbol(ctx->program, "images", YSYX_INPUT | YSYX_REQUIRED, 1, 4,
                (const uint32_t[]){1, 320, 320, 4})
            || !symbol(ctx->program, "c6_head_scales", YSYX_OUTPUT, 3, 1,
                       (const uint32_t[]){6})
            || !symbol(ctx->program, "c6_score_lut", YSYX_OUTPUT, 3, 2,
                       (const uint32_t[]){3, 256}))
        return EINVAL;
    ctx->input = malloc(320u * 320u * 4u);
    if (!ctx->input)
        return ENOMEM;
    for (unsigned level = 0; level < YSYX_YOLO_LEVELS; ++level) {
        const uint32_t side = 40u >> level;
        if (!symbol(ctx->program, box_names[level], YSYX_OUTPUT, 1, 4,
                    (const uint32_t[]){1, side, side, 4})
                || !symbol(ctx->program, score_names[level], YSYX_OUTPUT, 1, 4,
                           (const uint32_t[]){1, side, side, 80}))
            return EINVAL;
        ctx->boxes[level] = malloc(side * side * 4u);
        ctx->scores[level] = malloc(side * side * 80u);
        if (!ctx->boxes[level] || !ctx->scores[level])
            return ENOMEM;
        ctx->heads.boxes[level] = (const int8_t *)ctx->boxes[level];
        ctx->heads.scores[level] = ctx->scores[level];
    }
    return 0;
}

ysyx_yolo26n_ctx_t *ysyx_yolo26n_open(const char *yhb_path, int use_mock_backend)
{
    if (!yhb_path) {
        errno = EINVAL;
        return NULL;
    }
    ysyx_yolo26n_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return NULL;
    int failure = EBADMSG;
    size_t size = 0;
    const char *error = NULL;
    ctx->image = load_program(yhb_path, &size);
    if (!ctx->image || ysyx_program_open(ctx->image, size, &ctx->program, &error)
            || ysyx_program_target(ctx->program) != YSYX_TARGET_Z15_BALANCED)
        goto fail;
    failure = open_buffers(ctx);
    if (failure)
        goto fail;
    if (use_mock_backend) {
#ifdef YSYX_WITH_VERILATOR
        if (!ysyx_hal_verilator_open(&ctx->hal, &error))
            return ctx;
        failure = EIO;
#else
        if (!ysyx_hal_reference_c6_open(&ctx->hal, &error)) {
            ctx->software_reference = 1;
            return ctx;
        }
        failure = ENOMEM;
#endif
    } else {
        if (!ysyx_hal_linux_open(&ctx->hal, getenv("YSYX_CONTROL_UIO"), getenv("YSYX_DMA_UIO"), &error))
            return ctx;
        failure = ENODEV;
    }
fail:
    ysyx_yolo26n_close(ctx);
    errno = failure;
    return NULL;
}

static float little_float(const uint8_t *b)
{
    const uint32_t word = b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
    float result;
    memcpy(&result, &word, sizeof(result));
    return result;
}

static int read_heads(ysyx_yolo26n_ctx_t *ctx)
{
    uint8_t scales[6 * 4];
    uint8_t lut[YSYX_YOLO_LEVELS * 256 * 4];
    if (ysyx_read_output(ctx->program, &ctx->hal, "c6_head_scales", scales, sizeof(scales))
            || ysyx_read_output(ctx->program, &ctx->hal, "c6_score_lut", lut, sizeof(lut)))
        return -1;
    for (unsigned level = 0; level < YSYX_YOLO_LEVELS; ++level) {
        const size_t side = 40u >> level;
        const float scale = little_float(scales + 4 * level);
        if (!isfinite(scale) || scale <= 0.0f
                || ysyx_read_output(ctx->program, &ctx->hal, box_names[level], ctx->boxes[level], side * side * 4)
                || ysyx_read_output(ctx->program, &ctx->hal, score_names[level], ctx->scores[level], side * side * 80))
            return -1;
        ctx->heads.box_scale[level] = scale;
        for (unsigned i = 0; i < 256; ++i) {
            const float score = little_float(lut + 4 * (256 * level + i));
            if (!(score >= 0.0f && score <= 1.0f))
                return -1;
            ctx->heads.probabilities[level][i] = score;
        }
    }
    return 0;
}

static double elapsed_ms(const struct timespec *start)
{
    struct timespec end;
    if (!start->tv_sec || clock_gettime(CLOCK_MONOTONIC, &end))
        return 0.0;
    return (double)(end.tv_sec - start->tv_sec) * 1e3
           + (double)(end.tv_nsec - start->tv_nsec) * 1e-6;
}

int ysyx_yolo26n_infer_rgb320(ysyx_yolo26n_ctx_t *ctx, const uint8_t *rgb320_hwc,
                            float conf_thresh, ysyx_det_t *out_dets, int max_dets,
                            ysyx_perf_t *perf)
{
    if (!ctx || !rgb320_hwc || !(conf_thresh >= 0.0f && conf_thresh <= 1.0f)
            || max_dets < 0 || (max_dets && !out_dets)) {
        errno = EINVAL;
        return -1;
    }
    struct timespec begin = {0, 0};
    if (perf) {
        memset(perf, 0, sizeof(*perf));
        (void)clock_gettime(CLOCK_MONOTONIC, &begin);
    }
    for (size_t i = 0; i < (size_t)320 * 320; ++i) {
        for (unsigned c = 0; c < 3; ++c)
            ctx->input[4 * i + c] = (uint8_t)((127u * rgb320_hwc[3 * i + c] + 127u) / 255u);
        ctx->input[4 * i + 3] = 0;
    }
    const ysyx_input_t input = {"images", ctx->input, 320u * 320u * 4u};
    ysyx_stats_t stats = {0};
    const char *error = NULL;
    const int status = ysyx_execute(ctx->program, &ctx->hal, &input, 1, &stats, NULL, &error);
    if (perf) {
        perf->hardware_counters_valid = !ctx->software_reference;
        perf->npu_ticks = stats.npu_ticks;
        perf->gpu_ticks = stats.gpu_ticks;
        perf->wall_ticks = stats.wall_ticks;
        perf->overlap_ticks = stats.npu_gpu_overlap_ticks;
        perf->dma_bytes = stats.dma_bytes;
        perf->dma_operations = stats.dma_operations;
        perf->abi_error = stats.abi_error;
        perf->invalid_dma_descriptors = stats.invalid_dma_descriptors;
        perf->estimated_ms_at_125_mhz = (double)stats.wall_ticks / 125000.0;
        perf->estimated_ms_at_164_555_mhz = (double)stats.wall_ticks / 164555.0;
    }
    if (status || read_heads(ctx)) {
        errno = EIO;
        return -1;
    }
    const int count = ysyx_yolo_decode(&ctx->heads, 320, conf_thresh, out_dets, max_dets);
    if (perf)
        perf->io_and_compute_ms = elapsed_ms(&begin);
    return count;
}

void ysyx_yolo26n_close(ysyx_yolo26n_ctx_t *ctx)
{
    if (!ctx)
        return;
    if (ctx->hal.close)
        ctx->hal.close(ctx->hal.ctx);
    ysyx_program_close(ctx->program);
    for (unsigned level = 0; level < YSYX_YOLO_LEVELS; ++level) {
        free(ctx->boxes[level]);
        free(ctx->scores[level]);
    }
    free(ctx->input);
    free(ctx->image);
    free(ctx);
}
