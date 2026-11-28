#define _POSIX_C_SOURCE 200809L
#include "yolo26n_post.h"
#include "c17_simd.h"
#include "yolo26n_target.h"
#include "hal_reference_c6.h"
#include "c10_session.h"
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
    ysyx_c10_session_t *resident;
    int metadata_ready, c10_heads_ready;
    uint8_t quant[256];
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

static int fpga_operating(void)
{
    char state[32];
    FILE *stream = fopen("/sys/class/fpga_manager/fpga0/state", "r");
    if (!stream)
        return 0;
    const int operating = fgets(state, sizeof(state), stream) &&
                          !strcmp(state, "operating\n");
    fclose(stream);
    return operating;
}

static ysyx_yolo26n_ctx_t *open_target(const char *yhb_path, int use_mock_backend, int phase17)
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
    if (phase17) {
        if (!ctx->image || size < 4u || memcmp(ctx->image, "YHP1", 4))
            goto fail;
        /* The marker selects the Phase 17 entry point. The inner YHB CRC
           covers the standard magic restored in this private buffer. */
        memcpy(ctx->image, "YHB1", 4);
    }
    if (!ctx->image || ysyx_program_open(ctx->image, size, &ctx->program, &error)
            || ysyx_program_target(ctx->program) != YSYX_TARGET_Z15_BALANCED)
        goto fail;
    failure = open_buffers(ctx);
    if (failure)
        goto fail;
    for (unsigned v = 0; v < sizeof(ctx->quant); ++v)
        ctx->quant[v] = (uint8_t)((127u * v + 127u) / 255u);
    if (use_mock_backend) {
#ifdef YSYX_WITH_VERILATOR
        if (!ysyx_hal_verilator_open(&ctx->hal, &error))
            return ctx;
        failure = EIO;
#else
        const int rc = phase17 ? ysyx_hal_reference_p17_open(&ctx->hal, &error) :
                                 ysyx_hal_reference_c6_open(&ctx->hal, &error);
        if (!rc) {
            ctx->software_reference = 1;
            return ctx;
        }
        failure = ENOMEM;
#endif
    } else {
        if ((!phase17 || fpga_operating()) &&
            !ysyx_hal_linux_open(&ctx->hal, getenv("YSYX_CONTROL_UIO"), getenv("YSYX_DMA_UIO"), &error))
            return ctx;
        failure = ENODEV;
    }
fail:
    ysyx_yolo26n_close(ctx);
    errno = failure;
    return NULL;
}

ysyx_yolo26n_ctx_t *ysyx_yolo26n_open(const char *path, int use_mock_backend)
{
    return open_target(path, use_mock_backend, 0);
}

ysyx_yolo26n_ctx_t *ysyx_yolo26n_open_p17(const char *path, int use_reference)
{
    return open_target(path, use_reference, 1);
}

static float little_float(const uint8_t *b)
{
    const uint32_t word = b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
    float result;
    memcpy(&result, &word, sizeof(result));
    return result;
}

static int read_tensor(ysyx_yolo26n_ctx_t *ctx, const char *name, void *data, size_t size)
{
    if (ctx->resident)
        return ysyx_c10_session_read(ctx->resident, name, data, size);
    return ysyx_read_output(ctx->program, &ctx->hal, name, data, size);
}

static int read_metadata(ysyx_yolo26n_ctx_t *ctx)
{
    uint8_t scales[6 * 4];
    uint8_t lut[YSYX_YOLO_LEVELS * 256 * 4];
    if (read_tensor(ctx, "c6_head_scales", scales, sizeof(scales)) ||
        read_tensor(ctx, "c6_score_lut", lut, sizeof(lut)))
        return -1;
    for (unsigned level = 0; level < YSYX_YOLO_LEVELS; ++level) {
        const float scale = little_float(scales + 4 * level);
        if (!isfinite(scale) || scale <= 0.0f)
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

static int read_heads(ysyx_yolo26n_ctx_t *ctx)
{
    if (!ctx->metadata_ready) {
        if (read_metadata(ctx))
            return -1;
        ctx->metadata_ready = ysyx_c10_session_static_output(ctx->resident, "c6_head_scales") &&
                              ysyx_c10_session_static_output(ctx->resident, "c6_score_lut");
    }
    for (unsigned level = 0; level < YSYX_YOLO_LEVELS; ++level) {
        const size_t side = 40u >> level;
        if (read_tensor(ctx, box_names[level], ctx->boxes[level], side * side * 4) ||
            read_tensor(ctx, score_names[level], ctx->scores[level], side * side * 80))
            return -1;
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

static int valid_arguments(const ysyx_yolo26n_ctx_t *ctx, const uint8_t *rgb,
                           float threshold, const ysyx_det_t *out, int max_dets)
{
    return ctx && rgb && threshold >= 0.0f && threshold <= 1.0f &&
           max_dets >= 0 && (!max_dets || out);
}

static void quantize_rgb(ysyx_yolo26n_ctx_t *ctx, const uint8_t *rgb)
{
    c17_rgb_quantize(ctx->input, rgb, ctx->quant, (size_t)320 * 320);
}

static void record_perf(ysyx_perf_t *perf, const ysyx_stats_t *stats, int hardware)
{
    if (!perf)
        return;
    perf->hardware_counters_valid = hardware;
    perf->npu_ticks = stats->npu_ticks;
    perf->gpu_ticks = stats->gpu_ticks;
    perf->wall_ticks = stats->wall_ticks;
    perf->overlap_ticks = stats->npu_gpu_overlap_ticks;
    perf->dma_bytes = stats->dma_bytes;
    perf->dma_operations = stats->dma_operations;
    perf->abi_error = stats->abi_error;
    perf->invalid_dma_descriptors = stats->invalid_dma_descriptors;
    perf->estimated_ms_at_125_mhz = (double)stats->wall_ticks / 125000.0;
    perf->estimated_ms_at_164_555_mhz = (double)stats->wall_ticks / 164555.0;
}

int ysyx_yolo26n_infer_rgb320(ysyx_yolo26n_ctx_t *ctx, const uint8_t *rgb320_hwc,
                            float conf_thresh, ysyx_det_t *out_dets, int max_dets,
                            ysyx_perf_t *perf)
{
    if (!valid_arguments(ctx, rgb320_hwc, conf_thresh, out_dets, max_dets)) {
        errno = EINVAL;
        return -1;
    }
    if (ctx->resident) {
        ysyx_c10_session_close(ctx->resident);
        ctx->resident = NULL;
        ctx->metadata_ready = 0;
    }
    ctx->c10_heads_ready = 0;
    struct timespec begin = {0, 0};
    if (perf) {
        memset(perf, 0, sizeof(*perf));
        (void)clock_gettime(CLOCK_MONOTONIC, &begin);
    }
    quantize_rgb(ctx, rgb320_hwc);
    const ysyx_input_t input = {"images", ctx->input, 320u * 320u * 4u};
    ysyx_stats_t stats = {0};
    const int status = ysyx_execute(ctx->program, &ctx->hal, &input, 1, &stats, NULL, NULL);
    record_perf(perf, &stats, !ctx->software_reference);
    if (status || read_heads(ctx)) {
        errno = EIO;
        return -1;
    }
    const int count = ysyx_yolo_decode(&ctx->heads, 320, conf_thresh, out_dets, max_dets);
    if (perf)
        perf->io_and_compute_ms = elapsed_ms(&begin);
    return count;
}

int ysyx_c10_yolo_infer_rgb320(ysyx_yolo26n_ctx_t *ctx, const uint8_t *rgb,
                              float threshold, ysyx_det_t *dets, int max_dets,
                              ysyx_perf_t *perf, ysyx_c10_timing_t *timing, double npu_mhz)
{
    if (timing)
        *timing = (ysyx_c10_timing_t){0};
    if (perf)
        *perf = (ysyx_perf_t){0};
    if (ctx)
        ctx->c10_heads_ready = 0;
    if (!valid_arguments(ctx, rgb, threshold, dets, max_dets)) {
        errno = EINVAL;
        return -1;
    }
    struct timespec begin = {0, 0}, quant_begin = {0, 0}, post_begin = {0, 0};
    (void)clock_gettime(CLOCK_MONOTONIC, &begin);
    if (!ctx->resident && ysyx_c10_session_open(ctx->program, &ctx->hal, &ctx->resident, NULL)) {
        errno = EBADMSG;
        return -1;
    }
    (void)clock_gettime(CLOCK_MONOTONIC, &quant_begin);
    quantize_rgb(ctx, rgb);
    const double quant_ms = elapsed_ms(&quant_begin);
    const ysyx_input_t input = {"images", ctx->input, 320u * 320u * 4u};
    ysyx_c10_timing_t stage;
    const int run = ysyx_c10_session_run(ctx->resident, &input, 1, NULL, &stage, NULL, NULL);
    stage.hardware_counters_valid &= !ctx->software_reference;
    stage.t_capture_preproc_ms = quant_ms;
    if (stage.hardware_counters_valid && isfinite(npu_mhz) && npu_mhz > 0.0)
        stage.t_pl_hw_ms = (double)stage.stats.wall_ticks / (npu_mhz * 1000.0);
    record_perf(perf, &stage.stats, stage.hardware_counters_valid);
    int count = -1;
    if (!run) {
        (void)clock_gettime(CLOCK_MONOTONIC, &post_begin);
        if (!read_heads(ctx)) {
            count = ysyx_yolo_decode(&ctx->heads, 320, threshold, dets, max_dets);
            ctx->c10_heads_ready = count >= 0;
        } else {
            errno = EIO;
        }
        stage.t_d2h_post_ms = elapsed_ms(&post_begin);
    } else {
        errno = EIO;
    }
    stage.t_e2e_ms = elapsed_ms(&begin);
    if (perf)
        perf->io_and_compute_ms = stage.t_e2e_ms;
    if (timing)
        *timing = stage;
    return count;
}

int ysyx_c10_yolo_read_heads(const ysyx_yolo26n_ctx_t *ctx, void *data, size_t size)
{
    if (!ctx || !ctx->c10_heads_ready || !data || size != YSYX_C10_YOLO_HEAD_BYTES)
        return -1;
    uint8_t *write = data;
    for (unsigned level = 0; level < YSYX_YOLO_LEVELS; ++level) {
        const size_t side = 40u >> level, cells = side * side;
        memcpy(write, ctx->boxes[level], cells * 4);
        write += cells * 4;
        memcpy(write, ctx->scores[level], cells * YSYX_YOLO_CLASSES);
        write += cells * YSYX_YOLO_CLASSES;
    }
    return 0;
}

void ysyx_yolo26n_close(ysyx_yolo26n_ctx_t *ctx)
{
    if (!ctx)
        return;
    ysyx_c10_session_close(ctx->resident);
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
