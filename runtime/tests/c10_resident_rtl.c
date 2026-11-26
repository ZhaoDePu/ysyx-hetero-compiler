/* Execute two different photographs through a single frozen Z15 session. */
#include "c10_session.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

enum { PIXELS = 320 * 320, HEAD_BYTES = 176400 };

static const char *const heads[] = {
    "boxes_stride8", "scores_stride8", "boxes_stride16", "scores_stride16",
    "boxes_stride32", "scores_stride32",
};

static uint8_t *file_bytes(const char *path, size_t *len)
{
    FILE *file = fopen(path, "rb");
    if (!file)
        return NULL;
    uint8_t *data = NULL;
    long bytes;
    if (!fseek(file, 0, SEEK_END) && (bytes = ftell(file)) > 0 &&
            (uintmax_t)bytes <= SIZE_MAX && !fseek(file, 0, SEEK_SET)) {
        data = malloc((size_t)bytes);
        if (data && fread(data, 1, (size_t)bytes, file) == (size_t)bytes)
            *len = (size_t)bytes;
        else {
            free(data);
            data = NULL;
        }
    }
    fclose(file);
    return data;
}

static uint8_t *fixture(const char *folder, const char *scene, const char *suffix, size_t *len)
{
    char path[1024];
    int n = snprintf(path, sizeof(path), "%s/%s.%s", folder, scene, suffix);
    return n > 0 && (size_t)n < sizeof(path) ? file_bytes(path, len) : NULL;
}

static size_t check_heads(ysyx_c10_session_t *session, const char *folder, const char *scene,
                          unsigned long *fingerprint)
{
    size_t total = 0;
    *fingerprint = 0;
    for (unsigned i = 0; i < sizeof(heads) / sizeof(heads[0]); ++i) {
        char name[32];
        (void)snprintf(name, sizeof(name), "pl_out_%u.bin", i);
        size_t len = 0;
        uint8_t *expected = fixture(folder, scene, name, &len);
        uint8_t *got = len ? malloc(len) : NULL;
        int failed = !expected || !got || ysyx_c10_session_read(session, heads[i], got, len) ||
                     memcmp(expected, got, len);
        if (!failed)
            for (size_t j = 0; j < len; ++j)
                *fingerprint = *fingerprint * 33u + got[j];
        free(got);
        free(expected);
        if (failed) {
            fprintf(stderr, "%s: head verification failed for %s\n", scene, heads[i]);
            return 0;
        }
        total += len;
    }
    return total;
}

static int run_frame(ysyx_c10_session_t *session, const char *folder, const char *scene,
                     unsigned frame, unsigned long *fingerprint)
{
    size_t len = 0;
    uint8_t *rgb = fixture(folder, scene, "input_nhwc_s8.bin", &len);
    uint8_t *padded = calloc(PIXELS, 4);
    if (!rgb || !padded || len != PIXELS * 3u) {
        fprintf(stderr, "%s: expected signed packed RGB fixture\n", scene);
        free(rgb);
        free(padded);
        return -1;
    }
    for (size_t i = 0; i < PIXELS; ++i)
        memcpy(padded + 4 * i, rgb + 3 * i, 3);
    free(rgb);
    ysyx_input_t input = {"images", padded, PIXELS * 4u};
    ysyx_stats_t stats = {0};
    ysyx_c10_timing_t timing = {0};
    const char *error = NULL;
    int status = ysyx_c10_session_run(session, &input, 1, &stats, &timing, stderr, &error);
    free(padded);
    if (status) {
        fprintf(stderr, "%s: %s\n", scene, error ? error : "session frame failed");
        return -1;
    }
    size_t bytes = check_heads(session, folder, scene, fingerprint);
    if (bytes != HEAD_BYTES || stats.abi_error || stats.invalid_dma_descriptors ||
        !stats.has_observations || !stats.wall_ticks ||
        timing.dynamic_h2d_bytes != PIXELS * 4u || !!timing.static_h2d_bytes != (frame == 0)) {
        fprintf(stderr, "%s: frame, counters or residency verification failed "
                "(bytes=%zu, ERR=0x%" PRIx32 ", native=%" PRIu64 ")\n",
                scene, bytes, stats.abi_error, stats.npu_ticks);
        return -1;
    }
    printf("C10_RESIDENT_FRAME frame=%u scene=%s compared_bytes=%zu static_h2d_bytes=%" PRIu64
           " dynamic_h2d_bytes=%" PRIu64 " h2d_ms=%.6f execute_host_ms=%.6f"
           " wall_ticks=%" PRIu64 " native_ticks=%" PRIu64 " gpu_ticks=%" PRIu64
           " dma_ticks=%" PRIu64 " native_gpu_overlap_ticks=%" PRIu64 " dma_bytes=%" PRIu64
           " dma_descriptors=%" PRIu32 " ERROR=%" PRIu32 " invalid_dma=%" PRIu32 "\n",
           frame, scene, bytes, timing.static_h2d_bytes, timing.dynamic_h2d_bytes,
           timing.t_h2d_ms, timing.t_execute_ms, stats.wall_ticks, stats.npu_ticks, stats.gpu_ticks,
           stats.observed.dma_busy_ticks, stats.npu_gpu_overlap_ticks,
           stats.dma_bytes, stats.dma_operations, stats.abi_error, stats.invalid_dma_descriptors);
    fflush(stdout);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fputs("usage: c10_resident_rtl MODEL.yhb VECTORS\n", stderr);
        return 2;
    }
    const char *error = NULL;
    ysyx_program_t *model = NULL;
    ysyx_c10_session_t *session = NULL;
    ysyx_hal_t hal = {0};
    size_t len = 0;
    uint8_t *image = file_bytes(argv[1], &len);
    int rc = 1;
    unsigned long first = 0, next = 0;
    if (!image || ysyx_program_open(image, len, &model, &error) ||
        ysyx_hal_verilator_open(&hal, &error) ||
        ysyx_c10_session_open(model, &hal, &session, &error)) {
        fprintf(stderr, "C10 session setup: %s\n", error ? error : "could not open fixture");
        goto done;
    }
    if (!run_frame(session, argv[2], "bus", 0, &first) &&
        !run_frame(session, argv[2], "zidane", 1, &next) && first != next) {
        puts("C10_RESIDENT_RTL_PASS frames=2 head_bytes=352800 ERROR=0 invalid_dma=0");
        rc = 0;
    }
done:
    ysyx_c10_session_close(session);
    if (hal.close)
        hal.close(hal.ctx);
    ysyx_program_close(model);
    free(image);
    return rc;
}
