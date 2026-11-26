#define _POSIX_C_SOURCE 200809L
#include "c11_qwen_ref.h"

#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t))
        return 0;
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fputs("usage: c11b_full WEIGHTS TOKEN LOGITS_OUTPUT\n", stderr);
        return 2;
    }
    char *end;
    errno = 0;
    unsigned long token = strtoul(argv[2], &end, 10);
    if (errno || *end || token >= QWEN_VOCAB)
        return 2;

    qwen_reference *cpu = qwen_reference_open(argv[1]);
    qwen_reference *rtl = qwen_reference_open(argv[1]);
    c11b_z15 *dev = NULL;
    ysyx_hal_t hal = {0};
    const char *error = NULL;
    FILE *out = NULL;
    int rc = 1;
    if (!cpu || !rtl || qwen_reference_native(cpu, NULL))
        goto done;
    const int32_t *gold = qwen_reference_step(cpu, (unsigned)token);
    size_t bytes;
    const uint8_t *image = qwen_reference_image(rtl, &bytes);
    if (!gold || ysyx_hal_verilator_open(&hal, &error) ||
        hal.ddr_write(hal.ctx, 0, image, bytes) ||
        hal.cache_sync(hal.ctx, 0, bytes, YSYX_SYNC_FOR_DEVICE) ||
        !(dev = c11b_z15_open(&hal, image, bytes, &error)) ||
        qwen_reference_native(rtl, c11b_z15_ops(dev)))
        goto done;
    ysyx_stats_t before, after;
    if (c11b_z15_stats(dev, &before))
        goto done;
    const double begin = now();
    const int32_t *actual = qwen_reference_step(rtl, (unsigned)token);
    const double host = now() - begin;
    if (!actual || c11b_z15_stats(dev, &after))
        goto done;
    unsigned top = 0;
    for (unsigned i = 0; i < QWEN_VOCAB; ++i) {
        if (actual[i] != gold[i]) {
            fprintf(stderr, "Full model logit %u mismatch: %d vs %d\n", i, actual[i], gold[i]);
            goto done;
        }
        if (actual[i] > actual[top])
            top = i;
    }
    out = fopen(argv[3], "wb");
    if (!out || fwrite(actual, sizeof(*actual), QWEN_VOCAB, out) != QWEN_VOCAB || fflush(out))
        goto done;
    const uint64_t vpu = c11b_z15_vpu_ticks(dev);
    printf("C11B_FULL_PASS layers=24 token=%lu top=%u logits_bytes=%zu wall=%" PRIu64
           " npu=%" PRIu64 " vpu=%" PRIu64 " gpu=%" PRIu64 " dma=%" PRIu64
           " dma_compute=%" PRIu64 " dma_bytes=%" PRIu64 " ddr_peak=%" PRIu64
           " host_seconds=%.4f ERROR=%" PRIu32 " SLVERR=0 invalid_dma=%" PRIu32 "\n",
           token, top, QWEN_VOCAB * sizeof(*actual), after.wall_ticks - before.wall_ticks,
           after.npu_ticks - before.npu_ticks - vpu, vpu, after.gpu_ticks - before.gpu_ticks,
           after.observed.dma_busy_ticks - before.observed.dma_busy_ticks,
           after.observed.dma_compute_overlap_ticks - before.observed.dma_compute_overlap_ticks,
           after.dma_bytes - before.dma_bytes, c11b_z15_peak_bytes(dev), host, after.abi_error,
           after.invalid_dma_descriptors);
    rc = 0;
done:
    if (error)
        fprintf(stderr, "%s\n", error);
    if (out && fclose(out))
        rc = 1;
    c11b_z15_close(dev);
    if (hal.close)
        hal.close(hal.ctx);
    qwen_reference_close(cpu);
    qwen_reference_close(rtl);
    return rc;
}
