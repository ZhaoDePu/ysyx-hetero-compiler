#define _POSIX_C_SOURCE 200809L
#include "c11_qwen_ref.h"

#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { VOCAB_SLICE = 2048, GROUPS = QWEN_HIDDEN / QWEN_GROUP };

static double now(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t))
        return 0;
    return t.tv_sec + t.tv_nsec / 1e9;
}

static int read_exact(const char *name, void *value, size_t bytes)
{
    FILE *f = fopen(name, "rb");
    if (!f)
        return -1;
    int ok = fread(value, 1, bytes, f) == bytes && fgetc(f) == EOF && !ferror(f);
    return fclose(f) || !ok;
}

int main(int argc, char **argv)
{
    if ((argc != 6 && argc != 7) || (strcmp(argv[1], "serial") && strcmp(argv[1], "pipeline"))) {
        fputs("usage: c11b_projection serial|pipeline WEIGHTS SLICE_NIBBLES A8 ORIGINAL_PARTIAL [CHANNELS]\n", stderr);
        return 2;
    }
    unsigned width = VOCAB_SLICE;
    if (argc == 7) {
        char *end;
        errno = 0;
        unsigned long parsed = strtoul(argv[6], &end, 10);
        if (errno || *end || !parsed || parsed > VOCAB_SLICE || (parsed & 15u))
            return 2;
        width = (unsigned)parsed;
    }
    const size_t packed_size = (size_t)width * QWEN_HIDDEN / 2;
    const size_t partial_size = (size_t)width * GROUPS * sizeof(int32_t);
    uint8_t *packed = malloc(packed_size);
    int32_t *gold = malloc(partial_size), *got = malloc(partial_size);
    int8_t act[QWEN_HIDDEN];
    ysyx_hal_t hal = {0};
    c11b_z15 *device = NULL;
    qwen_reference *m = qwen_reference_open(argv[2]);
    const char *err = NULL;
    int rc = 1;
    if (!packed || !got || !gold || !m ||
        read_exact(argv[3], packed, packed_size) || read_exact(argv[4], act, sizeof(act)) ||
        read_exact(argv[5], gold, partial_size))
        goto done;
    size_t bytes;
    const uint8_t *image = qwen_reference_image(m, &bytes);
    if (ysyx_hal_verilator_open(&hal, &err) ||
        !(device = c11b_z15_open(&hal, image, bytes, &err)) ||
        qwen_reference_native(m, c11b_z15_ops(device)))
        goto done;
    c11b_z15_pipeline(device, !strcmp(argv[1], "pipeline"));
    ysyx_stats_t start, end;
    if (c11b_z15_stats(device, &start))
        goto done;
    const double begin = now();
    if (qwen_reference_projection(m, packed, width, QWEN_HIDDEN, act, got) ||
        c11b_z15_stats(device, &end))
        goto done;
    double host = now() - begin;
    if (memcmp(gold, got, partial_size)) {
        for (size_t i = 0; i < partial_size / sizeof(*got); ++i)
            if (gold[i] != got[i]) {
                fprintf(stderr, "raw dot mismatch index %zu got %d want %d\n", i, got[i], gold[i]);
                break;
            }
        goto done;
    }
    printf("C11B_HEAD_PASS mode=%s bytes=%zu wall=%" PRIu64 " npu=%" PRIu64 " gpu=%" PRIu64
           " vpu=0 dma=%" PRIu64 " dma_compute=%" PRIu64 " dma_bytes=%" PRIu64
           " ddr_peak=%" PRIu64 " expanded_host_bytes=%" PRIu64 " host_seconds=%.6f ERROR=%" PRIu32
           " SLVERR=0 illegal_dma=%" PRIu32 "\n",
           argv[1], partial_size, end.wall_ticks - start.wall_ticks, end.npu_ticks - start.npu_ticks,
           end.gpu_ticks - start.gpu_ticks, end.observed.dma_busy_ticks - start.observed.dma_busy_ticks,
           end.observed.dma_compute_overlap_ticks - start.observed.dma_compute_overlap_ticks,
           end.dma_bytes - start.dma_bytes, c11b_z15_peak_bytes(device), c11b_z15_expanded_bytes(device),
           host, end.abi_error, end.invalid_dma_descriptors);
    rc = 0;
done:
    if (err)
        fprintf(stderr, "%s\n", err);
    c11b_z15_close(device);
    if (hal.close)
        hal.close(hal.ctx);
    qwen_reference_close(m);
    free(packed);
    free(gold);
    free(got);
    return rc;
}
