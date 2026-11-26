#define _POSIX_C_SOURCE 200809L
#include "c11_qwen_ref.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double elapsed(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t))
        return 0;
    return t.tv_sec + t.tv_nsec / 1e9;
}

static unsigned top1(const int32_t *scores)
{
    unsigned best = 0;
    for (unsigned i = 1; i < QWEN_VOCAB; ++i)
        if (scores[best] < scores[i])
            best = i;
    return best;
}

static int run_model(qwen_reference *model, c11b_z15 *device, FILE *tokens, FILE *dump)
{
    const int greedy = dump == NULL;
    const int check = device != NULL;
    unsigned pos = 0, id;
    const int32_t *last = NULL;
    c11b_layer0 *native = malloc(sizeof(*native));
    c11b_layer0 *gold = malloc(sizeof(*gold));
    int rc = 1;
    if (!native || !gold)
        goto done;
    for (int got; (got = fscanf(tokens, "%u", &id)) != EOF; ++pos) {
        if (got != 1)
            goto done;
        if (greedy) {
            last = qwen_reference_step(model, id);
            if (!last)
                goto done;
            printf("%u\n", id);
            continue;
        }
        ysyx_stats_t before = {0}, after = {0};
        uint64_t vpu_before = 0;
        if (check) {
            if (c11b_z15_stats(device, &before))
                goto done;
            vpu_before = c11b_z15_vpu_ticks(device);
        }
        const double start = elapsed();
        if (qwen_reference_step_layer0(model, id, native)) {
            fprintf(stderr, "Layer 0 failed at position %u token %u.\n", pos, id);
            goto done;
        }
        const double host = elapsed() - start;
        if (check) {
            if (c11b_z15_stats(device, &after) || fread(gold, sizeof(*gold), 1, dump) != 1)
                goto done;
            if (memcmp(native, gold, sizeof(*gold))) {
                unsigned j = 0;
                const uint8_t *a = (const uint8_t *)native, *b = (const uint8_t *)gold;
                while (j < sizeof(*gold) && a[j] == b[j])
                    ++j;
                fprintf(stderr, "Mismatch position %u byte %u, got %u expected %u.\n",
                        pos, j, a[j], b[j]);
                goto done;
            }
            const uint64_t vpu = c11b_z15_vpu_ticks(device) - vpu_before;
            printf("C11B_STEP pos=%u token=%u hidden=%zu kv=%zu wall=%" PRIu64
                   " npu=%" PRIu64 " vpu=%" PRIu64 " gpu=%" PRIu64 " dma=%" PRIu64
                   " overlap=%" PRIu64 " dma_compute=%" PRIu64 " dma_bytes=%" PRIu64
                   " ddr_peak=%" PRIu64 " host_seconds=%.6f ERROR=%" PRIu32
                   " SLVERR=0 invalid_dma=%" PRIu32 "\n",
                   pos, id, sizeof(native->hidden), sizeof(native->keys) + sizeof(native->values),
                   after.wall_ticks - before.wall_ticks, after.npu_ticks - before.npu_ticks - vpu,
                   vpu, after.gpu_ticks - before.gpu_ticks,
                   after.observed.dma_busy_ticks - before.observed.dma_busy_ticks,
                   after.npu_gpu_overlap_ticks - before.npu_gpu_overlap_ticks,
                   after.observed.dma_compute_overlap_ticks - before.observed.dma_compute_overlap_ticks,
                   after.dma_bytes - before.dma_bytes, c11b_z15_peak_bytes(device), host,
                   after.abi_error, after.invalid_dma_descriptors);
            fflush(stdout);
        } else if (fwrite(native, sizeof(*native), 1, dump) != 1) {
            goto done;
        }
    }
    if (!pos || ferror(tokens))
        goto done;
    if (greedy) {
        for (unsigned i = 0; i < 2; ++i) {
            id = top1(last);
            printf("%u\n", id);
            last = qwen_reference_step(model, id);
            if (!last)
                goto done;
        }
        fprintf(stderr, "C11B_GREEDY2 prefill=%u decode=2\n", pos);
    } else if (check) {
        if (fgetc(dump) != EOF || ferror(dump))
            goto done;
        printf("C11B_LAYER0_PASS positions=%u bytes=%zu peak=%" PRIu64 "\n",
               pos, pos * sizeof(*native), c11b_z15_peak_bytes(device));
    }
    rc = 0;
done:
    free(native);
    free(gold);
    return rc;
}

int main(int argc, char **argv)
{
    if ((argc != 4 && argc != 5) || (strcmp(argv[1], "sequence") &&
        strcmp(argv[1], "cpu") && strcmp(argv[1], "rtl")) ||
        (argc == 4) != !strcmp(argv[1], "sequence")) {
        fputs("usage: c11b_layer0 sequence WEIGHTS PREFILL_IDS\n"
              "       c11b_layer0 cpu|rtl WEIGHTS TOKEN_IDS TENSOR_DUMP\n", stderr);
        return 2;
    }
    const int real = !strcmp(argv[1], "rtl");
    qwen_reference *model = qwen_reference_open(argv[2]);
    FILE *tokens = fopen(argv[3], "r");
    FILE *dump = argc == 5 ? fopen(argv[4], real ? "rb" : "wb") : NULL;
    ysyx_hal_t hal = {0};
    c11b_z15 *device = NULL;
    int rc = 1;
    if (!model || !tokens || (argc == 5 && !dump))
        goto done;
    if (real) {
#ifdef YSYX_WITH_VERILATOR
        const char *why = NULL;
        size_t bytes;
        const uint8_t *image = qwen_reference_image(model, &bytes);
        if (ysyx_hal_verilator_open(&hal, &why) || !(device = c11b_z15_open(&hal, image, bytes, &why))) {
            fprintf(stderr, "%s\n", why ? why : "could not open frozen HAL");
            goto done;
        }
#else
        goto done;
#endif
    }
    if (!qwen_reference_native(model, c11b_z15_ops(device)))
        rc = run_model(model, device, tokens, dump);
done:
    if (dump && fclose(dump))
        rc = 1;
    if (tokens)
        fclose(tokens);
    c11b_z15_close(device);
    if (hal.close)
        hal.close(hal.ctx);
    qwen_reference_close(model);
    return rc;
}
