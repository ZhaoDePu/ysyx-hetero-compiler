#include "c12_qwen_ref.h"
#include "c11b_qwen_z15_cmdq.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    enum { MAX_TOKENS = 32 };
    if (argc != 4)
        return 2;
    char *end;
    const unsigned long layer = strtoul(argv[3], &end, 10);
    if (*end || layer >= QWEN_LAYERS)
        return 2;
    uint32_t input[MAX_TOKENS], token;
    unsigned count = 0;
    FILE *file = fopen(argv[2], "r");
    if (!file)
        return 2;
    int scan;
    while ((scan = fscanf(file, "%" SCNu32, &token)) == 1 && count < MAX_TOKENS)
        input[count++] = token;
    fclose(file);
    if (scan != EOF || count < 16)
        return 2;

    const char *error = NULL;
    qwen_reference *cpu = qwen_reference_open(argv[1]);
    qwen_reference *rtl = qwen_reference_open(argv[1]);
    ysyx_hal_t hal = {0};
    c11b_z15 *device = NULL;
    int failed = 1;
    c12_qwen_capture want = {0}, actual = {0};
    if (!cpu || !rtl || c12_qwen_capture_layer(cpu, (unsigned)layer, &want) ||
        c12_qwen_capture_layer(rtl, (unsigned)layer, &actual) ||
        ysyx_hal_verilator_open(&hal, &error))
        goto done;
    size_t bytes;
    const uint8_t *weights = qwen_reference_image(rtl, &bytes);
    device = c11b_z15_open_target(&hal, weights, bytes, VALENCE_P17, &error);
    if (!device || c12_z15_attach(rtl, device, UINT32_C(1) << layer))
        goto done;
    ysyx_stats_t before, after;
    if (c11b_z15_stats(device, &before))
        goto done;
    const z15_cmdq_stats qb = c11b_z15_queue_stats(device);
    const int32_t *golden = c16_qwen_prefill(cpu, input, count);
    const int32_t *got = c16_qwen_prefill(rtl, input, count);
    if (!golden || !got || c11b_z15_stats(device, &after))
        goto done;
    if (want.cache_bytes != actual.cache_bytes ||
        memcmp(want.cache, actual.cache, want.cache_bytes) ||
        memcmp(want.hidden, actual.hidden, sizeof(want.hidden)) ||
        memcmp(golden, got, QWEN_VOCAB * sizeof(*got))) {
        error = "prefill KV, hidden or logits mismatch";
        goto done;
    }
    const z15_cmdq_stats qa = c11b_z15_queue_stats(device);
    printf("C16_PREFILL_PASS layer=%lu tokens=%u hidden=%zu kv=%zu logits=%zu wall=%" PRIu64
           " npu=%" PRIu64 " dma_bytes=%" PRIu64 " starts=%" PRIu64 " commands=%" PRIu64
           " error=%" PRIu32 " slverr=0 invalid_dma=%" PRIu32 "\n",
           layer, count, sizeof(want.hidden), want.cache_bytes, QWEN_VOCAB * sizeof(*got),
           after.wall_ticks - before.wall_ticks, after.npu_ticks - before.npu_ticks,
           after.dma_bytes - before.dma_bytes, qa.starts - qb.starts, qa.commands - qb.commands,
           after.abi_error, after.invalid_dma_descriptors);
    failed = 0;
done:
    if (failed)
        fprintf(stderr, "C16_PREFILL_FAIL layer=%lu cause=%s\n", layer, error ? error : "device or input failure");
    c11b_z15_close(device);
    if (hal.close)
        hal.close(hal.ctx);
    qwen_reference_close(rtl);
    qwen_reference_close(cpu);
    return failed;
}
