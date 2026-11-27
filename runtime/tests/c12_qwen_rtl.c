/* Replay real autoregressive input against a separate C99 reference. */
#include "c12_qwen_ref.h"
#include "c11b_qwen_z15_cmdq.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned greedy(const int32_t *scores)
{
    unsigned best = 0;
    for (unsigned i = 1; i < QWEN_VOCAB; ++i)
        if (scores[best] < scores[i])
            best = i;
    return best;
}

static int replay_step(qwen_reference *gold, qwen_reference *sim, c11b_z15 *dev,
                       unsigned pos, unsigned id, unsigned layer, int is_decode,
                       int check_head, unsigned *next)
{
    c12_qwen_capture reference = {0}, observed = {0};
    ysyx_stats_t before, after;
    if (c12_qwen_capture_layer(gold, layer, &reference) ||
        c12_qwen_capture_layer(sim, layer, &observed) ||
        c11b_z15_stats(dev, &before))
        return -1;
    const valence_submission_stats before_submit = c11b_z15_submission_stats(dev);
    const z15_cmdq_stats queue_before = c11b_z15_queue_stats(dev);
    const uint64_t before_vpu = c11b_z15_vpu_ticks(dev);
    const int32_t *want = qwen_reference_step(gold, id);
    const int32_t *got = qwen_reference_step(sim, id);
    if (!want || !got || c11b_z15_stats(dev, &after))
        return -1;
    if (memcmp(observed.hidden, reference.hidden, sizeof(reference.hidden)) ||
        observed.cache_bytes != reference.cache_bytes ||
        memcmp(observed.cache, reference.cache, reference.cache_bytes) ||
        memcmp(want, got, QWEN_VOCAB * sizeof(*want))) {
        fprintf(stderr, "Tensor mismatch layer=%u pos=%u token=%u\n", layer, pos, id);
        return -1;
    }
    *next = greedy(got);
    const valence_submission_stats after_submit = c11b_z15_submission_stats(dev);
    const z15_cmdq_stats queue_after = c11b_z15_queue_stats(dev);
    printf("C12_STEP layer=%u pos=%u token=%u next=%u phase=%s hidden=%zu cache=%zu "
           "logits=%zu head_npu=%d wall=%" PRIu64 " npu=%" PRIu64 " gpu=%" PRIu64
           " dma=%" PRIu64 " npu_gpu_overlap=%" PRIu64 " compute_dma_overlap=%" PRIu64
           " dma_bytes=%" PRIu64 " ddr_high=%" PRIu64 " error=%" PRIu32
           " slverr=0 invalid_dma=%" PRIu32 " vpu=%" PRIu64 " dma_descriptors=%" PRIu32
           " mmio_writes=%" PRIu64 " doorbells=%" PRIu64 " host_reads=%" PRIu64
           " host_writes=%" PRIu64 " arm_reads=%" PRIu64 " arm_writes=%" PRIu64
           " cmdq_batches=%" PRIu64 " cmdq_commands=%" PRIu64 " cmdq_cycles=%" PRIu64 "\n",
           layer, pos, id, *next, is_decode ? "decode" : "prefill", sizeof(reference.hidden),
           reference.cache_bytes, QWEN_VOCAB * sizeof(*want), check_head,
           after.wall_ticks - before.wall_ticks, after.npu_ticks - before.npu_ticks,
           after.gpu_ticks - before.gpu_ticks, after.observed.dma_busy_ticks - before.observed.dma_busy_ticks,
           after.npu_gpu_overlap_ticks - before.npu_gpu_overlap_ticks,
           after.observed.dma_compute_overlap_ticks - before.observed.dma_compute_overlap_ticks,
           after.dma_bytes - before.dma_bytes, c11b_z15_peak_bytes(dev),
           after.abi_error, after.invalid_dma_descriptors,
           c11b_z15_vpu_ticks(dev) - before_vpu, after.dma_operations - before.dma_operations,
           after_submit.mmio_writes - before_submit.mmio_writes,
           after_submit.doorbells - before_submit.doorbells,
           after_submit.host_reads - before_submit.host_reads,
           after_submit.host_writes - before_submit.host_writes,
           queue_after.reads - queue_before.reads, queue_after.writes - queue_before.writes,
           queue_after.starts - queue_before.starts, queue_after.commands - queue_before.commands,
           queue_after.cycles - queue_before.cycles);
    return fflush(stdout);
}

static int replay(qwen_reference *gold, qwen_reference *sim, c11b_z15 *dev,
                  FILE *tokens, unsigned layer, unsigned decode, int check_head)
{
    unsigned pos = 0, id, next = 0;
    for (int n; (n = fscanf(tokens, "%u", &id)) != EOF; ++pos) {
        if (n != 1 || replay_step(gold, sim, dev, pos, id, layer, 0, check_head, &next))
            return -1;
    }
    if (!pos || ferror(tokens))
        return -1;
    for (unsigned step = 0; step < decode; ++step) {
        if (replay_step(gold, sim, dev, pos + step, next, layer, 1, check_head, &next))
            return -1;
    }
    printf("C12_RTL_PASS layer=%u prefill=%u decode=%u head_npu=%d\n",
           layer, pos, decode, check_head);
    return 0;
}

int main(int argc, char **argv)
{
    valence_target target = VALENCE_STEP2;
    if (argc == 8 && !strcmp(argv[7], "p17"))
        target = VALENCE_P17;
    else if (argc != 7) {
        fputs("Usage: c12_qwen_rtl WEIGHTS TOKEN_IDS KV_BITS LAYER DECODE_STEPS HEAD_ON_NPU [p17]\n", stderr);
        return 2;
    }
    unsigned params[4];
    for (unsigned i = 0; i < 4; ++i) {
        char *end;
        unsigned long parsed = strtoul(argv[3 + i], &end, 10);
        if (!argv[3 + i][0] || *end || parsed > 2048)
            return 2;
        params[i] = (unsigned)parsed;
    }
    const unsigned bits = params[0], layer = params[1], decode = params[2];
    const int head = (int)params[3];
    if ((bits != 4 && bits != 8) || layer >= QWEN_LAYERS || head > 1)
        return 2;
    qwen_reference *cpu = qwen_reference_open(argv[1]);
    qwen_reference *rtl = qwen_reference_open(argv[1]);
    FILE *ids = fopen(argv[2], "r");
    ysyx_hal_t hal = {0};
    c11b_z15 *dev = NULL;
    const char *error = NULL;
    int rc = 1;
    if (!cpu || !rtl || !ids || c12_qwen_cache_bits(cpu, bits) ||
        c12_qwen_cache_bits(rtl, bits) || ysyx_hal_verilator_open(&hal, &error))
        goto done;
    size_t length;
    const uint8_t *image = qwen_reference_image(rtl, &length);
    dev = c11b_z15_open_target(&hal, image, length, target, &error);
    uint32_t mask = UINT32_C(1) << layer;
    if (head)
        mask |= UINT32_C(1) << QWEN_LAYERS;
    if (!dev || c12_z15_attach(rtl, dev, mask))
        goto done;
    rc = replay(cpu, rtl, dev, ids, layer, decode, head) != 0;
done:
    if (rc)
        fprintf(stderr, "C12 frozen replay failed: %s\n", error ? error : "check inputs and AXI log");
    c11b_z15_close(dev);
    if (hal.close)
        hal.close(hal.ctx);
    if (ids)
        fclose(ids);
    qwen_reference_close(cpu);
    qwen_reference_close(rtl);
    return rc;
}
