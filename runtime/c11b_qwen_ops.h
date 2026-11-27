#ifndef YSYX_C11B_QWEN_OPS_H
#define YSYX_C11B_QWEN_OPS_H

#include "ysyx_rt.h"

enum { C11B_PITCH = 128, C11B_KV_WIDTH = 128, C11B_TRIG_BYTES = 256 };

typedef struct {
    void *ctx;
    int (*project)(void *, const uint8_t *, unsigned, unsigned, const int8_t *, int32_t *);
    int (*rms)(void *, const int8_t *, const int8_t *, unsigned, unsigned, unsigned, int8_t *);
    int (*rope)(void *, int32_t *, int32_t *, const uint8_t *);
    int (*attention)(void *, unsigned, unsigned, const int8_t *, const int8_t *,
                     const int8_t *, const unsigned *, int8_t *, int8_t *, int8_t *);
    int (*swiglu)(void *, const int8_t *, const int8_t *, const int8_t *, unsigned, int8_t *);
    int (*faulted)(void *);
} c11b_ops;

typedef struct c11b_z15 c11b_z15;

c11b_z15 *c11b_z15_open(const ysyx_hal_t *, const uint8_t *image, size_t,
                       const char **error);
typedef enum { VALENCE_STEP2, VALENCE_P17 } valence_target;

typedef struct {
    uint64_t batches, descriptor_words, mmio_writes, doorbells, host_reads, host_writes;
} valence_submission_stats;

c11b_z15 *c11b_z15_open_target(const ysyx_hal_t *, const uint8_t *, size_t,
                              valence_target, const char **);
int c11b_z15_project_grouped(void *, const uint8_t *, unsigned n, unsigned k,
                            unsigned group, unsigned bits, const int8_t *, int32_t *);
valence_submission_stats c11b_z15_submission_stats(const c11b_z15 *);

const c11b_ops *c11b_z15_ops(c11b_z15 *);
void c11b_z15_close(c11b_z15 *);
void c11b_z15_pipeline(c11b_z15 *, int enable);

/* Native combined NPU/VPU totals are sampled from the frozen ABI. */
int c11b_z15_stats(c11b_z15 *, ysyx_stats_t *);
uint64_t c11b_z15_vpu_ticks(const c11b_z15 *);
uint64_t c11b_z15_peak_bytes(const c11b_z15 *);
uint64_t c11b_z15_expanded_bytes(const c11b_z15 *);

void c11b_cpu_rms(const int8_t *, const int8_t *, unsigned, unsigned, unsigned, int8_t *);
void c11b_cpu_attention(const int8_t *, const int8_t *, const int8_t *, const unsigned *,
                        unsigned, int8_t *);
void c11b_cpu_swiglu(const int8_t *, const int8_t *, const int8_t *, unsigned, int8_t *);

#endif
