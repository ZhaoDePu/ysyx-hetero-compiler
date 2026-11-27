#ifndef VALENCE_C13_QWEN_H
#define VALENCE_C13_QWEN_H

#include "c12_qwen_ref.h"

enum {
    C13_RMS = 1,
    C13_SWIGLU = 2,
    C13_ATTENTION = 4,
    C13_ACCUMULATE = C13_RMS | C13_SWIGLU | C13_ATTENTION
};

typedef struct {
    void *ctx;
    int (*rms)(void *, const int32_t *, const uint8_t *, int32_t *);
    int (*swiglu)(void *, int32_t *, const int32_t *, const uint8_t *);
    int (*attention)(void *, unsigned, unsigned, const int32_t *, const int32_t *,
                     const int32_t *, const uint8_t *, int32_t *, uint8_t *);
} c13_qwen_device;

int c13_qwen_set_device(qwen_reference *, const c13_qwen_device *, uint32_t layers);
unsigned c13_qwen_kv_bits(const qwen_reference *);
/* Verification-only prefill seeding. The cache must be primed separately. */
int c13_qwen_replay_from(qwen_reference *, unsigned first_device_position);

typedef struct {
    uint64_t host_reads, host_writes, read_bytes, write_bytes;
    uint64_t cache_reserved, gpu_launches;
} c13_io;

#ifdef YSYX_C11B_NATIVE
void c13_z15_io(const c11b_z15 *, c13_io *);
#ifdef YSYX_C13_PL
int c13_z15_kernels(c11b_z15 *, unsigned ops, c13_qwen_device *);
int c13_z15_seed_kv(c11b_z15 *, unsigned layer, unsigned pos, const uint8_t *record, size_t bytes);
int c13_z15_attach(qwen_reference *, c11b_z15 *, uint32_t layers, unsigned ops);
#endif
#endif
#endif
