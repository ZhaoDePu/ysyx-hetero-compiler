#ifndef YSYX_RT_H
#define YSYX_RT_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { YSYX_INPUT = 1, YSYX_OUTPUT = 2, YSYX_REQUIRED = 4 };
enum { YSYX_SYNC_FOR_DEVICE = 0, YSYX_SYNC_FOR_HOST = 1 };
enum { YSYX_TARGET_Z15_BALANCED = 1 };

typedef struct {
    uint64_t axi_transactions;
    uint64_t dma_busy_ticks;
    uint64_t dma_compute_overlap_ticks;
    uint64_t triple_overlap_ticks;
    uint64_t gpu_clock_busy_ticks;
    uint64_t dma_gpu_overlap_ticks;
    uint64_t dma_tensor_overlap_ticks;
} ysyx_observation_t;

typedef struct {
    void *ctx;
    int (*mmio_read32)(void *, uint32_t, uint32_t *);
    int (*mmio_write32)(void *, uint32_t, uint32_t);
    int (*ddr_write)(void *, uint32_t, const void *, size_t);
    int (*ddr_read)(void *, uint32_t, void *, size_t);
    int (*cache_sync)(void *, uint32_t, size_t, int);
    void (*observe)(void *, ysyx_observation_t *); /* optional */
    void (*close)(void *); /* optional */
} ysyx_hal_t;

typedef struct {
    const char *name; /* counted bytes, UTF-8, no terminating NUL */
    uint32_t name_bytes, flags, dtype, rank, dims[4], ddr_offset, bytes;
} ysyx_symbol_t;

typedef struct {
    const char *name;
    const uint8_t *data;
    size_t size;
} ysyx_input_t;

typedef struct {
    uint64_t wall_ticks, npu_ticks, gpu_ticks, npu_gpu_overlap_ticks;
    uint32_t dma_operations;
    uint64_t dma_bytes;
    ysyx_observation_t observed;
    int has_observations;
} ysyx_stats_t;

typedef struct ysyx_program ysyx_program_t;

/* The caller owns and retains the immutable byte image through program close. */
int ysyx_program_open(const void *, size_t, ysyx_program_t **, const char **error);
void ysyx_program_close(ysyx_program_t *);
uint32_t ysyx_program_target(const ysyx_program_t *);
size_t ysyx_program_symbol_count(const ysyx_program_t *);
int ysyx_program_symbol(const ysyx_program_t *, size_t, ysyx_symbol_t *);
int ysyx_find_symbol(const ysyx_program_t *, const char *, uint32_t required_flags, ysyx_symbol_t *);

int ysyx_execute(const ysyx_program_t *, const ysyx_hal_t *, const ysyx_input_t *, size_t,
                 ysyx_stats_t *, FILE *trace, const char **error);
int ysyx_read_output(const ysyx_program_t *, const ysyx_hal_t *, const char *, void *, size_t);

int ysyx_hal_linux_open(ysyx_hal_t *, const char *control_uio, const char *dma_uio, const char **error);
int ysyx_hal_verilator_open(ysyx_hal_t *, const char **error);
int ysyx_run_cli(int argc, char **argv, int simulation_available);

#ifdef __cplusplus
}
#endif

#endif
