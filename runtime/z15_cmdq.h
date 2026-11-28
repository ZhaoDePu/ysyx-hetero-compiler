#ifndef YSYX_Z15_CMDQ_H
#define YSYX_Z15_CMDQ_H

#include "ysyx_rt.h"

#define Z15_EXT_VERSION 0x120u
#define Z15_CAP_FLAGS 0x124u
#define Z15_P18_VERSION 0x0c0c0200u
#define Z15_CAP_CMDQ 16u
#define Z15_CAP_P19A 96u
enum {
    Z15_TMA_DDR = 0x180, Z15_TMA_SRAM = 0x184, Z15_TMA_BYTES = 0x188, Z15_TMA_ROWS = 0x18c,
    Z15_TMA_DDR_STRIDE = 0x190, Z15_TMA_SRAM_STRIDE = 0x194, Z15_TMA_CTRL = 0x198,
    Z15_TMA_STATUS = 0x19c, Z15_TMA_ERR = 0x1ac, Z15_NPU_BIND = 0x1c0, Z15_VPU_BIND = 0x1c4,
    Z15_TMA_BUSY = 1, Z15_TMA_FAULT = 4, Z15_TMA_START = 1, Z15_TMA_STORE = 4, Z15_TMA_WAIT = 16, Z15_TMA_SIGNAL = 256
};

enum {
    CMDQ_BASE_LO = 0x140, CMDQ_BASE_HI = 0x144, CMDQ_COUNT = 0x148,
    CMDQ_CTRL = 0x14c, CMDQ_STATUS = 0x150, CMDQ_COMPLETED = 0x154,
    CMDQ_CYCLES = 0x158, CMDQ_ERR_CODE = 0x160,
    CMDQ_START = 1, CMDQ_CLEAR = 2, CMDQ_BUSY = 1, CMDQ_DONE = 2, CMDQ_ERROR = 4,
    CMDQ_LIMIT = 262144
};

typedef enum {
    CMDQ_NOP = 0, CMDQ_WRITE = 1, CMDQ_ZERO = 2, CMDQ_ONE = 3,
    CMDQ_IDLE = 4, CMDQ_SIGNAL = 5, CMDQ_CONSUME = 6, CMDQ_RESET = 7, CMDQ_END = 15
} z15_cmdq_op;

typedef struct z15_cmdq_plan z15_cmdq_plan;

typedef struct {
    uint64_t cycles, commands, starts, reads, writes, upload_bytes;
    uint32_t status, completed, code, legacy_error, ddr_end;
} z15_cmdq_stats;

int z15_cmdq_validate(uint64_t word);
int z15_cmdq_p19(const ysyx_hal_t *hal);
void z15_cmdq_allow_p19(z15_cmdq_plan *plan);
int z15_cmdq_select(const ysyx_hal_t *hal);
int z15_cmdq_idle(const ysyx_hal_t *hal);
z15_cmdq_plan *z15_cmdq_create(void);
void z15_cmdq_reset(z15_cmdq_plan *plan);
void z15_cmdq_destroy(z15_cmdq_plan *plan);
int z15_cmdq_add(z15_cmdq_plan *plan, z15_cmdq_op op, uint32_t reg, uint32_t value);
int z15_cmdq_image(z15_cmdq_plan *plan, const uint8_t *image, size_t size);
size_t z15_cmdq_batches(const z15_cmdq_plan *plan);
unsigned z15_cmdq_count(const z15_cmdq_plan *plan);
int z15_cmdq_place_bounded(z15_cmdq_plan *plan, uint32_t base, uint32_t limit);
int z15_cmdq_place_tail(z15_cmdq_plan *plan, uint32_t low, uint32_t *upper);
void z15_cmdq_discard_uploaded_bytes(z15_cmdq_plan *plan);
int z15_cmdq_place(z15_cmdq_plan *plan, uint32_t high_water);
int z15_cmdq_run(z15_cmdq_plan *plan, const ysyx_hal_t *hal, z15_cmdq_stats *stats);
void z15_cmdq_report(FILE *out, const z15_cmdq_stats *stats);

#endif
