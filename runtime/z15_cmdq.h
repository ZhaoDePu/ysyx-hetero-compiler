#ifndef YSYX_Z15_CMDQ_H
#define YSYX_Z15_CMDQ_H

#include "ysyx_rt.h"

#define Z15_EXT_VERSION 0x120u
#define Z15_CAP_FLAGS 0x124u
#define Z15_P18_VERSION 0x0c0c0200u
#define Z15_CAP_CMDQ 16u

enum {
    CMDQ_BASE_LO = 0x140, CMDQ_BASE_HI = 0x144, CMDQ_COUNT = 0x148,
    CMDQ_CTRL = 0x14c, CMDQ_STATUS = 0x150, CMDQ_COMPLETED = 0x154,
    CMDQ_CYCLES = 0x158, CMDQ_ERR_CODE = 0x160,
    CMDQ_START = 1, CMDQ_CLEAR = 2, CMDQ_BUSY = 1, CMDQ_DONE = 2, CMDQ_ERROR = 4,
    CMDQ_LIMIT = 262144
};

typedef enum {
    CMDQ_NOP = 0, CMDQ_WRITE = 1, CMDQ_ZERO = 2, CMDQ_ONE = 3,
    CMDQ_IDLE = 4, CMDQ_END = 15
} z15_cmdq_op;

typedef struct z15_cmdq_plan z15_cmdq_plan;

typedef struct {
    uint64_t cycles, commands, starts, reads, writes, upload_bytes;
    uint32_t status, completed, code, legacy_error, ddr_end;
} z15_cmdq_stats;

int z15_cmdq_validate(uint64_t word);
int z15_cmdq_select(const ysyx_hal_t *hal);
int z15_cmdq_idle(const ysyx_hal_t *hal);
z15_cmdq_plan *z15_cmdq_create(void);
void z15_cmdq_reset(z15_cmdq_plan *plan);
void z15_cmdq_destroy(z15_cmdq_plan *plan);
int z15_cmdq_add(z15_cmdq_plan *plan, z15_cmdq_op op, uint32_t reg, uint32_t value);
int z15_cmdq_image(z15_cmdq_plan *plan, const uint8_t *image, size_t size);
int z15_cmdq_place(z15_cmdq_plan *plan, uint32_t high_water);
int z15_cmdq_run(z15_cmdq_plan *plan, const ysyx_hal_t *hal, z15_cmdq_stats *stats);
void z15_cmdq_report(FILE *out, const z15_cmdq_stats *stats);

#endif
