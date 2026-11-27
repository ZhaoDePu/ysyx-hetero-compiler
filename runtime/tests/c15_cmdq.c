#define _POSIX_C_SOURCE 200809L

#include "hal_reference_c6.h"
#include "z15_cmdq.h"
#include "z15_regs.h"

#include <stdio.h>
#include <stdlib.h>

#define WORD(op, reg, value) ((uint64_t)(op) << 60 | (uint64_t)(reg) << 32 | (value))
#define CHECK(test) do { \
    if (!(test)) { fprintf(stderr, "CMDQ test line %d: %s\n", __LINE__, #test); return 1; } \
} while (0)

static uint32_t read_reg(const ysyx_hal_t *hal, uint32_t offset)
{
    uint32_t value;
    return hal->mmio_read32(hal->ctx, offset, &value) ? UINT32_MAX : value;
}

static int write_reg(const ysyx_hal_t *hal, uint32_t offset, uint32_t value)
{
    return hal->mmio_write32(hal->ctx, offset, value);
}

static int set_descriptor(const ysyx_hal_t *hal, uint32_t base, uint32_t count)
{
    return write_reg(hal, CMDQ_CTRL, CMDQ_CLEAR) ||
           write_reg(hal, CMDQ_BASE_LO, base) || write_reg(hal, CMDQ_BASE_HI, 0) ||
           write_reg(hal, CMDQ_COUNT, count);
}

static int fill(const ysyx_hal_t *hal, uint64_t first, uint64_t second)
{
    uint8_t words[16];
    for (unsigned i = 0; i < 8; ++i) {
        words[i] = (uint8_t)(first >> (8u * i));
        words[8 + i] = (uint8_t)(second >> (8u * i));
    }
    return hal->ddr_write(hal->ctx, 0, words, sizeof(words));
}

static int error_case(const ysyx_hal_t *hal, unsigned code)
{
    const uint32_t base = code == 1 ? Z15_POOL_BASE - 32u :
                          code == 2 ? Z15_POOL_BASE + 4u : Z15_POOL_BASE;
    const uint32_t count = code == 3 ? 0u : code == 8 ? 1u : 2u;
    const uint64_t first = code == 4 ? WORD(6, Z15_SCRATCH, 9) :
                           code == 5 ? WORD(CMDQ_WRITE, Z15_PROGRAM, 9) :
                           code == 6 ? WORD(CMDQ_ONE, Z15_GPU_SLOTS, 1) :
                           code == 9 ? WORD(CMDQ_WRITE, Z15_CTRL, 12) : 0;
    CHECK(!set_descriptor(hal, base, count));
    CHECK(!fill(hal, first, WORD(CMDQ_END, 0, 0)));
    if (code == 7)
        ysyx_hal_reference_cmdq_fault(hal);
    int start = write_reg(hal, CMDQ_CTRL, code == 10 ? CMDQ_START | CMDQ_CLEAR : CMDQ_START);
    CHECK((start != 0) == (code <= 3 || code == 10));
    if (code == 6) {
        CHECK(read_reg(hal, CMDQ_STATUS) == CMDQ_BUSY);
        CHECK(write_reg(hal, CMDQ_CTRL, CMDQ_START));
    }
    CHECK(read_reg(hal, CMDQ_STATUS) & CMDQ_ERROR);
    CHECK(read_reg(hal, CMDQ_ERR_CODE) == code);
    CHECK(read_reg(hal, Z15_ERROR) == 128u);
    CHECK(read_reg(hal, Z15_SCRATCH) == 0);
    CHECK(read_reg(hal, CMDQ_COMPLETED) == (unsigned)(code == 8));
    CHECK(!write_reg(hal, CMDQ_CTRL, CMDQ_CLEAR));
    CHECK(!read_reg(hal, CMDQ_STATUS) && !read_reg(hal, CMDQ_ERR_CODE) && !read_reg(hal, Z15_ERROR));
    printf("C15 reference error=%u PASS\n", code);
    return 0;
}

static int turbo_read(void *ctx, uint32_t reg, uint32_t *value)
{
    unsigned *reads = ctx;
    ++*reads;
    if (reg != Z15_EXT_VERSION)
        return -1;
    *value = UINT32_C(0x0c0c0101);
    return 0;
}

int main(void)
{
    ysyx_hal_t hal;
    CHECK(!unsetenv("YSYX_EXEC_MODE"));
    CHECK(!unsetenv("YSYX_REFERENCE_CMDQ"));
    CHECK(!ysyx_hal_reference_c6_open(&hal, NULL));
    CHECK(z15_cmdq_select(&hal) == 0);
    CHECK(!setenv("YSYX_EXEC_MODE", "cmdq", 1));
    CHECK(z15_cmdq_select(&hal) == -1);
    hal.close(hal.ctx);
    unsigned reads = 0;
    const ysyx_hal_t turbo = {.ctx = &reads, .mmio_read32 = turbo_read};
    CHECK(!setenv("YSYX_EXEC_MODE", "auto", 1));
    CHECK(z15_cmdq_select(&turbo) == 0 && reads == 1);
    CHECK(!setenv("YSYX_EXEC_MODE", "mmio", 1));
    CHECK(z15_cmdq_select(&turbo) == 0 && reads == 1);
    CHECK(!setenv("YSYX_EXEC_MODE", "cmdq", 1));
    CHECK(z15_cmdq_select(&turbo) == -1 && reads == 2);
    CHECK(!setenv("YSYX_REFERENCE_CMDQ", "1", 1));
    CHECK(!ysyx_hal_reference_p17_open(&hal, NULL));
    CHECK(z15_cmdq_select(&hal) == 1);
    for (unsigned code = 1; code <= 10; ++code)
        CHECK(!error_case(&hal, code));

    z15_cmdq_plan *plan = z15_cmdq_create();
    CHECK(plan);
    CHECK(z15_cmdq_add(plan, CMDQ_WRITE, Z15_PROGRAM, 0));
    CHECK(z15_cmdq_add(plan, CMDQ_WRITE, Z15_CTRL, 1));
    CHECK(z15_cmdq_add(plan, CMDQ_ONE, Z15_PROGRAM, 1));
    for (unsigned i = 0; i < CMDQ_LIMIT; ++i)
        CHECK(!z15_cmdq_add(plan, CMDQ_NOP, 0, 0));
    CHECK(!z15_cmdq_add(plan, CMDQ_WRITE, Z15_SCRATCH, 0x55));
    CHECK(!z15_cmdq_add(plan, CMDQ_ZERO, Z15_ERROR, 0xff));
    CHECK(!z15_cmdq_add(plan, CMDQ_ONE, Z15_STATUS, Z15_LOCKED));
    CHECK(!z15_cmdq_place(plan, 17u));
    z15_cmdq_stats s;
    CHECK(!z15_cmdq_run(plan, &hal, &s));
    CHECK(s.starts == 2 && s.commands == CMDQ_LIMIT + 7 && s.upload_bytes == s.commands * 8u);
    CHECK(read_reg(&hal, Z15_SCRATCH) == 0x55);
    CHECK(!z15_cmdq_run(plan, &hal, &s));
    CHECK(s.starts == 2 && s.upload_bytes == 0);
    z15_cmdq_destroy(plan);
    hal.close(hal.ctx);
    puts("C15 reference legacy fallback, six opcodes, segmentation and resident replay PASS");
    return 0;
}
