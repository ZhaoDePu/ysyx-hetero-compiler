#define _POSIX_C_SOURCE 200809L

#include "z15_cmdq.h"
#include "z15_regs.h"

#include <assert.h>
#include <stdio.h>
#include <time.h>

typedef struct {
    unsigned busy_reads, pending, complete, clear;
} bus_t;

static unsigned slept;
static long seconds;

int __wrap_nanosleep(const struct timespec *request, struct timespec *remaining)
{
    (void)remaining;
    assert(request->tv_sec == 0 && request->tv_nsec == 50000);
    ++slept;
    return 0;
}

int __wrap_clock_gettime(clockid_t clock, struct timespec *time)
{
    assert(clock == CLOCK_MONOTONIC);
    *time = (struct timespec){seconds, 0};
    return 0;
}

static int read_reg(void *ctx, uint32_t reg, uint32_t *value)
{
    bus_t *bus = ctx;
    *value = 0;
    if (reg == CMDQ_STATUS) {
        if (bus->pending && bus->busy_reads) {
            --bus->busy_reads;
            *value = CMDQ_BUSY;
            if (slept >= 3)
                seconds = 31;
        } else if (bus->pending) {
            bus->pending = 0;
            bus->complete = 1;
        }
        if (bus->complete)
            *value = CMDQ_DONE;
    } else if (reg == CMDQ_COMPLETED && bus->complete) {
        *value = 3;
    }
    return 0;
}

static int write_reg(void *ctx, uint32_t reg, uint32_t value)
{
    bus_t *bus = ctx;
    if (reg == CMDQ_CTRL) {
        if (value == CMDQ_START)
            bus->pending = 1;
        else if (value == CMDQ_CLEAR) {
            bus->pending = 0;
            ++bus->clear;
        }
    }
    return 0;
}

static int ddr_write(void *ctx, uint32_t at, const void *data, size_t bytes)
{
    (void)ctx;
    (void)at;
    (void)data;
    return bytes != 24;
}

static int sync_ddr(void *ctx, uint32_t at, size_t bytes, int direction)
{
    (void)ctx;
    (void)at;
    return bytes != 24 || direction != YSYX_SYNC_FOR_DEVICE;
}

static void run(unsigned busy_reads, unsigned sleeps, int timeout)
{
    bus_t bus = {.busy_reads = busy_reads};
    const ysyx_hal_t hal = {.ctx = &bus, .mmio_read32 = read_reg, .mmio_write32 = write_reg,
                            .ddr_write = ddr_write, .cache_sync = sync_ddr};
    z15_cmdq_plan *plan = z15_cmdq_create();
    z15_cmdq_stats stats;
    slept = 0;
    seconds = 0;
    assert(plan && !z15_cmdq_add(plan, CMDQ_NOP, 0, 0) && !z15_cmdq_place(plan, 0));
    assert(!!z15_cmdq_run(plan, &hal, &stats) == timeout);
    assert(stats.starts == 1 && slept == sleeps && bus.clear == (unsigned)timeout);
    z15_cmdq_destroy(plan);
}

int main(void)
{
    run(2, 0, 0);
    run(5, 3, 0);
    run(100, 3, 1);
    puts("CMDQ_WAIT_PASS short_spin=2 sleep_ns=50000 timeout_s=30 clear=1 replay=0");
    return 0;
}
