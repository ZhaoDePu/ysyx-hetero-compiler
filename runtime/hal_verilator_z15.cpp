#include "ysyx_rt.h"

#define main frozen_board_regression_main
#include "observed_rig.inc"
#undef main

namespace {
int read_word(void *opaque, uint32_t address, uint32_t *value)
{
    try {
        *value = static_cast<Rig *>(opaque)->read(address);
        return 0;
    } catch (const std::exception &error) {
        fprintf(stderr, "Verilator read: %s\n", error.what());
        return -1;
    }
}

int write_word(void *opaque, uint32_t address, uint32_t value)
{
    try {
        int response = static_cast<Rig *>(opaque)->write(address, value);
        if (response) {
            const uint32_t error = static_cast<Rig *>(opaque)->read(Z15_ERROR);
            fprintf(stderr, "Verilator write offset=0x%x response=%d fault=0x%x\n",
                    unsigned(address), response, unsigned(error));
        }
        return response ? -1 : 0;
    } catch (const std::exception &error) {
        fprintf(stderr, "Verilator write: %s\n", error.what());
        return -1;
    }
}

int write_ddr(void *opaque, uint32_t offset, const void *data, size_t size)
{
    if (offset > Z15_POOL_SIZE || size > Z15_POOL_SIZE - offset)
        return -1;
    try {
        Rig::put(opaque, offset, static_cast<const uint8_t *>(data), size);
        return 0;
    } catch (const std::exception &error) {
        fprintf(stderr, "Verilator DDR write: %s\n", error.what());
        return -1;
    }
}

int read_ddr(void *opaque, uint32_t offset, void *data, size_t size)
{
    if (offset > Z15_POOL_SIZE || size > Z15_POOL_SIZE - offset)
        return -1;
    try {
        Rig::get(opaque, offset, static_cast<uint8_t *>(data), size);
        return 0;
    } catch (const std::exception &error) {
        fprintf(stderr, "Verilator DDR read: %s\n", error.what());
        return -1;
    }
}

int sync_ddr(void *, uint32_t offset, size_t size, int)
{
    return offset > Z15_POOL_SIZE || size > Z15_POOL_SIZE - offset ? -1 : 0;
}

void observe(void *opaque, ysyx_observation_t *stats)
{
    const Rig &rig = *static_cast<Rig *>(opaque);
    *stats = {rig.transactions, rig.c2_dma, rig.c2_compute_dma, rig.c2_triple,
              rig.c2_gpu_clock, rig.c2_dma_gpu, rig.c2_dma_tensor};
}

void close_sim(void *opaque)
{
    Rig *rig = static_cast<Rig *>(opaque);
    rig->dut.final();
    delete rig;
}
}

extern "C" int ysyx_hal_verilator_open(ysyx_hal_t *out, const char **error)
{
    try {
        auto *rig = new Rig;
        *out = {rig, read_word, write_word, write_ddr, read_ddr, sync_ddr, observe, close_sim};
        return 0;
    } catch (const std::exception &cause) {
        fprintf(stderr, "Verilator setup: %s\n", cause.what());
        if (error)
            *error = "frozen Z15 simulator setup failed";
        return -1;
    }
}

extern "C" uint64_t ysyx_verilator_tma_tensor(const ysyx_hal_t *hal)
{
    return static_cast<const Rig *>(hal->ctx)->c18_tma_tensor;
}
