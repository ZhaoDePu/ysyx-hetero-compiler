#define main board_regression_main
#include "c2_observed_loop.hpp"
#undef main
#include <fstream>
#include <string>
#include <vector>

static void checked_write(Rig &rig, unsigned reg, uint32_t value) {
    if (rig.write(reg, value))
        throw std::runtime_error("MMIO write rejected at register " + std::to_string(reg));
}

static void wait_for(Rig &rig, unsigned mask, unsigned busy_mask = 0) {
    for (unsigned attempts = 0; attempts < 200000; ++attempts) {
        uint32_t status = rig.read(Z15_STATUS);
        if (status & Z15_HAS_ERROR)
            throw std::runtime_error("fabric reported error " + std::to_string(rig.read(Z15_ERROR)));
        if ((status & mask) && !(status & busy_mask))
            return;
    }
    throw std::runtime_error("fabric operation timeout");
}

static void dma_start(Rig &rig, unsigned ddr, unsigned address, unsigned size, bool readback) {
    checked_write(rig, Z15_DMA_BASE, Z15_POOL_BASE);
    checked_write(rig, Z15_DMA_OFFSET, ddr);
    checked_write(rig, Z15_DMA_SCRATCH, address);
    checked_write(rig, Z15_DMA_BYTES, size);
    checked_write(rig, Z15_DMA_FLAGS, 2 | unsigned(readback));
    checked_write(rig, Z15_CTRL, Z15_START_DMA);
}

static void dma(Rig &rig, unsigned address, unsigned size, bool readback) {
    dma_start(rig, address, address, size, readback);
    wait_for(rig, Z15_DMA_DONE, Z15_DMA_BUSY);
}

static std::vector<uint8_t> load_image(const std::string &file, unsigned size) {
    std::vector<uint8_t> bytes(size);
    std::ifstream stream(file, std::ios::binary);
    if (!stream.read(reinterpret_cast<char *>(bytes.data()), size) || stream.peek() != EOF)
        throw std::runtime_error("incorrect scratch image size: " + file);
    return bytes;
}

static void extent(unsigned address, unsigned size) {
    const uint64_t end = uint64_t(address) + size;
    if (!size || size % 32 || address % 32 || end > (1u << 21) || (address >> 20) != ((end - 1) >> 20))
        throw std::runtime_error("invalid scratch DMA extent");
}

static void upload(Rig &rig, const std::string &name, unsigned address, unsigned size) {
    extent(address, size);
    auto bytes = load_image(name, size);
    Rig::put(&rig, address, bytes.data(), bytes.size());
    dma(rig, address, size, false);
}

static void submit_gpu(Rig &rig, unsigned entry, uint32_t job) {
    const unsigned warp = (job >> 24) & 3u;
    if (entry % 4 || entry >= 0x4000 || (job >> 26))
        throw std::runtime_error("invalid balanced-profile GPU job");
    for (unsigned attempts = 0; attempts < 200000; ++attempts) {
        if (rig.read(Z15_STATUS) & Z15_HAS_ERROR)
            throw std::runtime_error("fabric reported error " + std::to_string(rig.read(Z15_ERROR)));
        if (!(rig.read(Z15_GPU_SLOTS) & ((1u << warp) | (1u << 4)))) {
            checked_write(rig, Z15_GPU_PC, entry);
            checked_write(rig, Z15_GPU_JOB, job);
            checked_write(rig, Z15_CTRL, Z15_START_GPU);
            return;
        }
    }
    throw std::runtime_error("GPU slot wait timeout");
}

int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("expected path to run control file");
        Verilated::commandArgs(argc, argv);
        std::ifstream control(argv[1]);
        std::string source, expected, command, stage;
        unsigned address, size, commands = 0;
        if (!(control >> source >> address >> size))
            throw std::runtime_error("missing startup DMA extent");
        Rig rig;
        require(rig.read(Z15_MAGIC) == Z15_MAGIC_VALUE && rig.read(Z15_VERSION) == Z15_VERSION_VALUE,
                "wrong Z15 MMIO ABI");
        bool dma_in_flight = false;
        if (source != "-")
            upload(rig, source, address, size);
        unsigned prior_npu = 0, prior_gpu = 0;
        while (control >> command) {
            if (command == "write") {
                unsigned reg;
                uint32_t value;
                if (!(control >> reg >> value))
                    throw std::runtime_error("malformed MMIO write");
                checked_write(rig, reg, value);
            } else if (command == "submit_gpu") {
                unsigned entry;
                uint32_t job;
                if (!(control >> entry >> job))
                    throw std::runtime_error("malformed GPU submission");
                submit_gpu(rig, entry, job);
            } else if (command == "upload") {
                unsigned lo, bytes;
                if (!(control >> source >> lo >> bytes))
                    throw std::runtime_error("malformed DMA upload");
                upload(rig, source, lo, bytes);
            } else if (command == "ddr_put" || command == "compare_ddr") {
                unsigned lo, bytes;
                if (!(control >> expected >> lo >> bytes) || !bytes ||
                        uint64_t(lo) + bytes > Z15_POOL_SIZE)
                    throw std::runtime_error("malformed DDR extent");
                auto golden = load_image(expected, bytes);
                if (command == "ddr_put") {
                    Rig::put(&rig, lo, golden.data(), bytes);
                } else {
                    if (dma_in_flight) throw std::runtime_error("DDR comparison before DMA drain");
                    std::vector<uint8_t> actual(bytes);
                    Rig::get(&rig, lo, actual.data(), bytes);
                    require(actual == golden, "DDR output differs from independent oracle");
                    ++commands;
                    printf("Z15_DDR_PASS offset=%u bytes=%u\n", lo, bytes);
                }
            } else if (command == "dma_in" || command == "dma_out" || command == "dma_wait") {
                if (dma_in_flight) {
                    wait_for(rig, Z15_DMA_DONE, Z15_DMA_BUSY);
                    dma_in_flight = false;
                }
                if (command != "dma_wait") {
                    unsigned ddr, lo, bytes;
                    if (!(control >> ddr >> lo >> bytes) || ddr % 32 ||
                            uint64_t(ddr) + bytes > Z15_POOL_SIZE)
                        throw std::runtime_error("malformed DMA command");
                    extent(lo, bytes);
                    dma_start(rig, ddr, lo, bytes, command == "dma_out");
                    dma_in_flight = true;
                }
            } else if (command == "wait_npu" || command == "wait_gpu") {
                if (!(control >> stage))
                    throw std::runtime_error("malformed launch wait");
                const bool gpu = command == "wait_gpu";
                wait_for(rig, gpu ? Z15_GPU_DONE : Z15_NPU_DONE, gpu ? Z15_GPU_BUSY : Z15_NPU_BUSY);
                printf("Z15_PLAN_DRAIN stage=%s device=%s wall_npu_ticks=%u\n",
                       stage.c_str(), gpu ? "GPU" : "TENSOR", rig.tick_count);
            } else if (command == "download" || command == "check" || command == "check_npu" ||
                       command == "check_gpu") {
                unsigned lo = address, bytes = size;
                if (!(control >> stage >> expected) || (command != "check" && !(control >> lo >> bytes)))
                    throw std::runtime_error("malformed output check");
                extent(lo, bytes);
                bool gpu = command == "check_gpu";
                if (command != "download")
                    wait_for(rig, gpu ? Z15_GPU_DONE : Z15_NPU_DONE, gpu ? Z15_GPU_BUSY : Z15_NPU_BUSY);
                unsigned current = rig.read(gpu ? Z15_GPU_CYCLES_LO : Z15_NPU_CYCLES_LO);
                unsigned &previous = gpu ? prior_gpu : prior_npu;
                dma(rig, lo, bytes, true);
                auto golden = load_image(expected, bytes);
                std::vector<uint8_t> actual(bytes);
                Rig::get(&rig, lo, actual.data(), bytes);
                for (unsigned byte = 0; byte < bytes; ++byte)
                    if (actual[byte] != golden[byte])
                        throw std::runtime_error(stage + " differs at byte " + std::to_string(lo + byte) +
                                                 " got " + std::to_string(actual[byte]) +
                                                 " expected " + std::to_string(golden[byte]));
                ++commands;
                if (command == "download") {
                    printf("Z15_PLAN_DMA_PASS output=%s compared_bytes=%u\n", stage.c_str(), bytes);
                } else {
                    printf("Z15_%s_RTL_STAGE_PASS stage=%s busy_ticks=%u compared_bytes=%u\n",
                           gpu ? "GPU" : "NPU", stage.c_str(), current - previous, bytes);
                    previous = current;
                }
            } else {
                throw std::runtime_error("unknown control command");
            }
        }
        require(commands != 0, "no hardware checks performed");
        const unsigned wall = rig.tick_count;
        const unsigned overlap = rig.read(Z15_OVERLAP_CYCLES_LO);
        printf("Z15_RTL_PASS commands=%u wall_npu_ticks=%u axi_transactions=%u overlap_npu_ticks=%u\n",
               commands, wall, rig.transactions, overlap);
        const auto *root = rig.dut.rootp;
        printf("Z15_METRICS {\"wall_npu_ticks\":%u,\"axi_transactions\":%u,"
               "\"npu_busy_ticks\":%llu,\"gpu_busy_ticks\":%llu,\"gpu_busy_edges\":%llu,"
               "\"npu_gpu_overlap_ticks\":%u,\"dma_busy_ticks\":%llu,"
               "\"compute_dma_overlap_ticks\":%llu,\"gpu_dma_overlap_ticks\":%llu,"
               "\"npu_dma_overlap_ticks\":%llu,\"triple_overlap_ticks\":%llu}\n",
               wall, rig.transactions,
               (unsigned long long)root->z15_bd_axi_shell__DOT__implementation__DOT__npu_busy_cycles,
               (unsigned long long)root->z15_bd_axi_shell__DOT__implementation__DOT__gpu_busy_cycles,
               (unsigned long long)rig.c2_gpu_clock, overlap, (unsigned long long)rig.c2_dma,
               (unsigned long long)rig.c2_compute_dma, (unsigned long long)rig.c2_dma_gpu,
               (unsigned long long)rig.c2_dma_tensor, (unsigned long long)rig.c2_triple);
        rig.dut.final();
    } catch (const std::exception &e) {
        fprintf(stderr,"Z15_NPU_RTL_FAIL %s\n",e.what());
        return 1;
    }
}
