// Direct Phase 12C tensor-engine regression. No AXI or GPU is present.
#include "Vzu15eg_tensor_wide.h"
#include "verilated.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t Capacity = 2u << 20;
constexpr unsigned BurstWords = 16;
using Beat = std::array<std::uint32_t, BurstWords>;

std::vector<unsigned char> read_image(const char *path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot read tensor image");
    std::vector<unsigned char> data{std::istreambuf_iterator<char>(file),
                                   std::istreambuf_iterator<char>()};
    if (data.size() != Capacity)
        throw std::runtime_error("incorrect tensor image capacity");
    return data;
}

struct Rig {
    VerilatedContext context;
    Vzu15eg_tensor_wide dut{&context};
    std::vector<unsigned char> memory;
    Beat pending{}, read_data{};
    unsigned ticks = 0, issued = 0, read_words = 0, write_words = 0;

    explicit Rig(std::vector<unsigned char> bytes) : memory(std::move(bytes)) {}

    void tick()
    {
        dut.clk = 0;
        for (unsigned i = 0; i < BurstWords; ++i)
            dut.read_data[i] = read_data[i];
        dut.eval();

        const size_t base = 4u * size_t(dut.read_addr);
        const unsigned n = dut.read_count;
        const size_t out = 4u * size_t(dut.write_addr);
        const unsigned count = dut.write_count;
        if (n > BurstWords || base + n * 4u > Capacity || count > BurstWords ||
            out + count * 4u > Capacity)
            throw std::runtime_error("frozen tensor memory request out of range");
        Beat reply{};
        for (unsigned i = 0; i < n; ++i)
            for (unsigned j = 0; j < 4; ++j)
                reply[i] |= std::uint32_t(memory[base + 4u * i + j]) << (8u * j);
        for (unsigned i = 0; i < count; ++i)
            for (unsigned j = 0; j < 4; ++j)
                memory[out + 4u * i + j] = static_cast<unsigned char>(dut.write_data[i] >> (8u * j));

        // Registered Phase 12C wide memory has a two-cycle response.
        read_data = pending;
        pending = reply;
        read_words += n;
        write_words += count;
        issued += dut.mac_issue;
        context.timeInc(1);
        dut.clk = 1;
        dut.eval();
        ++ticks;
    }

    void setup(const std::array<std::uint32_t, 10> &cfg)
    {
        const auto box = cfg[0], size = cfg[1], flags = cfg[6], shape = cfg[7], slice = cfg[8];
        dut.cmd_m = box & 511u;
        dut.cmd_n = (box >> 16) & 511u;
        dut.cmd_k = size & 511u;
        dut.cmd_tile_k = (size >> 16) & 511u;
        dut.cmd_a = cfg[2] >> 2;
        dut.cmd_b = cfg[3] >> 2;
        dut.cmd_c = cfg[4] >> 2;
        dut.cmd_bias = cfg[5] >> 2;
        dut.cmd_unsigned_a = flags & 1u;
        dut.cmd_has_bias = (flags >> 1) & 1u;
        dut.cmd_requant = (flags >> 2) & 1u;
        dut.cmd_relu = (flags >> 3) & 1u;
        dut.cmd_shift = (flags >> 4) & 31u;
        dut.cmd_conv = (flags >> 9) & 1u;
        dut.cmd_kernel3 = (flags >> 10) & 1u;
        dut.cmd_stride2 = (flags >> 11) & 1u;
        dut.cmd_pad1 = (flags >> 12) & 1u;
        dut.cmd_conv_slice = (flags >> 13) & 1u;
        dut.cmd_stream_fast = (flags >> 14) & 1u;
        dut.cmd_per_channel = (flags >> 15) & 1u;
        dut.cmd_silu = (flags >> 16) & 1u;
        dut.cmd_packed_output = (flags >> 17) & 1u;
        if (dut.cmd_per_channel || dut.cmd_silu)
            throw std::runtime_error("direct GEMM fixture does not upload activation/scale LUTs");
        dut.cmd_height = shape & 511u;
        dut.cmd_width = (shape >> 9) & 511u;
        dut.cmd_channels = (shape >> 18) & 511u;
        dut.cmd_conv_first_row = slice & 511u;
        dut.cmd_conv_rows = (slice >> 16) & 511u;
        dut.cmd_partial = cfg[9] >> 2;
    }
};
} // namespace

int main(int argc, char **argv)
{
    try {
        if (argc != 4)
            throw std::runtime_error("usage: tensor-sim registers.txt initial.bin expected.bin");
        std::ifstream cfgfile(argv[1]);
        std::array<std::uint32_t, 10> cfg{};
        for (auto &word : cfg)
            if (!(cfgfile >> word))
                throw std::runtime_error("invalid direct tensor register image");
        Rig rig(read_image(argv[2]));
        const auto expected = read_image(argv[3]);
        rig.dut.reset = 1;
        rig.dut.start = 0;
        rig.tick();
        rig.tick();
        rig.dut.reset = 0;
        rig.setup(cfg);
        const unsigned start = rig.ticks;
        rig.dut.start = 1;
        rig.tick();
        rig.dut.start = 0;
        while (!rig.dut.done && rig.ticks - start < 1000000)
            rig.tick();
        if (!rig.dut.done || rig.dut.error)
            throw std::runtime_error("frozen tensor engine did not complete successfully");
        const auto result = std::mismatch(rig.memory.begin(), rig.memory.end(), expected.begin());
        if (result.first != rig.memory.end()) {
            const auto offset = result.first - rig.memory.begin();
            std::fprintf(stderr, "tensor differs at %#zx: RTL %u, expected %u\n",
                         size_t(offset), *result.first, *result.second);
            return 1;
        }
        std::printf("C3_TENSOR_PASS cycles=%u mac_issue_ticks=%u read_words=%u write_words=%u compared_bytes=%zu\n",
                    rig.ticks - start, rig.issued, rig.read_words, rig.write_words, expected.size());
        return 0;
    } catch (const std::exception &exc) {
        std::fprintf(stderr, "%s\n", exc.what());
        return 1;
    }
}
