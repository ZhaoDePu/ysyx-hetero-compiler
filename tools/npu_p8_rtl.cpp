#include "Vnpu_tensor_engine.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using Arena = std::array<uint8_t, 4096>;

Arena load(const std::string &name) {
    Arena bytes{};
    std::ifstream file(name, std::ios::binary);
    if (!file.read(reinterpret_cast<char *>(bytes.data()), bytes.size()) || file.peek() != EOF)
        throw std::runtime_error("invalid 4096-byte arena: " + name);
    return bytes;
}

class Rig {
    struct SerialContext : VerilatedContext {
        SerialContext() { threads(1); }
    } context;
    Vnpu_tensor_engine dut{&context};
public:
    Arena memory;
    unsigned cycles = 0, checkpoints = 0;

    explicit Rig(Arena initial) : memory(initial) {
        dut.peer_done = dut.peer_error = 0;
        dut.reset = 1;
        tick();
        dut.reset = 0;
        tick();
    }

    void tick() {
        dut.clk = 0;
        dut.eval();
        dut.read_ready = cycles % 5 == 0 ? 0b0111 : 0b1111;
        dut.write_ready = cycles % 7 == 0 ? 0b1011 : 0b1111;
        for (int lane = 0; lane < 4; ++lane) {
            unsigned addr = dut.address[lane] & 4092;
            dut.load_data[lane] = 0;
            for (int byte = 0; byte < 4; ++byte)
                dut.load_data[lane] |= uint32_t(memory[addr + byte]) << (8 * byte);
        }
        dut.eval();
        if (dut.peer_request)
            throw std::runtime_error("unexpected NPU peer request");
        for (int lane = 0; lane < 4; ++lane) {
            unsigned addr = dut.address[lane];
            unsigned selected = 1u << lane;
            if ((dut.read_valid | dut.write_valid) & selected)
                if (addr >= 4096 || addr % 4)
                    throw std::runtime_error("unaligned or out of range NPU bus request");
            if (dut.write_valid & dut.write_ready & selected)
                for (int byte = 0; byte < 4; ++byte)
                    if (dut.store_strb[lane] & (1u << byte))
                        memory[addr + byte] = dut.store_data[lane] >> (8 * byte);
        }
        dut.clk = 1;
        dut.eval();
        checkpoints += bool(dut.checkpoint);
        ++cycles;
    }

    void run(uint32_t a, uint32_t descriptor) {
        dut.a_base = a;
        dut.descriptor = descriptor;
        dut.extended = 1;
        dut.dimensions = dut.options = 0;
        dut.start = 1;
        tick();
        dut.start = 0;
        for (unsigned limit = 200000; !dut.done && limit; --limit)
            tick();
        if (!dut.done || dut.error)
            throw std::runtime_error("NPU execution failed at descriptor " + std::to_string(descriptor));
        tick();
    }
};

int main(int argc, char **argv) {
    try {
        if (argc != 2 && argc != 3)
            throw std::runtime_error("expected run control file and optional actual arena output");
        std::ifstream control(argv[1]);
        std::string source, expected, name;
        unsigned a, descriptor, slots;
        if (!(control >> source))
            throw std::runtime_error("missing initial arena");
        Rig rig(load(source));
        int commands = 0;
        while (control >> name >> a >> descriptor >> slots >> expected) {
            unsigned previous = rig.checkpoints, before = rig.cycles;
            rig.run(a, descriptor);
            if (rig.checkpoints - previous != slots)
                throw std::runtime_error("incorrect PROGRAM checkpoint count");
            Arena golden = load(expected);
            for (unsigned addr = 0; addr < golden.size(); ++addr)
                if (rig.memory[addr] != golden[addr])
                    throw std::runtime_error(name + " differs at byte " + std::to_string(addr) +
                                             " got " + std::to_string(rig.memory[addr]) +
                                             " expected " + std::to_string(golden[addr]));
            ++commands;
            std::cout << "NPU_RTL_STAGE_PASS stage=" << name << " descriptor=" << descriptor
                      << " cycles=" << rig.cycles - before << " checkpoints=" << slots
                      << " compared_bytes=4096\n";
        }
        if (!control.eof() || commands == 0)
            throw std::runtime_error("incomplete run control file");
        if (argc == 3) {
            std::ofstream actual(argv[2], std::ios::binary);
            if (!actual.write(reinterpret_cast<const char *>(rig.memory.data()), rig.memory.size()))
                throw std::runtime_error("could not write actual arena");
        }
        std::cout << "NPU_RTL_PASS commands=" << commands << " cycles=" << rig.cycles << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "NPU_RTL_FAIL " << e.what() << "\n";
        return 1;
    }
}
