// Loads bytecode and dispatch commands emitted by the compiler.
#ifdef FPGA_WARPS
#include "Vgpu_zu15eg_multiwarp.h"
#else
#include "Vgpu.h"
#include "Vgpu___024root.h"
#endif
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

using Bytes = std::vector<uint8_t>;
constexpr unsigned kWindow = 4096;
constexpr unsigned kExit = 0x00100073;

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static Bytes read(const char* path) {
    std::ifstream input(path, std::ios::binary);
    require(bool(input), "cannot open input");
    return {std::istreambuf_iterator<char>(input), {}};
}

static uint32_t load(const Bytes& data, unsigned at) {
    require(!(at & 3) && at <= data.size() && data.size() - at >= 4, "invalid word read");
    return uint32_t(data[at]) | uint32_t(data[at+1]) << 8 |
           uint32_t(data[at+2]) << 16 | uint32_t(data[at+3]) << 24;
}

static void store(Bytes& data, unsigned at, uint32_t value, unsigned mask) {
    require(!(at & 3) && at <= data.size() && data.size() - at >= 4, "invalid word write");
    for (unsigned b = 0; b < 4; ++b) {
        if (mask & (1u << b))
            data[at+b] = uint8_t(value >> (b*8));
    }
}

static uint32_t opcode(const Bytes& program, unsigned pc) {
    return pc < program.size() ? load(program, pc) : kExit;
}

struct Job {
    unsigned entry, threads, warp;
};

struct Measurement {
    unsigned cycles = 0, retired = 0, issue = 0, dual = 0, max_pending = 0;
};

#ifdef FPGA_WARPS
class Simulation {
    Vgpu_zu15eg_multiwarp dut;
    const Bytes& program;
    unsigned tick() {
        dut.clk = 0;
        dut.program_ready = 0;
        dut.eval();
        if (dut.program_valid) {
            dut.program_ready = 1;
            dut.program_data = opcode(program, dut.program_address);
            dut.eval();
        }
        const bool accepted = dut.job_valid && dut.job_ready;
        dut.clk = 1;
        dut.eval();
        return accepted;
    }

    uint32_t host(unsigned address, uint32_t value, unsigned mask) {
        dut.host_address = address;
        dut.host_write_data = value;
        dut.host_write_strb = mask;
        dut.host_valid = 1;
        for (unsigned cycle = 0; cycle < 32; ++cycle) {
            tick();
            if (dut.host_ready) {
                const auto data = dut.host_read_data;
                dut.host_valid = 0;
                tick();
                return data;
            }
        }
        throw std::runtime_error("FPGA BRAM host access timeout");
    }

public:
    explicit Simulation(const Bytes& code) : program(code) {
        dut.reset = 1;
        dut.job_valid = 0;
        dut.host_valid = 0;
        dut.data_delay = 2;
        tick();
        dut.reset = 0;
        for (unsigned i = 0; i < FPGA_WARPS+4; ++i)
            tick();
    }

    Measurement run(Bytes& ram, const std::vector<Job>& jobs) {
        for (unsigned at = 0; at < ram.size(); at += 4)
            host(at, load(ram, at), 15);
        Measurement result;
        std::array<bool, FPGA_WARPS> busy{};
        unsigned submitted = 0, completed = 0, pending = 0;
        const auto initial_retired = dut.retired_threads;
        const auto initial_issue = dut.issue_cycles;
        const auto initial_dual = dut.dual_retires;
        for (; result.cycles < 5000000 && completed < jobs.size(); ++result.cycles) {
            dut.job_valid = 0;
            for (unsigned w = 0; w < FPGA_WARPS; ++w) {
                if (busy[w] && ((dut.job_done >> w) & 1)) {
                    require(!((dut.job_fault >> w) & 1), "FPGA GPU job fault");
                    busy[w] = false;
                    --pending;
                    ++completed;
                }
            }
            if (submitted < jobs.size() && !busy.at(jobs[submitted].warp)) {
                const auto& job = jobs[submitted];
                dut.job_valid = 1;
                dut.job_warp = job.warp;
                dut.job_pc = job.entry;
                dut.job_threads = job.threads;
            }
            if (tick()) {
                busy[dut.job_warp] = true;
                ++submitted;
                result.max_pending = std::max(++pending, result.max_pending);
            }
        }
        require(completed == jobs.size(), "FPGA GPU job timeout");
        result.retired = dut.retired_threads - initial_retired;
        result.issue = dut.issue_cycles - initial_issue;
        result.dual = dut.dual_retires - initial_dual;
        for (unsigned at = 0; at < ram.size(); at += 4)
            store(ram, at, host(at, 0, 0), 15);
        return result;
    }
};
#else
class Simulation {
    Vgpu dut;
    const Bytes& program;
    unsigned dual = 0, retired = 0;

    void tick(Bytes& ram) {
        dut.clk = 0;
        dut.program_mem_read_ready = 0;
        dut.data_mem_read_ready = 0;
        dut.data_mem_write_ready = 0;
        dut.eval();
        if (dut.program_mem_read_valid) {
            dut.program_mem_read_ready = 1;
            dut.program_mem_read_data[0] = opcode(program, dut.program_mem_read_address[0]);
        }
        const unsigned writes = dut.data_mem_write_valid;
        std::array<unsigned, 4> addresses{}, masks{};
        std::array<uint32_t, 4> values{};
        for (unsigned p = 0; p < 4; ++p) {
            if (dut.data_mem_read_valid & (1u << p)) {
                dut.data_mem_read_ready |= 1u << p;
                dut.data_mem_read_data[p] = load(ram, dut.data_mem_read_address[p]);
            }
            if (writes & (1u << p)) {
                dut.data_mem_write_ready |= 1u << p;
                addresses[p] = dut.data_mem_write_address[p];
                values[p] = dut.data_mem_write_data[p];
                masks[p] = dut.data_mem_write_strb[p];
            }
        }
        dut.eval();
        dut.clk = 1;
        dut.eval();
        for (unsigned p = 0; p < 4; ++p) {
            if (writes & (1u << p))
                store(ram, addresses[p], values[p], masks[p]);
        }
        dual += __builtin_popcount(dut.rootp->gpu__DOT__debug_commit & dut.rootp->gpu__DOT__debug_scalar_commit);
        for (unsigned core = 0; core < 2; ++core)
            if (dut.rootp->gpu__DOT__debug_commit & (1u << core))
                retired += __builtin_popcount(dut.rootp->gpu__DOT__debug_commit_mask[core]);
    }

public:
    explicit Simulation(const Bytes& code) : program(code) {}

    Measurement run(Bytes& ram, const std::vector<Job>& jobs) {
        dut.reset = 1;
        dut.start = 0;
        dut.arena_mode = 1;
        dut.data_cache_retain = 0;
        dut.device_control_write_enable = 0;
        tick(ram);
        dut.reset = 0;
        Measurement result;
        for (const auto& job : jobs) {
            require(job.warp == 0, "v6 has one dispatch slot");
            dut.start = 0;
            dut.start_pc = job.entry;
            dut.device_control_write_enable = 1;
            dut.device_control_data = job.threads;
            tick(ram);
            dut.device_control_write_enable = 0;
            dut.start = 1;
            tick(ram);
            ++result.cycles;
            do {
                tick(ram);
                ++result.cycles;
                require(!dut.fault, "v6 GPU job fault");
            } while (!dut.done && result.cycles < 5000000);
            require(dut.done, "v6 GPU job timeout");
        }
        result.max_pending = 1;
        result.dual = dual;
        result.retired = retired;
        return result;
    }
};
#endif

int main(int argc, char** argv) {
    try {
        require(argc == 5, "usage: gpu-sim code.bin initial.bin jobs.txt result.bin");
        const auto code = read(argv[1]);
        auto ram = read(argv[2]);
        require(!code.empty() && !(code.size() & 3) && code.size() <= 262144, "bad GPU program");
        require(ram.size() == kWindow, "GPU input must be exactly 4096 bytes");
        std::ifstream commands(argv[3]);
        require(bool(commands), "cannot open scheduled jobs");
        std::vector<Job> jobs;
        Job job{};
        while (commands >> job.entry >> job.threads >> job.warp) {
            require(!(job.entry & 3) && job.entry < code.size() &&
                    job.threads && job.threads <= 255, "invalid job");
            jobs.push_back(job);
        }
        require(commands.eof() && !jobs.empty(), "invalid job file");
        auto result = Simulation(code).run(ram, jobs);
        std::ofstream output(argv[4], std::ios::binary);
        output.write(reinterpret_cast<const char*>(ram.data()), ram.size());
        require(bool(output), "cannot write GPU result");
        std::cout << "GPU_RTL cycles=" << result.cycles << " retired=" << result.retired
                  << " issue=" << result.issue << " dual=" << result.dual
                  << " max_pending=" << result.max_pending << " jobs=" << jobs.size() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
