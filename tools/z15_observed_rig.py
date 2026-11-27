"""Attach passive counters to the frozen board testbench at each simulated tick."""

import argparse
from pathlib import Path


def substitute(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"frozen rig changed around {old!r}")
    return source.replace(old, new)


def observe(source, gpu_step=2, gpu_divisor=3, *, p17=False):
    if not 0 < gpu_step < gpu_divisor:
        raise ValueError("expected GPU/NPU clock ratio between zero and one")
    result = substitute(source, '#include "Vz15_bd_axi_shell.h"',
                        '#include "Vz15_bd_axi_shell.h"\n#include "Vz15_bd_axi_shell___024root.h"')
    result = substitute(result, "    VerilatedContext context;",
                        "    struct SerialContext : VerilatedContext {\n"
                        "        SerialContext() { threads(1); }\n"
                        "    } context;")
    result = substitute(result, "    unsigned transactions=0;",
                        """    unsigned transactions=0;
    uint64_t c2_dma=0,c2_compute_dma=0,c2_triple=0,c2_gpu_clock=0;
    uint64_t c2_dma_gpu=0,c2_dma_tensor=0;
    uint64_t c3_read_bytes=0,c3_write_bytes=0;
    uint64_t c4_occupancy[8]={};
    uint64_t c4_read_overlap_beats=0,c4_write_overlap_beats=0;""")
    old_step, old_divisor = (3, 4) if p17 else (2, 3)
    result = substitute(result, f"        gpu_phase+={old_step};",
                        f"        const bool was_gpu_high=dut.clk_gpu;\n        gpu_phase+={gpu_step};")
    result = substitute(result, f"if(gpu_phase>={old_divisor}) {{ gpu_phase-={old_divisor};",
                        f"if(gpu_phase>={gpu_divisor}) {{ gpu_phase-={gpu_divisor};")
    result = substitute(result, "        context.timeInc(1);\n        dut.eval();",
                        """        context.timeInc(1);
        dut.eval();
        if(!was_gpu_high && dut.clk_gpu && dut.aresetn &&
           dut.rootp->z15_bd_axi_shell__DOT__implementation__DOT__gpu_busy)
            ++c2_gpu_clock;""")
    read_counter = "ddr_read_bytes+=32; " if p17 else ""
    result = substitute(result, f"            {read_counter}--read_left; ar_st+=32; read_hold=false;",
                        """            c3_read_bytes+=32;
            c4_read_overlap_beats+=dut.rootp->z15_bd_axi_shell__DOT__implementation__DOT__gpu_busy ||
                                   dut.rootp->z15_bd_axi_shell__DOT__implementation__DOT__npu_busy;
            """ + read_counter + "--read_left; ar_st+=32; read_hold=false;")
    result = substitute(result, "            require(write_left!=0,\"W with no AW\");",
                        """            require(write_left!=0,"W with no AW");
            c4_write_overlap_beats+=dut.rootp->z15_bd_axi_shell__DOT__implementation__DOT__gpu_busy ||
                                    dut.rootp->z15_bd_axi_shell__DOT__implementation__DOT__npu_busy;""")
    result = substitute(result, "                ddr[aw_st+j]=uint8_t(dut.m_axi_wdata[j/4]>>(8*(j%4)));",
                        """                {ddr[aw_st+j]=uint8_t(dut.m_axi_wdata[j/4]>>(8*(j%4)));
                 ++c3_write_bytes;}""")
    return substitute(result, "        ++tick_count;",
                      """        ++tick_count;
        if(dut.aresetn) {
            const auto *r=dut.rootp;
            const bool dma=r->z15_bd_axi_shell__DOT__implementation__DOT__dma_busy;
            const bool gpu=r->z15_bd_axi_shell__DOT__implementation__DOT__gpu_busy;
            const bool npu=r->z15_bd_axi_shell__DOT__implementation__DOT__npu_busy;
            ++c4_occupancy[unsigned(npu) | unsigned(gpu)*2u | unsigned(dma)*4u];
            c2_dma+=dma;
            c2_compute_dma+=dma && (gpu || npu);
            c2_dma_gpu+=dma && gpu;
            c2_dma_tensor+=dma && npu;
            c2_triple+=dma && gpu && npu;
        }""")


def paged_ddr(source):
    """Preserve zero-filled DDR bytes and the BFM's AXI timing, without byte maps."""
    return substitute(source, "    std::map<uint32_t,uint8_t> ddr;",
                      """    struct PagedDdr {
        static constexpr unsigned page_size = 4096;
        using Page = std::array<uint8_t, page_size>;
        std::vector<std::unique_ptr<Page>> pages =
            std::vector<std::unique_ptr<Page>>(Z15_POOL_SIZE / page_size);

        uint8_t &operator[](uint32_t address) {
            require(address >= Z15_POOL_BASE && address - Z15_POOL_BASE < Z15_POOL_SIZE,
                    "DDR BFM address outside aperture");
            const unsigned local = address - Z15_POOL_BASE;
            auto &page = pages[local / page_size];
            if (!page) page = std::make_unique<Page>();
            return (*page)[local % page_size];
        }
    } ddr;""")

if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("src", type=Path)
    cli.add_argument("dst", type=Path)
    cli.add_argument("--gpu-clock-ratio", choices=("2:3", "4:5", "24840:32911", "3:4", "161:221"), default="2:3")
    cli.add_argument("--paged-ddr", action="store_true")
    cli.add_argument("--p17", action="store_true", help="instrument the pinned Phase 17 board rig")
    args = cli.parse_args()
    step, divisor = map(int, args.gpu_clock_ratio.split(":"))
    instrumented = observe(args.src.read_text(encoding="utf-8"), step, divisor, p17=args.p17)
    if args.paged_ddr:
        instrumented = "#include <array>\n#include <memory>\n#include <vector>\n" + paged_ddr(instrumented)
    args.dst.write_text(instrumented, encoding="utf-8")
