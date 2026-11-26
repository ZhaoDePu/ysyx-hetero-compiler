"""Attach passive counters to the frozen board testbench at each simulated tick."""

import argparse
from pathlib import Path


def substitute(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"frozen rig changed around {old!r}")
    return source.replace(old, new)


def observe(source, gpu_step=2, gpu_divisor=3):
    if not 0 < gpu_step < gpu_divisor:
        raise ValueError("expected GPU/NPU clock ratio between zero and one")
    result = substitute(source, '#include "Vz15_bd_axi_shell.h"',
                        '#include "Vz15_bd_axi_shell.h"\n#include "Vz15_bd_axi_shell___024root.h"')
    result = substitute(result, "    unsigned transactions=0;",
                        """    unsigned transactions=0;
    uint64_t c2_dma=0,c2_compute_dma=0,c2_triple=0,c2_gpu_clock=0;
    uint64_t c2_dma_gpu=0,c2_dma_tensor=0;
    uint64_t c3_read_bytes=0,c3_write_bytes=0;
    uint64_t c4_occupancy[8]={};
    uint64_t c4_read_overlap_beats=0,c4_write_overlap_beats=0;""")
    result = substitute(result, "        gpu_phase+=2;",
                        f"        const bool was_gpu_high=dut.clk_gpu;\n        gpu_phase+={gpu_step};")
    result = substitute(result, "if(gpu_phase>=3) { gpu_phase-=3;",
                        f"if(gpu_phase>={gpu_divisor}) {{ gpu_phase-={gpu_divisor};")
    result = substitute(result, "        context.timeInc(1);\n        dut.eval();",
                        """        context.timeInc(1);
        dut.eval();
        if(!was_gpu_high && dut.clk_gpu && dut.aresetn &&
           dut.rootp->z15_bd_axi_shell__DOT__implementation__DOT__gpu_busy)
            ++c2_gpu_clock;""")
    result = substitute(result, "            --read_left; ar_st+=32; read_hold=false;",
                        """            c3_read_bytes+=32;
            c4_read_overlap_beats+=dut.rootp->z15_bd_axi_shell__DOT__implementation__DOT__gpu_busy ||
                                   dut.rootp->z15_bd_axi_shell__DOT__implementation__DOT__npu_busy;
            --read_left; ar_st+=32; read_hold=false;""")
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


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("src", type=Path)
    cli.add_argument("dst", type=Path)
    cli.add_argument("--gpu-clock-ratio", choices=("2:3", "4:5", "24840:32911", "3:4", "161:221"), default="2:3")
    args = cli.parse_args()
    step, divisor = map(int, args.gpu_clock_ratio.split(":"))
    args.dst.write_text(observe(args.src.read_text(encoding="utf-8"), step, divisor), encoding="utf-8")
