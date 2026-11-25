#!/usr/bin/env bash
set -euo pipefail
work=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
s=/dev/shm/ysyx_compiler_c5
base=/home/zdp/ysyx-migration
pin=6fe0028a7b7def51c1f2754a058c1984830ee6d0
source "$work/tools/c5_resources.sh"
compiler_build_slots
root="$s/asic/a3"
[[ ! -e "$root" && ! -L "$s/asic" ]]
mkdir -p "$root"/{src,tmp}
export TMPDIR="$root/tmp" CCACHE_DISABLE=1 OMP_NUM_THREADS=1
git -C "$base/work-asic-soc" archive "$pin"     tiny-gpu/asic/rtl tiny-gpu/asic/test/sram_graph.cpp tiny-gpu/asic/test/sram_vpu_cases.h     tiny-gpu/src tiny-gpu/csrc |
    tar -xf - -C "$root/src"
printf 'source_commit=%s\n' "$pin"
cd "$root/src/tiny-gpu"
sha256sum asic/rtl/asic_soc_sram_top.sv asic/rtl/vpu/*.sv asic/test/{sram_graph.cpp,sram_vpu_cases.h}
v="$base/tools/verilator/bin/verilator"
"$v" --version
exec nice -n 10 ionice -c2 -n7 "$v" --cc --exe --build -j "$compiler_jobs" --assert     --public-flat-rw --output-split 5000 -Wno-fatal --top-module asic_soc_sram_top     --Mdir "$root/obj" -MAKEFLAGS 'OBJCACHE= VM_PARALLEL_BUILDS=1 OPT_FAST=-O1 OPT_SLOW=-O0'     -CFLAGS "-std=c++17 -I$PWD/csrc"     src/dcr.sv src/npu_tensor_engine.sv asic/rtl/gpu_v6/*.sv     asic/rtl/asic_exec_owner.sv asic/rtl/asic_compute.sv     asic/rtl/asic_sram_sp_wrapper.sv asic/rtl/asic_sram_banks.sv asic/rtl/asic_soc_sram_top.sv     asic/rtl/vpu/*.sv "$PWD/asic/test/sram_graph.cpp" "$PWD/csrc/elf.cpp" "$PWD/csrc/rv32.cpp"
