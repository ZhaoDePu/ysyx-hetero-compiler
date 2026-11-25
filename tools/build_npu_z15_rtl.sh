#!/usr/bin/env bash
set -euo pipefail
work=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
base=/home/zdp/ysyx-migration
board="$base/work-z15-bringup"
scratch=/dev/shm/ysyx_compiler/npu
commit=4d410d72508a5822bc5c8014981e5b55ef8faa56
[[ "$(realpath -e -- "$scratch")" == "$scratch" && -O "$scratch" ]]
mkdir -p -- "$scratch/z15-src"
[[ "$(cat "$board/FROZEN_COMMIT")" == "$commit" ]]
git -C "$base/repos/ysyx-npu-ext" archive "$commit" tiny-gpu/src fpga/vivado2020 | tar -xf - -C "$scratch/z15-src"
cp -a -- "$board/rtl" "$board/sim" "$board/app" "$scratch/z15-src/"
printf 'source_commit=%s\n' "$commit"
cd "$scratch/z15-src"
sha256sum rtl/gen/g3_cluster.v rtl/gen/g3_external.v rtl/z15_hetero_pl_wrapper.sv rtl/z15_bd_axi_shell.v sim/closed_loop.cpp
export TMPDIR="$scratch" CCACHE_DISABLE=1 OMP_NUM_THREADS=1
verilator="$base/tools/oss-cad-suite/bin/verilator"
"$verilator" --version
exec nice -n 15 ionice -c 3 "$verilator" --cc --exe --build -j 2 -MAKEFLAGS OBJCACHE= -O2 -Wno-fatal --assert \
  --top-module z15_bd_axi_shell --Mdir "$scratch/z15-build" \
  -Ifpga/vivado2020 -Itiny-gpu -Itiny-gpu/src -CFLAGS "-std=c++17 -I$PWD/app -I$PWD/sim -O2" \
  -f fpga/vivado2020/filelist.f rtl/gen/g3_cluster.v rtl/gen/g3_external.v \
  fpga/vivado2020/soc_phase12b_balanced_top.sv rtl/z15_hetero_pl_wrapper.sv rtl/z15_bd_axi_shell.v \
  "$work/tools/npu_z15_rtl.cpp" -o run-npu-z15
