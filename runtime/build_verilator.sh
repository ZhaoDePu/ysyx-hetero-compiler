#!/usr/bin/env bash
set -euo pipefail
runtime=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=/dev/shm/ysyx_compiler_c2
out="$root/runtime/verilator"
build="$root/npu/z15-build"
src="$root/npu/z15-src"
sdk=/home/zdp/ysyx-migration/tools/oss-cad-suite
[[ "$(realpath -e "$root")" == "$root" && -O "$root" ]]
[[ -f "$build/Vz15_bd_axi_shell__ALL.a" && -f "$src/sim/closed_loop.cpp" ]]
test ! -L "$root/runtime" && test ! -L "$out"
mkdir -p "$out"
export TMPDIR="$out" CCACHE_DISABLE=1 OMP_NUM_THREADS=1
low=(nice -n 15 ionice -c 3)
"${low[@]}" python3 -B "$runtime/../tools/z15_observed_rig.py" "$src/sim/closed_loop.cpp" "$out/observed_rig.inc"
c99=(-std=c99 -O2 -Wall -Wextra -Werror -I"$runtime")
for source in ysyx_rt hal_linux_uio ysyx_run; do
    "${low[@]}" gcc "${c99[@]}" -DYSYX_WITH_VERILATOR -c "$runtime/$source.c" -o "$out/$source.o"
done
"${low[@]}" g++ -std=c++17 -O2 -I"$out" -I"$build" -I"$src/app" \
    -isystem "$sdk/include" -isystem "$sdk/include/vltstd" \
    -c "$runtime/hal_verilator_z15.cpp" -o "$out/hal_verilator_z15.o"
"${low[@]}" g++ -pthread "$out/ysyx_run.o" "$out/ysyx_rt.o" \
    "$out/hal_linux_uio.o" "$out/hal_verilator_z15.o" \
    "$build/Vz15_bd_axi_shell__ALL.a" "$build/verilated.o" \
    "$build/verilated_dpi.o" "$build/verilated_threads.o" -o "$out/ysyx_run"
file "$out/ysyx_run"
