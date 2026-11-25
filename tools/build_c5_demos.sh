#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
root=/dev/shm/ysyx_compiler_c5
source "$repo/tools/c5_resources.sh"
compiler_build_slots

kind=${1:?expected x86, aarch64 or verilator}
targets=(all)
case "$kind" in
  x86)
    options=(CC=gcc AR=ar)
    ;;
  aarch64)
    toolchain=${AARCH64_PREFIX:-/opt/Xilinx/2019/SDK/2019.1/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-}
    options=(CC="$toolchain"gcc AR="$toolchain"ar)
    ;;
  verilator)
    targets=(demos)
    rtl="$root/npu/z15-build"
    hal="$root/runtime/verilator/hal_verilator_z15.o"
    test -f "$hal"
    options=(CC=gcc AR=ar CPPFLAGS=-DYSYX_WITH_VERILATOR DEMO_LINK=g++ LDFLAGS=-pthread
        "DEMO_LIBS=$hal $rtl/Vz15_bd_axi_shell__ALL.a $rtl/verilated.o $rtl/verilated_dpi.o $rtl/verilated_threads.o")
    ;;
  *) echo "expected x86, aarch64 or verilator" >&2; exit 2 ;;
esac

kit="$repo/deploy/z15_board_kit"
out="$root/runtime/demos-$kind"
export TMPDIR="$root"
nice -n 10 ionice -c2 -n7 make -C "$kit" -j"$compiler_jobs" OUT="$out" "${options[@]}" "${targets[@]}"
file "$out/z15_tinylm_demo" "$out/z15_yolo26_demo"
