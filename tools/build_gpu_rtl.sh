#!/usr/bin/env bash
set -euo pipefail
workspace=$(cd "$(dirname "$0")/.." && pwd -P)
hardware=/home/zdp/ysyx-migration/repos/ysyx-gpu-ext
variant=$1
base=/dev/shm/ysyx_compiler_c2/gpu
case "$variant" in
    v6) pin=19852c9a370828fa06aa1cfb90ad5fcd668cad30;;
    g4-16t4w|g4-8t8w) pin=df2b2b732c0786acc3dd0e207be3e39c2a6a9e28;;
    *) echo "invalid GPU simulation target: $variant" >&2; exit 2;;
esac
mkdir -m 700 -p "$base"
[[ $(realpath -e "$base") == "$base" ]]
build="$base/$variant"
mkdir -m 700 -p "$build"
[[ $(realpath -e "$build") == "$build" ]]
exec 9>"$base/build.lock"
flock 9
if [[ -x "$build/sim" && "$build/sim" -nt "$workspace/tools/gpu_rtl.cpp" &&
      "$build/sim" -nt "$workspace/tools/build_gpu_rtl.sh" ]]; then
    exit 0
fi
export TMPDIR="$build" CCACHE_DISABLE=1 OBJCACHE= MAKEFLAGS=-j2 PYTHONDONTWRITEBYTECODE=1
cd "$build"
git -C "$hardware" archive "$pin" tiny-gpu/src tiny-gpu/test | tar -xf -
cd tiny-gpu
src=(src/rv32_*.sv src/{gpu,dcr,dcache,icache}.sv)
if [[ "$variant" == v6 ]]; then
    vflags=(--top-module gpu)
else
    if [[ "$variant" == g4-16t4w ]]; then warps=4; lanes=8; else warps=8; lanes=4; fi
    python3 -B -c 'import sys; sys.path.insert(0, "test/g4"); from model import model; print(model(int(sys.argv[1]), int(sys.argv[2])))' \
        "$warps" "$lanes" > "$build/context_sim.sv"
    vflags=(--top-module gpu_zu15eg_multiwarp -GNUM_WARPS="$warps" -GDATA_ADDR_BITS=18
            -GBANKS=4 -GNUM_CORES=2 -GLANES="$lanes" -GSTACK_BYTES=256 -CFLAGS "-DFPGA_WARPS=$warps")
    src+=("$build/context_sim.sv" src/{gpu_zu15eg_multiwarp,g1_shared_bram,g1_warp_scheduler,g2_banked_bram,g3_lock_allocator}.sv)
fi
nice -n 15 ionice -c 3 verilator --cc --exe --build -j 2 --threads 1 --output-split 5000 \
    -Wno-fatal --Mdir "$build/obj" -o "$build/sim" "${vflags[@]}" \
    "${src[@]}" "$workspace/tools/gpu_rtl.cpp" > "$build/build.log" 2>&1
