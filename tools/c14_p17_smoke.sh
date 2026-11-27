#!/usr/bin/env bash
# Build and replay the publisher's Phase 17 projection from the archived sources.
set -euo pipefail
umask 077
work=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
root=/dev/shm/ysyx_compiler_c14
base=/home/zdp/ysyx-migration
pin=5545b9d1f4e70c25a7e15d8f9d863ac012096e85
archive=$(realpath -e -- "$1")
[[ $# == 1 || $# == 2 || $# == 4 ]]
[[ $archive == "$base"/artifacts/compiler-phase-c14-p17-trinity-* && -O $archive ]]
[[ $(realpath -e -- "$root") == "$root" && -O $root ]]
rtl="$root/p17_rtl"
build="$root/p17_build"
gpu="$build/gpu/balanced"
rig="$build/rig"
mkdir -p "$build" "$rig" "$root/tmp" "$archive/m0"
export PATH="$base/tools/oss-cad-suite/bin:$PATH" TMPDIR="$root/tmp"
export CCACHE_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
priority=(nice -n 15 ionice -c2 -n7)

wait_memory() {
    local available
    while :; do
        read -r available < <(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        printf '%s %s MemAvailable_KiB=%s\n' "$(date -Is)" "$1" "$available" |
            tee -a "$archive/m0/memory.log"
        (( available >= 6 * 1024 * 1024 )) && break
        sleep 30
    done
    jobs=1
    if (( available >= 10 * 1024 * 1024 )); then jobs=2; fi
}

exec 9>"$root/probe.lock"
flock -n 9
if [[ ! -d $rtl ]]; then
    mkdir -m 700 "$rtl"
    git -C "$base/repos/ysyx-npu-ext" archive "$pin" | tar -xf - -C "$rtl"
fi
w="$rtl/fpga/z15_step2_v2"
wait_memory g6
if [[ ! -s $gpu/core.v || ! -s $gpu/external.v ]]; then
    # This script resolves sources relative to itself. Change its output allowlist
    # in a private scratch copy beside the archived sources.
    generator="$rtl/tiny-gpu/scripts/c14_gen_g6.sh"
    sed 's@/dev/shm/ysyx_soc_p14/build/gpu/@/dev/shm/ysyx_compiler_c14/p17_build/gpu/@' \
        "$rtl/tiny-gpu/scripts/phase14_gen_g6.sh" >"$generator"
    "${priority[@]}" bash "$generator" "$gpu" balanced
fi
sha256sum -c <(printf '%s  %s\n' \
    ada88ffb2a8f316a33ae20ed154861e87da309d42105f3f5144b789681be427a "$gpu/core.v" \
    2417ff05832ad2a2b50ce8b2559e142361efdad10503f174097dc4664ba59e6c "$gpu/external.v")
cp -- "$w/sim/phase16_probe.cpp" "$w/sim/phase17_probe.cpp" "$rig/"
sed 's/    VerilatedContext context;/    struct SerialContext : VerilatedContext { SerialContext() { threads(1); } } context;/' \
    "$w/sim/closed_loop.cpp" >"$rig/closed_loop.cpp"
cd "$rtl"
wait_memory verilator_build
(
    ulimit -v 2097152
    "${priority[@]}" verilator --cc --exe --build -j "$jobs" --threads 1 \
        -MAKEFLAGS OBJCACHE= -O2 -Wno-fatal --assert --output-split 5000 \
        --top-module z15_bd_axi_shell --Mdir "$build/obj" \
        +define+NPU_NATIVE_DSP +define+Z15_PHASE17 +define+Z15_PHASE17_MONITOR \
        -Ifpga/vivado2020 -Itiny-gpu -Itiny-gpu/src -f fpga/vivado2020/phase17_filelist.f \
        -CFLAGS "-std=c++17 -O2 -I$w/app -DPHASE16=1 -DZ15_PHASE17_MONITOR" \
        "$gpu/core.v" "$gpu/external.v" "$w/rtl/z15_hetero_pl_wrapper.sv" \
        "$w/rtl/z15_bd_axi_shell.v" "$rig/phase17_probe.cpp" -o "$build/probe"
) >"$archive/m0/verilator-build.log" 2>&1
wait_memory projection
(ulimit -v 2097152; "${priority[@]}" "$build/probe" 1 896 64 4) |
    tee "$archive/m0/phase17-projection.log"
awk -F, '$1 == "P17_PROJECTION" && $6 == "W4A8" && $14 == 109323 && $15 == "PASS" {
    passed = 1
} END { exit !passed }' "$archive/m0/phase17-projection.log"

if [[ -n "${2:-}" ]]; then
    replay=$(realpath -e -- "$2")
    [[ $replay == "$root"/* ]]
    wait_memory yolo_build
    "${priority[@]}" python3 -B "$work/tools/z15_observed_rig.py" "$w/sim/closed_loop.cpp" \
        "$rig/c2_observed_loop.hpp" --p17 --paged-ddr --gpu-clock-ratio 3:4
    obj="$build/obj"
    sdk="$base/tools/oss-cad-suite"
    (
        ulimit -v 2097152
        "${priority[@]}" g++ -std=c++17 -O2 -I"$rig" -I"$w/app" -I"$obj" \
            -isystem "$sdk/include" -isystem "$sdk/include/vltstd" \
            "$work/tools/npu_z15_rtl.cpp" "$obj/Vz15_bd_axi_shell__ALL.a" "$obj/verilated.o" \
            "$obj/verilated_dpi.o" "$obj/verilated_threads.o" -pthread -latomic -o "$build/replay"
    ) >"$archive/m1/replay-build.log" 2>&1
    wait_memory yolo_replay
    (ulimit -v 2097152; "${priority[@]}" "$build/replay" "$replay") >"$archive/m1/replay.log" 2>&1
    awk '/^Z15_DDR_PASS/ { n++; sub(/bytes=/, "", $3); bytes += $3 }
         END { exit !(n == 6 && bytes == 176400) }' "$archive/m1/replay.log"
fi

if [[ -n "${3:-}" ]]; then
    weights=$(realpath -e -- "$3")
    ids=$(realpath -e -- "$4")
    wait_memory qwen_build
    app="$build/runtime"
    mkdir -p "$app"
    "${priority[@]}" python3 -B "$work/tools/z15_observed_rig.py" "$w/sim/closed_loop.cpp" \
        "$app/observed_rig.inc" --p17 --paged-ddr --gpu-clock-ratio 3:4
    (
        ulimit -v 2097152
        "${priority[@]}" make -C "$work/runtime" -f verilator.mk -j"$jobs" \
            OUT="$app" RTL_BUILD="$build/obj" RTL_SRC="$w" SDK="$base/tools/oss-cad-suite" \
            "$app/c12_qwen_rtl"
    ) >"$archive/m2/replay-build.log" 2>&1
    for layer in 0 12 23; do
        decode=0
        head=0
        if (( layer == 0 )); then decode=2; fi
        if (( layer == 23 )); then head=1; fi
        wait_memory "qwen_layer_$layer"
        (ulimit -v 2097152; "${priority[@]}" "$app/c12_qwen_rtl" "$weights" "$ids" 8 "$layer" \
            "$decode" "$head" p17) >"$archive/m2/layer-$layer.log" 2>&1
    done
fi
