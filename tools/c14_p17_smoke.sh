#!/usr/bin/env bash
# Build and replay the publisher's Phase 17 projection from the archived sources.
set -euo pipefail
umask 077
work=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
root=/dev/shm/ysyx_compiler_c14v2
base=/home/zdp/ysyx-migration
pin=5545b9d1f4e70c25a7e15d8f9d863ac012096e85
archive=$(realpath -e -- "$1")
[[ $# == 1 || $# == 2 || $# == 4 ]]
[[ $archive == "$base"/artifacts/compiler-phase-c14v2-p17-trinity-* && -O $archive ]]
[[ $(realpath -e -- "$root") == "$root" && -O $root ]]
rtl="$root/p17_rtl"
build="$root/p17_build"
gpu="$build/gpu/balanced"
rig="$build/rig"
mkdir -p "$build" "$rig" "$root/tmp" "$archive/m3"
export PATH="$base/tools/oss-cad-suite/bin:$PATH" TMPDIR="$root/tmp"
export CCACHE_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
priority=(nice -n 15 ionice -c3)

deadline_file="$root/memory-deadline"
if [[ ! -s $deadline_file ]]; then
    printf '%s\n' "$(( $(date +%s) + 10800 ))" >"$deadline_file"
fi
read -r deadline <"$deadline_file"

wait_memory() {
    local available cgroup path maximum used remaining
    while :; do
        read -r available < <(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
        cgroup=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
        path="/sys/fs/cgroup$cgroup"
        remaining=$((available * 1024))
        while [[ $path == /sys/fs/cgroup* ]]; do
            if [[ -r $path/memory.max && -r $path/memory.current ]]; then
                read -r maximum <"$path/memory.max"
                read -r used <"$path/memory.current"
                if [[ $maximum =~ ^[0-9]+$ ]] && (( maximum - used < remaining )); then
                    remaining=$((maximum - used))
                fi
            fi
            [[ $path != /sys/fs/cgroup ]] || break
            path=$(dirname "$path")
        done
        printf '%s %s MemAvailable_KiB=%s effective_available_B=%s\n' \
            "$(date -Is)" "$1" "$available" "$remaining" | tee -a "$archive/m3/memory.log"
        (( remaining >= 6 * 1024 * 1024 * 1024 )) && break
        local left=$((deadline - $(date +%s)))
        (( left > 0 )) || { echo 'memory gate timed out'; return 76; }
        sleep "$((left < 90 ? left : 90))"
    done
    jobs=2
}

exec 9>"$root/probe.lock"
flock -n 9
if [[ ! -d $rtl ]]; then
    mkdir -m 700 "$rtl"
    git -C "$base/repos/ysyx-npu-ext" archive "$pin" | tar -xf - -C "$rtl"
fi
w="$rtl/fpga/z15_step2_v2"
if [[ ! -s $gpu/core.v || ! -s $gpu/external.v ]]; then
    wait_memory g6
    # This script resolves sources relative to itself. Change its output allowlist
    # in a private scratch copy beside the archived sources.
    generator="$rtl/tiny-gpu/scripts/c14_gen_g6.sh"
    sed 's@/dev/shm/ysyx_soc_p14/build/gpu/@/dev/shm/ysyx_compiler_c14v2/p17_build/gpu/@' \
        "$rtl/tiny-gpu/scripts/phase14_gen_g6.sh" >"$generator"
    "${priority[@]}" bash "$generator" "$gpu" balanced
fi
# Yosys embeds the scratch output path in escaped signal identifiers.
{
    sha256sum "$gpu/core.v" "$gpu/external.v"
    sed "s@$gpu@/dev/shm/ysyx_npu_p17/build/gpu/balanced@g" "$gpu/core.v" |
        sha256sum | sed 's/  -/  core.v (publisher path normalized)/'
} >"$archive/m3/lowered-rtl.sha256"
grep -q '^ada88ffb2a8f316a33ae20ed154861e87da309d42105f3f5144b789681be427a ' \
    "$archive/m3/lowered-rtl.sha256"
grep -q '^2417ff05832ad2a2b50ce8b2559e142361efdad10503f174097dc4664ba59e6c ' \
    "$archive/m3/lowered-rtl.sha256"
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
) >"$archive/m3/verilator-build.log" 2>&1
wait_memory projection
(ulimit -v 2097152; "${priority[@]}" "$build/probe" 1 896 64 4) |
    tee "$archive/m3/phase17-projection.log"
awk -F, '$1 == "P17_PROJECTION" && $6 == "W4A8" && $14 == 109323 && $15 == "PASS" {
    passed = 1
} END { exit !passed }' "$archive/m3/phase17-projection.log"

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
    if (ulimit -v 2097152; "${priority[@]}" "$build/replay" "$replay") >"$archive/m1/replay.log" 2>&1 &&
        awk '/^Z15_DDR_PASS/ { n++; sub(/bytes=/, "", $3); bytes += $3 }
             END { exit !(n == 6 && bytes == 176400) }' "$archive/m1/replay.log"; then
        printf 'YOLO byte gate PASS\n'
    else
        printf 'YOLO byte gate FAIL (see replay.log)\n'
    fi
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
    variant=c12-control
    if [[ -s $root/qualified-image ]]; then
        read -r weights <"$root/qualified-image"
        [[ $(realpath -e -- "$weights") == "$archive"/m2/.models-cache/* ]]
        variant=block16-qualified
    fi
    printf 'Qwen RTL variant=%s\n' "$variant"
    for layer in 0 12 23; do
        decode=0
        head=0
        if (( layer == 0 )); then decode=2; fi
        if (( layer == 23 )); then head=1; fi
        wait_memory "qwen_layer_$layer"
        if (ulimit -v 2097152; "${priority[@]}" "$app/c12_qwen_rtl" "$weights" "$ids" 8 "$layer" \
            "$decode" "$head" p17) >"$archive/m2/layer-$layer.log" 2>&1; then
            printf 'Qwen layer %s PASS\n' "$layer"
        else
            printf 'Qwen layer %s FAIL (see layer-%s.log)\n' "$layer" "$layer"
        fi
    done
fi

if [[ -n "${3:-}" ]]; then
    cd "$work"
    export PYTHONPATH="$work" PYTHONDONTWRITEBYTECODE=1
    wait_memory legacy_yhb_build
    "${priority[@]}" make -C "$work/runtime" -f verilator.mk -j"$jobs" OUT="$app" \
        RTL_BUILD="$build/obj" RTL_SRC="$w" SDK="$base/tools/oss-cad-suite" "$app/ysyx_run" \
        >"$archive/m3/old-yhb-build.log" 2>&1
    if ! "${priority[@]}" python3 -B tools/c14_legacy_replay.py "$root" "$root/native/ysyx_run" \
        "$app/ysyx_run" >"$archive/m3/old-yhb.log" 2>&1; then
        printf 'Legacy YHB qualification FAIL (see old-yhb.log)\n'
    fi
    if [[ -s "$root/legacy-yhb/result.json" ]]; then
        cp "$root/legacy-yhb/result.json" "$archive/m3/old-yhb.json"
    fi
    wait_memory old_o4_p8_build
    mkdir -p "$root/npu/p8-src"
    git -C "$base/repos/ysyx-npu-ext" archive 2741540496f6f683843a231361bfa88743d7fabe tiny-gpu/src |
        tar -xf - -C "$root/npu/p8-src"
    "${priority[@]}" verilator --cc --exe --build -j "$jobs" --threads 1 --assert -Wno-fatal \
        --top-module npu_tensor_engine --Mdir "$root/npu/p8-build" -CFLAGS '-std=c++17' \
        "$root/npu/p8-src/tiny-gpu/src/npu_tensor_engine.sv" "$work/tools/npu_p8_rtl.cpp" \
        -o run-npu-p8 >"$archive/m3/old-o4-p8-build.log" 2>&1
    wait_memory old_o4_gpu_build
    mkdir -p "$root/gpu/v6"
    git -C "$base/repos/ysyx-gpu-ext" archive 19852c9a370828fa06aa1cfb90ad5fcd668cad30 tiny-gpu/src |
        tar -xf - -C "$root/gpu/v6"
    (
        cd "$root/gpu/v6/tiny-gpu"
        "${priority[@]}" verilator --cc --exe --build -j "$jobs" --threads 1 --output-split 5000 \
            -Wno-fatal --Mdir "$root/gpu/v6/obj" -o "$root/gpu/v6/sim" --top-module gpu \
            src/rv32_*.sv src/{gpu,dcr,dcache,icache}.sv "$work/tools/gpu_rtl.cpp"
    ) >"$archive/m3/old-o4-gpu-build.log" 2>&1
    "${priority[@]}" python3 -B tools/c14_legacy_replay.py "$root" \
        "$base/artifacts/compiler-phase-c1-20261125-161952/models/legacy" --o4 >"$archive/m3/old-o4.log" 2>&1
fi
