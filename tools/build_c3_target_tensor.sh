#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
root=/dev/shm/ysyx_compiler_c3/targets
hardware=/home/zdp/ysyx-migration/repos/ysyx-npu-ext
commit=4d410d72508a5822bc5c8014981e5b55ef8faa56
sdk=/home/zdp/ysyx-migration/tools/oss-cad-suite
test ! -L "$root"
mkdir -m 700 -p "$root/full-src"
[[ "$(realpath -e "$root")" == "$root" && -O "$root" ]]
git -C "$hardware" archive "$commit" tiny-gpu/src/fpga | tar -xf - -C "$root/full-src"
src="$root/full-src/tiny-gpu/src/fpga"
export TMPDIR="$root" CCACHE_DISABLE=1 OMP_NUM_THREADS=1
low=(nice -n 10 ionice -c2 -n7)
for dim in 16 24; do
    build="$root/tensor-$dim"
    test ! -L "$build"
    "${low[@]}" "$sdk/bin/verilator" --cc --exe --build -j 2 -MAKEFLAGS OBJCACHE= --threads 1 \
        --output-split 5000 --assert -Wno-fatal --top-module zu15eg_tensor_wide \
        --Mdir "$build" -GADDR_BITS=21 -GROWS="$dim" -GCOLS="$dim" -GPANEL_K=128 \
        -GWIDE_WORDS=16 -GENABLE_DSP_INT8_PACK2=1 -GHALF_GUARD=1 -GSTAGE_OPERANDS=1 \
        -GMEMORY_READ_LATENCY=2 "$src/zu15eg_tensor_wide.sv" "$src/zu15eg_conv_fetch.sv" \
        "$src/zu15eg_dot_pair.sv" "$src/zu15eg_epilogue_wide.sv" \
        "$repo/tools/c3_target_tensor_rtl.cpp" -o tensor-sim > "$root/build-tensor-$dim.log" 2>&1
done
