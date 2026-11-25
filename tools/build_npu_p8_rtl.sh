#!/usr/bin/env bash
set -euo pipefail
work=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
hardware=/home/zdp/ysyx-migration/repos/ysyx-npu-ext
commit=2741540496f6f683843a231361bfa88743d7fabe
scratch=/dev/shm/ysyx_compiler_c4vpu/npu
[[ "$(realpath -e -- "$scratch")" == "$scratch" && -O "$scratch" ]]
mkdir -p -- "$scratch/p8-src"
git -C "$hardware" archive "$commit" tiny-gpu/src | tar -xf - -C "$scratch/p8-src"
printf 'source_commit=%s\n' "$commit"
cd "$scratch/p8-src/tiny-gpu"
export TMPDIR="$scratch" CCACHE_DISABLE=1 OMP_NUM_THREADS=1
exec nice -n 10 ionice -c2 -n7 verilator --cc --exe --build -j 4 --assert -Wno-fatal \
  --top-module npu_tensor_engine \
  --Mdir "$scratch/p8-build" -CFLAGS '-std=c++17' \
  src/npu_tensor_engine.sv "$work/tools/npu_p8_rtl.cpp" -o run-npu-p8
