# Z15 Balanced deployment kit

Copy this directory to the PetaLinux build or target machine. It contains
the complete C99 loader and Linux UIO backend in `source/`, plus Balanced
YHB1 programs, little-endian input tensors, expected output bytes and
bindings in `cases/`. Each case supplies explicit inputs. The default first
case is a short mixed NPU/GPU smoke.

Build on x86-64 to check compiler portability:

```sh
make -j4 CC=gcc AR=ar
```

With the PetaLinux cross toolchain configured:

```sh
make -j4 CC=aarch64-xilinx-linux-gcc AR=aarch64-xilinx-linux-ar OUT=build-aarch64
file build-aarch64/ysyx_run
readelf -h build-aarch64/ysyx_run
```

Deploy the executable with the unchanged kit cases. On the board, boot the
matching frozen Balanced PL and PetaLinux package. The Linux HAL needs the
64 KiB control UIO at physical `0x80000000` and the noncached reserved
512 MiB DDR pool starting at `0x40000000`. The loader validates their UIO
map metadata and the frozen hardware magic and version.

```sh
YSYX_RUN=./build-aarch64/ysyx_run ./run_z15_bench.sh 00-mixed
YSYX_RUN=./build-aarch64/ysyx_run ./run_z15_bench.sh
```

The script can use auto-discovery or `YSYX_CONTROL_UIO` and `YSYX_DMA_UIO`
with the UIO name or exact `/dev/uioN` path. Run as an account with access
to those UIO devices. Set `YSYX_LOG_DIR` to keep raw logs. Each case checks
the YHB CRC, compares all output bytes and prints monotonic host elapsed time
and item throughput. This clock covers runtime execution and output writes,
including requested statistics, after opening the backend. Hardware counter
values are printed separately. The source build uses
`-std=c99 -O2 -Wall -Wextra -Werror`.

The same script accepts `YSYX_BACKEND=verilator` with the compiler repository's
C99/Verilator executable. Reports identify the backend. Verilator host
throughput and simulated NPU ticks describe the simulation.
