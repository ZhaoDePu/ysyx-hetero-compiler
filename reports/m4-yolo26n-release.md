# M4 Z15 推理套件与回归验收

官方 320×320 YOLO26n 的 C99 接口、RGB／P6 CLI 和跨架构静态库已交付。
x86-64 GCC 和 Xilinx 2019.1 AArch64 GCC 均以 `-std=c99 -O2 -Wall -Wextra -Werror` 编译。
AArch64 ELF 仅依赖 `libc.so.6`。从归档源码重新编译得到相同的 AArch64 CLI。

- 独立 C99 定点运行时及 CLI 在四张真图上与定点黄金逐字节一致。
- 同一 `.yhb` 在冻结 Step 2 100:125 时钟比下整帧通过，ERROR、SLVERR、非法 DMA 均为 0。
- 新发布包的 CRC 篡改、修复 CRC 后跨半区 DMA 与 DDR 回绕描述符均被拒收。
- 软件单测 118/118，C1–C5 历史回放通过，C5 原 15 组 YHB 全部逐字节一致，30 次历史 CRC 篡改全部拒收。
- 冻结 Step 2 和 Step 3-v2 的核验文件哈希匹配。对外部工作区仅执行只读检查，未启动 Vivado 或 Yosys。
- 远端 `/dev/shm/ysyx_compiler_c6` 和 Windows C6 导出临时目录已清空。

部署套件位于
`xu4090:/home/zdp/ysyx-migration/artifacts/compiler-phase-c6-yolo26n-20261126-013759/z15_yolo26n_320_kit.tar.gz`。
SHA-256 为 `192953c5447c44d794b5ec1d19528352332561497cf460573012a3ae3b2de301`。
Windows `D:/VMshare/z15-yolo26n-320-kit/` 含 tar.gz、README 和 SHA256SUMS，三文件总计 3,054,296 B。
