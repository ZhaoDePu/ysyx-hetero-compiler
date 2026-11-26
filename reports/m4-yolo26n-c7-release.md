# Phase C7 发布验收

C7 复用冻结 Z15 Step 2 提交 4d410d7、z15-mmio-abi-v1 与 YHB1。编译器未改 RTL，C99 公开头文件与 C6 逐字节一致，原 15 组部署套件继续使用原格式。

- 132 项软件单测全过，覆盖 C1–C6 原 118 项。C1 GPU 42/42、原生 NPU/VPU 13/13、旧 O4 13/13、C2 消融 12/12、C2 C99 8/8、C3 流式 8/8、训练模型 32/32、Q/DQ Softmax 5/5、C4-VPU 下沉 20/20。C4-A、C5 与 ASIC 的具体回放计数和原始日志见归档的 m4/regression/。
- 原 15 组 Z15 YHB 重新回放全数逐字节通过，CRC 篡改 30/30、非法 DMA 20/20 均被拒绝。C6 原 128×128 和 320×320 包的四图 C99 定点回放各 4/4 通过。C6 无 C7 开关的重新编译产物与旧版本逐字节一致。
- C99 严格编译通过，x86-64 和 AArch64 的静态库及 CLI 均重新构建。AArch64 可执行文件的 ELF 动态依赖只有 libc.so.6，部署包内源码重编译结果与包内产物一致。
- 主发布模型的 SHA-256 为 c21fff68a06aec7dec8920ed99bce87162a1c705b81d74d5bcefcdd4f72517fa。其四档冻结 RTL 回放均通过六个检测头 176,400 B 逐字节校验。125/100 MHz 口径下，17,785,743 拍对应 142.286 ms／帧、7.028 FPS。实板 Linux 端到端耗时尚待测量。
- Windows 新套件目录为 D:/VMshare/z15-yolo26n-320-kit-c7/，总计 3,070,403 B。已校验新文件的 SHA-256 及原 C6 压缩包哈希。C7 压缩包 z15_yolo26n_320_kit_c7.tar.gz 的 SHA-256 为 dff810184c99049853574c907d24c4ca410567f4f76155a2773e57af184625a1。

远端归档位于 /home/zdp/ysyx-migration/artifacts/compiler-phase-c7-yolo26n-fast-20261126-140127/，包含真图精度、逐阶段时间、旧包回放、包重编和冻结来源校验。顶层 SHA256SUMS.txt 用于校验归档内文件。
