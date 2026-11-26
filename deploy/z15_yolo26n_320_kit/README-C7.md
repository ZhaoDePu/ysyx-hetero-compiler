# YOLO26n 320×320 INT8 Phase C7

C7 保留冻结 Z15 Step 2 的 YHB1 和 z15-mmio-abi-v1。头文件和 API 签名与 C6 完全一致。部署包的模型、CLI 和静态库保持原有相对路径，可分别更新板端现有进程使用的文件。

    bin/aarch64/ysyx_yolo26n_cli models/yolo26n_320_int8.yhb --ppm samples/bus.ppm --conf 0.4

C7 将深 K 卷积的宽 Requant 和两处注意力 Softmax 交由 NPU／VPU。四张样例的精度账本为 samples/accuracy.json。相同输入、设备选择、坐标和计数器口径见 [C99 使用说明](README-YOLO26N-Z15.md)。

在 x86-64 Linux 可先执行以下命令校验归档和四图之一。软件参考模式不产生硬件周期计数。

    sha256sum -c SHA256SUMS.txt
    bin/x86-64/ysyx_yolo26n_cli models/yolo26n_320_int8.yhb --ppm samples/bus.ppm --sim --conf 0.4
