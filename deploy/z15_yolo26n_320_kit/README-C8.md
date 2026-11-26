# YOLO26n 320×320 Phase C8 部署套件

沿用 C6/C7 的 z15-mmio-abi-v1、YHB1、C99 公开头文件和原有文件布局。建议先核验包内 SHA-256。

    sha256sum -c SHA256SUMS.txt

在 Step 2 或兼容的 Step 2-v2 环境下，运行板端 CLI。以下命令同时给出旧 125 MHz 口径和自选 200 MHz 口径的毫秒/帧与 FPS 折算。保持原命令行即可获得原 JSON 字段。

    bin/aarch64/ysyx_yolo26n_cli models/yolo26n_320_int8.yhb --ppm samples/bus.ppm --conf 0.40 --npu-mhz 200

x86-64 Linux 上可运行软件参考，核对检测结果。软件模式的硬件计数无效，自定义频率折算返回 null。

    bin/x86-64/ysyx_yolo26n_cli models/yolo26n_320_int8.yhb --ppm samples/bus.ppm --conf 0.40 --sim

本包附有原始运行时源码和双架构静态库。板端绑定、输入布局和应用接口见 [C99 使用说明](README-YOLO26N-Z15.md)。C8 冻结 Verilator 数据由发布归档给出，频率折算不含板端 PS 开销。
