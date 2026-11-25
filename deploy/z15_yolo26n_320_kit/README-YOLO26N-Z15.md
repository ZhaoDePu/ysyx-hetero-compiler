# Z15 Step 2 官方 YOLO26n 320×320 推理套件

## 在 PetaLinux 上运行

启动已加载冻结 Step 2 Balanced PL 的 Z15，并确认控制 UIO 和非缓存 DMA UIO 已就绪。程序通过 sysfs 识别窗口，控制地址为 0x80000000，保留 DDR 池为 0x40000000–0x5FFFFFFF。运行时独占控制 UIO，设备忙或 ABI 不匹配时在 DMA 之前退出。

    bin/aarch64/ysyx_yolo26n_cli models/yolo26n_320_int8.yhb --ppm samples/bus.ppm --conf 0.4

如果需要指定设备，可为这个进程设置 `YSYX_CONTROL_UIO` 和 `YSYX_DMA_UIO`。它们接受系统中的 UIO 名称或 `/dev/uioN`。`--rgb` 接受 307,200 字节的紧密 RGB 数据，`--ppm` 接受 320×320、最大值 255 的 P6 文件。CLI 输出检测框、COCO 类别号、置信度和性能计数。类别号可在 `models/coco80.json` 中查询。

## 集成摄像头服务

在服务编译时包含 `include/ysyx_yolo26n_api.h` 并链接 `bin/aarch64/libysyx_yolo26n.a`。单进程调用 `ysyx_yolo26n_open` 一次，逐帧调用 `ysyx_yolo26n_infer_rgb320`，退出时调用 `ysyx_yolo26n_close`。每个上下文由一个线程使用。

调用方以 RGB 顺序提供 320×320 图像。要与样例一致，请先对原图保比例双线性缩放并居中填充 RGB 114。API 输出的 xyxy 位于 320×320 网络坐标系，逆变换时将坐标减去填充并除以缩放倍数，再裁剪至原图。末端两级 TopK 与坐标计算在 PS 上执行，模型不使用 NMS。

`io_and_compute_ms` 是本次调用包含 RGB 量化、输入与结果搬运、PL 运行及 PS 解码的单调时钟耗时。NPU、GPU 和重叠计数以 NPU 域为单位。按 125 和 164.555 MHz 的列是芯片计数折算，板端服务 FPS 可根据连续调用的实际耗时测量。

## x86-64 离线验证

    sha256sum -c SHA256SUMS.txt
    bin/x86-64/ysyx_yolo26n_cli models/yolo26n_320_int8.yhb --ppm samples/bus.ppm --sim --conf 0.4

本包的 `--sim` 使用 C99 定点参考，通过相同 YHB1 加载与调度路径回放。JSON 中 `hardware_counters_valid` 为 false，硬件计数为 0。源码可使用工具链重编译。

    make -C runtime CC=aarch64-linux-gnu-gcc AR=aarch64-linux-gnu-ar OUT="$PWD/build"

`samples/accuracy.json` 记录四张真实场景的 FP32 与定点匹配结果。`z15-mmio-abi-v1.md` 来自冻结 Step 2 交付。源代码的编译参数为 `-std=c99 -O2 -Wall -Wextra -Werror`。
