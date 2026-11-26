# C10 YOLO26n 实时画框

服务端每次启动创建一个 C8 推理会话。模型与静态 DDR 数据在后续帧以及 TCP 客户端重连时继续复用。以下命令供套件交付后的板端操作。

## TCP 喂图与 USB 直采

TCP 喂图使用 Step 4.5 的端口。Windows 查看器先连接 9000 再连接 9001。相机模式只监听 9000。

    ./z15_yolo_live_c10 --model yolo26n_320_int8.yhb --source tcp --freq-profile 125_100 --live --csv c10_live.csv

    ./z15_yolo_live_c10 --model yolo26n_320_int8.yhb --source v4l2 --device /dev/video0 --width 640 --height 480 --fps 30 --freq-profile 125_100 --live --csv c10_live.csv

V4L2 使用单平面 mmap 队列，依次请求 YUYV、RGB24 和基线 MJPEG。日志记录驱动实际协商的分辨率和帧间隔。推理慢于摄像头时优先取最新已完成帧。YUYV 按 BT.601 有限范围转换。MJPEG 支持基线编码以及常见的省略标准 Huffman 表的 UVC 帧。缓冲区异常或解码失败会在日志显示。

Windows 需要 Python、NumPy、OpenCV，中文标签需要 Pillow 和中文字体。将 IP 替换成已就绪的板端地址。TCP 模式下可用文件或 Windows 摄像头索引作为输入，V4L2 模式下省略 Feed。

    .\View-Z15C10.ps1 -BoardHost <IP> -Feed C:\images\bus.jpg
    .\View-Z15C10.ps1 -BoardHost <IP> -Feed 0
    .\View-Z15C10.ps1 -BoardHost <IP>

## 验证与计时

- Live 默认置信度阈值为 0.40。验证使用 `--verify --golden DIRECTORY`，逐帧比对 `DIRECTORY/1.heads.bin`、`DIRECTORY/2.heads.bin` 等。文件按 8、16、32 步幅的 boxes/scores 顺序拼接，每帧 176,400 字节。遇到不一致的字节会发送故障帧，记录首个偏差位置，并用非零状态退出。可用 `--frames N` 设置帧数。
- CSV 包含采集与预处理、H2D、PL、D2H 与后处理、TCP 发送、端到端时间，以及常驻权重和动态输入字节数。TCP 发送指把数据提交到本机套接字的时间。端到端从开始取输入到发送结束。CSV 文件写入与 Windows 解码显示在这个区间之外。
- `t_pl_hw_ms` 以硬件 WALL 拍数除以选择的 native 时钟提取，包含其计数窗口内 PS 发令间隙。`t_execute_ms` 是主机观察到的提交与等待区间，两者有重合。查看器的前一帧 PS 时间按该帧的端到端时间减去 PL WALL 时间计算。
- Step 2 ABI 提供 `native_npu_vpu_busy` 合并计数、GPU、WALL 和重叠。独立 `npu_busy`、`vpu_busy`、DMA 忙拍在来源不可用时留空。C99 定点软件参考也将缺失的硬件字段留空。
- 离线可通过 `--sim --bind 127.0.0.1` 使用定点参考。`--source v4l2 --recorded-yuyv FILE --width W --height H` 读取串接的紧密打包 YUYV 帧，共用色彩转换与 Letterbox。USB 实机协商需在板端验证。

## Z15F 兼容性

TCP 输入保留 16 字节小端 Z15I RGB24 或 Z15J 基线 JPEG 帧头。Z15F v1 保留原 152 字节帧头、图像与 24 字节检测框记录。TCP 模式发送原尺寸画面和归一化到原图的坐标，V4L2 模式发送 320×320 Letterbox 彩图和归一化到该画面的坐标。

增强遥测放在原 64 字节模型名字段的 NUL 之后，偏移 104。旧查看器读取到第一个 NUL 即结束。`C10T` v1 的小端布局为 `<4sQ6I2H2B`，依次包含四字节标记、前一帧序号、采集预处理/H2D/PL/D2H 后处理/TCP 发送/端到端的整数微秒、配置的两个时钟 MHz、模式位和版本。模式位从低到高为 verify、V4L2、sim、recorded YUYV。微秒值 0xffffffff 表示无值。发送时间在发送完成后可测，因此扩展携带上一帧完整遥测，首帧扩展序号为 0。每帧 CSV 保留完整记录。
