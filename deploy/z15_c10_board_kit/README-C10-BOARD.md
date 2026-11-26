# Z15 C10 板端部署与基准套件

套件包含 C8 YOLO26n 320×320 的常驻运行时、双输入画框服务和 C9 四模型基准。目标平台为已加载兼容 PL 镜像的 Z15 PetaLinux。C10 开发及验收使用离线 C99 参考后端和冻结 Step 2 Verilator 壳。新套件的实板延迟、FPS 和摄像头协商结果需要在实板开放后采集。

## 文件

| 文件 | 用途 |
|---|---|
| bin/z15_yolo_live_c10 | AArch64 服务，接收 TCP RGB 或经 V4L2 采集，执行 YOLO 并回传 Z15F |
| bin/z15_c10_benchmark | AArch64 四模型推理与字节校验，含底层自检 |
| models/ | 四个 YHB1 模型、输入与黄金张量、坏 CRC 和非法 DMA 拒收样本 |
| run_z15_c10_bench.sh | 板端自检、逐模型基准、稳态采样与 CSV 汇总 |
| sim_baselines.csv | C9 归档中与两档时钟配比一致的冻结 RTL 数据 |
| sim_vs_board_comparison.csv | 四模型、两档时钟的 8 行对照模板，实测前 Board 状态为 not_run |
| Deploy-Z15C10.ps1、View-Z15C10.ps1、pc_yolo_viewer_c10.py | 部署、启动服务、回收结果，以及画框与遥测显示 |
| SHA256SUMS.txt | 解压后文件校验清单 |

## 部署前检查

确认板卡可用、网络可达，并有可执行 AArch64 ELF 的用户空间。板端服务通过 UIO 独占控制窗和 DDR 池。运行时核验 ABI v1 MAGIC、版本与就绪状态。基准另检查 FPGA Manager 为 operating、SCRATCH 与 ERROR，并用拒收样本校验 CRC 和 DMA 描述符门禁。运行前结束已占用同一 UIO 控制窗的视觉服务。

建议在确认时钟配比后显式指定 125_100 或 200_150。自动模式采样 WALL 计数器辨认原生域频率，并按已知配比推定 GPU 域频率。ABI v1 空闲时没有独立 GPU 时钟读数。服务和板端脚本均沿用已就绪的 PL 镜像。

## Windows 一键运行

板卡开放后，在套件目录的 PowerShell 里将示例主机地址改成实际 SSH 主机。部署脚本在板端 /tmp 新建本次目录，校验上传文件并回收 CSV 到套件内 results/。输入模式命令会先跑基准，再启动服务和查看器。用 -Plan 可离线检查参数和流程。

```powershell
Set-Location 'D:/VMshare/z15-c10-board-kit'
$Board = 'root@z15'
.\Deploy-Z15C10.ps1 -BoardHost $Board -Action Bench -FreqProfile auto
.\Deploy-Z15C10.ps1 -BoardHost $Board -Action Tcp -Feed 'C:/path/to/bus.jpg'
.\Deploy-Z15C10.ps1 -BoardHost $Board -Action V4l2 -Camera /dev/video0 -Width 640 -Height 480 -Fps 30
```

如服务已经启动，可用 View-Z15C10.ps1 只连接画框窗口。服务监听 TCP 端口时，请使用受信任的板卡网段。

## 输入模式

- TCP 模式在 9001 端口接收 Step 4.5 Z15I，9000 端口发送原有长度的 Z15F 帧头、图像和检测框。检测框按输入图像尺寸归一化。增强查看器能解析 Z15F 模型名尾部的 C10 扩展。旧查看器仍可读取检测画面。
- V4L2 模式默认请求 /dev/video0 的 640×480、30 FPS，根据协商结果选择 YUYV、RGB24 或基线 MJPEG。V4L2 使用 mmap 缓冲队列，服务只依赖板端基础动态库。Windows 查看器使用本机 Python、NumPy 和 OpenCV，显示中文标签时另用 Pillow 和中文字体。
- 板端 --live 为常规推理模式。--verify --golden DIR 另外核对每帧的六个原始检测头，期望文件名为 DIR/序号.heads.bin，每份 176400 字节。黄金张量按对应的精确输入生成。
- 现场启用任何 UI 或摄像头前先用板端基准完成固定输入核验。部署与运行期间只让一个进程持有 UIO 设备。

## 计时口径

服务 CSV 每帧记录采集及预处理、主机至 DDR、PL WALL、回读及后处理、发送和端到端的毫秒数。Z15F 扩展中的上一帧探针包含已经完成的发送耗时，第一帧的上一帧信息留空。在参考后端，硬件拍数留空。

PL WALL 由 ABI v1 的硬件计数器差值换算，覆盖发起和等待命令期间的空档。主机单调时钟另记录 execute 与端到端时间。基准的 ps_outside_exec_ms 只统计 execute 区间以外的主机工作，MMIO 提交、轮询和传输同步中的一部分开销仍计入 execute。两类时间可能交叠。Board/Sim 周期比较使用 WALL 计数器，FPS 和 tok/s 使用主机端到端时间。

冻结 ABI 的 NPU_CYCLES 合计 NPU 与 VPU。实板 CSV 的单独 NPU/VPU 忙拍、DMA 忙拍及独立 SLVERR 列留空。可直接记录组合忙拍、GPU 忙拍、计算重叠、DMA 字节数与描述符数、ABI ERROR 及非法 DMA。仿真 BFM 检查 MMIO 响应。其 host 时间只反映服务器执行仿真的开销。

## 常驻与复现

YOLO26n 在首帧上传 2,591,424 B 静态 DDR 数据和 409,600 B 图像输入。后续帧只上传图像输入。YHB 程序在 PS 进程内保留，GPU 阶段仍遵循冻结 ABI 的 16 KiB 指令窗，NPU/VPU 阶段配置由 MMIO 提交。会话在发现 ABI 或 I/O 故障后拒绝继续执行。TinyLM 用现有连续 KV 检查点接口复用一次执行中的权重和 KV。

基准使用 C8 两张不同真图、TinyLM 的三条 prompt（合计 27 个输出 token）、ResNet-18 和 MobileNetV2 各 1,000 B logits。YOLO 首帧独立记录，其后默认 10 帧计入稳态统计。板端基准用与 C9 对照一致的 bus 输入测速，套件同时提供 zidane，供跨帧校验。仿真 CSV 中保留 C9 归档来源，板端执行后在独立结果目录生成 board_c10_summary.csv 和填入实测值的 sim_vs_board_comparison.csv。
