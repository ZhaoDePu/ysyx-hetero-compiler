# Qwen2.5-0.5B-Instruct C11B Z15 套件

本套件使用同一份官方权重的 W4A8 导出和同一套 C99 数值契约。Z15 路径在冻结 MMIO ABI v1 上提交 NPU 投影、VPU RMSNorm/注意力/SwiGLU 和 GPU RoPE。主机负责 W4 展开、分组缩放与偏置、残差、输出归一化和贪心选择。151,936 通道共享 LM Head 在 NPU 上按 2,048 通道切分，最后一块为 384 通道。

## 板端使用

在授权使用的 Z15 上，从已就绪的 SD 卡 DATA 分区解压套件。需要兼容 Step 2 或 Step 2-v2 的 ABI v1 PL 镜像、UIO 控制窗和 512 MiB DDR 池。启动脚本先检查 FPGA Manager 为 operating。硬件运行时核验 ABI MAGIC、VERSION、复位就绪状态和 ERROR。套件沿用已加载的 PL 镜像。

在解压后的 z15_qwen25_05b_board_kit 目录运行：

    ./run_z15_qwen25.sh --verify
    ./run_z15_qwen25.sh --prompt '什么是 FPGA？'
    ./run_z15_qwen25.sh --chat

离线参考可以在相同架构 Linux 上运行：

    ./run_z15_qwen25.sh --backend cpu --verify

交互支持 /reset 和 /quit。原生注意力的缓存长度为 128 位置，提示词和生成预算共享这一容量。需要缩短输出时，可直接调用 bin/z15_qwen25_chat 并传入 --max-new-tokens。

## 校验与测量

一键校验比对全部 151,936 个 Q16 logits。vectors 还保留完整 Layer 0 的真实短提示词、两个贪心反馈 token 和相应隐藏态、K/V 快照。历史 YHB 和公开 C99 头文件沿用原版。

PL 只读权重映像为 250.67 MiB。主机把当前投影的 W4 面板展开到固定 16 MiB DDR 区域，DMA 与 NPU 在不相交的 SRAM 半区重叠。K/V 持续存放在属于各层与各组的片上槽位。

版本验收在冻结 RTL 和 CPU 上完成。现有 ABI 给出 WALL、NPU/VPU 合计、GPU 和重叠。运行时在向量命令边界补充分离 VPU，DMA 忙拍来自采用观测器的仿真。报告中的全模型速率是从冻结 RTL 组件合成的 PL 口径，实板还会计入 AArch64 解包、重定标、预处理、MMIO 和缓存同步开销。

三条提示的官方 FP32 对照显示，原生字节域的生成质量明显低于 C11 的 Q16 软件参考。技术回答可能跑题、复读或混入无关字符，本套件供硬件闭环和性能验证使用。完整样本和数值差异见归档报告。
