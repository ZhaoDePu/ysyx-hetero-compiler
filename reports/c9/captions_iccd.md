# ICCD figure and table captions / 图表说明

- **Fig. C1 — Ablation and latency.** Bars show NPU-domain RTL cycles converted at the stated clock pair. Labels give same-workload speedup over C6 or A0. Classifier A0 uses complete isolated operator replays grouped by execution geometry; hatched bars mark their multiplicity-weighted aggregate. The dagger marks speedups relative to such an aggregate. The real small-subgraph replays independently check operator composition.
  **图 C1 — 消融与延迟。** 根据对应实测时钟比的 NPU 域拍数换算延迟，柱上加速比以同模型 C6 或 A0 为基线。分类器 A0 按执行几何测完整代表层，以出现次数聚合，斜线区分该口径。加速比后的 † 表示分母来自逐层聚合。小子图整段回放单独验证算子串接。

- **Fig. C2 — Workload overlap and utilization.** The left panel partitions wall time into disjoint classes. Compute and DMA overlap includes the triple-overlap interval when present. The right panel shows busy fractions of engines against wall time, so fractions can sum above unity. Fused attention is a native tensor command with internal vector work; its entire busy interval is not double counted.
  **图 C2 — 重叠与利用率。** 左图互斥分解整段时间。计算与 DMA 重叠包含三方同忙。右图将各引擎忙拍除以整段拍数，比例之和可以超过整段。融合 Attention 按原生命令计数，其内部向量工作不重复计入。

- **Fig. C3 — Memory and command traffic.** The baseline is C6 for detection and A1 for the other workloads. We compare simultaneous SRAM high-water marks, DDR DMA bytes and submitted DMA descriptors. These quantities do not include Linux preprocessing or CPU-side memory copies.
  **图 C3 — 显存与命令流量。** 检测基线为 C6，其余基线为 A1。比较 SRAM 同时活跃峰值、DDR DMA 字节和发射描述符，不含 Linux 预处理或 CPU 拷贝。

- **Fig. C4 — Empirical roofline and Phase 16 opportunities.** Operational intensity uses two logical ops per MAC divided by recorded DDR DMA bytes. Circle and square markers refer to the lower and higher measured clock pairs, respectively. Each point uses a verified whole-frame or prompt-suite run. The NPU memory-compute reference curves use the frozen array peak and either one full DMA beat per NPU cycle or the best observed DMA busy-window bandwidth. The curves are a common NPU reference for heterogeneous graphs. The codesign bars bound removable busy time for two hardware features while holding the other measured services fixed. They do not add across features. Treat the entire mixed attention interval as an upper bound when a VPU-only event split is unavailable.
  **图 C4 — 经验 Roofline 与 Phase 16 改进。** 算术强度为每 MAC 按两个逻辑运算除以实测 DDR DMA 字节。模型点来自整帧或完整提示序列回放。Roofline 分别使用冻结阵列峰值加理想满拍 DMA，或加实测的最高 DMA 忙窗口带宽。右图给出保持其他实测工作量不变时，两项硬件改进可消除的忙拍上界。这些上界不能直接相加。融合 Attention 缺少内部事件时，整个命令忙窗口只作为上界。

- **Table I — Benchmarks.** The two classifiers use official ImageNet weights and a downsampled held-out natural image. This is a hardware workload check rather than an ImageNet accuracy study. TinyLM rates count generated tokens over the full prefill-plus-decode session. Clocks refer to ratio-matched frozen Step 2 simulations and do not include board-side host latency.
  **表 I — 基准。** 两款分类器使用官方 ImageNet 权重和下采样的独立真实图像。对应硬件负载校验，不用于报告 ImageNet 准确率。TinyLM 的速率以完整 Prefill＋Decode 窗口产生的 token 计。时钟为冻结 Step 2 的比例匹配仿真，不含板端主机时延。

- **Table II — Ablations.** End-to-end and layer-aggregate measurements use separate evidence categories. A0 layer geometry includes input/output shape, physical layout, operation attributes and quantized-tile role. Representative layer counts in the sweep are multiplied by the corresponding topology multiplicity. This method preserves the complete representative-layer measurements and does not estimate a layer from a smaller tile.
  **表 II — 消融。** 区分端到端与逐层聚合。A0 的分组依据包括输入输出形状、物理布局、算子属性及量化路径。每组用完整代表层实测乘以拓扑中的出现次数，不从缩小 Tile 外推整层。
