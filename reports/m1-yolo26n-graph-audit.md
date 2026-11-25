# M1 官方 YOLO26n ONNX 图审计

审计输入 `yolo26n_320.onnx`，ONNX SHA-256 为 `1f4ece0b17fc3094c5c1684718808036ed7cd5e166e72638a0ceceaef3bb692a`。

官方预训练权重 `yolo26n.pt` 的 SHA-256 为 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。

逐节点记录在 [m1-yolo26n-graph-nodes.csv](m1-yolo26n-graph-nodes.csv)，逐张量记录在 [m1-yolo26n-graph-tensors.csv](m1-yolo26n-graph-tensors.csv)。节点从 0 编号，输入在节点 −1 进入，最终输出存活至图末端。

## 输入、输出与算子清单

ONNX IR 版本 `9`。模型生产者 `pytorch 2.12.0`。算子集 {"ai.onnx": 19}。

| 角色 | 张量 | 形状 | 数据类型 |
| --- | --- | --- | --- |
| 输入 | images | [1, 3, 320, 320] | float32 |
| 输出 | output0 | [1, 300, 6] | float32 |

整图 384 个节点、23 种 OpType、629 个张量。

| ONNX OpType | 节点数 |
| --- | --- |
| Add | 21 |
| Cast | 1 |
| Concat | 24 |
| Conv | 102 |
| Div | 1 |
| Expand | 2 |
| Flatten | 1 |
| GatherElements | 3 |
| MatMul | 4 |
| MaxPool | 3 |
| Mod | 1 |
| Mul | 90 |
| ReduceMax | 1 |
| Reshape | 12 |
| Resize | 2 |
| Sigmoid | 88 |
| Slice | 2 |
| Softmax | 2 |
| Split | 12 |
| Sub | 1 |
| TopK | 2 |
| Transpose | 5 |
| Unsqueeze | 4 |

## 卷积规格

| 卷积类别 | Kernel | Stride | Padding | Dilation | Groups | 节点数 |
| --- | --- | --- | --- | --- | --- | --- |
| depthwise | [3, 3] | [1, 1] | [1, 1, 1, 1] | [1, 1] | 64 | 1 |
| depthwise | [3, 3] | [1, 1] | [1, 1, 1, 1] | [1, 1] | 80 | 3 |
| depthwise | [3, 3] | [1, 1] | [1, 1, 1, 1] | [1, 1] | 128 | 3 |
| depthwise | [3, 3] | [1, 1] | [1, 1, 1, 1] | [1, 1] | 256 | 1 |
| ordinary | [3, 3] | [1, 1] | [1, 1, 1, 1] | [1, 1] | 1 | 32 |
| ordinary | [3, 3] | [2, 2] | [1, 1, 1, 1] | [1, 1] | 1 | 7 |
| pointwise | [1, 1] | [1, 1] | [0, 0, 0, 0] | [1, 1] | 1 | 55 |

类别由权重形状和分组推导。普通与逐点卷积的 groups 为 1，深度卷积的 groups 等于输入通道数。

## 连接与计算结构

| 导出节点作用域 | 可见结构 | 该作用域节点数 |
| --- | --- | --- |
| model.2 | 1 处等形残差候选，Split×1、Concat×1 | 15 |
| model.4 | 1 处等形残差候选，Split×1、Concat×1 | 15 |
| model.6 | 2 处等形残差候选，Split×1、Concat×2 | 32 |
| model.8 | 2 处等形残差候选，Split×1、Concat×2 | 32 |
| model.9 | SPPF 池化链（MaxPool×3），1 处等形残差候选，Split×0、Concat×1 | 9 |
| model.10 | 自注意力（QK、Softmax、PV），3 处等形残差候选，Split×2、Concat×1 | 28 |
| model.11 | 上采样或重采样 | 1 |
| model.12 | Split×0、Concat×1 | 1 |
| model.13 | 2 处等形残差候选，Split×1、Concat×2 | 32 |
| model.14 | 上采样或重采样 | 1 |
| model.15 | Split×0、Concat×1 | 1 |
| model.16 | 2 处等形残差候选，Split×1、Concat×2 | 32 |
| model.23 | Detect（TopK×2、GatherElements×3），Split×1、Concat×5 | 95 |
| model.18 | Split×0、Concat×1 | 1 |
| model.19 | 2 处等形残差候选，Split×1、Concat×2 | 32 |
| model.21 | Split×0、Concat×1 | 1 |
| model.22 | 自注意力（QK、Softmax、PV），4 处等形残差候选，Split×2、Concat×1 | 35 |

| 节点 | 算子 | 主输入 | 主输出 | 关键属性 |
| --- | --- | --- | --- | --- |
| /model.9/m/MaxPool | MaxPool | [1, 128, 10, 10] | [1, 128, 10, 10] | ceil_mode=0，dilations=[1, 1]，kernel_shape=[5, 5]，pads=[2, 2, 2, 2]，strides=[1, 1] |
| /model.9/m_1/MaxPool | MaxPool | [1, 128, 10, 10] | [1, 128, 10, 10] | ceil_mode=0，dilations=[1, 1]，kernel_shape=[5, 5]，pads=[2, 2, 2, 2]，strides=[1, 1] |
| /model.9/m_2/MaxPool | MaxPool | [1, 128, 10, 10] | [1, 128, 10, 10] | ceil_mode=0，dilations=[1, 1]，kernel_shape=[5, 5]，pads=[2, 2, 2, 2]，strides=[1, 1] |
| /model.10/m/m.0/attn/MatMul | MatMul | [1, 2, 100, 32] | [1, 2, 100, 100] | — |
| /model.10/m/m.0/attn/Softmax | Softmax | [1, 2, 100, 100] | [1, 2, 100, 100] | axis=-1 |
| /model.10/m/m.0/attn/MatMul_1 | MatMul | [1, 2, 64, 100] | [1, 2, 64, 100] | — |
| /model.11/Resize | Resize | [1, 256, 10, 10] | [1, 256, 20, 20] | coordinate_transformation_mode=asymmetric，mode=nearest，nearest_mode=floor |
| /model.14/Resize | Resize | [1, 128, 20, 20] | [1, 128, 40, 40] | coordinate_transformation_mode=asymmetric，mode=nearest，nearest_mode=floor |
| /model.22/m.0/m.0.1/attn/MatMul | MatMul | [1, 2, 100, 32] | [1, 2, 100, 100] | — |
| /model.22/m.0/m.0.1/attn/Softmax | Softmax | [1, 2, 100, 100] | [1, 2, 100, 100] | axis=-1 |
| /model.22/m.0/m.0.1/attn/MatMul_1 | MatMul | [1, 2, 64, 100] | [1, 2, 64, 100] | — |
| /model.23/TopK | TopK | [1, 2100] | [1, 300] | axis=1，largest=1，sorted=1 |
| /model.23/TopK_1 | TopK | [1, 24000] | [1, 300] | axis=1，largest=1，sorted=1 |

ONNX 图中未出现 `NonMaxSuppression` 节点，存在 0 个显式 `BatchNormalization` 节点。检测到 87 处 `Sigmoid(x) × x` SiLU 模式。权重和张量形状导出的逻辑运算量为 Conv 669,678,400 MAC、MatMul 3,840,000 MAC。卷积 MAC 计入边缘补零，这些数值用于核对后续映射，实际硬件指令与周期由编译和 RTL 测量给出。

## 活跃张量与两级存储压力

ONNX 初始化张量共 2,419,477 个元素、9,678,068 B。常量按浮点转一字节、其余宽度不变的假设估算为 2,419,757 B。

逻辑活跃量统计包含消费节点还未释放的输入和同一节点刚产生的输出，不计入常量区、重排、对齐、宽累加或硬件内部缓冲。INT8 列仅将浮点激活按每元素 1 B 估算，保留整数索引的宽度。实际量化位宽和实板缓冲由 M2/M3 决定。

| 逻辑表示 | 峰值活跃 B | 峰值节点 | 超过 1 MiB 半区 | 超过 2 MiB 片上容量 |
| --- | --- | --- | --- | --- |
| 导出 ONNX | 4,915,200 | 2 | 9 | 0 |
| INT8 容量敏感度 | 1,228,800 | 2 | 0 | 0 |

FP32 单张量上限与峰值活跃量是两个容量检查。INT8 最大片上需求还受一次计算中的输入权重、输出暂存、宽累加和双半区分配影响。

INT8 假设下有 235 个跨层活跃张量（存在多次消费或跨过中间节点），建议结合其拓扑编号评估 DDR 持久化。超出半区的完整张量需要分块、跨半区安排或外存暂存。CSV 提供全部张量的首末节点、消费边、两种容量分类和 DDR 候选标记，供两级规划使用。

| 最大活跃张量（前八） | 形状 | ONNX B | INT8 估算 B | 活跃期 |
| --- | --- | --- | --- | --- |
| /model.0/act/Mul_output_0 | [1, 16, 160, 160] | 1,638,400 | 409,600 | 2–3 |
| /model.0/act/Sigmoid_output_0 | [1, 16, 160, 160] | 1,638,400 | 409,600 | 1–2 |
| /model.0/conv/Conv_output_0 | [1, 16, 160, 160] | 1,638,400 | 409,600 | 0–2 |
| /model.15/Concat_output_0 | [1, 256, 40, 40] | 1,638,400 | 409,600 | 181–182 |
| /model.2/cv2/act/Mul_output_0 | [1, 64, 80, 80] | 1,638,400 | 409,600 | 20–21 |
| /model.2/cv2/act/Sigmoid_output_0 | [1, 64, 80, 80] | 1,638,400 | 409,600 | 19–20 |
| /model.2/cv2/conv/Conv_output_0 | [1, 64, 80, 80] | 1,638,400 | 409,600 | 18–20 |
| /model.2/Concat_output_0 | [1, 48, 80, 80] | 1,228,800 | 307,200 | 17–18 |

| 跨作用域激活（按 INT8 大小前八） | 产出 | 消费 | INT8 估算 B | 活跃期 |
| --- | --- | --- | --- | --- |
| /model.0/act/Mul_output_0 | model.0 | model.1 | 409,600 | 2–3 |
| /model.15/Concat_output_0 | model.15 | model.16 | 409,600 | 181–182 |
| /model.2/cv2/act/Mul_output_0 | model.2 | model.3 | 409,600 | 20–21 |
| /model.1/act/Mul_output_0 | model.1 | model.2 | 204,800 | 5–6 |
| /model.14/Resize_output_0 | model.14 | model.15 | 204,800 | 180–181 |
| /model.4/cv2/act/Mul_output_0 | model.4 | model.5、model.15 | 204,800 | 38–181 |
| /model.12/Concat_output_0 | model.12 | model.13 | 153,600 | 147–148 |
| /model.11/Resize_output_0 | model.11 | model.12 | 102,400 | 146–147 |

按图顺序逐层执行且每层只在自己的半区保留输入和输出时，跨作用域的激活进入 DDR 工作区。融合与跨层常驻可缩减这一集合。DMA 总量、驻留 DDR 工作区与片上峰值需按最终物理分块、权重布局、外存复用和指令调度核验。
