# Phase C6 M2：官方 YOLO26n 定点量化

320×320 模型达到精度门禁。双方均取置信度 ≥0.40 时，14 个 FP32 目标中有 13 个匹配框类别一致（92.86%），匹配框平均 IoU 为 0.9601。官方权重 `yolo26n.pt` 的 SHA-256 为 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`，320×320 静态 ONNX 的 SHA-256 为 `1f4ece0b17fc3094c5c1684718808036ed7cd5e166e72638a0ceceaef3bb692a`。

## 真图精度

两侧独立按相同置信度阈值筛选。以源图像裁剪后的坐标计算 IoU，按 IoU 从高到低建立一对一匹配，匹配要求 IoU ≥0.50。类别一致率为类别相同的匹配对数除以 FP32 检出数，未匹配的 FP32 目标计为失败。漏检和额外框均以 FP32 为比较基准。

| 320×320 测试图（conf ≥0.40） | FP32 / INT8 检出 | 类别一致 | 匹配框平均 IoU | INT8 代表目标 |
| --- | ---: | ---: | ---: | --- |
| bus.jpg | 4 / 5 | 4 / 4 | 0.9568 | bus、person |
| COCO 241 | 6 / 8 | 6 / 6 | 0.9643 | person、bottle、cup、couch |
| COCO 384 | 2 / 1 | 1 / 2 | 0.9618 | suitcase |
| zidane.jpg | 2 / 2 | 2 / 2 | 0.9533 | person |
| 合计 | 14 / 16 | **13 / 14** | **0.9601** | 1 个未匹配 FP32 框，3 个额外 INT8 框 |

320×320 在 conf ≥0.25 时，FP32 检出 17 个，INT8 检出 22 个，17 个框实现空间匹配，16 个类别一致（94.12%），匹配框平均 IoU 为 0.9550。5 个额外 INT8 框占 22.73%，无 FP32 框空间漏检。COCO 384 的一个 FP32 `couch` 框在 INT8 对应为 `bed`，阈值升至 0.40 后该框被筛掉。128×128 回放尺寸使用同一量化流程，conf ≥0.40 时为 5/5 一致、平均 IoU 0.9424，另有两个额外 INT8 框。逐图细项见 [m2-yolo26n-accuracy.csv](m2-yolo26n-accuracy.csv)。

## 量化配置与敏感度

ONNX 检查器与静态 shape inference 通过，且量化入口将尚含 BatchNormalization 的模型拒收。102 层 Conv 使用逐输出通道对称有符号 INT8 权重，激活以每张量零点 0 的有符号 INT8 表示。32 张 COCO128 自然校准图与评估图按 SHA-256 排除交集。每幅图每个浮点激活张量最多以固定种子采 4,096 个绝对值，另记录全样本峰值。320×320 选择 P99.95，128×128 选择 P99.99。卷积保持冻结 RTL 的 32 位累加、无符号 16 位乘子和 5 位算术右移，融合的 SiLU 使用 256 字节 LUT。注意力采用 Q23 指数表与整数归一化，概率以 U8、尺度 1/256 表示。V×P 可按 `(V×P)ᵀ=Pᵀ×Vᵀ` 通过冻结 DOT4 的 unsigned-A 路径实现。

| 320×320 配置（conf ≥0.40） | 类别一致 | 平均 IoU |
| --- | ---: | ---: |
| 原始 ASR，校准集所选 P99.90 和 Q23 | 8 / 14 | 0.8509 |
| 将半个输出 LSB 折入每通道 Conv 累加器偏置，P99.95 和 Q23 | 11 / 14 | 0.9284 |
| 末端三个分类头用 ±12 的细尺度 | 11 / 14 | 0.9284 |
| 同上，用 26 张校准图拟合三个分类头的前景 logits 偏置 | **13 / 14** | **0.9601** |

将半个输出 LSB 写入原有 INT32 bias 减轻 ASR 向下取整的累计误差。末端分类的离群背景 logits 将 3 个尺度扩大到约 0.20、0.76 和 0.54，按 ±12 限幅后每码约 0.0945。对应的 sigmoid 概率在饱和处偏差约 6×10⁻⁶，置信度阈值附近有更细的分辨率。拟合使用校准集同位置的 FP32 与定点 logits，并按 sigmoid 置信度梯度加权。保留 6 张校准图筛选候选，26 张图拟合偏置，4 张评估图只用于精度复核。拟合分支平均偏置约为 +0.693、+0.679 和 +0.478 logits，已折入原有硬件累加器 INT32 bias。报告中的额外框计数保留了这项校准的精度代价。

320×320 量化图归属为 NPU 106、VPU 95、GPU 44、PS 35。以 MAC 计作两次标量操作，结合尾部筛选上界估算，规划给 PL 的计算量为 99.26%。图中 PS 尾部用 FP32 完成排序和坐标解码，卷积、注意力 MatMul 及其上游都通过定点参考执行。该占比属于静态规划口径，实际硬件归属、逐层回放和时延由 M3 验收。

## 复现

在有 ONNX、ONNX Runtime、NumPy、OpenCV 的官方导出环境，工作目录放编译器存档代码。以下示例使用 Windows 导出路径中的输入文件。Linux 验收只需要 NumPy。

```powershell
python -B -m tools.c6_ptq --onnx yolo26n_320.onnx --calibration calibration --eval eval --golden fp32_320.json --metadata model.json --out ptq320 --percentiles 99.95 --softmax q23 --classifier-bounds 12 --classifier-bias-correction
```

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python3 -B -m tools.verify_c6_quant /dev/shm/ysyx_compiler_c6/m2-320
```

`quant.json` 和 `weights.npz` 构成静态量化 IR 资产。320×320 成品 SHA-256 分别为 `8a7e466088cd8c2d76e16b2efb3ebc51d90667828c9fbc1cc7a01eb5f18ebd47` 和 `fa96bb9db70962142e4284059dcf7cd372f3aabb0a87959526d55ec46979ab5f`。128×128 分别为 `37f077bbe10570f90d4d6076b95b61246c552201fd2e77d4c41c23f3e0d0205b` 和 `1f2ca47e307bdf6d61a83d0fd484ca6ec9255ff8dc8b30a0d79ff19f34264486`。

xu4090 的 Linux NumPy 参考对两种尺寸、各四张测试图重复执行。RGB 预处理码值、每张图 6 个 PL 边界输出及最终 Q16.16 坐标、Q16.16 分数、INT64 类别均与 Windows 参考逐字节一致。各尺寸 `vectors/pl_outputs.json` 定义 PL 边界张量名、布局与逐图二进制文件名，`accuracy.json` 包含原始框和匹配数据，`overlays/` 保存源图尺寸的 FP32 与 INT8 并排叠图。
