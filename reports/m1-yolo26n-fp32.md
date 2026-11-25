# M1 官方模型与四张真图 FP32 基线

使用 Ultralytics 8.4.218 的官方 COCO 80 类 YOLO26n 检测权重。原始检查点含 2,572,280 个参数，Ultralytics 导出日志报告融合后的推理网络含 2,408,932 个参数。权重及两张示例图来自 Ultralytics，另外两张评估图取自 COCO128 中的 train2017 图像。

- 官方权重 SHA-256：9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
- 静态 FP32 320 ONNX SHA-256：1f4ece0b17fc3094c5c1684718808036ed7cd5e166e72638a0ceceaef3bb692a
- 导出参数：`format=onnx imgsz=320 batch=1 dynamic=False simplify=True nms=False opset=19 device=cpu half=False`。
- ONNX 1.23.2 完整 checker 与严格 shape inference 通过。ONNX Runtime 1.32.0 CPU 回放的输入为 RGB [0,1]、`[1,3,320,320]`，输出 `[1,300,6]`。
- 预处理为保比例双线性缩放并用 RGB 114 填充，检测框逆映射并裁剪到原图像素范围。

| 图像 | conf ≥ 0.25 | conf ≥ 0.40 | conf ≥ 0.40 检出标签 |
| --- | ---: | ---: | --- |
| bus.jpg | 5 | 4 | person, person, bus, person |
| coco000000000241.jpg | 8 | 6 | person, person, bottle, person, cup, couch |
| coco000000000384.jpg | 2 | 2 | suitcase, couch |
| zidane.jpg | 2 | 2 | person, person |

[模型与导出环境](m1-yolo26n-model.json) · [图像来源及逐文件 SHA-256](m1-yolo26n-images.json) · [检测原始数值](m1-yolo26n-fp32_320.json) · [ONNX 图审计](m1-yolo26n-graph-audit.md)

32 张校准图与这四张评估图无重复文件。两张室内评估图来自 COCO train2017，因此四图结果用于部署对齐。
