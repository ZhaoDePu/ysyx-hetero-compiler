# C4-A Z15 Balanced 流式 ONNX 消融

- `EdgeConv_Stream_C4A.onnx` 处理 16 个 `10×10×8` 图像 Tile，每个 Tile 完成五层量化空间卷积、ReLU、残差和 GPU ArgMax。图像接口采用 NHWC，ONNX 卷积采用 NCHW。导入后消去内部相邻的逆转置，保留可观察的输入输出布局。
- `LongPrefix_Stream_C4A.onnx` 处理 24 个各长 48 token 的独立上下文。每个上下文计算末两行真实因果 Attention，其可见历史分别为 47 和 48 token。图中使用标准 `DequantizeLinear → Add(因果掩码) → Softmax → QuantizeLinear`，导入后生成 GPU QSoftmax。后续包含全连接投影、FFN、残差、logits 与 ArgMax。
- 两个 ONNX 采用确定性的合成输入与权重。从仓库根目录运行 `python3 -B -m examples.c4a_streams.generate` 可重新生成。

## 编译与验证

```sh
python3 -B -m hetero_compiler \
  --graph examples/c4a_streams/EdgeConv_Stream_C4A.onnx \
  --profile fpga_zu15eg_balanced \
  --out /dev/shm/ysyx_compiler_c4a/edgeconv_bundle \
  --pipeline-tiles --pipeline-scheduler bank-aware

PYTHONHASHSEED=0 nice -n 10 python3 -B -m tools.run_c4a_ablation \
  --out /dev/shm/ysyx_compiler_c4a/stream_ablation \
  examples/c4a_streams/EdgeConv_Stream_C4A.onnx \
  examples/c4a_streams/LongPrefix_Stream_C4A.onnx
```

消融工具通过同一个冻结 Z15 Balanced Verilator MMIO runner 检查每个输出张量，统计 NPU 时钟域拍数、实际 AXI DMA 字节和引擎占用。A0 到 A3 共用规范化的布局预处理与可选 GPU IMEM 程序驻留。IMEM 预装使用实际 MMIO 写入，计入周期。用固定的 `PYTHONHASHSEED` 重放可使继承的 ONNX 导入顺序与 DDR 地址可复现。C99 的 `program.yhb` 路径沿用部署 ABI。

复用启用时，分配器优先把同一 GPU QSoftmax 的分数与概率放在本地窗口上端的连续区域。冻结 RTL 实测表明，该布局保持原有的大立即数地址生成指令形态，避免复用把缓冲推到低地址后增加 QSoftmax 拍数。容量或原生缓冲约束冲突时回退到原布局。

A3 逐半区追踪活跃 DMA 或计算，并在发射下一项任务前等待对应 Fence。对各引擎优先发射较早 Tile，避免两个半区的 NPU 主干锁步结束，从而让一半的 GPU 尾部覆盖另一半的 DMA 或 NPU 主干。冻结壳的两个半区各只有一个 owner，DMA 只有一个在途传输槽，因此同一半区的 GPU 与 DMA 必须依次执行。C3 的调度器仍可通过 `--pipeline-scheduler c3` 选择，也是未指定调度器时的默认值。
