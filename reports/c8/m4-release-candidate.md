# C8 候选交付与版控

C8 候选在 125/100 MHz 达到 15,052,683 拍，较 C6 3.570×，较 C7 1.182×。由于 3.8× 和 1.25× 两道底线未达到，按约定推送代码和证据，保留正式封板标签 v9-compiler-phase-c8 未创建。

## 兼容性

- C6/C7 C99 公开 API 头文件、z15-mmio-abi-v1 与 YHB1 保持兼容。未改动硬件 RTL。Phase C7、冻结 Step 2 和 Step 2-v2 的关键原始文件已按只读归档校验和复核。
- x86-64、Xilinx 2019.1 SDK AArch64 静态库和 CLI 均以 C99、-O2、-Wall、-Wextra、-Werror 重新构建。AArch64 CLI 的唯一 NEEDED 是 libc.so.6。含样例的部署包解压后逐文件 SHA-256 校验均通过。
- 使用本包 CLI 的软件定点后端执行四图，全部 8,400 B/API 结果与 C7 黄金一致。CRC 损坏、半区越界与 DDR 回绕报错。Step 2 冻结 RTL 的四档回放和四图 C99 原始检测头均逐字节通过。
- 历史回放用 C8 同版工具和固定 RTL。旧 C6 128/320 与 C7 320 模型通过 C99 四图回放，C7 旧 YHB 还通过本轮 125/100 MHz 整帧 RTL。C3 八点流式回放的吞吐单调性、逐字节校验和 DMA 字节一致；五档相对 C7 有小幅拍数漂移，最大 576 拍、低于 0.02%，原因尚未确认。

## 路径和校验

- 远端归档 /home/zdp/ysyx-migration/artifacts/compiler-phase-c8-yolo26n-20261126-162320/。m1/node-breakdown.csv 为 C7 节点归属，m3/ablation.csv 为整帧消融，m3/node-ablation.csv 为各版本逐节点计数，m3/roofline.json 为发射关键路径模型。m4/regression.csv 汇总回放门禁。
- Windows D:/VMshare/z15-yolo26n-320-kit-c8/，共 3,034,989 B。新套件已对照 SHA256SUMS.txt 验证，同步前后 C6、C7 原包哈希一致。
- z15_yolo26n_320_kit_c8.tar.gz SHA-256 为 49bedb97049d77ed61a899ccbab55610348a79610b1ea5d34bbbae332e566fec。
- 包内 YHB 的 SHA-256 为 1684bd2a81336b284b167a6cc137e860cb40d26207cfe4da18a82a5bee266ae9。

共 38 项回放门禁全绿，含 148 项软件测试、原 15 组 YHB、30 项 CRC 和 20 项非法 DMA 拒收。具体计数和原始日志见归档。
