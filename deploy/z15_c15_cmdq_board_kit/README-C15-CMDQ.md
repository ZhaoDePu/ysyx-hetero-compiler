# Valence · 蔚澜 C15 CMDQ 上板套件

本目录的 AArch64 程序用于已由操作者安装的 Z15 UIO 控制设备和 DDR 池。
程序支持 --exec-mode auto、cmdq、mmio，默认 auto。Phase 18
必须同时报告 0x0120 等于 0x0c0c0200 和 0x0124 的 bit 4。
需要读取发令数、命令数、轮询次数和 CMDQ 拍数时，传入 --exec-stats。

| 程序 | 输入与目标 |
| --- | --- |
| z15_yolo26n_service | C14 的 YHP1 模型需传 --target p17。旧 C8/C10 的 YHB1 模型沿用默认目标。TCP 输入 9001，输出 9000。V4L2 使用 --source v4l2。 |
| z15_c10_benchmark | 使用 C10 kit 的 models 子目录和既有 io.tsv。C14 YHP1 用例指定 --target p17，并把模型命名为 program.yhp。 |
| z15_qwen25_chat | 复用 C14 S3-H2 量化包与 C12 BPE。硬件模式传 --backend z15 --target p17。Qwen 在各层内部根据主机依赖切分。 |
| ysyx_rt_cli | 运行原有 YHB1 模型，输入输出名称和 KV 脚本沿用 C10 以前的格式。 |

先在解包目录执行 sha256sum -c SHA256SUMS.txt。
install.sh 将四个程序复制到指定目录。compare.sh 依次在同一用例上执行
MMIO 和 CMDQ，把 CSV 和执行日志写入指定目录。

示例命令

    ./install.sh /path/to/new/c15
    ./compare.sh /path/to/c10/models/resnet18 /path/to/result
    ./compare.sh /path/to/c14-yolo-case /path/to/result-p17 p17 2

C14 双帧用例可以复用 C10 YOLO 用例中的 bus、zidane 输入、黄金检测头
和 io.tsv，并使用 C14 的 program.yhp。切换固件时重启运行时会话，以便
重新探测扩展寄存器和装载常驻资产。

C15 正式封板接受 C14 混合精度路径的层内切批。Qwen 首个 Decode
在无词表头场景实测每层 14 次 START。可在离线机器用
--backend cmdq-reference --target p17 配合 --dump-logits 验证相同
CMDQ 序列和完整 logits。参考后端的周期字段不用于上板性能估算。
