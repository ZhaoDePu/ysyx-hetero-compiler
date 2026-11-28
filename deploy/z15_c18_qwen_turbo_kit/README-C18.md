# Valence C18 Qwen 候选板端套件

在 Z15 Linux 上运行 sh install.sh /DATA/valence-c18，将 C18 可执行文件安装到独立目录。
安装脚本先检查 SHA-256。模型、tokenizer 和现有固件可沿用原有校验版本。

    /DATA/valence-c18/bin/chat.sh /DATA/models/h2.bin /DATA/models/tokenizer.bpe \
        /DATA/models/model.w4a8.bin --exec-mode p18b --trace-steps on \
        --max-new-tokens 64 --prompt 'Explain how a systolic array works.'

对 Phase 19A 的 TMA 试验使用同一组参数，将 p18b 换为 p19a。可用 `--exec-mode auto`
在 Phase 18B 和 19A 上选择已经验证的驻留 CMDQ 模式。旧 Turbo MMIO 使用第三个参数给出的 C12 权重。
p19a 试验要求硬件报告 TMA 和 Barrier 能力并且 TMA 空闲。

逐步日志的 decode_upload_bytes 只计新写入的命令字节，host_cpu_ms 是主机线程 CPU 时间。
首轮记录、驻留复用、PL WALL 以及输出读取各有不同口径。REPORT-C18.md 给出冻结 RTL 和
软件参考的对照，并记录 p19a 性能门禁。
