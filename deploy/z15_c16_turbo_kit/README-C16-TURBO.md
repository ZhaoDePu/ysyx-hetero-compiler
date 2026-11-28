# Valence C16-Turbo Qwen2.5-0.5B-Instruct

适用于 Z15 Linux AArch64。安装脚本校验本包并更新指定 C16 目录下的两个文件。
请指定独立的 C16 目录，沿用已有权重和 tokenizer。

```sh
sh ./install.sh /DATA/valence-c16
/DATA/valence-c16/bin/chat.sh /DATA/models/h2.bin /DATA/models/tokenizer.bpe \
    /DATA/models/model.w4a8.bin --max-new-tokens 64 --prompt '请解释什么是 FPGA。'
```

去掉 `--prompt` 可交互聊天。已有 Phase 18 和 S3-H2 时第三个参数可用 `-`。
仅有旧版固件和 C12 时，第一个参数可填尚未存在的 S3-H2 路径，程序探测旧版后
直接读取第三个参数指向的 C12 权重。CLI 输出 CMDQ START 和硬件计数。

| 固件 | 权重 | 默认调度 |
| --- | --- | --- |
| Phase 18，扩展版本 0x0c0c0200 且 CMDQ 能力位有效 | C14-v2/C15 S3-H2，YSQC14H | CMDQ，最多 16-token 同批 Prefill，每层 Decode 4 次 START，全词表头额外 10 次 |
| Step 2-v3 Turbo，扩展版本 0x0c0c0101 | C12 KV8，YSQC12 | Turbo MMIO |
| Step 2 v1，无扩展版本 | C12 KV8，YSQC12 | v1 MMIO |

从已有部署目录指定模型路径。源资产位于 xu4090 的以下目录。

- S3-H2 对应 `/home/zdp/ysyx-migration/artifacts/compiler-phase-c14v2-p17-trinity-20261127-181930/m2/.models-cache/h2.bin`。
- C12 KV8 与 tokenizer 对应 `/home/zdp/ysyx-migration/artifacts/compiler-phase-c12-qwen25-smoothquant-20261127-130522/z15-qwen25-05b-c12-kit/`。

| 资产 | SHA-256 |
| --- | --- |
| S3-H2 | `fb6bc64c1e831bae530e19b15937e8475b8fe506ea8dab0372345065f98ed787` |
| C12 KV8 | `141372e0b798df279f07bb783bd6cb9a37b86d85223df31583ff25923c1a1f33` |
| tokenizer.bpe | `c197c4d96a15faf52cc8147a85a1610c41765a1d080ecf9aa87670aaf634c33a` |

P1/P2/P3 有损压缩留在研究候选分支。推理使用已装载固件。
