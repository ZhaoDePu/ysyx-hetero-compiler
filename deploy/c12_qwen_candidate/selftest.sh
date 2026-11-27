#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$root"
sha256sum -c SHA256SUMS.txt
chat=./z15_qwen25_chat
case "$(uname -m)" in
    x86_64) chat=./z15_qwen25_chat.x86_64 ;;
    aarch64) ;;
    *) printf '%s\n' 'Unsupported CPU architecture' >&2; exit 2 ;;
esac

for bits in 4 8; do
    "$chat" --weights model.w4a8.bin --tokenizer tokenizer.bpe \
        --backend cpu --kv-bits "$bits" --mode verify --token-id 151644 \
        --golden "prefill-kv$bits.q16"
done
