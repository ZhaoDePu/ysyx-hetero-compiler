#!/bin/sh
# Run from a copy of the verified kit on an authorized Z15 or an AArch64 CPU.
set -eu

kit=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
backend=z15
if [ "$#" -ge 2 ] && [ "$1" = --backend ]; then
    backend=$2
    shift 2
fi
if [ "${1:-}" = --help ]; then
    printf '%s\n' 'run_z15_qwen25.sh [--backend cpu|z15] [--verify|--chat|--prompt TEXT]'
    exit 0
fi
case "$backend" in
    cpu) ;;
    z15)
        test "$(cat /sys/class/fpga_manager/fpga0/state)" = operating ||
            { echo 'FPGA Manager is not operating.' >&2; exit 1; }
        ;;
    *) echo 'Expected --backend cpu|z15.' >&2; exit 2 ;;
esac

case "@{1:---verify}" in
    --verify)
        [ "$#" -le 1 ] || exit 2
        (cd "$kit" && sha256sum -c SHA256SUMS.txt)
        exec "$kit/bin/z15_qwen25_chat" --backend "$backend" --weights "$kit/model/qwen25-05b.w4a8" \
            --tokenizer "$kit/model/tokenizer.bpe" --mode verify --token-id 151644 \
            --golden "$kit/vectors/token-151644-logits.q16"
        ;;
    --chat)
        [ "$#" -eq 1 ] || exit 2
        exec "$kit/bin/z15_qwen25_chat" --backend "$backend" --weights "$kit/model/qwen25-05b.w4a8" \
            --tokenizer "$kit/model/tokenizer.bpe"
        ;;
    --prompt)
        [ "$#" -eq 2 ] || exit 2
        exec "$kit/bin/z15_qwen25_chat" --backend "$backend" --weights "$kit/model/qwen25-05b.w4a8" \
            --tokenizer "$kit/model/tokenizer.bpe" --prompt "$2"
        ;;
    *) exit 2 ;;
esac
