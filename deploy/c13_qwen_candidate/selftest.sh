#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$root"
sha256sum --quiet -c SHA256SUMS.txt
c12=$(CDPATH= cd -- "${VALENCE_C12_KIT:-$root/../z15-qwen25-05b-c12-kit}" && pwd -P)
(
    cd "$c12"
    sha256sum --quiet -c "$root/C12-ASSETS-SHA256.txt"
)
case "$(uname -m)" in
    aarch64) cli=./z15_qwen25_chat_c13 ;;
    x86_64) cli=./z15_qwen25_chat_c13.x86_64 ;;
    *) printf '%s\n' 'Unsupported CPU architecture' >&2; exit 2 ;;
esac
"$cli" --backend cpu --weights "$c12/model.w4a8.bin" --tokenizer "$c12/tokenizer.bpe" \
    --mode verify --token-id 151644 --golden "$c12/prefill-kv8.q16"
