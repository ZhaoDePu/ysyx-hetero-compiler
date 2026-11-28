#!/bin/sh
set -eu

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 S3_H2_WEIGHTS TOKENIZER C12_FALLBACK_WEIGHTS|- [chat options]" >&2
    exit 2
fi
weights=$1
tokenizer=$2
fallback=$3
shift 3

verify() {
    actual=$(sha256sum < "$1")
    if [ "${actual%% *}" != "$2" ]; then
        echo "Unexpected SHA-256: $1" >&2
        exit 1
    fi
}
verify "$tokenizer" c197c4d96a15faf52cc8147a85a1610c41765a1d080ecf9aa87670aaf634c33a
if [ -f "$weights" ]; then
    verify "$weights" fb6bc64c1e831bae530e19b15937e8475b8fe506ea8dab0372345065f98ed787
elif [ "$fallback" = - ]; then
    echo "S3-H2 weights are missing: $weights" >&2
    exit 1
fi
for opt do
    if [ "$opt" = "--fallback-weights" ] || [ "$opt" = "--" ]; then
        echo "Supply the C12 fallback path as the third argument." >&2
        exit 2
    fi
done
if [ "$fallback" != - ]; then
    verify "$fallback" 141372e0b798df279f07bb783bd6cb9a37b86d85223df31583ff25923c1a1f33
    set -- "$@" --fallback-weights "$fallback"
fi
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
unset YSYX_QWEN_C16
exec "$here/z15_qwen25_chat" "$@" --backend z15 --target auto --exec-mode auto --exec-stats \
    --weights "$weights" --tokenizer "$tokenizer" --kv-bits 8
