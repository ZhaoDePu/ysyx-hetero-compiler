#!/bin/sh
set -eu
if [ "$#" -ne 1 ]; then
    printf 'Usage: %s C18_INSTALL_DIRECTORY\n' "$0" >&2
    exit 2
fi
kit=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
(cd "$kit" && sha256sum -c SHA256SUMS.txt)
mkdir -p -- "$1/bin"
for name in z15_qwen25_chat z15_yolo26n_service z15_c10_benchmark ysyx_rt_cli chat.sh; do
    install -m 755 -- "$kit/bin/$name" "$1/bin/$name"
done
