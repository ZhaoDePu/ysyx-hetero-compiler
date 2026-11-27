#!/bin/sh
set -eu
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 NEW_DESTINATION" >&2
    exit 2
fi
kit=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
(cd "$kit" && sha256sum -c SHA256SUMS.txt)
mkdir -p -- "$1/bin"
for name in z15_yolo26n_service z15_c10_benchmark z15_qwen25_chat ysyx_rt_cli; do
    install -m 755 -- "$kit/bin/$name" "$1/bin/$name"
done
