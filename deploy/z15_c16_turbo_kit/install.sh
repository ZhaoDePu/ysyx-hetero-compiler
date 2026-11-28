#!/bin/sh
set -eu
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 C16_INSTALL_DIRECTORY" >&2
    exit 2
fi
kit=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
(cd "$kit" && sha256sum -c SHA256SUMS.txt)
mkdir -p -- "$1/bin"
for file in z15_qwen25_chat chat.sh; do
    install -m 755 -- "$kit/bin/$file" "$1/bin/$file"
done
