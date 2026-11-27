#!/bin/sh
set -eu
if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 CASE_DIR RESULT_DIR [step2|p17] [FRAMES]" >&2
    exit 2
fi
kit=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
freq=${YSYX_FREQ_PROFILE:-200_150}
mkdir -p -- "$2"
for mode in mmio cmdq; do
    "$kit/bin/z15_c10_benchmark" --case "$1" --backend linux \
        --target "${3:-step2}" --frames "${4:-1}" --freq-profile "$freq" \
        --exec-mode "$mode" --exec-stats >"$2/$mode.csv" 2>"$2/$mode.log"
done
