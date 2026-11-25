#!/bin/sh
set -eu
kit=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
runner=${YSYX_RUN:-"$kit/build/ysyx_run"}
backend=${YSYX_BACKEND:-linux}
if [ "$#" -eq 0 ]; then
    set --
    for i in "$kit"/cases/*; do set -- "$@" "$(basename "$i")"; done
fi
work=$(mktemp -d "${TMPDIR:-/tmp}/ysyx-z15-bench.XXXXXX")
trap 'rm -rf -- "$work"' EXIT HUP INT TERM
for name do
    case "$name" in ""|*[!A-Za-z0-9_-]*) printf '%s\n' "invalid case name $name" >&2; exit 2;; esac
    case_dir="$kit/cases/$name"
    test -f "$case_dir/io.tsv"
    mkdir "$work/$name"
    (
        set -- "$case_dir/program.yhb" --backend "$backend" --host-timing --dump-stats
        if [ -n "${YSYX_CONTROL_UIO:-}" ]; then set -- "$@" --uio-control "$YSYX_CONTROL_UIO"; fi
        if [ -n "${YSYX_DMA_UIO:-}" ]; then set -- "$@" --uio-dma "$YSYX_DMA_UIO"; fi
        tab=$(printf '\t')
        while IFS="$tab" read -r role symbol file; do
            if [ "$role" = input ]; then
                set -- "$@" --input "$symbol=$case_dir/$file"
            elif [ "$role" = output ]; then
                set -- "$@" --output "$symbol=$work/$name/$file"
            else
                printf '%s\n' "invalid binding $role" >&2
                exit 1
            fi
        done < "$case_dir/io.tsv"
        "$runner" "$case_dir/program.yhb" --inspect > "$work/$name/inspect.log"
        "$runner" "$@" > "$work/$name/run.log" 2>&1 || { cat "$work/$name/run.log" >&2; exit 1; }
        while IFS="$tab" read -r role symbol file; do
            if [ "$role" = output ] && ! cmp -s "$case_dir/$file" "$work/$name/$file"; then
                printf '%s\n' "$name: $symbol output differs" >&2
                exit 1
            fi
        done < "$case_dir/io.tsv"
        samples=$(cat "$case_dir/samples.txt")
        ns=$(awk '/^HOST_TIMING elapsed_ns=/ {split($2,x,"="); print x[2]}' "$work/$name/run.log")
        awk -v name="$name" -v backend="$backend" -v ns="$ns" -v n="$samples" \
            'BEGIN { if(ns<=0 || n<=0) exit 1; printf "%s %s BYTE_EXACT wall_ms=%.3f items_per_sec=%.3f items=%d\n",name,backend,ns/1e6,(n*1e9)/ns,n }'
        grep '^STATS ' "$work/$name/run.log"
        if [ -n "${YSYX_LOG_DIR:-}" ]; then
            mkdir -p "$YSYX_LOG_DIR"
            cp "$work/$name/run.log" "$YSYX_LOG_DIR/$name.log"
        fi
    )
done
