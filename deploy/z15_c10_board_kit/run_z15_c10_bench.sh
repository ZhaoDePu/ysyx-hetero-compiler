#!/bin/sh
# Run the supplied byte-exact cases. Board access begins only in Linux mode.
set -eu

kit=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
backend=linux
profile=auto
frames=10
out=
benchmark="$kit/bin/z15_c10_benchmark"
mode=run

usage() {
    cat <<'HELP'
Usage: run_z15_c10_bench.sh [--freq-profile auto|125_100|200_150]
       [--steady-frames N] [--out DIR] [--probe-only|--profile-only]
       [--backend linux|reference|verilator] [--benchmark-bin PATH]

Linux mode first checks FPGA Manager and the v1 MMIO health. Auto measures the
NPU fabric counter and associates its frequency with a supported clock pair.
Use the explicit profile for a separately verified bitstream clock pair.
The default 10 steady YOLO frames follow one separately logged first frame.
YOLO uses the C9 bus input for direct comparison against its frozen RTL row.
HELP
}

fail() {
    printf '%s\n' "$*" >&2
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --probe-only) [ "$mode" = run ] || fail "Select one action"; mode=probe; shift ;;
        --profile-only) [ "$mode" = run ] || fail "Select one action"; mode=profile; shift ;;
        --backend|--freq-profile|--steady-frames|--out|--benchmark-bin)
            [ "$#" -ge 2 ] || fail "Missing value for $1"
            case "$1" in
                --backend) backend=$2 ;;
                --freq-profile) profile=$2 ;;
                --steady-frames) frames=$2 ;;
                --out) out=$2 ;;
                --benchmark-bin) benchmark=$2 ;;
            esac
            shift 2 ;;
        *) fail "Unknown argument: $1" ;;
    esac
done

case "$backend" in linux|reference|verilator) ;; *) fail "Invalid backend: $backend";; esac
case "$profile" in auto|125_100|200_150) ;; *) fail "Invalid clock profile: $profile";; esac
case "$frames" in ''|*[!0-9]*) fail "Steady frame count must be 1 to 1000";; esac
[ "$frames" -ge 1 ] && [ "$frames" -le 1000 ] || fail "Steady frame count must be 1 to 1000"
[ -x "$benchmark" ] || fail "Missing benchmark executable: $benchmark"

if [ "$backend" = linux ]; then
    manager=/sys/class/fpga_manager/fpga0/state
    [ -r "$manager" ] && [ "$(cat "$manager")" = operating ] ||
        fail "FPGA Manager is not operating"
elif [ "$profile" = auto ] && [ "$mode" != probe ]; then
    fail "An explicit clock profile is required for an offline backend"
fi

run_bench() { "$benchmark" --backend "$backend" "$@"; }

if [ "$mode" = probe ]; then
    run_bench --probe
    exit
fi

if [ "$profile" = auto ]; then
    profile=$(run_bench --detect-clock) ||
        fail "Clock probe failed. Supply --freq-profile after verifying the loaded clock pair."
elif [ "$mode" = profile ]; then
    run_bench --probe >&2
fi

if [ "$mode" = profile ]; then
    printf '%s\n' "$profile"
    exit
fi

[ -n "$out" ] || out="$kit/results-$backend-$profile"
case "$out" in /*) ;; *) out="$PWD/$out" ;; esac
[ ! -e "$out" ] || fail "Result directory already exists: $out"
mkdir -p -- "$out"
{
    run_bench --self-test &&
    run_bench --reject "$kit/models/reject/bad_crc.yhb" "YHB CRC32 mismatch" &&
    run_bench --reject "$kit/models/reject/bad_dma.yhb" "invalid YHB DMA extent"
} > "$out/mmio-selftest.log" 2>&1 || fail "Self-test failed. See $out/mmio-selftest.log"

case "$backend" in
    reference)
        models="yolo26n resnet18 mobilenetv2"
        printf "%s\n" "Reference backend omits TinyLM native attention. Use Verilator for the four-model gate." >&2 ;;
    *) models="yolo26n tinylm resnet18 mobilenetv2" ;;
esac
set --
for model in $models; do
    count=1
    if [ "$model" = yolo26n ]; then
        count=$((frames + 1))
        run_bench --case "$kit/models/$model" --freq-profile "$profile" --frames "$count" \
            --fixture bus > "$out/$model.csv" 2> "$out/$model.log" ||
            fail "$model failed. See $out/$model.log"
    else
        run_bench --case "$kit/models/$model" --freq-profile "$profile" --frames "$count" \
            > "$out/$model.csv" 2> "$out/$model.log" ||
            fail "$model failed. See $out/$model.log"
    fi
    set -- "$@" "$out/$model.csv"
done

case "$backend" in
    linux) summary="$out/board_c10_summary.csv" ;;
    reference) summary="$out/reference_c10_summary.csv" ;;
    verilator) summary="$out/verilator_c10_summary.csv" ;;
esac

LC_ALL=C awk -F, '
BEGIN {
    OFS = ",";
    print "model,freq_profile,measurement_kind,unit,work_units,samples,scope,wall_ms,pl_hw_ms,execute_host_ms,ps_overhead_ms,ps_overhead_pct,wall_cycles,native_gpu_union_cycles,native_vpu_busy_cycles,gpu_busy_cycles,native_gpu_overlap_cycles,npu_busy_cycles,vpu_busy_cycles,dma_busy_cycles,dma_bytes,dma_descriptors,throughput,verification,compared_bytes,abi_error,slverr,invalid_dma";
}
function mean(column) { return seen[column] == samples ? sprintf("%.6f", sum[column]/samples) : "" }
function reset(   col) {
    for (col in sum) delete sum[col];
    for (col in seen) delete seen[col];
    samples = 0;
}
function emit(scope) {
    print model,profile,kind,unit,work,samples,scope,
          mean(8),mean(13),mean(10),mean(12),
          (sum[8]>0?sprintf("%.6f",100*sum[12]/sum[8]):""),
          mean(14),mean(15),mean(16),mean(17),mean(18),mean(19),mean(20),mean(21),
          mean(23),mean(24),(sum[8]>0?sprintf("%.6f",1000*sum[7]/sum[8]):""),
          "byte_exact",sum[28],sum[30],mean(31),sum[32];
    reset();
}
FNR == 1 {
    if (samples) emit(model=="yolo26n"?"resident_steady":"complete_session");
    next;
}
{
    if (NF != 32 || $29 != "byte_exact" || $30 != 0 || $32 != 0 ||
        ($31 != "" && $31 != 0) || $8 <= 0 || $7 <= 0) {
        print "Malformed or failed benchmark row" > "/dev/stderr";
        bad=1;
        exit 1;
    }
    model=$1; profile=$2; kind=$3; unit=$6; work=$7;
    for (i=7; i<=NF; ++i)
        if ($i != "" && $i ~ /^[0-9]+(\.[0-9]+)?$/) {
            sum[i]+=$i; seen[i]++;
        }
    ++samples;
    if (model == "yolo26n" && $4 == 1) emit("first_frame");
}
END {
    if (!bad && samples) emit(model=="yolo26n"?"resident_steady":"complete_session");
}' "$@" > "$summary"

if [ "$backend" = linux ]; then
    LC_ALL=C awk -F, '
        BEGIN { OFS="," }
        NR==FNR {
            if (FNR>1 && $3=="board" && $7!="first_frame" && $24=="byte_exact") {
                key=$1 FS $2;
                for (i=1;i<=NF;++i) board[key,i]=$i;
            }
            next;
        }
        FNR==1 { print; next }
        {
            key=$1 FS $2;
            if (board[key,6]=="" || $3!=board[key,4] || $4!=board[key,5]) {print; next;}
            ++joined;
            sim=$5; ratio=board[key,13]/sim; delta=(ratio-1)*100;
            agreement=100-(delta<0?-delta:delta);
            print $1,$2,$3,$4,$5,$6,$7,$8,"measured",board[key,6],
                  board[key,13],board[key,9],board[key,8],board[key,10],board[key,11],
                  board[key,14],board[key,15],board[key,16],board[key,17],board[key,18],
                  board[key,19],board[key,20],board[key,21],board[key,22],board[key,23],
                  sprintf("%.6f",ratio),sprintf("%.6f",delta),
                  sprintf("%.6f",(agreement>0?agreement:0)),board[key,12];
        }
        END { if (joined!=4) exit 1 }
    ' "$summary" "$kit/sim_vs_board_comparison.csv" > "$out/sim_vs_board_comparison.csv"
else
    cp "$kit/sim_vs_board_comparison.csv" "$out/sim_vs_board_comparison.csv"
fi
printf 'Reports: %s\n' "$out"
