# Reserve one compiler build at a time and size make/Verilator to available RAM.
compiler_build_slots() {
    compiler_scratch="${YSYX_COMPILER_SCRATCH:-/dev/shm/ysyx_compiler_c5}"
    local root="$compiler_scratch"
    [[ -d "$root" && ! -L "$root" && -O "$root" && "$(realpath -e -- "$root")" == "$root" ]]
    exec {compiler_build_lock}>"$root/build.lock"
    flock "$compiler_build_lock"
    local available
    available=$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo)
    if [[ ! "$available" =~ ^[0-9]+$ ]] || (( available < 15*1024*1024 )); then
        compiler_jobs=2
    elif [[ "$root" == /dev/shm/ysyx_compiler_c[67] ]]; then
        compiler_jobs=8
    else
        compiler_jobs=4
    fi
    if [[ "$root" == /dev/shm/ysyx_compiler_c[67] ]]; then
        ulimit -v 3145728
    fi
    export MAKEFLAGS="-j$compiler_jobs"
    printf 'compiler_build jobs=%s mem_available_kib=%s\n' "$compiler_jobs" "${available:-unavailable}" >&2
}
