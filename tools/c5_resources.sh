# Reserve one compiler build at a time and size make/Verilator to available RAM.
compiler_build_slots() {
    local root=/dev/shm/ysyx_compiler_c5
    [[ -d "$root" && ! -L "$root" && -O "$root" && "$(realpath -e -- "$root")" == "$root" ]]
    exec {compiler_build_lock}>"$root/build.lock"
    flock "$compiler_build_lock"
    local available
    available=$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo)
    if [[ ! "$available" =~ ^[0-9]+$ ]] || (( available < 15*1024*1024 )); then
        compiler_jobs=2
    else
        compiler_jobs=4
    fi
    export MAKEFLAGS="-j$compiler_jobs"
    printf 'compiler_build jobs=%s mem_available_kib=%s\n' "$compiler_jobs" "${available:-unavailable}" >&2
}
