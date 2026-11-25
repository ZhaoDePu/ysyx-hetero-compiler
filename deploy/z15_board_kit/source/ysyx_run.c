#define _POSIX_C_SOURCE 200809L

#include "ysyx_rt.h"
#include "z15_regs.h"

#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    char *name;
    const char *path;
} file_binding_t;

static int decode_binding(file_binding_t *dst, const char *entry)
{
    const char *eq = strchr(entry, '=');
    if (!eq || eq == entry || !eq[1])
        return -1;
    const size_t length = (size_t)(eq - entry);
    dst->name = malloc(length + 1);
    if (!dst->name)
        return -1;
    memcpy(dst->name, entry, length);
    dst->name[length] = 0;
    dst->path = eq + 1;
    return 0;
}

static uint8_t *read_file(const char *path, size_t max_size, size_t *n)
{
    FILE *file = fopen(path, "rb");
    if (!file)
        return NULL;
    uint8_t *data = NULL;
    if (!fseeko(file, 0, SEEK_END)) {
        off_t size = ftello(file);
        if (size >= 0 && (uint64_t)size <= max_size && !fseeko(file, 0, SEEK_SET)) {
            *n = (size_t)size;
            data = malloc(*n ? *n : 1u);
            if (data && fread(data, 1, *n, file) != *n) {
                free(data);
                data = NULL;
            }
        }
    }
    if (fclose(file)) {
        free(data);
        data = NULL;
    }
    return data;
}

static int write_file(const char *path, const void *data, size_t n)
{
    FILE *file = fopen(path, "wb");
    if (!file)
        return -1;
    int failed = fwrite(data, 1, n, file) != n;
    return fclose(file) || failed ? -1 : 0;
}

static void usage(FILE *f)
{
    fputs("usage: ysyx_run PROGRAM.yhb [--backend linux|verilator]\n"
          "       [--input tensor=file.bin] [--output tensor=file.bin] [--dump-stats] [--host-timing]\n"
          "       [--uio-control /dev/uioN|name] [--uio-dma /dev/uioN|name] [--inspect]\n"
          "       [--kv-script TSV_FILE]\n", f);
}

static void inspect(const ysyx_program_t *program, size_t size)
{
    printf("YHB1 target=%" PRIu32 " bytes=%zu\n", ysyx_program_target(program), size);
    if (ysyx_kv_step_count(program))
        printf("KV_STEPS count=%" PRIu32 "\n", ysyx_kv_step_count(program));
    for (size_t i = 0; i < ysyx_program_symbol_count(program); ++i) {
        ysyx_symbol_t s;
        if (ysyx_program_symbol(program, i, &s))
            break;
        printf("SYMBOL name=%.*s flags=%" PRIu32 " dtype=%" PRIu32 " shape=",
               (int)s.name_bytes, s.name, s.flags, s.dtype);
        for (uint32_t d = 0; d < s.rank; ++d)
            printf("%s%" PRIu32, d ? "x" : "", s.dims[d]);
        printf(" bytes=%" PRIu32 " ddr_offset=%" PRIu32 "\n", s.bytes, s.ddr_offset);
    }
}

static int monotonic_ns(uint64_t *result)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t))
        return -1;
    *result = (uint64_t)t.tv_sec * UINT64_C(1000000000) + (uint64_t)t.tv_nsec;
    return 0;
}

static void add_stats(ysyx_stats_t *total, const ysyx_stats_t *step)
{
    total->wall_ticks += step->wall_ticks;
    total->npu_ticks += step->npu_ticks;
    total->gpu_ticks += step->gpu_ticks;
    total->npu_gpu_overlap_ticks += step->npu_gpu_overlap_ticks;
    total->dma_operations += step->dma_operations;
    total->dma_bytes += step->dma_bytes;
    total->has_observations = step->has_observations;
    total->abi_error |= step->abi_error;
    total->invalid_dma_descriptors += step->invalid_dma_descriptors;
    total->observed.axi_transactions += step->observed.axi_transactions;
    total->observed.dma_busy_ticks += step->observed.dma_busy_ticks;
    total->observed.dma_compute_overlap_ticks += step->observed.dma_compute_overlap_ticks;
    total->observed.triple_overlap_ticks += step->observed.triple_overlap_ticks;
    total->observed.gpu_clock_busy_ticks += step->observed.gpu_clock_busy_ticks;
    total->observed.dma_gpu_overlap_ticks += step->observed.dma_gpu_overlap_ticks;
    total->observed.dma_tensor_overlap_ticks += step->observed.dma_tensor_overlap_ticks;
}

static int scripted_tensor(ysyx_kv_session_t *session, const ysyx_program_t *p, const char *name,
                           const char *file, int input, const char **error)
{
    ysyx_symbol_t symbol;
    *error = "KV script tensor symbol or file is invalid";
    if (ysyx_find_symbol(p, name, input ? YSYX_INPUT : YSYX_OUTPUT, &symbol))
        return -1;
    if (input) {
        size_t size;
        uint8_t *data = read_file(file, symbol.bytes, &size);
        if (!data)
            return -1;
        const ysyx_input_t binding = {name, data, size};
        int rc = ysyx_kv_write_input(session, &binding, error);
        free(data);
        return rc;
    }
    uint8_t *data = malloc(symbol.bytes);
    if (!data)
        return -1;
    int rc = ysyx_kv_read_output(session, name, data, symbol.bytes) ||
             write_file(file, data, symbol.bytes);
    free(data);
    return rc ? -1 : 0;
}

static int run_kv_script(const char *path, const ysyx_program_t *p, const ysyx_hal_t *hal,
                         const ysyx_input_t *inputs, size_t count, ysyx_stats_t *stats,
                         FILE *trace, const char **error)
{
    FILE *script = fopen(path, "r");
    *error = "cannot open KV script";
    if (!script)
        return -1;
    ysyx_kv_session_t *session = NULL;
    char *line = NULL;
    size_t capacity = 0, line_index = 0;
    uint32_t next = 0;
    int rc = -1;
    if (ysyx_kv_begin(p, hal, inputs, count, &session, error))
        goto done;
    memset(stats, 0, sizeof(*stats));
    ssize_t length;
    while ((length = getline(&line, &capacity, script)) >= 0) {
        ++line_index;
        *error = "malformed KV script line (expected tab-separated fields)";
        if (memchr(line, 0, (size_t)length))
            goto done;
        while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r'))
            line[--length] = 0;
        if (!length || line[0] == '#')
            continue;
        char *arg = strchr(line, '\t');
        if (!arg)
            goto done;
        *arg++ = 0;
        char *value = strchr(arg, '\t');
        if (value)
            *value++ = 0;
        if (!*arg || (value && (!*value || strchr(value, '\t'))))
            goto done;

        if (!strcmp(line, "step") && !value) {
            if (strspn(arg, "0123456789") != strlen(arg))
                goto done;
            errno = 0;
            unsigned long index = strtoul(arg, NULL, 10);
            if (errno || index != next || next >= ysyx_kv_step_count(p))
                goto done;
            ysyx_stats_t part;
            if (ysyx_kv_step(session, next++, &part, trace, error))
                goto done;
            add_stats(stats, &part);
            if (trace)
                fprintf(trace, "KV_STEP step=%lu npu_domain_wall_ticks=%" PRIu64
                        " dma_bytes=%" PRIu64 "\n", index, part.wall_ticks, part.dma_bytes);
        } else if (value && !strcmp(line, "feedback")) {
            if (ysyx_kv_feedback(session, arg, value, error))
                goto done;
        } else if (value && (!strcmp(line, "input") || !strcmp(line, "output"))) {
            if (scripted_tensor(session, p, arg, value, !strcmp(line, "input"), error))
                goto done;
        } else {
            goto done;
        }
    }
    *error = "KV script must execute all steps in order";
    if (!ferror(script) && next == ysyx_kv_step_count(p))
        rc = 0;
done:
    if (rc && line_index)
        fprintf(stderr, "KV script %s line %zu\n", path, line_index);
    free(line);
    ysyx_kv_close(session);
    if (fclose(script) && !rc) {
        *error = "could not close KV script";
        rc = -1;
    }
    return rc;
}

int ysyx_run_cli(int argc, char **argv, int simulation_available)
{
    if (argc == 2 && !strcmp(argv[1], "--help")) {
        usage(stdout);
        return 0;
    }
    if (argc < 2) {
        usage(stderr);
        return 2;
    }
    const char *error = "invalid argument list";
    int status = 2, dump = 0, show = 0, timing = 0;
    const char *control = NULL, *dma = NULL, *kv_script = NULL;
    const char *backend = simulation_available ? "verilator" : "linux";
    uint8_t *data = NULL;
    ysyx_program_t *program = NULL;
    ysyx_hal_t hal = {0};
    size_t ninputs = 0, noutputs = 0;
    file_binding_t *inputs = calloc((size_t)argc, sizeof(*inputs));
    file_binding_t *outputs = calloc((size_t)argc, sizeof(*outputs));
    ysyx_input_t *values = calloc((size_t)argc, sizeof(*values));
    if (!inputs || !outputs || !values) {
        error = "out of memory at startup";
        goto done;
    }
    for (int i = 2; i < argc; ++i) {
        const char *opt = argv[i];
        if (!strcmp(opt, "--dump-stats")) {
            dump = 1;
        } else if (!strcmp(opt, "--host-timing")) {
            timing = 1;
        } else if (!strcmp(opt, "--inspect")) {
            show = 1;
        } else {
            if (i + 1 == argc)
                goto done;
            const char *value = argv[++i];
            if (!strcmp(opt, "--input")) {
                if (decode_binding(&inputs[ninputs], value))
                    goto done;
                ++ninputs;
            } else if (!strcmp(opt, "--output")) {
                if (decode_binding(&outputs[noutputs], value))
                    goto done;
                ++noutputs;
            } else if (!strcmp(opt, "--backend")) {
                backend = value;
            } else if (!strcmp(opt, "--uio-control")) {
                control = value;
            } else if (!strcmp(opt, "--uio-dma")) {
                dma = value;
            } else if (!strcmp(opt, "--kv-script")) {
                kv_script = value;
            } else {
                goto done;
            }
        }
    }

    error = "cannot read bounded YHB file";
    size_t size;
    data = read_file(argv[1], Z15_POOL_SIZE, &size);
    if (!data || ysyx_program_open(data, size, &program, &error))
        goto done;
    if (show) {
        inspect(program, size);
        status = 0;
        goto done;
    }
    if (ysyx_program_target(program) != YSYX_TARGET_Z15_BALANCED) {
        error = "bundle target lacks a runtime hardware binding";
        goto done;
    }
    for (size_t i = 0; i < ninputs; ++i) {
        size_t length;
        ysyx_symbol_t symbol;
        error = "unknown input name or input byte count does not match bundle";
        if (ysyx_find_symbol(program, inputs[i].name, YSYX_INPUT, &symbol))
            goto done;
        uint8_t *payload = read_file(inputs[i].path, symbol.bytes, &length);
        if (!payload || length != symbol.bytes) {
            free(payload);
            goto done;
        }
        values[i] = (ysyx_input_t){inputs[i].name, payload, length};
    }
    for (size_t i = 0; i < noutputs; ++i) {
        ysyx_symbol_t symbol;
        if (ysyx_find_symbol(program, outputs[i].name, YSYX_OUTPUT, &symbol)) {
            error = "unknown output name";
            goto done;
        }
    }
    if (!strcmp(backend, "verilator")) {
#ifdef YSYX_WITH_VERILATOR
        if (ysyx_hal_verilator_open(&hal, &error))
            goto done;
#else
        error = "this executable was built without the Verilator HAL";
        goto done;
#endif
    } else if (!strcmp(backend, "linux")) {
        if (ysyx_hal_linux_open(&hal, control, dma, &error))
            goto done;
    } else {
        error = "unknown backend name";
        goto done;
    }

    ysyx_stats_t stats;
    uint64_t start_ns = 0, end_ns;
    if (timing && monotonic_ns(&start_ns)) {
        error = "cannot read monotonic clock before execution";
        goto done;
    }
    FILE *trace = dump ? stdout : NULL;
    if ((kv_script ? run_kv_script(kv_script, program, &hal, values, ninputs, &stats, trace, &error) :
                     ysyx_execute(program, &hal, values, ninputs, &stats, trace, &error)))
        goto done;
    for (size_t i = 0; i < noutputs; ++i) {
        ysyx_symbol_t symbol;
        error = "cannot read output from DDR or write output file";
        if (ysyx_find_symbol(program, outputs[i].name, YSYX_OUTPUT, &symbol))
            goto done;
        uint8_t *payload = malloc(symbol.bytes);
        if (!payload)
            goto done;
        int rc = ysyx_read_output(program, &hal, outputs[i].name, payload, symbol.bytes) ||
                 write_file(outputs[i].path, payload, symbol.bytes);
        free(payload);
        if (rc)
            goto done;
    }
    if (timing) {
        if (monotonic_ns(&end_ns)) {
            error = "cannot read monotonic clock after output";
            goto done;
        }
        printf("HOST_TIMING elapsed_ns=%" PRIu64 "\n", end_ns - start_ns);
    }
    if (dump) {
        printf("STATS backend=%s npu_domain_wall_ticks=%" PRIu64
               " npu_busy_ticks=%" PRIu64 " gpu_busy_ticks=%" PRIu64 " npu_gpu_overlap_ticks=%" PRIu64
               " dma_operations=%" PRIu32 " dma_bytes=%" PRIu64
               " abi_error=0x%08" PRIx32 " invalid_dma_descriptors=%" PRIu32,
               backend, stats.wall_ticks, stats.npu_ticks, stats.gpu_ticks,
               stats.npu_gpu_overlap_ticks, stats.dma_operations, stats.dma_bytes,
               stats.abi_error, stats.invalid_dma_descriptors);
        if (stats.has_observations)
            printf(" axi_transactions=%" PRIu64 " dma_busy_ticks=%" PRIu64
                   " dma_compute_overlap_ticks=%" PRIu64 " triple_overlap_ticks=%" PRIu64
                   " gpu_clock_busy_ticks=%" PRIu64 " dma_gpu_overlap_ticks=%" PRIu64
                   " dma_tensor_overlap_ticks=%" PRIu64,
                   stats.observed.axi_transactions, stats.observed.dma_busy_ticks,
                   stats.observed.dma_compute_overlap_ticks, stats.observed.triple_overlap_ticks,
                   stats.observed.gpu_clock_busy_ticks, stats.observed.dma_gpu_overlap_ticks,
                   stats.observed.dma_tensor_overlap_ticks);
        putchar('\n');
    }
    status = 0;

done:
    if (status)
        fprintf(stderr, "ysyx_run: %s\n", error ? error : "unknown failure");
    if (hal.close)
        hal.close(hal.ctx);
    ysyx_program_close(program);
    free(data);
    for (size_t i = 0; i < ninputs; ++i) {
        free((void *)values[i].data);
        free(inputs[i].name);
    }
    for (size_t i = 0; i < noutputs; ++i)
        free(outputs[i].name);
    free(values);
    free(inputs);
    free(outputs);
    return status;
}

int main(int argc, char **argv)
{
#ifdef YSYX_WITH_VERILATOR
    return ysyx_run_cli(argc, argv, 1);
#else
    return ysyx_run_cli(argc, argv, 0);
#endif
}
