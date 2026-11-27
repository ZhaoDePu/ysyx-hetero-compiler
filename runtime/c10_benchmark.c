/* Frozen Z15 C10 fixture benchmark. Stdout is CSV except for probe modes. */
#define _POSIX_C_SOURCE 200809L
#include "z15_exec_option.h"
#include "c10_session.h"
#include "hal_reference_c6.h"
#include "z15_regs.h"

#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MAX_FRAMES = 16, MAX_INPUTS = 16, MAX_OUTPUTS = 192 };
enum backend { LINUX, REFERENCE, VERILATOR };

typedef struct {
    char *name;
    uint8_t *bytes;
    size_t size;
} binding_t;

typedef struct {
    char *id;
    binding_t inputs[MAX_INPUTS], outputs[MAX_OUTPUTS];
    size_t input_count, output_count;
} frame_t;

typedef struct {
    ysyx_program_t *program;
    uint8_t *image;
    frame_t frames[MAX_FRAMES];
    size_t frame_count, max_output;
} fixture_t;

typedef struct {
    const char *directory, *profile, *control, *dma, *reject, *fixed_fixture;
    const char *reason;
    unsigned frames;
    enum backend backend;
    int probe, selftest, scan_clock, phase17;
} options_t;

typedef struct {
    ysyx_stats_t stats;
    uint64_t compare_bytes, static_bytes, input_bytes;
    double h2d, exec, read, total;
    int has_counters, has_upload;
} result_t;

static double now_ms(void)
{
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value))
        return -1;
    return (double)value.tv_sec * 1000.0 + value.tv_nsec / 1000000.0;
}

static const char *model_name(const char *directory)
{
    const char *name = strrchr(directory, '/');
    return name ? name + 1 : directory;
}

static int simple_name(const char *name)
{
    return *name && strcmp(name, ".") && strcmp(name, "..") &&
           strspn(name, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") ==
           strlen(name);
}

static FILE *open_path(const char *directory, const char *name)
{
    char path[4096];
    if (directory) {
        int length = snprintf(path, sizeof(path), "%s/%s", directory, name);
        if (length < 0 || (size_t)length >= sizeof(path))
            return NULL;
        name = path;
    }
    return fopen(name, "rb");
}

static uint8_t *load_file(const char *directory, const char *name, size_t *size)
{
    FILE *file = open_path(directory, name);
    if (!file) {
        fprintf(stderr, "cannot open fixture file %s\n", name);
        return NULL;
    }
    uint8_t *data = NULL;
    long bytes;
    if (!fseek(file, 0, SEEK_END) && (bytes = ftell(file)) > 0 &&
        bytes <= 512L * 1024L * 1024L && !fseek(file, 0, SEEK_SET)) {
        data = malloc((size_t)bytes);
        if (data && fread(data, 1, (size_t)bytes, file) == (size_t)bytes)
            *size = (size_t)bytes;
        else {
            free(data);
            data = NULL;
        }
    }
    if (!data)
        fprintf(stderr, "cannot read fixture file %s\n", name);
    fclose(file);
    return data;
}

static void free_bindings(binding_t *bindings, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        free(bindings[i].name);
        free(bindings[i].bytes);
    }
}

static void close_fixture(fixture_t *fixture)
{
    for (size_t i = 0; i < fixture->frame_count; ++i) {
        frame_t *frame = &fixture->frames[i];
        free(frame->id);
        free_bindings(frame->inputs, frame->input_count);
        free_bindings(frame->outputs, frame->output_count);
    }
    ysyx_program_close(fixture->program);
    free(fixture->image);
}

static frame_t *find_frame(fixture_t *fixture, const char *id)
{
    for (size_t i = 0; i < fixture->frame_count; ++i)
        if (!strcmp(fixture->frames[i].id, id))
            return &fixture->frames[i];
    if (fixture->frame_count == MAX_FRAMES)
        return NULL;
    frame_t *frame = &fixture->frames[fixture->frame_count];
    frame->id = strdup(id);
    if (!frame->id)
        return NULL;
    ++fixture->frame_count;
    return frame;
}

static int add_binding(fixture_t *fixture, const char *directory, char *line)
{
    char *saved = NULL;
    const char *direction = strtok_r(line, "\t", &saved);
    const char *id = strtok_r(NULL, "\t", &saved);
    const char *symbol = strtok_r(NULL, "\t", &saved);
    const char *filename = strtok_r(NULL, "\t", &saved);
    if (!direction || !id || !symbol || !filename || strtok_r(NULL, "\t", &saved) ||
        !simple_name(id) || !simple_name(symbol) || !simple_name(filename))
        return -1;
    int input = !strcmp(direction, "input");
    if (!input && strcmp(direction, "output"))
        return -1;
    ysyx_symbol_t spec;
    if (ysyx_find_symbol(fixture->program, symbol, input ? YSYX_INPUT : YSYX_OUTPUT, &spec))
        return -1;
    frame_t *frame = find_frame(fixture, id);
    if (!frame)
        return -1;
    size_t *count = input ? &frame->input_count : &frame->output_count;
    binding_t *bindings = input ? frame->inputs : frame->outputs;
    if (*count == (input ? MAX_INPUTS : MAX_OUTPUTS))
        return -1;
    for (size_t i = 0; i < *count; ++i)
        if (!strcmp(symbol, bindings[i].name))
            return -1;
    binding_t *binding = &bindings[(*count)++];
    binding->name = strdup(symbol);
    binding->bytes = load_file(directory, filename, &binding->size);
    if (!binding->name || !binding->bytes || binding->size != spec.bytes)
        return -1;
    if (!input && binding->size > fixture->max_output)
        fixture->max_output = binding->size;
    return 0;
}

static int open_fixture(fixture_t *fixture, const char *directory, int phase17)
{
    size_t image_bytes;
    const char *error = NULL;
    const char *file = phase17 ? "program.yhp" : "program.yhb";
    fixture->image = load_file(directory, file, &image_bytes);
    if (!fixture->image)
        return -1;
    if (phase17) {
        if (image_bytes < 4u || memcmp(fixture->image, "YHP1", 4))
            return -1;
        memcpy(fixture->image, "YHB1", 4);
    }
    if (ysyx_program_open(fixture->image, image_bytes, &fixture->program, &error)) {
        fprintf(stderr, "invalid %s: %s\n", file, error ? error : "unknown error");
        return -1;
    }
    if (ysyx_program_target(fixture->program) != YSYX_TARGET_Z15_BALANCED)
        return -1;
    FILE *table = open_path(directory, "io.tsv");
    if (!table)
        return -1;
    char *line = NULL;
    size_t size = 0;
    int rc = 0;
    while (getline(&line, &size, table) >= 0) {
        line[strcspn(line, "\r\n")] = '\0';
        if (add_binding(fixture, directory, line)) {
            rc = -1;
            break;
        }
    }
    if (ferror(table) || !fixture->frame_count)
        rc = -1;
    for (size_t i = 0; i < fixture->frame_count; ++i)
        if (!fixture->frames[i].input_count || !fixture->frames[i].output_count)
            rc = -1;
    free(line);
    fclose(table);
    return rc;
}

static int reject_fixture(const options_t *options)
{
    size_t size;
    uint8_t *image = load_file(NULL, options->reject, &size);
    if (!image)
        return -1;
    ysyx_program_t *program = NULL;
    const char *error = NULL;
    int rejected = ysyx_program_open(image, size, &program, &error);
    int pass = rejected && error && !strcmp(error, options->reason);
    if (pass)
        printf("PASS rejected %s: %s\n", options->reject, error);
    else
        fprintf(stderr, "unexpected result opening %s: %s\n", options->reject,
                error ? error : "accepted");
    ysyx_program_close(program);
    free(image);
    return pass ? 0 : -1;
}

static int open_backend(const options_t *options, ysyx_hal_t *hal, const char **error)
{
    switch (options->backend) {
    case LINUX:
        return ysyx_hal_linux_open(hal, options->control, options->dma, error);
    case REFERENCE:
        return options->phase17 ? ysyx_hal_reference_p17_open(hal, error) :
                                  ysyx_hal_reference_c6_open(hal, error);
    case VERILATOR:
#ifdef YSYX_WITH_VERILATOR
        return ysyx_hal_verilator_open(hal, error);
#else
        *error = "this binary was built without the Verilator HAL";
        return -1;
#endif
    }
    return -1;
}

static int probe(const ysyx_hal_t *hal, int scratch_test, int quiet)
{
    uint32_t magic, version, status, error;
    if (hal->mmio_read32(hal->ctx, Z15_MAGIC, &magic) ||
        hal->mmio_read32(hal->ctx, Z15_VERSION, &version) ||
        hal->mmio_read32(hal->ctx, Z15_STATUS, &status) ||
        hal->mmio_read32(hal->ctx, Z15_ERROR, &error))
        return -1;
    if (magic != Z15_MAGIC_VALUE || (version & Z15_VERSION_COMPAT_MASK) != Z15_VERSION_VALUE ||
        !(status & Z15_LOCKED) || (status & (Z15_DMA_BUSY | Z15_NPU_BUSY | Z15_GPU_BUSY |
                                           Z15_HAS_ERROR)) || error) {
        fprintf(stderr, "Z15 probe rejected MAGIC=0x%08" PRIx32 " VERSION=0x%08" PRIx32
                " STATUS=0x%08" PRIx32 " ERROR=0x%08" PRIx32 "\n",
                magic, version, status, error);
        return -1;
    }
    if (scratch_test) {
        uint32_t original, readback;
        if (hal->mmio_read32(hal->ctx, Z15_SCRATCH, &original))
            return -1;
        int failed = 0;
        const uint32_t patterns[] = {UINT32_C(0x55aa33cc), UINT32_C(0xaa55cc33)};
        for (size_t i = 0; i < sizeof(patterns) / sizeof(patterns[0]); ++i) {
            if (hal->mmio_write32(hal->ctx, Z15_SCRATCH, patterns[i]) ||
                hal->mmio_read32(hal->ctx, Z15_SCRATCH, &readback) ||
                readback != patterns[i]) {
                failed = 1;
                break;
            }
        }
        if (hal->mmio_write32(hal->ctx, Z15_SCRATCH, original) ||
            hal->mmio_read32(hal->ctx, Z15_SCRATCH, &readback) || readback != original ||
            hal->mmio_read32(hal->ctx, Z15_ERROR, &error) || error || failed) {
            fputs("Z15 SCRATCH round trip or restoration failed\n", stderr);
            return -1;
        }
    }
    if (!quiet)
        printf("PASS MAGIC=0x%08" PRIx32 " VERSION=0x%08" PRIx32
               " STATUS=0x%08" PRIx32 " ERROR=%" PRIu32 " SCRATCH=%s\n",
               magic, version, status, error, scratch_test ? "restored" : "not_tested");
    return 0;
}

static int read_wall(const ysyx_hal_t *hal, uint64_t *counter)
{
    for (unsigned attempt = 0; attempt < 8; ++attempt) {
        uint32_t hi, lo, hi2;
        if (hal->mmio_read32(hal->ctx, Z15_WALL_CYCLES + 4, &hi) ||
            hal->mmio_read32(hal->ctx, Z15_WALL_CYCLES, &lo) ||
            hal->mmio_read32(hal->ctx, Z15_WALL_CYCLES + 4, &hi2))
            return -1;
        if (hi == hi2) {
            *counter = ((uint64_t)hi << 32) | lo;
            return 0;
        }
    }
    return -1;
}

static int detect_clock(const ysyx_hal_t *hal)
{
    int profile = 0;
    double average = 0;
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
        uint64_t first, last;
        double start = now_ms();
        if (start < 0 || read_wall(hal, &first))
            return -1;
        struct timespec delay = {0, 30000000};
        while (nanosleep(&delay, &delay))
            if (errno != EINTR)
                return -1;
        if (read_wall(hal, &last))
            return -1;
        double finish = now_ms();
        if (finish <= start || last <= first)
            return -1;
        double mhz = (double)(last - first) / ((finish - start) * 1000);
        int choice = (mhz >= 121 && mhz <= 129) ? 125 :
                     (mhz >= 194 && mhz <= 206) ? 200 : 0;
        if (!choice || (profile && profile != choice)) {
            fprintf(stderr, "unsupported or unstable NPU clock (sample %.3f MHz)\n", mhz);
            return -1;
        }
        profile = choice;
        average += mhz / 3;
    }
    fprintf(stderr, "WALL measured %.3f MHz NPU; matched supported pair %d/%d MHz "
            "(GPU clock inferred from profile)\n", average, profile,
            profile == 125 ? 100 : 150);
    puts(profile == 125 ? "125_100" : "200_150");
    return 0;
}

static void add_stats(ysyx_stats_t *sum, const ysyx_stats_t *step)
{
    sum->wall_ticks += step->wall_ticks;
    sum->npu_ticks += step->npu_ticks;
    sum->gpu_ticks += step->gpu_ticks;
    sum->npu_gpu_overlap_ticks += step->npu_gpu_overlap_ticks;
    sum->dma_operations += step->dma_operations;
    sum->dma_bytes += step->dma_bytes;
    sum->observed.dma_busy_ticks += step->observed.dma_busy_ticks;
    sum->observed.dma_compute_overlap_ticks += step->observed.dma_compute_overlap_ticks;
    sum->has_observations = step->has_observations;
    sum->abi_error |= step->abi_error;
    sum->invalid_dma_descriptors += step->invalid_dma_descriptors;
}

static int run_kv(const fixture_t *fixture, const ysyx_hal_t *hal, const ysyx_input_t *inputs,
                  size_t count, ysyx_kv_session_t **session, result_t *result, const char **error)
{
    double start = now_ms();
    if (ysyx_kv_begin(fixture->program, hal, inputs, count, session, error))
        return -1;
    double execute_start = now_ms();
    result->h2d = execute_start - start;
    uint32_t steps = ysyx_kv_step_count(fixture->program);
    for (uint32_t i = 0; i < steps; ++i) {
        ysyx_stats_t step = {0};
        if (ysyx_kv_step(*session, i, &step, NULL, error))
            return -1;
        add_stats(&result->stats, &step);
    }
    result->exec = now_ms() - execute_start;
    return 0;
}

static int check_outputs(const frame_t *frame, uint8_t *buffer, ysyx_c10_session_t *resident,
                         ysyx_kv_session_t *kv, result_t *result)
{
    double start = now_ms();
    for (size_t i = 0; i < frame->output_count; ++i) {
        const binding_t *output = &frame->outputs[i];
        int rc = kv ? ysyx_kv_read_output(kv, output->name, buffer, output->size) :
                      ysyx_c10_session_read(resident, output->name, buffer, output->size);
        if (rc) {
            fprintf(stderr, "%s: cannot read output %s\n", frame->id, output->name);
            return -1;
        }
        if (memcmp(buffer, output->bytes, output->size)) {
            size_t offset = 0;
            while (buffer[offset] == output->bytes[offset])
                ++offset;
            fprintf(stderr, "%s: %s differs at byte %zu (%02x vs %02x)\n", frame->id,
                    output->name, offset, (unsigned)buffer[offset],
                    (unsigned)output->bytes[offset]);
            return -1;
        }
        result->compare_bytes += output->size;
    }
    result->read = now_ms() - start;
    return 0;
}

static int execute_frame(const fixture_t *fixture, const frame_t *frame, const ysyx_hal_t *hal,
                         ysyx_c10_session_t *resident, uint8_t *buffer, result_t *result,
                         const char **error)
{
    ysyx_input_t inputs[MAX_INPUTS];
    for (size_t i = 0; i < frame->input_count; ++i)
        inputs[i] = (ysyx_input_t){frame->inputs[i].name, frame->inputs[i].bytes,
                                   frame->inputs[i].size};
    double start = now_ms();
    ysyx_kv_session_t *kv = NULL;
    int rc;
    if (resident) {
        ysyx_c10_timing_t timing = {0};
        rc = ysyx_c10_session_run(resident, inputs, frame->input_count, &result->stats,
                                  &timing, NULL, error);
        result->h2d = timing.t_h2d_ms;
        result->exec = timing.t_execute_ms;
        result->static_bytes = timing.static_h2d_bytes;
        result->input_bytes = timing.dynamic_h2d_bytes;
        result->has_upload = 1;
    } else {
        rc = run_kv(fixture, hal, inputs, frame->input_count, &kv, result, error);
    }
    if (!rc && (result->stats.abi_error || result->stats.invalid_dma_descriptors)) {
        *error = "nonzero ABI error or illegal DMA";
        rc = -1;
    }
    if (!rc)
        rc = check_outputs(frame, buffer, resident, kv, result);
    ysyx_kv_close(kv);
    result->total = now_ms() - start;
    return rc;
}

static void print_row(const options_t *options, const frame_t *frame, unsigned index,
                      uint32_t kv_steps, const result_t *result)
{
    const char *backends[] = {"board", "software_reference", "rtl_simulator"};
    const ysyx_stats_t *s = &result->stats;
    size_t work = kv_steps ? frame->input_count * kv_steps : 1;
    printf("%s,%s,%s,%u,%s,%s,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,",
           model_name(options->directory), options->profile, backends[options->backend],
           index + 1, frame->id, kv_steps ? "generated_token" : "frame", work, result->total,
           result->h2d, result->exec, result->read,
           result->total > result->exec ? result->total - result->exec : 0.0);
    if (result->has_counters) {
        double mhz = !strcmp(options->profile, "125_100") ? 125.0 : 200.0;
        uint64_t busy = s->npu_ticks + s->gpu_ticks - s->npu_gpu_overlap_ticks;
        printf("%.6f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",,,",
               s->wall_ticks / (mhz * 1000), s->wall_ticks, busy, s->npu_ticks, s->gpu_ticks,
               s->npu_gpu_overlap_ticks);
    } else {
        fputs(",,,,,,,,", stdout);
    }
    if (s->has_observations)
        printf("%" PRIu64 ",%" PRIu64, s->observed.dma_busy_ticks,
               s->observed.dma_compute_overlap_ticks);
    else
        putchar(',');
    printf(",%" PRIu64 ",%" PRIu32 ",", s->dma_bytes, s->dma_operations);
    if (result->has_upload)
        printf("%" PRIu64 ",%" PRIu64, result->static_bytes, result->input_bytes);
    else
        putchar(',');
    printf(",%zu,%" PRIu64 ",byte_exact,%" PRIu32 ",%s,%" PRIu32 "\n",
           frame->output_count, result->compare_bytes, s->abi_error,
           options->backend == VERILATOR ? "0" : "", s->invalid_dma_descriptors);
}

static int benchmark(const options_t *options, const ysyx_hal_t *hal)
{
    fixture_t *fixture = calloc(1, sizeof(*fixture));
    if (!fixture)
        return -1;
    int rc = -1;
    uint8_t *buffer = NULL;
    ysyx_c10_session_t *resident = NULL;
    const char *error = NULL;
    if (open_fixture(fixture, options->directory, options->phase17)) {
        fputs("fixture program, io.tsv or tensor file is invalid\n", stderr);
        goto done;
    }
    uint32_t kv_steps = ysyx_kv_step_count(fixture->program);
    buffer = malloc(fixture->max_output);
    if (!buffer || (!kv_steps && ysyx_c10_session_open(fixture->program, hal, &resident, &error)))
        goto done;
    const frame_t *fixed = NULL;
    if (options->fixed_fixture) {
        for (size_t i = 0; i < fixture->frame_count; ++i) {
            if (!strcmp(fixture->frames[i].id, options->fixed_fixture))
                fixed = &fixture->frames[i];
        }
        if (!fixed) {
            fprintf(stderr, "unknown fixture ID: %s\n", options->fixed_fixture);
            goto done;
        }
    }
    puts("model,freq_profile,measurement_kind,sample,fixture,unit,work_units,wall_ms,h2d_ms,"
         "execute_host_ms,d2h_verify_ms,ps_outside_exec_ms,pl_hw_ms,wall_cycles,"
         "native_gpu_union_cycles,native_vpu_busy_cycles,gpu_busy_cycles,"
         "native_gpu_overlap_cycles,npu_busy_cycles,vpu_busy_cycles,dma_busy_cycles,"
         "dma_compute_overlap_cycles,dma_bytes,dma_descriptors,static_h2d_bytes,"
         "dynamic_h2d_bytes,outputs,compared_bytes,verification,abi_error,slverr,invalid_dma");
    for (unsigned i = 0; i < options->frames; ++i) {
        const frame_t *frame = fixed ? fixed : &fixture->frames[i % fixture->frame_count];
        result_t result = {0};
        result.has_counters = options->backend != REFERENCE;
        if (execute_frame(fixture, frame, hal, resident, buffer, &result, &error)) {
            fprintf(stderr, "sample %u (%s) did not validate\n", i + 1, frame->id);
            goto done;
        }
        print_row(options, frame, i, kv_steps, &result);
        if (fflush(stdout))
            goto done;
    }
    rc = 0;
done:
    if (error)
        fprintf(stderr, "%s\n", error);
    ysyx_c10_session_close(resident);
    free(buffer);
    close_fixture(fixture);
    free(fixture);
    return rc;
}

static int parse(char *const *args, int count, options_t *options)
{
    *options = (options_t){.profile = "125_100", .frames = 1};
    for (int i = 1; i < count; ++i) {
        const char *arg = args[i];
        if (!strcmp(arg, "--probe"))
            options->probe = 1;
        else if (!strcmp(arg, "--self-test"))
            options->selftest = 1;
        else if (!strcmp(arg, "--detect-clock"))
            options->scan_clock = 1;
        else if (!strcmp(arg, "--reject") && i + 2 < count) {
            options->reject = args[++i];
            options->reason = args[++i];
        } else if (i + 1 < count) {
            const char *value = args[++i];
            if (!strcmp(arg, "--case"))
                options->directory = value;
            else if (!strcmp(arg, "--target") && (!strcmp(value, "step2") || !strcmp(value, "p17")))
                options->phase17 = !strcmp(value, "p17");
            else if (!strcmp(arg, "--fixture") && simple_name(value))
                options->fixed_fixture = value;
            else if (!strcmp(arg, "--freq-profile") &&
                     (!strcmp(value, "125_100") || !strcmp(value, "200_150")))
                options->profile = value;
            else if (!strcmp(arg, "--uio-control"))
                options->control = value;
            else if (!strcmp(arg, "--uio-dma"))
                options->dma = value;
            else if (!strcmp(arg, "--backend") && !strcmp(value, "linux"))
                options->backend = LINUX;
            else if (!strcmp(arg, "--backend") && !strcmp(value, "reference"))
                options->backend = REFERENCE;
            else if (!strcmp(arg, "--backend") && !strcmp(value, "verilator"))
                options->backend = VERILATOR;
            else if (!strcmp(arg, "--frames")) {
                char *end;
                errno = 0;
                unsigned long frames = strtoul(value, &end, 10);
                if (errno || !*value || *end || !frames || frames > 100000ul)
                    return -1;
                options->frames = (unsigned)frames;
            } else
                return -1;
        } else
            return -1;
    }
    if (!!options->directory + !!options->reject + options->probe + options->selftest +
            options->scan_clock != 1 ||
        (options->directory && !simple_name(model_name(options->directory))) ||
        (options->fixed_fixture && !options->directory) ||
        (options->scan_clock && options->backend != LINUX))
        return -1;
    return 0;
}

int main(int argc, char **argv)
{
    if (z15_exec_option(&argc, argv))
        return 2;
    options_t options;
    if (parse(argv, argc, &options)) {
        fputs("usage: z15_c10_benchmark --case DIR [--backend linux|reference|verilator]\n"
              "       [--frames N] [--fixture ID] [--freq-profile 125_100|200_150]\n"
              "       [--target step2|p17] [--exec-mode auto|cmdq|mmio] [--exec-stats]\n"
              "       [--uio-control ID] [--uio-dma ID]\n"
              "       z15_c10_benchmark --probe|--self-test|--detect-clock [--backend linux]\n"
              "       z15_c10_benchmark --reject YHB EXPECTED_ERROR\n", stderr);
        return 2;
    }
    if (options.reject)
        return reject_fixture(&options) ? 1 : 0;
    ysyx_hal_t hal = {0};
    const char *error = NULL;
    if (open_backend(&options, &hal, &error)) {
        fprintf(stderr, "cannot open HAL: %s\n", error ? error : "unknown error");
        return 1;
    }
    int rc = probe(&hal, options.selftest, options.directory || options.scan_clock);
    if (!rc && options.directory)
        rc = benchmark(&options, &hal);
    else if (!rc && options.scan_clock)
        rc = detect_clock(&hal);
    if (hal.close)
        hal.close(hal.ctx);
    return rc ? 1 : 0;
}
