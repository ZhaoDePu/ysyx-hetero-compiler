#include "demo_support.h"
#include "z15_regs.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

int demo_options(int argc, char **argv, const char *input_option, demo_options_t *opt)
{
    if (argc < 4 || argc % 2)
        return -1;
    opt->directory = argv[1];
    opt->backend = getenv("YSYX_BACKEND");
    opt->control_uio = getenv("YSYX_CONTROL_UIO");
    opt->dma_uio = getenv("YSYX_DMA_UIO");
    opt->input = NULL;
    if (!opt->backend)
        opt->backend = "linux";
    for (int i = 2; i < argc; i += 2) {
        if (!strcmp(argv[i], input_option) && !opt->input)
            opt->input = argv[i + 1];
        else if (!strcmp(argv[i], "--backend"))
            opt->backend = argv[i + 1];
        else if (!strcmp(argv[i], "--uio-control"))
            opt->control_uio = argv[i + 1];
        else if (!strcmp(argv[i], "--uio-dma"))
            opt->dma_uio = argv[i + 1];
        else
            return -1;
    }
    if (!opt->input || (strcmp(opt->backend, "linux") && strcmp(opt->backend, "verilator")))
        return -1;
    return 0;
}

int demo_path(char out[DEMO_PATH_BYTES], const char *directory, const char *leaf)
{
    int length = snprintf(out, DEMO_PATH_BYTES, "%s/%s", directory, leaf);
    return length < 0 || (size_t)length >= DEMO_PATH_BYTES ? -1 : 0;
}

int demo_label(const char *name)
{
    if (!*name || strlen(name) >= DEMO_WORD_BYTES)
        return 0;
    for (; *name; ++name)
        if (!isalnum((unsigned char)*name) && *name != '-' && *name != '_')
            return 0;
    return 1;
}

uint8_t *demo_read(const char *path, size_t limit, size_t *size)
{
    FILE *stream = fopen(path, "rb");
    if (!stream)
        return NULL;
    uint8_t *data = NULL;
    if (!fseek(stream, 0, SEEK_END)) {
        long length = ftell(stream);
        if (length > 0 && (unsigned long)length <= limit && !fseek(stream, 0, SEEK_SET)) {
            *size = (size_t)length;
            data = malloc(*size);
            if (data && fread(data, 1, *size, stream) != *size) {
                free(data);
                data = NULL;
            }
        }
    }
    if (fclose(stream)) {
        free(data);
        data = NULL;
    }
    return data;
}

int demo_open(demo_t *demo, const demo_options_t *opt, const char **error)
{
    char path[DEMO_PATH_BYTES];
    size_t length = 0;
    *error = "could not read program.yhb";
    if (demo_path(path, opt->directory, "program.yhb"))
        return -1;
    demo->image = demo_read(path, Z15_POOL_SIZE, &length);
    if (!demo->image || ysyx_program_open(demo->image, length, &demo->program, error))
        return -1;
    if (ysyx_program_target(demo->program) != YSYX_TARGET_Z15_BALANCED) {
        *error = "demo requires a Balanced YHB";
        return -1;
    }
    if (!strcmp(opt->backend, "linux"))
        return ysyx_hal_linux_open(&demo->hal, opt->control_uio, opt->dma_uio, error);
#ifdef YSYX_WITH_VERILATOR
    return ysyx_hal_verilator_open(&demo->hal, error);
#else
    *error = "this demo was built without the Verilator HAL";
    return -1;
#endif
}

void demo_close(demo_t *demo)
{
    if (demo->hal.close)
        demo->hal.close(demo->hal.ctx);
    ysyx_program_close(demo->program);
    free(demo->image);
}

void demo_accumulate(ysyx_stats_t *total, const ysyx_stats_t *step)
{
    total->wall_ticks += step->wall_ticks;
    total->npu_ticks += step->npu_ticks;
    total->gpu_ticks += step->gpu_ticks;
    total->npu_gpu_overlap_ticks += step->npu_gpu_overlap_ticks;
    total->dma_bytes += step->dma_bytes;
    total->dma_operations += step->dma_operations;
    total->abi_error |= step->abi_error;
    total->invalid_dma_descriptors += step->invalid_dma_descriptors;
}

void demo_print_stats(const ysyx_stats_t *stats, const char *items_name, unsigned items)
{
    const double ms164 = stats->wall_ticks / 164555.0;
    const double ms125 = stats->wall_ticks / 125000.0;
    printf("STATS wall=%" PRIu64 " npu=%" PRIu64 " gpu=%" PRIu64 " overlap=%" PRIu64
           " dma_bytes=%" PRIu64 " error=0x%08" PRIx32 " invalid_dma=%" PRIu32 "\n",
           stats->wall_ticks, stats->npu_ticks, stats->gpu_ticks, stats->npu_gpu_overlap_ticks,
           stats->dma_bytes, stats->abi_error, stats->invalid_dma_descriptors);
    if (items && stats->wall_ticks) {
        printf("164.555 MHz: %.6f ms/%s %.3f %s/s\n", ms164/items, items_name, 1000.0*items/ms164, items_name);
        printf("125.00 MHz: %.6f ms/%s %.3f %s/s\n", ms125/items, items_name, 1000.0*items/ms125, items_name);
    }
}

int demo_dictionary(const char *path, char entries[DEMO_VOCAB_SIZE][DEMO_WORD_BYTES])
{
    FILE *stream = fopen(path, "rb");
    if (!stream)
        return -1;
    char line[DEMO_WORD_BYTES+24];
    int status = 0, count = 0;
    while (fgets(line, sizeof(line), stream)) {
        char *end;
        errno = 0;
        long id = strtol(line, &end, 10);
        if (end == line || *end != '\t' || errno || id < 0 || id >= DEMO_VOCAB_SIZE || entries[id][0]) {
            status = -1;
            break;
        }
        char *term = strpbrk(end + 1, "\r\n");
        if (!term || (size_t)(term - end - 1) >= DEMO_WORD_BYTES || term == end + 1) {
            status = -1;
            break;
        }
        *term = '\0';
        memcpy(entries[id], end + 1, strlen(end + 1) + 1);
        ++count;
    }
    if (ferror(stream))
        status = -1;
    if (fclose(stream) || !count)
        status = -1;
    return status;
}

size_t demo_integer_width(uint32_t dtype)
{
    switch (dtype) {
    case 1: case 2: return 1;
    case 3: return 4;
    case 4: return 8;
    case 5: case 6: return 2;
    default: return 0;
    }
}

int64_t demo_integer(const uint8_t *data, uint32_t dtype)
{
    size_t width = demo_integer_width(dtype);
    uint64_t bits = 0;
    for (size_t i = 0; i < width; ++i)
        bits |= (uint64_t)data[i] << (8 * i);
    if (width == 0 || dtype == 2 || dtype == 5)
        return (int64_t)bits;
    if (width < 8 && (bits & (UINT64_C(1) << (width*8-1))))
        bits |= UINT64_MAX << (width*8);
    return bits <= INT64_MAX ? (int64_t)bits : -1 - (int64_t)(UINT64_MAX - bits);
}
