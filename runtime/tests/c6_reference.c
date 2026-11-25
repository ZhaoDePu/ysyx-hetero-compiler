/* Run one external-input C6 YHB across independent PTQ reference frames. */
#include "hal_reference_c6.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

static uint8_t *read_binary(const char *path, size_t *bytes)
{
    FILE *file = fopen(path, "rb");
    if (!file)
        return NULL;
    uint8_t *data = NULL;
    long end;
    if (!fseek(file, 0, SEEK_END) && (end = ftell(file)) > 0 &&
        (uintmax_t)end <= SIZE_MAX && !fseek(file, 0, SEEK_SET)) {
        *bytes = (size_t)end;
        data = malloc(*bytes);
        if (data && fread(data, 1, *bytes, file) != *bytes) {
            free(data);
            data = NULL;
        }
    }
    fclose(file);
    return data;
}

static int check(const ysyx_program_t *program, const ysyx_hal_t *hal, const char *folder, const char *scene)
{
    static const char *const outputs[] = {
        "boxes_stride8", "scores_stride8", "boxes_stride16", "scores_stride16",
        "boxes_stride32", "scores_stride32"
    };
    char path[1024];
    size_t pixels, input_bytes = 0;
    int n = snprintf(path, sizeof(path), "%s/%s.input_nhwc_s8.bin", folder, scene);
    uint8_t *rgb = NULL, *padded = NULL;
    int ok = -1;
    const char *error = NULL;
    ysyx_symbol_t shape;
    if (n < 0 || (size_t)n >= sizeof(path) ||
        ysyx_find_symbol(program, "images", YSYX_INPUT | YSYX_REQUIRED, &shape) ||
        shape.rank != 4 || shape.dims[0] != 1 || shape.dims[3] != 4)
        goto done;

    pixels = shape.bytes / 4u;
    rgb = read_binary(path, &input_bytes);
    padded = calloc(shape.bytes, 1);
    if (!padded || !rgb || input_bytes != 3u * pixels)
        goto done;
    for (size_t i = 0; i < pixels; ++i)
        memcpy(padded + 4u * i, rgb + 3u * i, 3);
    const ysyx_input_t input = {.name = "images", .data = padded, .size = shape.bytes};
    ysyx_stats_t stats = {0};
    if (ysyx_execute(program, hal, &input, 1, &stats, NULL, &error)) {
        fprintf(stderr, "%s: %s (ERROR=0x%" PRIx32 ")\n", scene, error ? error : "reference replay failed", stats.abi_error);
        goto done;
    }
    size_t total = 0;
    for (size_t i = 0; i < sizeof(outputs) / sizeof(outputs[0]); ++i) {
        n = snprintf(path, sizeof(path), "%s/%s.pl_out_%zu.bin", folder, scene, i);
        if (n < 0 || (size_t)n >= sizeof(path))
            goto done;
        size_t bytes = 0;
        uint8_t *expected = read_binary(path, &bytes);
        uint8_t *actual = expected ? malloc(bytes) : NULL;
        const int differs = !actual || ysyx_read_output(program, hal, outputs[i], actual, bytes) ||
                            memcmp(expected, actual, bytes);
        free(actual);
        free(expected);
        if (differs) {
            fprintf(stderr, "%s: mismatched software reference tensor %s\n", scene, outputs[i]);
            goto done;
        }
        total += bytes;
    }
    if (stats.abi_error || stats.invalid_dma_descriptors)
        goto done;
    printf("C6_REFERENCE_PASS scene=%s outputs=%zu bytes=%zu dma_bytes=%" PRIu64 "\n",
           scene, sizeof(outputs) / sizeof(outputs[0]), total, stats.dma_bytes);
    ok = 0;
done:
    free(padded);
    free(rgb);
    return ok;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fputs("usage: c6_reference PROGRAM.yhb VECTORS SCENE [SCENE...]\n", stderr);
        return 2;
    }
    size_t size = 0;
    uint8_t *image = read_binary(argv[1], &size);
    const char *error = NULL;
    ysyx_program_t *program = NULL;
    ysyx_hal_t hal = {0};
    int result = 1;
    if (!image || ysyx_program_open(image, size, &program, &error) ||
        ysyx_hal_reference_c6_open(&hal, &error)) {
        fprintf(stderr, "C6 reference startup: %s\n", error ? error : "cannot read YHB");
        goto done;
    }
    for (int i = 3; i < argc; ++i)
        if (check(program, &hal, argv[2], argv[i]))
            goto done;
    result = 0;
done:
    if (hal.close)
        hal.close(hal.ctx);
    ysyx_program_close(program);
    free(image);
    return result;
}
