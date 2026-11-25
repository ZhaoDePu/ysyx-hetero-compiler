#include "demo_support.h"

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    ysyx_symbol_t symbol;
    uint8_t *data;
} output_t;

static int read_output(output_t *out, const demo_t *demo, const char *name)
{
    if (ysyx_find_symbol(demo->program, name, YSYX_OUTPUT, &out->symbol) ||
            !demo_integer_width(out->symbol.dtype) || out->symbol.bytes > (1u << 20))
        return -1;
    out->data = malloc(out->symbol.bytes);
    return !out->data || ysyx_read_output(demo->program, &demo->hal, name, out->data, out->symbol.bytes);
}

static size_t elements(const output_t *out)
{
    return out->symbol.bytes / demo_integer_width(out->symbol.dtype);
}

static int64_t read_integer(const output_t *out, size_t i)
{
    return demo_integer(out->data + i * demo_integer_width(out->symbol.dtype), out->symbol.dtype);
}

static int calibration(const char *path, double *score_scale, double *threshold, double *box_scale)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    int count = fscanf(f, "score_scale%lf score_threshold%lf box_codes_per_pixel%lf",
                       score_scale, threshold, box_scale);
    int bad = fclose(f);
    if (bad || count != 3 || !isfinite(*score_scale) || !isfinite(*threshold) ||
            !isfinite(*box_scale) || *score_scale <= 0 || *threshold < 0 || *threshold > 1 || *box_scale <= 0)
        return -1;
    return 0;
}

int main(int argc, char **argv)
{
    static const char usage[] = "z15_yolo26_demo CASE_DIR --image NAME [--backend linux|verilator]\n";
    if (argc == 2 && !strcmp(argv[1], "--help")) {
        fputs(usage, stdout);
        return 0;
    }
    demo_options_t opt;
    if (demo_options(argc, argv, "--image", &opt) || !demo_label(opt.input)) {
        fputs(usage, stderr);
        return 2;
    }

    demo_t demo = {0};
    char path[DEMO_PATH_BYTES], leaf[DEMO_WORD_BYTES+12];
    char labels[DEMO_VOCAB_SIZE][DEMO_WORD_BYTES] = {{0}};
    double score_scale, threshold, box_scale;
    uint8_t *image = NULL;
    output_t scores = {0}, classes = {0}, boxes = {0}, indices = {0};
    ysyx_symbol_t image_symbol;
    const char *error = "could not load case labels or quantization";
    int result = 1;
    if (demo_path(path, opt.directory, "labels.tsv") || demo_dictionary(path, labels) ||
            demo_path(path, opt.directory, "detection.tsv") ||
            calibration(path, &score_scale, &threshold, &box_scale))
        goto done;
    if (demo_open(&demo, &opt, &error))
        goto done;
    size_t size = 0;
    error = "unknown or incorrectly sized sample image";
    snprintf(leaf, sizeof(leaf), "images/%s.bin", opt.input);
    if (ysyx_find_symbol(demo.program, "image", YSYX_INPUT, &image_symbol) ||
            demo_path(path, opt.directory, leaf))
        goto done;
    image = demo_read(path, image_symbol.bytes, &size);
    if (!image || size != image_symbol.bytes)
        goto done;
    ysyx_input_t input = {"image", image, size};
    ysyx_stats_t stats;
    if (ysyx_execute(demo.program, &demo.hal, &input, 1, &stats, NULL, &error))
        goto done;
    error = "missing or inconsistent detector output tensors";
    if (read_output(&scores, &demo, "scores") || read_output(&classes, &demo, "classes") ||
            read_output(&boxes, &demo, "boxes") || read_output(&indices, &demo, "indices"))
        goto done;
    const size_t count = elements(&scores);
    if (!count || count != elements(&classes) || count != elements(&indices) ||
            boxes.symbol.rank != 2 || boxes.symbol.dims[1] != 4 || elements(&boxes) != 4*count)
        goto done;
    printf("IMAGE %s\n", opt.input);
    unsigned detections = 0;
    for (size_t i = 0; i < count; ++i) {
        double score = 1.0 / (1.0 + exp(-read_integer(&scores, i)*score_scale));
        if (score <= threshold)
            continue;
        int64_t cls = read_integer(&classes, i);
        if (cls < 0 || cls >= DEMO_VOCAB_SIZE || !labels[cls][0]) {
            error = "device class is outside the case labels";
            goto done;
        }
        printf("BOX class=%s score=%.5f x1=%.2f y1=%.2f x2=%.2f y2=%.2f index=%" PRId64 "\n",
               labels[cls], score, read_integer(&boxes, 4*i)/box_scale,
               read_integer(&boxes, 4*i+1)/box_scale, read_integer(&boxes, 4*i+2)/box_scale,
               read_integer(&boxes, 4*i+3)/box_scale, read_integer(&indices, i));
        ++detections;
    }
    printf("DETECTIONS %u\n", detections);
    demo_print_stats(&stats, "image", 1);
    result = 0;

done:
    free(scores.data);
    free(classes.data);
    free(boxes.data);
    free(indices.data);
    free(image);
    demo_close(&demo);
    if (result)
        fprintf(stderr, "z15_yolo26_demo: %s\n", error);
    return result;
}
