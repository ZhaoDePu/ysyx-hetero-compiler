#include "yolo26n_target.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ppm_token(FILE *in, char *out, size_t capacity)
{
    int ch;
    do {
        ch = fgetc(in);
        if (ch == '#')
            while (ch != EOF && ch != '\n')
                ch = fgetc(in);
    } while (ch != EOF && isspace(ch));
    size_t length = 0;
    while (ch != EOF && !isspace(ch)) {
        if (length + 1 >= capacity)
            return -1;
        out[length++] = (char)ch;
        ch = fgetc(in);
    }
    if (!length || ch == EOF)
        return -1;
    if (ch == '\r') {
        ch = fgetc(in);
        if (ch != EOF && ch != '\n')
            (void)ungetc(ch, in);
    }
    out[length] = 0;
    return 0;
}

static int read_rgb(const char *path, int ppm, uint8_t *rgb)
{
    FILE *in = fopen(path, "rb");
    if (!in)
        return -1;
    int status = -1;
    if (ppm) {
        static const char *const expected[] = {"P6", "320", "320", "255"};
        char token[32];
        for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i)
            if (ppm_token(in, token, sizeof(token)) || strcmp(token, expected[i]))
                goto done;
    }
    if (fread(rgb, 1, YSYX_YOLO26N_RGB_BYTES, in) == YSYX_YOLO26N_RGB_BYTES
            && fgetc(in) == EOF && !ferror(in))
        status = 0;
done:
    fclose(in);
    if (status)
        errno = EBADMSG;
    return status;
}

static void print_result(const ysyx_det_t *d, int count, const ysyx_perf_t *p, double npu_mhz)
{
    printf("{\"detections\":[");
    for (int i = 0; i < count; ++i) {
        printf("%s{\"x1\":%.9g,\"y1\":%.9g,\"x2\":%.9g,\"y2\":%.9g,"
               "\"score\":%.9g,\"class_id\":%d}", i ? "," : "", d[i].x1, d[i].y1,
               d[i].x2, d[i].y2, d[i].score, d[i].class_id);
    }
    printf("],\"perf\":{\"wall_ticks\":%llu,\"npu_ticks\":%llu,\"gpu_ticks\":%llu,"
           "\"overlap_ticks\":%llu,\"dma_bytes\":%llu,\"dma_operations\":%u,"
           "\"abi_error\":%u,\"invalid_dma_descriptors\":%u,"
           "\"hardware_counters_valid\":%s,\"io_and_compute_ms\":%.9g,"
           "\"estimated_ms_at_125_mhz\":%.9g,\"estimated_ms_at_164_555_mhz\":%.9g",
           (unsigned long long)p->wall_ticks, (unsigned long long)p->npu_ticks,
           (unsigned long long)p->gpu_ticks, (unsigned long long)p->overlap_ticks,
           (unsigned long long)p->dma_bytes, p->dma_operations, p->abi_error,
           p->invalid_dma_descriptors, p->hardware_counters_valid ? "true" : "false",
           p->io_and_compute_ms, p->estimated_ms_at_125_mhz,
           p->estimated_ms_at_164_555_mhz);
    if (npu_mhz > 0.0) {
        const double ms = (double)p->wall_ticks / 1000.0 / npu_mhz;
        printf(",\"npu_mhz\":%.15g,\"estimated_ms_at_npu_mhz\":", npu_mhz);
        if (p->hardware_counters_valid && isfinite(ms))
            printf("%.9g", ms);
        else
            fputs("null", stdout);
        fputs(",\"estimated_fps_at_125_mhz\":", stdout);
        if (p->hardware_counters_valid && p->wall_ticks)
            printf("%.9g", 125000000.0 / (double)p->wall_ticks);
        else
            fputs("null", stdout);
        fputs(",\"estimated_fps_at_npu_mhz\":", stdout);
        if (p->hardware_counters_valid && p->wall_ticks && isfinite(1000.0 / ms))
            printf("%.9g", 1000.0 / ms);
        else
            fputs("null", stdout);
    }
    fputs("}}\n", stdout);
}

int main(int argc, char **argv)
{
    const char *path = NULL;
    int sim = 0, ppm = 0, phase17 = 0;
    float threshold = 0.25f;
    double npu_mhz = 0.0;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--sim")) {
            sim = 1;
        } else if (!strcmp(argv[i], "--target") && i + 1 < argc && !phase17) {
            if (strcmp(argv[++i], "p17"))
                goto usage;
            phase17 = 1;
        } else if ((!strcmp(argv[i], "--rgb") || !strcmp(argv[i], "--ppm")) && i + 1 < argc && !path) {
            ppm = !strcmp(argv[i], "--ppm");
            path = argv[++i];
        } else if (!strcmp(argv[i], "--conf") && i + 1 < argc) {
            char *end;
            const char *number = argv[++i];
            threshold = strtof(number, &end);
            if (end == number || *end || !(threshold >= 0.0f && threshold <= 1.0f))
                goto usage;
        } else if (!strcmp(argv[i], "--npu-mhz") && i + 1 < argc && npu_mhz == 0.0) {
            char *end;
            const char *number = argv[++i];
            errno = 0;
            npu_mhz = strtod(number, &end);
            if (end == number || *end || errno == ERANGE || !isfinite(npu_mhz) || npu_mhz <= 0.0)
                goto usage;
        } else {
            goto usage;
        }
    }
    if (argc < 4 || !path)
        goto usage;
    uint8_t *rgb = malloc(YSYX_YOLO26N_RGB_BYTES);
    if (!rgb || read_rgb(path, ppm, rgb)) {
        perror("read RGB/PPM");
        free(rgb);
        return 1;
    }
    ysyx_yolo26n_ctx_t *ctx = phase17 ? ysyx_yolo26n_open_p17(argv[1], sim) :
                                       ysyx_yolo26n_open(argv[1], sim);
    if (!ctx) {
        perror("open YOLO26n");
        free(rgb);
        return 1;
    }
    ysyx_det_t detections[YSYX_YOLO26N_MAX_DETECTIONS];
    ysyx_perf_t perf;
    const int count = ysyx_yolo26n_infer_rgb320(ctx, rgb, threshold, detections,
                                             YSYX_YOLO26N_MAX_DETECTIONS, &perf);
    if (count < 0)
        perror("execute YOLO26n");
    else
        print_result(detections, count, &perf, npu_mhz);
    ysyx_yolo26n_close(ctx);
    free(rgb);
    return count < 0 ? 1 : 0;

usage:
    fprintf(stderr, "usage: %s MODEL (--rgb RGB320 | --ppm FRAME.ppm) [--conf FLOAT] [--npu-mhz MHz] [--sim] [--target p17]\n",
            argv[0]);
    return 2;
}
