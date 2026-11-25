#include "../yolo26n_post.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_exact(const char *path, void *buffer, size_t size)
{
    FILE *stream = fopen(path, "rb");
    if (!stream)
        return -1;
    const int status = fread(buffer, 1, size, stream) == size && fgetc(stream) == EOF ? 0 : -1;
    fclose(stream);
    return status;
}

static int fixture(const char *prefix, const char *suffix, void *buffer, size_t size)
{
    char path[4096];
    const int length = snprintf(path, sizeof(path), "%s.%s.bin", prefix, suffix);
    if (length < 0 || (size_t)length >= sizeof(path))
        return -1;
    return read_exact(path, buffer, size);
}

static uint32_t le32(const uint8_t *p)
{
    return p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static float read_float(const uint8_t *p)
{
    const uint32_t bits = le32(p);
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static int compare(int size, const char *prefix, const char *scales_path, const char *lut_path)
{
    uint8_t scales[24], lut[3 * 256 * 4];
    ysyx_yolo_heads_t heads = {0};
    uint8_t *bytes[6] = {0};
    int status = -1;
    if (read_exact(scales_path, scales, sizeof(scales)) || read_exact(lut_path, lut, sizeof(lut)))
        goto done;
    for (int i = 0; i < 3; ++i) {
        heads.box_scale[i] = read_float(scales + i * 4);
        for (int j = 0; j < 256; ++j)
            heads.probabilities[i][j] = read_float(lut + (i * 256 + j) * 4);
        const size_t side = (size_t)size >> (3 + i);
        for (int kind = 0; kind < 2; ++kind) {
            const size_t bytes_needed = side * side * (kind ? 80u : 4u);
            const int idx = 2 * i + kind;
            char suffix[16];
            (void)snprintf(suffix, sizeof(suffix), "pl_out_%d", idx);
            bytes[idx] = malloc(bytes_needed);
            if (!bytes[idx] || fixture(prefix, suffix, bytes[idx], bytes_needed))
                goto done;
        }
        heads.boxes[i] = (const int8_t *)bytes[2 * i];
        heads.scores[i] = bytes[2 * i + 1];
    }
    ysyx_det_t detections[300];
    uint8_t coords[300 * 4 * 4], scores[300 * 4], classes[300 * 8];
    const int n = ysyx_yolo_decode(&heads, size, 0.0f, detections, 300);
    if (n != 300 || fixture(prefix, "coords_q16_16", coords, sizeof(coords))
            || fixture(prefix, "scores_q16_16", scores, sizeof(scores))
            || fixture(prefix, "classes_i64", classes, sizeof(classes)))
        goto done;
    for (int i = 0; i < n; ++i) {
        const ysyx_det_t d = detections[i];
        const float xyxy[] = {d.x1, d.y1, d.x2, d.y2};
        if ((uint32_t)d.class_id != le32(classes + i * 8) || le32(classes + i * 8 + 4)
                || (uint32_t)lrintf(d.score * 65536.0f) != le32(scores + i * 4)) {
            fprintf(stderr, "C6 C99 class/score mismatch at %d\n", i);
            goto done;
        }
        for (int j = 0; j < 4; ++j) {
            if ((uint32_t)lrintf(xyxy[j] * 65536.0f) != le32(coords + (i * 4 + j) * 4)) {
                fprintf(stderr, "C6 C99 coordinate mismatch at detection %d axis %d\n", i, j);
                goto done;
            }
        }
    }
    printf("C6_C99_POST_PASS size=%d detections=%d bytes=%zu scene=%s\n", size, n,
           sizeof(coords) + sizeof(scores) + sizeof(classes), prefix);
    status = 0;
done:
    for (int i = 0; i < 6; ++i)
        free(bytes[i]);
    return status;
}

int main(int argc, char **argv)
{
    if (argc != 5 || (strcmp(argv[1], "128") && strcmp(argv[1], "320"))) {
        fprintf(stderr, "usage: %s SIZE VECTOR_PREFIX SCALES SCORE_LUT\n", argv[0]);
        return 2;
    }
    return compare(atoi(argv[1]), argv[2], argv[3], argv[4]) ? 1 : 0;
}
