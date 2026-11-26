#ifndef YSYX_C10_PIXELS_H
#define YSYX_C10_PIXELS_H

#include "../include/ysyx_yolo26n_api.h"
#include <stddef.h>

enum { C10_MAX_WIDTH = 1920, C10_MAX_HEIGHT = 1080 };
enum { C10_MAX_RGB = C10_MAX_WIDTH * C10_MAX_HEIGHT * 3 };

typedef struct {
    unsigned width, height;
    const uint8_t *pixels;
} c10_rgb_frame;

typedef struct {
    unsigned width, height, out_width, out_height, left, top;
    double scale;
} c10_letterbox_geometry;

typedef enum { C10_YUYV, C10_RGB24, C10_JPEG } c10_pixfmt;

int c10_frame_size(unsigned w, unsigned h, size_t *size);
int c10_pixels_rgb(c10_pixfmt format, unsigned w, unsigned h, size_t stride,
                   const uint8_t *source, size_t used, uint8_t *rgb, size_t capacity);
void c10_letterbox_rgb320(const c10_rgb_frame *, uint8_t *, c10_letterbox_geometry *);
float c10_letterbox_unscale(float value, unsigned pad, unsigned extent, double scale);

#endif
