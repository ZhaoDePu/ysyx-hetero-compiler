#include "c10_pixels.h"
#include "c10_jpeg.h"
#include <string.h>

enum { SIZE = YSYX_YOLO26N_SIZE, PAD = 114 };

typedef struct {
    unsigned lo, hi, fraction;
} sample_axis;

/* Same dimension rounding as round() in Python. The extra pixel goes right/bottom. */
static unsigned fit_dimension(unsigned extent, unsigned longest)
{
    unsigned value = SIZE * extent;
    unsigned whole = value / longest, remainder = value % longest;
    whole += 2 * remainder > longest || (2 * remainder == longest && (whole & 1));
    return whole ? whole : 1;
}

static void axis_samples(sample_axis *axis, unsigned from, unsigned to)
{
    unsigned denominator = 2 * to;
    for (unsigned j = 0; j < to; ++j) {
        /* Inverse pixel-center transform with replicated image borders. */
        int numerator = (int)((2 * j + 1) * from) - (int)to;
        unsigned source = numerator < 0 ? 0 : (unsigned)numerator;
        unsigned lo = source / denominator;
        axis[j].lo = lo;
        axis[j].hi = lo + (lo + 1 < from);
        axis[j].fraction = source % denominator;
    }
}

void c10_letterbox_rgb320(const c10_rgb_frame *frame, uint8_t *buffer, c10_letterbox_geometry *g)
{
    unsigned longest = frame->width > frame->height ? frame->width : frame->height;
    *g = (c10_letterbox_geometry){.width = frame->width, .height = frame->height,
                              .out_width = fit_dimension(frame->width, longest),
                              .out_height = fit_dimension(frame->height, longest),
                              .scale = (double)SIZE / longest};
    g->left = (SIZE - g->out_width) / 2;
    g->top = (SIZE - g->out_height) / 2;
    if (g->width == SIZE && g->height == SIZE) {
        memcpy(buffer, frame->pixels, YSYX_YOLO26N_RGB_BYTES);
        return;
    }
    memset(buffer, PAD, YSYX_YOLO26N_RGB_BYTES);
    const unsigned dx = 2 * g->out_width, dy = 2 * g->out_height;
    const unsigned denominator = dx * dy;
    sample_axis xs[SIZE], ys[SIZE];
    axis_samples(xs, frame->width, g->out_width);
    axis_samples(ys, frame->height, g->out_height);
    for (unsigned y = 0; y < g->out_height; ++y) {
        const uint8_t *a = frame->pixels + (size_t)ys[y].lo * frame->width * 3;
        const uint8_t *b = frame->pixels + (size_t)ys[y].hi * frame->width * 3;
        uint8_t *dest = buffer + ((size_t)(g->top + y) * SIZE + g->left) * 3;
        const unsigned fy = ys[y].fraction;
        for (unsigned x = 0; x < g->out_width; ++x) {
            const unsigned lo = xs[x].lo * 3, hi = xs[x].hi * 3, fx = xs[x].fraction;
            for (unsigned k = 0; k < 3; ++k) {
                unsigned top = a[lo + k] * (dx - fx) + a[hi + k] * fx;
                unsigned bottom = b[lo + k] * (dx - fx) + b[hi + k] * fx;
                *dest++ = (uint8_t)((top * (dy - fy) + bottom * fy + denominator / 2) / denominator);
            }
        }
    }
}

float c10_letterbox_unscale(float value, unsigned pad, unsigned extent, double scale)
{
    double pixel = ((double)value - (double)pad) / scale;
    if (pixel <= 0)
        return 0.0f;
    if (pixel >= extent)
        return 1.0f;
    return (float)(pixel / extent);
}

int c10_frame_size(unsigned w, unsigned h, size_t *size)
{
    if (!w || !h || w > C10_MAX_WIDTH || h > C10_MAX_HEIGHT)
        return -1;
    *size = (size_t)w * h * 3;
    return 0;
}

static uint8_t clip_yuv(int value)
{
    return value <= 0 ? 0 : value >= 255 * 256 ? 255 : (uint8_t)(value / 256);
}

static void yuyv_row(const uint8_t *src, uint8_t *dst, unsigned w)
{
    for (unsigned x = 0; x < w; x += 2, src += 4, dst += 6) {
        int y0 = 298 * ((int)src[0] - 16);
        int y1 = 298 * ((int)src[2] - 16);
        int u = (int)src[1] - 128, v = (int)src[3] - 128;
        int r = 409 * v + 128, g = -100 * u - 208 * v + 128, b = 516 * u + 128;
        dst[0] = clip_yuv(y0 + r);
        dst[1] = clip_yuv(y0 + g);
        dst[2] = clip_yuv(y0 + b);
        dst[3] = clip_yuv(y1 + r);
        dst[4] = clip_yuv(y1 + g);
        dst[5] = clip_yuv(y1 + b);
    }
}

int c10_pixels_rgb(c10_pixfmt format, unsigned w, unsigned h, size_t stride,
                   const uint8_t *source, size_t used, uint8_t *rgb, size_t capacity)
{
    size_t size;
    if (!source || !rgb || c10_frame_size(w, h, &size) || size > capacity)
        return -1;
    if (format == C10_JPEG)
        return used <= C10_MAX_RGB ? c10_decode_jpeg(source, used, rgb, w, h) : -1;
    if ((format != C10_YUYV && format != C10_RGB24) || (format == C10_YUYV && (w & 1)))
        return -1;
    size_t row_bytes = (size_t)w * (format == C10_YUYV ? 2 : 3);
    if (!stride)
        stride = row_bytes;
    if (stride < row_bytes || used < row_bytes ||
        (h > 1 && stride > (used - row_bytes) / (h - 1)))
        return -1;
    for (unsigned y = 0; y < h; ++y) {
        const uint8_t *row = source + y * stride;
        uint8_t *out = rgb + (size_t)y * w * 3;
        if (format == C10_YUYV)
            yuyv_row(row, out, w);
        else
            memcpy(out, row, row_bytes);
    }
    return 0;
}
