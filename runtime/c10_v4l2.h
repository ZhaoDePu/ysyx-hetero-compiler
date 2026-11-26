#ifndef YSYX_C10_V4L2_H
#define YSYX_C10_V4L2_H

#include "c10_pixels.h"

typedef struct c10_camera c10_camera;

c10_camera *c10_camera_open(const char *device, unsigned w, unsigned h, unsigned fps);
int c10_camera_payload_valid(unsigned flags, size_t used, size_t mapped);
int c10_camera_rgb(c10_camera *, uint8_t *rgb, size_t capacity, unsigned *w, unsigned *h);
void c10_camera_close(c10_camera *);

#endif
