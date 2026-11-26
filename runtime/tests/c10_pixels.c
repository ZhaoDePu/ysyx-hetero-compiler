#define _POSIX_C_SOURCE 200809L

#include "../c10_pixels.h"
#include "../c10_v4l2.h"
#include <time.h>
#include <linux/videodev2.h>

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
    assert(c10_camera_payload_valid(0, 10, 16));
    assert(!c10_camera_payload_valid(V4L2_BUF_FLAG_ERROR, 10, 16));
    assert(!c10_camera_payload_valid(0, 0, 16));
    assert(!c10_camera_payload_valid(0, 17, 16));
    const uint8_t yuyv[] = { 16, 128, 235, 128, 17, 18, 81, 90, 81, 240 };
    const uint8_t expected[] = { 0, 0, 0, 255, 255, 255, 255, 0, 0, 255, 0, 0 };
    uint8_t rgb[sizeof(expected)];
    assert(!c10_pixels_rgb(C10_YUYV, 2, 2, 6, yuyv, sizeof(yuyv), rgb, sizeof(rgb)));
    assert(!memcmp(rgb, expected, sizeof(rgb)));
    assert(c10_pixels_rgb(C10_YUYV, 2, 2, 6, yuyv, sizeof(yuyv) - 1, rgb, sizeof(rgb)) < 0);
    assert(c10_pixels_rgb(C10_YUYV, 3, 1, 0, yuyv, sizeof(yuyv), rgb, sizeof(rgb)) < 0);
    assert(c10_pixels_rgb(C10_YUYV, 2, 2, SIZE_MAX, yuyv, sizeof(yuyv), rgb, sizeof(rgb)) < 0);
    assert(c10_pixels_rgb(C10_YUYV, 2, 2, 6, yuyv, sizeof(yuyv), rgb, sizeof(rgb) - 1) < 0);
    assert(c10_pixels_rgb(C10_YUYV, 2, 2, 3, yuyv, sizeof(yuyv), rgb, sizeof(rgb)) < 0);
    assert(c10_pixels_rgb(C10_YUYV, 1922, 2, 0, yuyv, sizeof(yuyv), rgb, sizeof(rgb)) < 0);
    assert(c10_pixels_rgb(C10_YUYV, 0, 2, 0, yuyv, sizeof(yuyv), rgb, sizeof(rgb)) < 0);

    const uint8_t padded[] = { 1, 2, 3, 99, 4, 5, 6 };
    assert(!c10_pixels_rgb(C10_RGB24, 1, 2, 4, padded, sizeof(padded), rgb, sizeof(rgb)));
    assert(!memcmp(rgb, "\x01\x02\x03\x04\x05\x06", 6));

    static uint8_t letterbox[YSYX_YOLO26N_RGB_BYTES], identity[YSYX_YOLO26N_RGB_BYTES];
    for (size_t i = 0; i < sizeof(identity); ++i)
        identity[i] = (uint8_t)i;
    c10_rgb_frame frame = { 320, 320, identity };
    c10_letterbox_geometry geom;
    c10_letterbox_rgb320(&frame, letterbox, &geom);
    assert(!memcmp(letterbox, identity, sizeof(identity)));
    const uint8_t two_pixels[] = { 4, 8, 16, 249, 230, 210 };
    frame = (c10_rgb_frame){ 2, 1, two_pixels };
    c10_letterbox_rgb320(&frame, letterbox, &geom);
    assert(geom.left == 0 && geom.top == 80 && geom.out_width == 320 && geom.out_height == 160);
    assert(letterbox[3 * 320 * 79] == 114 && letterbox[3 * 320 * 240] == 114);
    assert(!memcmp(letterbox + 3 * 320 * 80, two_pixels, 3));
    assert(!memcmp(letterbox + 3 * (320 * 239 + 319), two_pixels + 3, 3));
    assert(c10_letterbox_unscale(160, 80, 1, geom.scale) == 0.5f);
    assert(c10_letterbox_unscale(40, 80, 1, geom.scale) == 0.0f);
    assert(c10_letterbox_unscale(300, 80, 1, geom.scale) == 1.0f);
    puts("C10 V4L2 payload flags, YUYV BT.601, row stride, bytes used, size bounds, RGB and letterbox PASS");
    return 0;
}
