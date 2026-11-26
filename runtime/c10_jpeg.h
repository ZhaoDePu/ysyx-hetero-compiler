#ifndef C10_JPEG_H
#define C10_JPEG_H
#include <stddef.h>
#include <stdint.h>
/* Baseline JPEG decoder. Output belongs to caller; input and output must not alias. */
int c10_decode_jpeg(const uint8_t *data, size_t length, uint8_t *rgb, unsigned width, unsigned height);
#endif
