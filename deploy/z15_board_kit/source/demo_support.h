#ifndef Z15_DEMO_SUPPORT_H
#define Z15_DEMO_SUPPORT_H

#include "ysyx_rt.h"

enum { DEMO_PATH_BYTES = 4096, DEMO_VOCAB_SIZE = 256, DEMO_WORD_BYTES = 64 };

typedef struct {
    const char *directory;
    const char *input;
    const char *backend;
    const char *control_uio;
    const char *dma_uio;
} demo_options_t;

typedef struct {
    ysyx_program_t *program;
    uint8_t *image;
    ysyx_hal_t hal;
} demo_t;

int demo_options(int argc, char **argv, const char *input_option, demo_options_t *);
int demo_path(char out[DEMO_PATH_BYTES], const char *directory, const char *leaf);
int demo_label(const char *name);
uint8_t *demo_read(const char *path, size_t limit, size_t *size);
int demo_open(demo_t *, const demo_options_t *, const char **error);
void demo_close(demo_t *);
void demo_accumulate(ysyx_stats_t *, const ysyx_stats_t *);
void demo_print_stats(const ysyx_stats_t *, const char *items_name, unsigned items);
int demo_dictionary(const char *path, char entries[DEMO_VOCAB_SIZE][DEMO_WORD_BYTES]);
size_t demo_integer_width(uint32_t dtype);
int64_t demo_integer(const uint8_t *data, uint32_t dtype);

#endif
