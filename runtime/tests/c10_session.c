#include "c10_session.h"

#include <string.h>

#define main run_legacy_abi_test
#include "z15_abi.c"
#undef main

enum { INPUT_OFFSET = 128, OUTPUT_OFFSET = 160, TEST_BYTES = 32 };

typedef struct {
    model_t abi;
    uint8_t ddr[224], scratch[128];
    uint32_t offset, local, bytes, flags;
    int fail_read;
} session_hal_t;

static int session_write(void *opaque, uint32_t reg, uint32_t value)
{
    session_hal_t *m = opaque;
    if (write_reg(&m->abi, reg, value))
        return -1;
    switch (reg) {
    case Z15_DMA_OFFSET: m->offset = value; break;
    case Z15_DMA_SCRATCH: m->local = value; break;
    case Z15_DMA_BYTES: m->bytes = value; break;
    case Z15_DMA_FLAGS: m->flags = value; break;
    case Z15_CTRL:
        if (value == Z15_START_DMA) {
            if (m->offset + m->bytes > sizeof(m->ddr) || m->local + m->bytes > sizeof(m->scratch))
                return -1;
            if (m->flags & 1u)
                memcpy(m->ddr + m->offset, m->scratch + m->local, m->bytes);
            else
                memcpy(m->scratch + m->local, m->ddr + m->offset, m->bytes);
        }
        break;
    default: break;
    }
    return 0;
}

static int session_put(void *opaque, uint32_t offset, const void *data, size_t size)
{
    session_hal_t *m = opaque;
    if (offset > sizeof(m->ddr) || size > sizeof(m->ddr) - offset ||
        put_ddr(&m->abi, offset, data, size))
        return -1;
    memcpy(m->ddr + offset, data, size);
    return 0;
}

static int session_get(void *opaque, uint32_t offset, void *data, size_t size)
{
    session_hal_t *m = opaque;
    if (m->fail_read || offset > sizeof(m->ddr) || size > sizeof(m->ddr) - offset)
        return -1;
    memcpy(data, m->ddr + offset, size);
    return 0;
}

static uint32_t le32(const uint8_t *p)
{
    return p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static void put32(uint8_t *p, uint32_t x)
{
    for (unsigned i = 0; i < 4; ++i)
        p[i] = (uint8_t)(x >> (8u * i));
}

static void correct_crc(uint8_t *image, size_t size)
{
    put32(image + 16, 0);
    uint32_t crc = UINT32_MAX;
    for (size_t i = 0; i < size; ++i) {
        crc ^= image[i];
        for (unsigned bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (crc & 1u ? UINT32_C(0xedb88320) : 0);
    }
    put32(image + 16, ~crc);
}

static int check_resident(ysyx_program_t *p, const ysyx_hal_t *hal, session_hal_t *m)
{
    const char *error = NULL;
    ysyx_c10_session_t *session = NULL;
    ysyx_c10_timing_t timing;
    uint8_t a[TEST_BYTES], b[TEST_BYTES], out[TEST_BYTES];
    memset(a, 0x39, sizeof(a));
    memset(b, 0xb4, sizeof(b));
    const ysyx_input_t first = {"x", a, sizeof(a)}, second = {"x", b, sizeof(b)};
    int rc = 1;
    if (ysyx_c10_session_open(p, hal, &session, &error) || m->abi.initial_writes ||
        !ysyx_c10_session_static_output(session, "b") || ysyx_c10_session_static_output(session, "y") ||
        !ysyx_c10_session_read(session, "y", out, 4))
        goto done;
    if (ysyx_c10_session_run(session, &first, 1, NULL, &timing, NULL, &error) ||
        ysyx_c10_session_read(session, "y", out, 4) || memcmp(out, a, 4) ||
        timing.static_h2d_bytes != 96 || timing.dynamic_h2d_bytes != sizeof(a) ||
        !timing.hardware_counters_valid || !timing.stats.wall_ticks || m->abi.initial_writes != 3)
        goto done;
    const uint64_t first_ticks = timing.stats.wall_ticks;
    if (!ysyx_c10_session_run(session, NULL, 0, NULL, &timing, NULL, &error) ||
        m->abi.initial_writes != 3 || !ysyx_c10_session_read(session, "y", out, 4))
        goto done;
    if (ysyx_c10_session_run(session, &second, 1, NULL, &timing, NULL, &error) ||
        ysyx_c10_session_read(session, "y", out, 4) || memcmp(out, b, 4) ||
        m->abi.initial_writes != 4 || timing.static_h2d_bytes || timing.dynamic_h2d_bytes != sizeof(b) ||
        timing.stats.wall_ticks != first_ticks)
        goto done;
    m->abi.inject = 1;
    if (!ysyx_c10_session_run(session, &first, 1, NULL, &timing, NULL, &error) ||
        timing.hardware_counters_valid || timing.stats.abi_error != Z15_ERR_DMA_DESC ||
        timing.stats.invalid_dma_descriptors != 1 || !ysyx_c10_session_read(session, "y", out, 4))
        goto done;
    m->abi.error = 0;
    m->abi.inject = 0;
    const unsigned writes_after_fault = m->abi.initial_writes;
    if (!ysyx_c10_session_run(session, &first, 1, NULL, &timing, NULL, &error) ||
        m->abi.initial_writes != writes_after_fault)
        goto done;
    ysyx_c10_session_close(session);
    session = NULL;
    if (ysyx_c10_session_open(p, hal, &session, &error) ||
        ysyx_c10_session_run(session, &first, 1, NULL, &timing, NULL, &error))
        goto done;
    m->fail_read = 1;
    if (!ysyx_c10_session_read(session, "y", out, 4) ||
        !ysyx_c10_session_run(session, &second, 1, NULL, &timing, NULL, &error))
        goto done;
    rc = 0;
done:
    if (rc)
        fprintf(stderr, "resident layout, frame or fault gate failed (%s)\n", error ? error : "counter/write/read");
    ysyx_c10_session_close(session);
    return rc;
}

static int reject_layout(uint8_t *image, size_t size, const ysyx_hal_t *hal, const char *reason)
{
    correct_crc(image, size);
    const char *error = NULL;
    ysyx_program_t *model = NULL;
    ysyx_c10_session_t *session = NULL;
    int rc = ysyx_program_open(image, size, &model, &error) ||
             !ysyx_c10_session_open(model, hal, &session, &error) ||
             !error || !strstr(error, reason);
    if (rc)
        fprintf(stderr, "expected layout rejection %s (got %s)\n", reason, error ? error : "none");
    ysyx_c10_session_close(session);
    ysyx_program_close(model);
    return rc;
}

int main(int argc, char **argv)
{
    if (argc != 2)
        return 1;
    ysyx_program_t *p = NULL;
    uint8_t *image = open_fixture(argv[1], &p);
    if (!image)
        return 1;
    ysyx_program_close(p);
    p = NULL;
    const size_t size = le32(image + 12);
    uint8_t *symbols = image + le32(image + 52);
    uint8_t *moves = image + le32(image + 116);
    session_hal_t model = {.abi = {.version = Z15_VERSION_VALUE, .status = Z15_LOCKED}};
    const ysyx_hal_t h = {&model, read_reg, session_write, session_put, session_get, synchronize, NULL, NULL};
    if (reject_layout(image, size, &h, "input aliases")) {
        free(image);
        return 1;
    }

    put32(symbols + 4u * 2u, YSYX_INPUT | YSYX_REQUIRED);
    put32(symbols + 4u * 5u, TEST_BYTES / 4u);
    put32(symbols + 4u * 9u, INPUT_OFFSET);
    put32(symbols + 4u * 10u, TEST_BYTES);
    put32(symbols + 48u + 4u * 9u, 64u);
    put32(symbols + 2u * 48u + 4u * 9u, OUTPUT_OFFSET);
    put32(moves + 4u, INPUT_OFFSET);
    int rc = reject_layout(image, size, &h, "DMA store aliases");
    put32(moves + 2u * 24u + 4u, OUTPUT_OFFSET);
    put32(symbols + 4u * 2u, YSYX_INPUT);
    rc |= reject_layout(image, size, &h, "current-frame DDR source");
    put32(symbols + 4u * 2u, YSYX_INPUT | YSYX_REQUIRED);
    correct_crc(image, size);
    if (!rc)
        rc = ysyx_program_open(image, size, &p, NULL) || check_resident(p, &h, &model);
    ysyx_program_close(p);
    free(image);
    if (!rc)
        puts("C10_SESSION_PASS DDR alias and freshness, distinct frames, per-frame deltas, sticky I/O fault");
    return rc;
}
