#include "c10_jpeg.h"
#include "c10_jpeg_tables.h"
#include <math.h>
#include <string.h>

static const uint8_t zigzag[64] = {
    0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,
    12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,
    35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63
};

typedef struct {
    unsigned first[16], after[16], offset[16];
    uint8_t values[256];
    int ready;
} huffman;

typedef struct {
    unsigned id, h, v, quant, dc, ac;
    int prediction;
    uint8_t samples[16 * 16];
} component;

typedef struct {
    const uint8_t *p, *end;
    unsigned word, bits, w, h, count, mh, mv, interval;
    unsigned order[3];
    uint8_t quant[4][64];
    huffman ht[2][4];
    component comp[3];
    float matrix[8][8];
} decoder;

static unsigned be16(const uint8_t *p)
{
    return (unsigned)p[0] * 256 + p[1];
}

static int tables(decoder *d, const uint8_t *p, size_t size, int only_missing)
{
    while (size) {
        if (size < 17 || (p[0] & 0xec) != 0)
            return -1;
        huffman t = {0};
        unsigned next = 0, used = 0;
        for (unsigned i = 0; i < 16; ++i) {
            t.first[i] = next;
            t.offset[i] = used;
            used += p[i + 1];
            next += p[i + 1];
            if (next > (2u << i))
                return -1;
            t.after[i] = next;
            next <<= 1;
        }
        if (!used || used > 256 || size < 17 + used)
            return -1;
        memcpy(t.values, p + 17, used);
        t.ready = 1;
        huffman *selected = &d->ht[p[0] >> 4][p[0] & 3];
        if (!only_missing || !selected->ready)
            *selected = t;
        p += 17 + used;
        size -= 17 + used;
    }
    return 0;
}

static int bits(decoder *d, unsigned n)
{
    int result = 0;
    for (unsigned i = 0; i < n; ++i) {
        if (!d->bits) {
            if (d->p == d->end)
                return -1;
            d->word = *d->p++;
            if (d->word == 255) {
                if (d->p == d->end || *d->p != 0)
                    return -1;
                ++d->p;
            }
            d->bits = 8;
        }
        result = (result << 1) | (int)((d->word >> --d->bits) & 1);
    }
    return result;
}

static int code(decoder *d, const huffman *t)
{
    unsigned key = 0;
    for (unsigned i = 0; i < 16; ++i) {
        int bit = bits(d, 1);
        if (bit < 0)
            return -1;
        key = 2 * key + (unsigned)bit;
        if (key >= t->first[i] && key < t->after[i])
            return t->values[t->offset[i] + key - t->first[i]];
    }
    return -1;
}

static int value(decoder *d, unsigned n, int *out)
{
    if (!n) {
        *out = 0;
        return 0;
    }
    int v = bits(d, n);
    if (v < 0)
        return -1;
    *out = v < (1 << (n - 1)) ? v - (1 << n) + 1 : v;
    return 0;
}

static uint8_t clip(int x)
{
    return x < 0 ? 0 : x > 255 ? 255 : (uint8_t)x;
}

static void idct(const decoder *d, const float *coeff, uint8_t *pixels, unsigned stride)
{
    float tmp[8][8];
    for (unsigned y = 0; y < 8; ++y) {
        for (unsigned x = 0; x < 8; ++x) {
            float v = 0;
            for (unsigned u = 0; u < 8; ++u)
                v += d->matrix[x][u] * coeff[y * 8 + u];
            tmp[y][x] = v;
        }
    }
    for (unsigned y = 0; y < 8; ++y) {
        for (unsigned x = 0; x < 8; ++x) {
            float v = 0;
            for (unsigned u = 0; u < 8; ++u)
                v += d->matrix[y][u] * tmp[u][x];
            pixels[y * stride + x] = clip((int)(128.5f + v));
        }
    }
}

static int block(decoder *d, component *c, uint8_t *pixels, unsigned stride)
{
    float coeff[64] = {0};
    int dc = code(d, &d->ht[0][c->dc]), delta;
    if (dc < 0 || dc > 11 || value(d, (unsigned)dc, &delta))
        return -1;
    c->prediction += delta;
    if (c->prediction < -32768 || c->prediction > 32767)
        return -1;
    coeff[0] = (float)c->prediction * d->quant[c->quant][0];
    unsigned nonzero = 0;
    for (unsigned i = 1; i < 64;) {
        int ac = code(d, &d->ht[1][c->ac]);
        if (ac < 0)
            return -1;
        if (ac == 0)
            break;
        unsigned n = (unsigned)ac & 15;
        if ((!n && ac != 0xf0) || n > 10)
            return -1;
        i += (unsigned)ac >> 4;
        if (i >= 64 || value(d, n, &delta))
            return -1;
        coeff[zigzag[i]] = (float)delta * d->quant[c->quant][i];
        ++i;
        nonzero |= n;
    }
    if (nonzero) {
        idct(d, coeff, pixels, stride);
    } else {
        uint8_t gray = clip((int)(128.5f + coeff[0] * .125f));
        for (unsigned y = 0; y < 8; ++y)
            memset(pixels + y * stride, gray, 8);
    }
    return 0;
}

static uint8_t sample(const decoder *d, unsigned k, unsigned x, unsigned y)
{
    const component *c = &d->comp[k];
    return c->samples[(y * c->v / d->mv) * (8 * c->h) + x * c->h / d->mh];
}

static int restart(decoder *d, unsigned expected)
{
    d->bits = 0;
    if (d->p == d->end || *d->p++ != 255)
        return -1;
    while (d->p < d->end && *d->p == 255)
        ++d->p;
    if (d->p == d->end || *d->p++ != 0xd0 + (expected & 7))
        return -1;
    for (unsigned i = 0; i < d->count; ++i)
        d->comp[i].prediction = 0;
    return 0;
}

static int decode_scan(decoder *d, uint8_t *frame)
{
    for (unsigned x = 0; x < 8; ++x) {
        for (unsigned u = 0; u < 8; ++u)
            d->matrix[x][u] = .5f * (u ? 1.f : .70710678f) *
                             cosf((float)((2 * x + 1) * u) * .19634954f);
    }
    unsigned span_x = 8 * d->mh, span_y = 8 * d->mv;
    unsigned counter = 0, marker = 0;
    for (unsigned by = 0; by < d->h; by += span_y) {
        for (unsigned bx = 0; bx < d->w; bx += span_x) {
            if (d->interval && counter == d->interval) {
                if (restart(d, marker++))
                    return -1;
                counter = 0;
            }
            ++counter;
            for (unsigned i = 0; i < d->count; ++i) {
                component *c = &d->comp[d->order[i]];
                for (unsigned row = 0; row < c->v; ++row) {
                    for (unsigned col = 0; col < c->h; ++col) {
                        if (block(d, c, c->samples + row * 64 * c->h + col * 8, c->h * 8))
                            return -1;
                    }
                }
            }
            for (unsigned y = 0; y < span_y && by + y < d->h; ++y) {
                uint8_t *pixel = frame + ((size_t)(by + y) * d->w + bx) * 3;
                for (unsigned x = 0; x < span_x && bx + x < d->w; ++x, pixel += 3) {
                    int luma = sample(d, 0, x, y);
                    int cb = d->count == 3 ? sample(d, 1, x, y) - 128 : 0;
                    int cr = d->count == 3 ? sample(d, 2, x, y) - 128 : 0;
                    pixel[0] = clip(luma + 359 * cr / 256);
                    pixel[1] = clip(luma - (88 * cb + 183 * cr) / 256);
                    pixel[2] = clip(luma + 454 * cb / 256);
                }
            }
        }
    }
    if (d->p == d->end || *d->p++ != 255)
        return -1;
    while (d->p < d->end && *d->p == 255)
        ++d->p;
    return d->p < d->end && *d->p == 0xd9 ? 0 : -1;
}

static int frame_header(decoder *d, const uint8_t *p, size_t n)
{
    if (d->count || n < 6 || p[0] != 8 || be16(p + 1) != d->h || be16(p + 3) != d->w ||
        (p[5] != 1 && p[5] != 3) || n != 6u + 3u * p[5])
        return -1;
    d->count = p[5];
    p += 6;
    for (unsigned i = 0; i < d->count; ++i, p += 3) {
        component *c = &d->comp[i];
        c->id = p[0];
        c->h = p[1] >> 4;
        c->v = p[1] & 15;
        c->quant = p[2];
        if ((d->count == 3 && c->id != i + 1) || !c->h || !c->v ||
            c->h > 2 || c->v > 2 || c->quant >= 4 || (d->count == 1 && (c->h != 1 || c->v != 1)))
            return -1;
        if (c->h > d->mh) d->mh = c->h;
        if (c->v > d->mv) d->mv = c->v;
    }
    return 0;
}

static int scan_header(decoder *d, const uint8_t *p, size_t n)
{
    if (!d->count || n != 4 + 2 * d->count || p[0] != d->count ||
        p[n - 3] != 0 || p[n - 2] != 63 || p[n - 1] != 0)
        return -1;
    if (tables(d, default_huffman, sizeof(default_huffman), 1))
        return -1;
    unsigned seen = 0;
    for (unsigned i = 0; i < d->count; ++i) {
        unsigned k = 0;
        while (k < d->count && d->comp[k].id != p[1 + 2 * i])
            ++k;
        if (k == d->count || (seen & (1u << k)))
            return -1;
        seen |= 1u << k;
        d->order[i] = k;
        component *c = &d->comp[k];
        c->dc = p[2 + 2 * i] >> 4;
        c->ac = p[2 + 2 * i] & 15;
        if (c->dc >= 4 || c->ac >= 4 || !d->ht[0][c->dc].ready ||
            !d->ht[1][c->ac].ready || !d->quant[c->quant][0])
            return -1;
    }
    return 0;
}

int c10_decode_jpeg(const uint8_t *data, size_t length, uint8_t *rgb, unsigned width, unsigned height)
{
    if (length < 4 || data[0] != 255 || data[1] != 216 || !width || !height)
        return -1;
    decoder d = {.p = data + 2, .end = data + length, .w = width, .h = height};
    while (d.end - d.p >= 4 && *d.p++ == 255) {
        while (d.p < d.end && *d.p == 255)
            ++d.p;
        if (d.end - d.p < 3)
            break;
        unsigned marker = *d.p++;
        unsigned total = be16(d.p);
        if (total < 2 || total > (size_t)(d.end - d.p))
            break;
        const uint8_t *p = d.p + 2;
        size_t n = total - 2;
        d.p += total;
        switch (marker) {
        case 0xc0:
            if (frame_header(&d, p, n)) return -1;
            break;
        case 0xc4:
            if (tables(&d, p, n, 0)) return -1;
            break;
        case 0xdb:
            while (n) {
                if (n < 65 || p[0] >= 4) return -1;
                for (unsigned i = 1; i <= 64; ++i)
                    if (!p[i]) return -1;
                memcpy(d.quant[p[0]], p + 1, 64);
                p += 65;
                n -= 65;
            }
            break;
        case 0xdd:
            if (n != 2) return -1;
            d.interval = be16(p);
            break;
        case 0xda:
            if (scan_header(&d, p, n)) return -1;
            return decode_scan(&d, rgb);
        default:
            if (marker != 0xfe && (marker < 0xe0 || marker > 0xef)) return -1;
            break;
        }
    }
    return -1;
}
