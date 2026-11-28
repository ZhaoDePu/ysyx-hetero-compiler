#include "c17_simd.h"
#define _POSIX_C_SOURCE 200809L
#include "c11_qwen_ref.h"
#include "c12_qwen_ref.h"
#include "c12_qwen_cache.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

enum {
    KV_WIDTH = QWEN_KV_HEADS * QWEN_HEAD_DIM,
    EXP_POINTS = 4097, RMS_EPS_Q32 = 4295, Q24 = 1 << 24,
    STATE_BOUND = 1 << 26
};

typedef struct {
    const uint8_t *packed, *scale, *bias, *permutation, *equalizer, *fine;
    unsigned inputs, outputs, group, bits, panel_major;
} matrix;

typedef struct {
    const uint8_t *rms_in, *rms_ffn;
    matrix q, k, v, o, gate, up, down;
} block;

struct qwen_reference {
    const uint8_t *image, *rms, *rope, *exp, *kv_scale;
    size_t bytes, layout;
    unsigned context, position, group, protection, kv_bits, grouped_a8, kv4_outliers;
    int block16;
    uint8_t *packed_kv;
    c12_qwen_device c12_device;
    c12_qwen_capture *c12_capture, *prefill_trace;
    unsigned trace_start, trace_count;
    unsigned c12_layer;
    uint32_t c12_hardware_layers;
    int32_t *c12_partials;
    uint8_t *c12_panel;
    matrix embedding;
    qwen_layer0_rope *capture;
    qwen_layer0_vectors *vectors;
#ifdef YSYX_C11B_NATIVE
    int native, native_failed;
    const c11b_ops *device;
    uint32_t ks[QWEN_LAYERS], vs[QWEN_LAYERS];
#endif
    block layer[QWEN_LAYERS];
    int32_t *keys, *values, *scores;
    int32_t x[QWEN_HIDDEN], norm[QWEN_HIDDEN], q[QWEN_HIDDEN], mix[QWEN_HIDDEN];
    int32_t k[KV_WIDTH], v[KV_WIDTH];
    int32_t gate[QWEN_INTER], up[QWEN_INTER], logits[QWEN_VOCAB];
};

static uint32_t u32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static int32_t s32(const uint8_t *p)
{
    uint32_t value = u32(p);
    return value <= INT32_MAX ? (int32_t)value : -(int32_t)(UINT32_MAX - value) - 1;
}

static int32_t bounded(int64_t value)
{
    if (value > STATE_BOUND)
        return STATE_BOUND;
    if (value < -STATE_BOUND)
        return -STATE_BOUND;
    return (int32_t)value;
}

/* Nearest integer, ties away from zero. Callers keep products below INT64_MAX. */
static int64_t rounded(int64_t value, int64_t divisor)
{
    return value < 0 ? -((-value + divisor / 2) / divisor) : (value + divisor / 2) / divisor;
}

static const uint8_t *extent(qwen_reference *model, size_t bytes)
{
    size_t start = (model->layout + 63u) & ~(size_t)63;
    if (start > model->bytes || bytes > model->bytes - start)
        return NULL;
    model->layout = start + bytes;
    return model->image + start;
}

static int read_matrix(qwen_reference *model, matrix *mat, unsigned outputs,
                       unsigned inputs, int bias, unsigned bits)
{
    mat->inputs = inputs;
    mat->outputs = outputs;
    mat->group = model->group;
    mat->bits = bits;
    size_t weights = (size_t)outputs * inputs;
    mat->packed = extent(model, weights * bits / 8);
    mat->scale = extent(model, weights / mat->group * 4);
    mat->bias = bias ? extent(model, outputs * 4u) : NULL;
    if (!mat->packed || !mat->scale || (bias && !mat->bias))
        return -1;
    const size_t scale_extent = weights / mat->group * 4;
    if (model->block16 && bits == 4 &&
        (model->block16 != 4 || u32(mat->scale + scale_extent - 4) == UINT32_C(0xb1600014))) {
        if (inputs != QWEN_HIDDEN || outputs != QWEN_INTER || bias || outputs % 16)
            return -1;
        const size_t scale_bytes = 4u * (inputs / mat->group) * (outputs / 16);
        mat->permutation = mat->scale + scale_bytes;
        mat->panel_major = model->block16 >= 2;
        mat->equalizer = mat->permutation + outputs * 4u;
        uint8_t seen[QWEN_INTER] = {0};
        for (unsigned i = 0; i < outputs; ++i) {
            const uint32_t row = u32(mat->permutation + i * 4u);
            const uint32_t gain = u32(mat->equalizer + i * 4u);
            if (row >= outputs || seen[row]++ || !gain || gain > 16u * QWEN_Q16)
                return -1;
        }
        for (size_t i = 0; i < scale_bytes; i += 4)
            if (!u32(mat->scale + i) || u32(mat->scale + i) > STATE_BOUND)
                return -1;
    }
    return 0;
}

static int read_layout(qwen_reference *m)
{
    static const uint8_t magic[8] = {'Y','S','Q','W','4','A','8',0};
    static const uint8_t mixed_magic[8] = {'Y','S','Q','C','1','2',0,0};
    static const uint32_t header[] = {
        QWEN_LAYERS, QWEN_HIDDEN, QWEN_INTER, QWEN_HEADS,
        QWEN_KV_HEADS, QWEN_HEAD_DIM, QWEN_VOCAB
    };
    if (m->bytes < 64 || u32(m->image + 8) != 1)
        return -1;
    static const uint8_t block16_magic[8] = {'Y','S','Q','C','1','4',0,0};
    static const uint8_t panel_magic[8] = {'Y','S','Q','C','1','4','P',0};
    static const uint8_t residual_magic[8] = {'Y','S','Q','C','1','4','D',0};
    static const uint8_t hybrid_magic[8] = {'Y','S','Q','C','1','4','H',0};
    m->block16 = !memcmp(m->image, hybrid_magic, sizeof(hybrid_magic)) ? 4 :
                 !memcmp(m->image, residual_magic, sizeof(residual_magic)) ? 3 :
                 !memcmp(m->image, panel_magic, sizeof(panel_magic)) ? 2 :
                 !memcmp(m->image, block16_magic, sizeof(block16_magic));
    const int mixed = m->block16 || !memcmp(m->image, mixed_magic, sizeof(mixed_magic));
    m->group = u32(m->image + 12);
    const unsigned policy = u32(m->image + 48);
    if (mixed) {
        if ((m->group != 64 && m->group != 128) || (policy & ~63u))
            return -1;
        if (m->block16 && (policy & 15u) != 15u)
            return -1;
        m->protection = policy & 7u;
        m->grouped_a8 = !!(policy & 8u);
        m->kv4_outliers = !!(policy & 16u);
        m->kv_bits = 8;
    } else if (memcmp(m->image, magic, sizeof(magic)) || m->group != QWEN_GROUP || policy != 16) {
        return -1;
    }
    for (unsigned i = 0; i < sizeof(header) / sizeof(header[0]); ++i)
        if (u32(m->image + 16 + i * 4u) != header[i])
            return -1;
    m->context = u32(m->image + 44);
    if (m->context < 32 || m->context > 2048 ||
        u32(m->image + 52) != 24 || u32(m->image + 56) != EXP_POINTS ||
        u32(m->image + 60) != RMS_EPS_Q32)
        return -1;

    m->layout = 64;
    const unsigned attention = m->protection & 1u ? 8u : 4u;
    const unsigned down = m->protection & 2u ? 8u : 4u;
    const unsigned head = m->protection & 4u ? 8u : 4u;
    if (read_matrix(m, &m->embedding, QWEN_VOCAB, QWEN_HIDDEN, 0, head))
        return -1;
    for (unsigned i = 0; i < QWEN_LAYERS; ++i) {
        block *b = m->layer + i;
        b->rms_in = extent(m, QWEN_HIDDEN * 4u);
        b->rms_ffn = extent(m, QWEN_HIDDEN * 4u);
        if (!b->rms_in || !b->rms_ffn ||
            read_matrix(m, &b->q, QWEN_HIDDEN, QWEN_HIDDEN, 1, attention) ||
            read_matrix(m, &b->k, KV_WIDTH, QWEN_HIDDEN, 1, attention) ||
            read_matrix(m, &b->v, KV_WIDTH, QWEN_HIDDEN, 1, attention) ||
            read_matrix(m, &b->o, QWEN_HIDDEN, QWEN_HIDDEN, 0, attention) ||
            read_matrix(m, &b->gate, QWEN_INTER, QWEN_HIDDEN, 0, 4) ||
            read_matrix(m, &b->up, QWEN_INTER, QWEN_HIDDEN, 0, 4) ||
            read_matrix(m, &b->down, QWEN_HIDDEN, QWEN_INTER, 0, down))
            return -1;
    }
    m->rms = extent(m, QWEN_HIDDEN * 4u);
    m->rope = extent(m, (size_t)m->context * QWEN_HEAD_DIM * 4);
    m->exp = extent(m, EXP_POINTS * 4u);
    if (mixed && (policy & 32u)) {
        m->kv_scale = extent(m, QWEN_LAYERS * 2 * KV_WIDTH);
        if (!m->kv_scale)
            return -1;
        for (size_t i = 0; i < QWEN_LAYERS * 2 * KV_WIDTH; ++i)
            if (m->kv_scale[i] > 8 && m->kv_scale[i] < 248)
                return -1;
    }
    if (m->block16 >= 3) {
        for (unsigned i = 0; i < QWEN_LAYERS; ++i) {
            matrix *layers[] = {&m->layer[i].gate, &m->layer[i].up};
            for (unsigned x = 0; x < 2; ++x)
                if (layers[x]->permutation) {
                    layers[x]->fine = extent(m, (size_t)QWEN_HIDDEN * QWEN_INTER / 2);
                    if (!layers[x]->fine)
                        return -1;
                }
        }
    }
    if (!m->rms || !m->rope || !m->exp || ((m->layout + 63u) & ~(size_t)63) != m->bytes ||
        u32(m->exp) != Q24)
        return -1;
    for (unsigned i = 1; i < EXP_POINTS; ++i)
        if (u32(m->exp + 4 * i) > u32(m->exp + 4 * (i - 1)))
            return -1;
    return 0;
}

qwen_reference *qwen_reference_open(const char *path)
{
    struct stat info;
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return NULL;
    if (fstat(fd, &info) || info.st_size <= 0 || info.st_size > 502 * 1024 * 1024) {
        close(fd);
        return NULL;
    }
    const size_t bytes = (size_t)info.st_size;
    const void *ptr = mmap(NULL, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED)
        return NULL;
    qwen_reference *m = calloc(1, sizeof(*m));
    if (!m) {
        munmap((void *)ptr, bytes);
        return NULL;
    }
    m->image = ptr;
    m->bytes = bytes;
    if (read_layout(m)) {
        qwen_reference_close(m);
        return NULL;
    }
    const size_t kv = (size_t)(m->kv_bits ? 1 : QWEN_LAYERS) * m->context * KV_WIDTH;
    if (m->kv_bits && c12_qwen_cache_bits(m, m->kv_bits)) {
        qwen_reference_close(m);
        return NULL;
    }
    m->keys = malloc(kv * sizeof(*m->keys));
    m->values = malloc(kv * sizeof(*m->values));
    m->scores = malloc(m->context * sizeof(*m->scores));
    if (!m->keys || !m->values || !m->scores) {
        qwen_reference_close(m);
        return NULL;
    }
    return m;
}

unsigned qwen_reference_context(const qwen_reference *m)
{
#ifdef YSYX_C11B_NATIVE
    if (m && m->native && m->context > C11B_PITCH)
        return C11B_PITCH;
#endif
    return m ? m->context : 0;
}

void qwen_reference_reset(qwen_reference *m)
{
    if (m)
        m->position = 0;
}

void qwen_reference_close(qwen_reference *m)
{
    if (!m)
        return;
    munmap((void *)m->image, m->bytes);
    free(m->keys);
    free(m->values);
    free(m->scores);
    free(m->packed_kv);
    free(m->c12_partials);
    free(m->c12_panel);
    free(m);
}

static int nibble(uint8_t byte)
{
    return (int)((byte & 15u) ^ 8u) - 8;
}

typedef struct {
    int8_t a8[QWEN_INTER];
    int32_t scale;
    int32_t groups[QWEN_INTER / 64];
    int grouped;
} activation;

static int32_t quantize_block(int8_t *out, const int32_t *src, unsigned width)
{
    const int32_t peak = (int32_t)c17_peak(src, width);
    int32_t scale = (peak + 63) / 127;
    if (!scale)
        scale = 1;
    unsigned i = 0;
#if C17_NEON
    const uint32_t reciprocal = scale > 1 ? (uint32_t)((UINT64_C(1) << 32) / scale) : 0;
    for (; i + 8 <= width; i += 8) {
        const int16x4_t lo = vmovn_s32(c17_quant4(vld1q_s32(src + i), (uint32_t)scale, reciprocal, 127));
        const int16x4_t hi = vmovn_s32(c17_quant4(vld1q_s32(src + i + 4), (uint32_t)scale, reciprocal, 127));
        vst1_s8(out + i, vmovn_s16(vcombine_s16(lo, hi)));
    }
#endif
    for (; i < width; ++i) {
        const int32_t v = src[i], magnitude = v < 0 ? -v : v;
        int32_t a = (magnitude + scale / 2) / scale;
        if (v < 0)
            a = -a;
        a = a < -127 ? -127 : a > 127 ? 127 : a;
        out[i] = (int8_t)a;
    }
    return scale;
}

static void quantize(activation *out, const int32_t *src, unsigned width)
{
    out->scale = quantize_block(out->a8, src, width);
    out->grouped = 0;
}

static void quantize_q16(const qwen_reference *m, activation *out,
                         const int32_t *src, unsigned width)
{
    if (!m->grouped_a8) {
        quantize(out, src, width);
        return;
    }
    out->grouped = 1;
    out->scale = 1;
    for (unsigned i = 0; i < width; i += m->group)
        out->groups[i / m->group] = quantize_block(out->a8 + i, src + i, m->group);
}

static int64_t rescale_q24(int64_t value, int32_t scale_q16)
{
    return (value / Q24) * scale_q16 + rounded((value % Q24) * scale_q16, Q24);
}

static int group_dot(const uint8_t *w, const int8_t *a, unsigned group,
                     unsigned bits, int sum_a)
{
#if defined(__AVX2__)
    __m256i mask = _mm256_set1_epi8(15);
    __m256i offset = _mm256_set1_epi8(8);
    __m256i one = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    if (bits == 8) {
        for (unsigned i = 0; i < group; i += 16) {
            __m128i w8 = _mm_loadu_si128((const __m128i *)(const void *)(w + i));
            __m128i a8 = _mm_loadu_si128((const __m128i *)(const void *)(a + i));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_cvtepi8_epi16(w8),
                                                        _mm256_cvtepi8_epi16(a8)));
        }
    } else {
        for (unsigned i = 0; i < group / 2; i += 32) {
            __m256i packed = _mm256_loadu_si256((const __m256i *)(const void *)(w + i));
            __m256i lo = _mm256_xor_si256(_mm256_and_si256(packed, mask), offset);
            __m256i hi = _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(packed, 4), mask), offset);
            __m256i left = _mm256_unpacklo_epi8(lo, hi);
            __m256i right = _mm256_unpackhi_epi8(lo, hi);
            __m256i wl = _mm256_permute2x128_si256(left, right, 0x20);
            __m256i wr = _mm256_permute2x128_si256(left, right, 0x31);
            __m256i al = _mm256_loadu_si256((const __m256i *)(const void *)(a + 2 * i));
            __m256i ar = _mm256_loadu_si256((const __m256i *)(const void *)(a + 2 * i + 32));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(wl, al), one));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(wr, ar), one));
        }
    }
    int32_t lanes[8];
    _mm256_storeu_si256((__m256i *)(void *)lanes, acc);
    int dot = bits == 4 ? -8 * sum_a : 0;
    for (unsigned i = 0; i < 8; ++i)
        dot += lanes[i];
    return dot;
#else
    int dot = 0;
    (void)sum_a;
    if (bits == 8) {
        for (unsigned i = 0; i < group; ++i)
            dot += ((int)(w[i] ^ 128u) - 128) * a[i];
    } else {
        for (unsigned i = 0; i < group / 2; ++i)
            dot += nibble(w[i]) * a[2 * i] + nibble(w[i] >> 4) * a[2 * i + 1];
    }
    return dot;
#endif
}

static int64_t round_even(int64_t value, uint64_t divisor)
{
    const uint64_t magnitude = value < 0 ? (uint64_t)-value : (uint64_t)value;
    const uint64_t q = magnitude / divisor, r = magnitude % divisor;
    const int64_t result = (int64_t)(q + (r > divisor - r || (r == divisor - r && (q & 1))));
    return value < 0 ? -result : result;
}

static void block16_coefficients(const matrix *mat, const activation *input,
                                 uint64_t *units, uint16_t *coeffs)
{
    const unsigned groups = mat->inputs / mat->group, panels = mat->outputs / 16;
    const unsigned passes = mat->fine ? 2 : 1;
    for (unsigned panel = 0; panel < panels; ++panel) {
        uint64_t gain[QWEN_HIDDEN / 64], unit = 1;
        for (unsigned g = 0; g < groups; ++g) {
            const size_t at = (size_t)g * panels + panel;
            gain[g] = (uint64_t)u32(mat->scale + at * 4u) * (uint32_t)input->groups[g];
            while (unit < gain[g])
                unit <<= 1;
        }
        units[panel] = unit;
        for (unsigned g = 0; g < groups; ++g) {
            const unsigned at = (panel * groups + g) * passes;
            coeffs[at] = (uint16_t)round_even((int64_t)(gain[g] << 8), unit);
            if (passes == 2)
                coeffs[at + 1] = (uint16_t)round_even((int64_t)(gain[g] << 4), unit);
        }
    }

}

static void block16_finish(const matrix *mat, const int32_t *accum, const uint64_t *units, int32_t *out)
{
    const uint64_t fixed_units = UINT64_C(1) << 40;
    for (unsigned physical = 0; physical < mat->outputs; ++physical) {
        const uint64_t unit = units[physical / 16];
        const unsigned logical = u32(mat->permutation + 4u * physical);
        const uint32_t equalizer = u32(mat->equalizer + 4u * physical);
        const int64_t product = (int64_t)accum[physical] * equalizer;
        const int64_t value = unit <= fixed_units
            ? round_even(product, fixed_units / unit)
            : product * (int64_t)(unit / fixed_units);
        out[logical] = bounded(value);
    }
}

static int linear_block16(const matrix *mat, const activation *input, int32_t *out,
                          const c12_qwen_device *device)
{
    const unsigned groups = mat->inputs / mat->group;
    const unsigned panels = mat->outputs / 16;
    const unsigned row_bytes = mat->group / 2;
    uint64_t units[QWEN_INTER / 16];
    const unsigned passes = mat->fine ? 2 : 1;
    uint16_t coeffs[2 * (QWEN_INTER / 16) * (QWEN_HIDDEN / 64)];
    int32_t accum[QWEN_INTER];
    block16_coefficients(mat, input, units, coeffs);

    if (device && device->project_block16 && mat->panel_major) {
        if (device->project_block16(device->ctx, mat->packed, mat->fine, mat->outputs, mat->inputs,
                                    mat->group, input->a8, coeffs, accum))
            return -1;
    } else {
        int sums[QWEN_HIDDEN / 64] = {0};
        memset(accum, 0, (size_t)mat->outputs * sizeof(*accum));
        for (unsigned g = 0; g < groups; ++g)
            for (unsigned k = 0; k < mat->group; ++k)
                sums[g] += input->a8[g * mat->group + k];
        for (unsigned panel = 0; panel < panels; ++panel)
            for (unsigned g = 0; g < groups; ++g) {
                const size_t index = mat->panel_major ? (size_t)panel * groups + g : (size_t)g * panels + panel;
                const int8_t *x = input->a8 + g * mat->group;
                for (unsigned pass = 0; pass < passes; ++pass) {
                    const uint8_t *w = (pass ? mat->fine : mat->packed) + index * 16 * row_bytes;
                    const unsigned coefficient = coeffs[(panel * groups + g) * passes + pass];
                    for (unsigned lane = 0; lane < 16; ++lane) {
                        const int dot = group_dot(w + lane * row_bytes, x, mat->group, 4, sums[g]);
                        accum[panel * 16 + lane] += (int32_t)round_even((int64_t)dot * coefficient, 256);
                    }
                }
            }
    }
    block16_finish(mat, accum, units, out);
    return 0;
}

static void linear(const matrix *mat, const activation *input, int32_t *out)
{
    const unsigned groups = mat->inputs / mat->group;
    const unsigned stride = mat->group * mat->bits / 8;
    int sums[QWEN_INTER / 64] = {0};
    if (mat->bits == 4) {
        for (unsigned g = 0; g < groups; ++g)
            for (unsigned i = 0; i < mat->group; ++i)
                sums[g] += input->a8[g * mat->group + i];
    }
    enum { ROWS = 16 };
    for (unsigned first = 0; first < mat->outputs; first += ROWS) {
        int64_t acc[ROWS] = {0};
        unsigned rows = mat->outputs - first;
        if (rows > ROWS)
            rows = ROWS;
        for (unsigned g = 0; g < groups; ++g) {
            const size_t panel = (size_t)g * mat->outputs + first;
            const uint8_t *weights = mat->packed + panel * stride;
            const uint8_t *scale = mat->scale + panel * 4;
            for (unsigned i = 0; i < rows; ++i) {
                int dot = group_dot(weights + i * stride, input->a8 + g * mat->group,
                                    mat->group, mat->bits, sums[g]);
                int64_t value = (int64_t)dot * u32(scale + 4 * i);
                acc[i] += input->grouped ? rescale_q24(value, input->groups[g]) : value;
            }
        }
        for (unsigned i = 0; i < rows; ++i) {
            unsigned row = first + i;
            /* Split the product before rescaling to keep worst-case math in int64. */
            int64_t scaled = input->grouped ? acc[i] : rescale_q24(acc[i], input->scale);
            if (mat->bias)
                scaled += s32(mat->bias + 4 * row);
            out[row] = bounded(scaled);
        }
    }
}

static void embed(const matrix *mat, unsigned token, int32_t *out)
{
    const unsigned stride = mat->group * mat->bits / 8;
    for (unsigned g = 0; g < QWEN_HIDDEN / mat->group; ++g) {
        size_t panel = (size_t)g * mat->outputs + token;
        const uint8_t *w = mat->packed + panel * stride;
        uint32_t scale = u32(mat->scale + panel * 4);
        for (unsigned j = 0; j < mat->group; ++j) {
            int q = mat->bits == 4 ? nibble(w[j / 2] >> (4 * (j & 1u))) : (int)(w[j] ^ 128u) - 128;
            out[g * mat->group + j] = bounded(rounded((int64_t)q * scale, 256));
        }
    }
}

static uint64_t integer_sqrt(uint64_t x)
{
    uint64_t result = 0, bit = UINT64_C(1) << 62;
    while (bit > x)
        bit >>= 2;
    while (bit) {
        if (x >= result + bit) {
            x -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }
        bit >>= 2;
    }
    return result;
}

static void rms(const int32_t *x, const uint8_t *gamma, int32_t *out)
{
    const uint64_t sum = (uint64_t)c17_dot(x, x, QWEN_HIDDEN);
    uint64_t root = integer_sqrt(sum / QWEN_HIDDEN + RMS_EPS_Q32);
    for (unsigned i = 0; i < QWEN_HIDDEN; ++i)
        out[i] = bounded(rounded((int64_t)x[i] * s32(gamma + 4 * i), (int64_t)root));
}

static void rotate(int32_t *x, unsigned heads, const uint8_t *table)
{
    for (unsigned head = 0; head < heads; ++head, x += QWEN_HEAD_DIM) {
        unsigned i = 0;
#if C17_NEON && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        const int32x2_t lower = vdup_n_s32(-STATE_BOUND), upper = vdup_n_s32(STATE_BOUND);
        for (; i < QWEN_HEAD_DIM / 2; i += 2) {
            const int32x4_t coeff = vreinterpretq_s32_u8(vld1q_u8(table + 8 * i));
            const int32x2_t c = vget_low_s32(vuzp1q_s32(coeff, coeff));
            const int32x2_t s = vget_low_s32(vuzp2q_s32(coeff, coeff));
            const int32x2_t u = vld1_s32(x + i), v = vld1_s32(x + i + QWEN_HEAD_DIM / 2);
            const int64x2_t real = vmlsl_s32(vmull_s32(c, u), s, v);
            const int64x2_t imag = vmlal_s32(vmull_s32(s, u), c, v);
            vst1_s32(x + i, vmax_s32(lower, vmin_s32(upper, vqmovn_s64(c17_round_q16(real)))));
            vst1_s32(x + i + QWEN_HEAD_DIM / 2, vmax_s32(lower, vmin_s32(upper, vqmovn_s64(c17_round_q16(imag)))));
        }
#endif
        for (; i < QWEN_HEAD_DIM / 2; ++i) {
            int32_t u = x[i], v = x[i + QWEN_HEAD_DIM / 2];
            int64_t c = s32(table + 8 * i), s = s32(table + 8 * i + 4);
            x[i] = bounded(rounded(c * u - s * v, QWEN_Q16));
            x[i + QWEN_HEAD_DIM / 2] = bounded(rounded(s * u + c * v, QWEN_Q16));
        }
    }
}

static uint32_t exp_negative(const uint8_t *table, int64_t magnitude)
{
    if (magnitude >= (EXP_POINTS - 1) * 256)
        return 0;
    unsigned i = (unsigned)magnitude >> 8;
    unsigned f = (unsigned)magnitude & 255u;
    return (uint32_t)(((uint64_t)u32(table + 4 * i) * (256 - f) +
                       (uint64_t)u32(table + 4 * (i + 1)) * f + 128) >> 8);
}

static void causal_attention(qwen_reference *m, const int32_t *keys, const int32_t *values)
{
    for (unsigned head = 0; head < QWEN_HEADS; ++head) {
        unsigned group = head / (QWEN_HEADS / QWEN_KV_HEADS);
        const int32_t *q = m->q + head * QWEN_HEAD_DIM;
        int32_t maximum = -STATE_BOUND;
        for (unsigned p = 0; p <= m->position; ++p) {
            const int32_t *k = keys + (size_t)p * KV_WIDTH + group * QWEN_HEAD_DIM;
            const int64_t dot = c17_dot(q, k, QWEN_HEAD_DIM);
            m->scores[p] = bounded(rounded(dot, 8 * QWEN_Q16));
            if (m->scores[p] > maximum)
                maximum = m->scores[p];
        }
        int64_t total = 0, mix[QWEN_HEAD_DIM] = {0};
        for (unsigned p = 0; p <= m->position; ++p) {
            uint32_t weight = exp_negative(m->exp, (int64_t)maximum - m->scores[p]);
            const int32_t *v = values + (size_t)p * KV_WIDTH + group * QWEN_HEAD_DIM;
            total += weight;
            c17_accumulate(mix, v, (int32_t)weight, QWEN_HEAD_DIM);
        }
        for (unsigned d = 0; d < QWEN_HEAD_DIM; ++d)
            m->mix[head * QWEN_HEAD_DIM + d] = bounded(rounded(mix[d], total));
    }
}

static void swiglu(int32_t *gate, const int32_t *up, const uint8_t *exp)
{
    unsigned i = 0;
#if C17_NEON
    const int32x4_t lower = vdupq_n_s32(-STATE_BOUND), upper = vdupq_n_s32(STATE_BOUND);
    for (; i + 4 <= QWEN_INTER; i += 4) {
        int32_t silu[4];
        for (unsigned j = 0; j < 4; ++j) {
            const int32_t x = gate[i + j];
            const uint32_t e = exp_negative(exp, x < 0 ? -x : x);
            silu[j] = (int32_t)rounded((int64_t)(x < 0 ? e : Q24) * x, Q24 + e);
        }
        const int32x4_t gate_vec = vld1q_s32(silu), up_vec = vld1q_s32(up + i);
        const int64x2_t lo = c17_round_q16(vmull_s32(vget_low_s32(gate_vec), vget_low_s32(up_vec)));
        const int64x2_t hi = c17_round_q16(vmull_high_s32(gate_vec, up_vec));
        const int32x4_t result = vcombine_s32(vqmovn_s64(lo), vqmovn_s64(hi));
        vst1q_s32(gate + i, vmaxq_s32(lower, vminq_s32(upper, result)));
    }
#endif
    for (; i < QWEN_INTER; ++i) {
        int32_t x = gate[i];
        uint32_t e = exp_negative(exp, x < 0 ? -x : x);
        int64_t p = x < 0 ? e : Q24;
        int64_t silu = rounded(p * x, Q24 + e);
        gate[i] = bounded(rounded(silu * up[i], QWEN_Q16));
    }
}

static void captured_qk(int32_t output[QWEN_HEADS + QWEN_KV_HEADS][QWEN_HEAD_DIM],
                        const int32_t *q, const int32_t *k)
{
    enum { QUERIES = QWEN_HEADS / QWEN_KV_HEADS };
    for (unsigned group = 0; group < QWEN_KV_HEADS; ++group) {
        memcpy(output[group * (QUERIES + 1)], q + group * QUERIES * QWEN_HEAD_DIM,
               QUERIES * QWEN_HEAD_DIM * sizeof(*q));
        memcpy(output[group * (QUERIES + 1) + QUERIES], k + group * QWEN_HEAD_DIM,
               QWEN_HEAD_DIM * sizeof(*k));
    }
}

void qwen_reference_capture_rope(qwen_reference *model, qwen_layer0_rope *capture)
{
    model->capture = capture;
}

void qwen_reference_capture_vectors(qwen_reference *model, qwen_layer0_vectors *capture)
{
    model->vectors = capture;
}

static void captured_norm(qwen_layer0_vectors *trace, unsigned unit,
                          const int32_t *in, const int32_t *out)
{
    memcpy(trace->input[unit], in, sizeof(trace->input[unit]));
    memcpy(trace->norm[unit], out, sizeof(trace->norm[unit]));
}

int c12_qwen_cache_bits(qwen_reference *m, unsigned bits)
{
    if (!m || !m->kv_bits || m->position || (bits != 4 && bits != 8))
        return -1;
    const size_t bytes = (size_t)QWEN_LAYERS * m->context * 2 * QWEN_KV_HEADS *
                         c12_cache_record_size(bits, m->kv4_outliers);
    uint8_t *block = malloc(bytes);
    if (!block)
        return -1;
    free(m->packed_kv);
    m->packed_kv = block;
    m->kv_bits = bits;
    return 0;
}

int c14_qwen_block16(const qwen_reference *m)
{
    return m ? m->block16 : 0;
}

int c12_qwen_mixed(const qwen_reference *m)
{
    return m && m->kv_bits;
}

static int32_t shift_cache(int32_t value, int exponent)
{
    return exponent > 0 ? bounded((int64_t)value * (1 << exponent)) :
                          (int32_t)rounded(value, 1 << -exponent);
}

static void pack_cache(const qwen_reference *m, uint8_t *record, const int32_t *head,
                       const uint8_t *exponents)
{
    int32_t scaled[QWEN_HEAD_DIM];
    if (exponents) {
        for (unsigned i = 0; i < QWEN_HEAD_DIM; ++i)
            scaled[i] = shift_cache(head[i], 128 - (int)(exponents[i] ^ 128u));
        head = scaled;
    }
    if (m->kv_bits == 4 && m->kv4_outliers)
        c12_cache_pack_outliers(record, head);
    else
        c12_cache_pack(record, head, m->kv_bits);
}

static void unpack_cache(const qwen_reference *m, const uint8_t *record, int32_t *head,
                         const uint8_t *exponents)
{
    if (m->kv_bits == 4 && m->kv4_outliers)
        c12_cache_unpack_outliers(record, head);
    else
        c12_cache_unpack(record, head, m->kv_bits);
    if (exponents)
        for (unsigned i = 0; i < QWEN_HEAD_DIM; ++i)
            head[i] = shift_cache(head[i], (int)(exponents[i] ^ 128u) - 128);
}

static c12_qwen_capture *prefill_capture(qwen_reference *m, unsigned layer, unsigned position)
{
    if (!m->prefill_trace || layer != m->c12_layer || position < m->trace_start ||
        position - m->trace_start >= m->trace_count)
        return NULL;
    return m->prefill_trace + position - m->trace_start;
}

static void store_cache(qwen_reference *m, unsigned layer, unsigned unpack_from)
{
    const size_t head_bytes = c12_cache_record_size(m->kv_bits, m->kv4_outliers);
    const size_t slot = 2 * QWEN_KV_HEADS * head_bytes;
    uint8_t *start = m->packed_kv + (size_t)layer * m->context * slot;
    uint8_t *record = start + (size_t)m->position * slot;
    const uint8_t *exponents = m->kv_bits == 4 && m->kv_scale ?
                              m->kv_scale + (size_t)layer * 2 * KV_WIDTH : NULL;
    for (unsigned h = 0; h < QWEN_KV_HEADS; ++h) {
        const unsigned d = h * QWEN_HEAD_DIM;
        uint8_t *k_head = record + h * head_bytes;
        uint8_t *v_head = k_head + QWEN_KV_HEADS * head_bytes;
        pack_cache(m, k_head, m->k + d, exponents ? exponents + d : NULL);
        pack_cache(m, v_head, m->v + d, exponents ? exponents + d + KV_WIDTH : NULL);
    }
    if (m->c12_capture && m->c12_layer == layer) {
        memcpy(m->c12_capture->cache, record, slot);
        m->c12_capture->cache_bytes = slot;
    }
    c12_qwen_capture *capture = prefill_capture(m, layer, m->position);
    if (capture) {
        memcpy(capture->cache, record, slot);
        capture->cache_bytes = slot;
    }
    for (unsigned pos = unpack_from; pos <= m->position; ++pos) {
        const uint8_t *p = start + pos * slot;
        for (unsigned h = 0; h < QWEN_KV_HEADS; ++h) {
            size_t index = (size_t)pos * KV_WIDTH + h * QWEN_HEAD_DIM;
            const uint8_t *k_head = p + h * head_bytes;
            const uint8_t *v_head = k_head + QWEN_KV_HEADS * head_bytes;
            const unsigned d = h * QWEN_HEAD_DIM;
            unpack_cache(m, k_head, m->keys + index, exponents ? exponents + d : NULL);
            unpack_cache(m, v_head, m->values + index, exponents ? exponents + d + KV_WIDTH : NULL);
        }
    }
}

enum { C12_LM_PANEL = 2048 };

static unsigned head_panel(const c12_qwen_device *ops)
{
    if (!ops->project_head)
        return C12_LM_PANEL;
    return ops->head_panel ? ops->head_panel : C16_HEAD_PANEL;
}

int c12_qwen_set_device(qwen_reference *m, const c12_qwen_device *ops, uint32_t layers)
{
    if (!m || !m->kv_bits || m->position || !ops || !ops->project ||
        !layers || layers >> (QWEN_LAYERS + 1))
        return -1;
    size_t partials = (size_t)QWEN_HIDDEN * QWEN_INTER / m->group;
    if (ops->project_rows)
        partials *= C16_PREFILL_ROWS * (ops->project_block16_rows ? 2u : 1u);
    if (layers >> QWEN_LAYERS) {
        const size_t head = (size_t)QWEN_HIDDEN * head_panel(ops) / m->group;
        if (partials < head)
            partials = head;
    }
    int32_t *buffer = realloc(m->c12_partials, partials * sizeof(*m->c12_partials));
    if (!buffer)
        return -1;
    m->c12_partials = buffer;
    const int copy_head = (layers >> QWEN_LAYERS) && !ops->project_head;
    if (copy_head && !m->c12_panel)
        m->c12_panel = malloc((size_t)C12_LM_PANEL * QWEN_HIDDEN);
    if (copy_head && !m->c12_panel)
        return -1;
    m->c12_device = *ops;
    m->c12_hardware_layers = layers;
    return 0;
}

static void apply_partials(const matrix *mat, const activation *input, const int32_t *partials,
                           unsigned first, unsigned count, int32_t *output)
{
    const unsigned groups = mat->inputs / mat->group;
    for (unsigned c = 0; c < count; ++c) {
        int64_t sum = 0;
        for (unsigned g = 0; g < groups; ++g) {
            const size_t at = (size_t)g * mat->outputs + first + c;
            int64_t value = (int64_t)partials[(size_t)g * count + c] * u32(mat->scale + 4 * at);
            sum += input->grouped ? rescale_q24(value, input->groups[g]) : value;
        }
        int64_t value = input->grouped ? sum : rescale_q24(sum, input->scale);
        if (mat->bias)
            value += s32(mat->bias + 4 * (first + c));
        output[first + c] = bounded(value);
    }
}

static int project_q16(qwen_reference *m, const matrix *mat, const activation *input,
                       int32_t *output, unsigned layer)
{
    if (mat->permutation)
        return linear_block16(mat, input, output,
               m->c12_hardware_layers & (UINT32_C(1) << layer) ? &m->c12_device : NULL);
    if (!(m->c12_hardware_layers & (UINT32_C(1) << layer))) {
        linear(mat, input, output);
        return 0;
    }

    int vocab = layer == QWEN_LAYERS;
    const unsigned groups = mat->inputs / mat->group;
    const unsigned stride = mat->group * mat->bits / 8;
    const int resident_head = vocab && m->c12_device.project_head;
    const unsigned panel = vocab ? head_panel(&m->c12_device) : mat->outputs;
    for (unsigned first = 0; first < mat->outputs; first += panel) {
        unsigned count = mat->outputs - first;
        if (count > panel)
            count = panel;
        const uint8_t *packed = mat->packed;
        if (vocab && !resident_head) {
            for (unsigned g = 0; g < groups; ++g)
                memcpy(m->c12_panel + (size_t)g * count * stride,
                       packed + ((size_t)g * mat->outputs + first) * stride, (size_t)count * stride);
            packed = m->c12_panel;
        }
        if (resident_head ?
            m->c12_device.project_head(m->c12_device.ctx, mat->packed, first, count, mat->outputs,
                                       mat->group, mat->bits, input->a8, m->c12_partials) :
            m->c12_device.project(m->c12_device.ctx, packed, count, mat->inputs,
                                  mat->group, mat->bits, input->a8, m->c12_partials))
            return -1;
        apply_partials(mat, input, m->c12_partials, first, count, output);
    }
    return 0;
}

static int project_parallel(qwen_reference *m, const activation *input, unsigned layer,
                            const matrix *const *weights, int32_t *const *outputs, unsigned count)
{
    if (!(m->c12_hardware_layers & (UINT32_C(1) << layer)) || !m->c12_device.project_parallel) {
        for (unsigned i = 0; i < count; ++i)
            if (project_q16(m, weights[i], input, outputs[i], layer))
                return -1;
        return 0;
    }
    valence_raw_projection jobs[3];
    int32_t *buffer = m->c12_partials;
    for (unsigned i = 0; i < count; ++i) {
        const matrix *mat = weights[i];
        jobs[i] = (valence_raw_projection){mat->packed, buffer, mat->outputs, mat->bits};
        buffer += mat->outputs * (QWEN_HIDDEN / m->group);
    }
    if (m->c12_device.project_parallel(m->c12_device.ctx, jobs, count, m->group, input->a8))
        return -1;
    for (unsigned i = 0; i < count; ++i)
        apply_partials(weights[i], input, jobs[i].partials, 0, jobs[i].rows, outputs[i]);
    return 0;
}

static int project_gate_up(qwen_reference *m, const block *b, const activation *input, unsigned layer)
{
    if (!(m->c12_hardware_layers & (UINT32_C(1) << layer)) || !m->c12_device.project_parallel)
        return project_q16(m, &b->gate, input, m->gate, layer) ||
               project_q16(m, &b->up, input, m->up, layer);
    const matrix *const mat[] = {&b->gate, &b->up};
    int32_t *const out[] = {m->gate, m->up};
    if (!mat[0]->permutation && !mat[1]->permutation)
        return project_parallel(m, input, layer, mat, out, 2);
    if (!m->c12_device.project_block16_pair || !mat[0]->permutation || !mat[1]->permutation)
        return project_q16(m, mat[0], input, out[0], layer) ||
               project_q16(m, mat[1], input, out[1], layer);
    uint64_t units[2][QWEN_INTER / 16];
    uint16_t coefficients[2][2 * (QWEN_INTER / 16) * (QWEN_HIDDEN / 64)];
    int32_t accum[2][QWEN_INTER];
    valence_block16_projection jobs[2];
    for (unsigned i = 0; i < 2; ++i) {
        block16_coefficients(mat[i], input, units[i], coefficients[i]);
        jobs[i] = (valence_block16_projection){mat[i]->packed, mat[i]->fine, coefficients[i],
                                              accum[i], mat[i]->outputs};
    }
    if (m->c12_device.project_block16_pair(m->c12_device.ctx, jobs, m->group, input->a8))
        return -1;
    for (unsigned i = 0; i < 2; ++i)
        block16_finish(mat[i], accum[i], units[i], out[i]);
    return 0;
}

int c12_qwen_capture_layer(qwen_reference *m, unsigned layer, c12_qwen_capture *capture)
{
    if (!m || !m->kv_bits || layer >= QWEN_LAYERS)
        return -1;
    m->c12_capture = capture;
    m->c12_layer = layer;
    return 0;
}

int c16_qwen_capture_prefill(qwen_reference *m, unsigned layer, c12_qwen_capture *trace, unsigned count)
{
    if (!m || !m->kv_bits || layer >= QWEN_LAYERS || !trace || !count || count > m->context - m->position)
        return -1;
    m->c12_layer = layer;
    m->prefill_trace = trace;
    m->trace_start = m->position;
    m->trace_count = count;
    return 0;
}

static int add_residual(qwen_reference *m, unsigned layer)
{
    if (m->c12_device.residual && (m->c12_hardware_layers & (UINT32_C(1) << layer)))
        return m->c12_device.residual(m->c12_device.ctx, m->x, m->norm);
    for (unsigned d = 0; d < QWEN_HIDDEN; ++d)
        m->x[d] = bounded((int64_t)m->x[d] + m->norm[d]);
    return 0;
}

static int emit_logits(qwen_reference *m, activation *input)
{
    rms(m->x, m->rms, m->norm);
    quantize_q16(m, input, m->norm, QWEN_HIDDEN);
    return project_q16(m, &m->embedding, input, m->logits, QWEN_LAYERS);
}

static const int32_t *process_q16(qwen_reference *m, unsigned token, int logits)
{
    if (!m || token >= QWEN_VOCAB || m->position >= m->context)
        return NULL;
    activation input;
    embed(&m->embedding, token, m->x);
    const uint8_t *rope = m->rope + (size_t)m->position * QWEN_HEAD_DIM * 4;
    for (unsigned i = 0; i < QWEN_LAYERS; ++i) {
        const block *b = m->layer + i;
        rms(m->x, b->rms_in, m->norm);
        if (i == 0 && m->vectors)
            captured_norm(m->vectors, 0, m->x, m->norm);
        quantize_q16(m, &input, m->norm, QWEN_HIDDEN);
        const matrix *const qkv[] = {&b->q, &b->k, &b->v};
        int32_t *const outputs[] = {m->q, m->k, m->v};
        if (project_parallel(m, &input, i, qkv, outputs, 3))
            return NULL;
        if (i == 0 && m->capture)
            captured_qk(m->capture->before, m->q, m->k);
        if (m->c12_device.rope && (m->c12_hardware_layers & (UINT32_C(1) << i))) {
            if (m->c12_device.rope(m->c12_device.ctx, m->q, m->k, rope))
                return NULL;
        } else {
            rotate(m->q, QWEN_HEADS, rope);
            rotate(m->k, QWEN_KV_HEADS, rope);
        }
        if (i == 0 && m->capture) {
            captured_qk(m->capture->after, m->q, m->k);
            memcpy(m->capture->values, m->v, sizeof(m->v));
        }
        int32_t *keys = m->keys;
        int32_t *values = m->values;
        if (m->kv_bits) {
            store_cache(m, i, 0);
        } else {
            keys += (size_t)i * m->context * KV_WIDTH;
            values += (size_t)i * m->context * KV_WIDTH;
            memcpy(keys + (size_t)m->position * KV_WIDTH, m->k, sizeof(m->k));
            memcpy(values + (size_t)m->position * KV_WIDTH, m->v, sizeof(m->v));
        }
        causal_attention(m, keys, values);
        quantize_q16(m, &input, m->mix, QWEN_HIDDEN);
        if (project_q16(m, &b->o, &input, m->norm, i))
            return NULL;
        if (add_residual(m, i))
            return NULL;

        rms(m->x, b->rms_ffn, m->norm);
        if (i == 0 && m->vectors)
            captured_norm(m->vectors, 1, m->x, m->norm);
        quantize_q16(m, &input, m->norm, QWEN_HIDDEN);
        if (project_gate_up(m, b, &input, i))
            return NULL;
        if (i == 0 && m->vectors) {
            memcpy(m->vectors->gate, m->gate, sizeof(m->gate));
            memcpy(m->vectors->up, m->up, sizeof(m->up));
        }
        swiglu(m->gate, m->up, m->exp);
        if (i == 0 && m->vectors)
            memcpy(m->vectors->swiglu, m->gate, sizeof(m->gate));
        quantize_q16(m, &input, m->gate, QWEN_INTER);
        if (project_q16(m, &b->down, &input, m->norm, i))
            return NULL;
        if (add_residual(m, i))
            return NULL;
        if (m->c12_capture && m->c12_layer == i)
            memcpy(m->c12_capture->hidden, m->x, sizeof(m->x));
        c12_qwen_capture *capture = prefill_capture(m, i, m->position);
        if (capture)
            memcpy(capture->hidden, m->x, sizeof(m->x));
    }
    if (logits && emit_logits(m, &input))
        return NULL;
    ++m->position;
    return m->logits;
}

#ifdef YSYX_C11B_NATIVE
#include "c11b_reference.inc"
#endif

const int32_t *qwen_reference_step(qwen_reference *m, unsigned token)
{
#ifdef YSYX_C11B_NATIVE
    if (m && m->native)
        return step_native(m, token);
#endif
    return process_q16(m, token, 1);
}

typedef struct {
    activation inputs[C16_PREFILL_ROWS];
    int8_t grouped[C16_PREFILL_ROWS * QWEN_INTER];
    int32_t state[C16_PREFILL_ROWS][QWEN_HIDDEN];
    int32_t norm[C16_PREFILL_ROWS * QWEN_HIDDEN], q[C16_PREFILL_ROWS * QWEN_HIDDEN];
    int32_t k[C16_PREFILL_ROWS * KV_WIDTH], v[C16_PREFILL_ROWS * KV_WIDTH];
    int32_t gate[C16_PREFILL_ROWS * QWEN_INTER], up[C16_PREFILL_ROWS * QWEN_INTER];
} prefill_work;

static int block16_prefill(qwen_reference *m, const matrix *mat, prefill_work *work,
                            int32_t *output, unsigned rows)
{
    if (m->c12_device.project_block16_rows(m->c12_device.ctx, mat->packed, mat->fine, mat->outputs,
                                          mat->group, rows, work->grouped, m->c12_partials))
        return -1;
    const unsigned chunks = (mat->inputs / mat->group) * (mat->fine ? 2 : 1);
    for (unsigned row = 0; row < rows; ++row) {
        uint64_t units[QWEN_INTER / 16];
        uint16_t coeffs[2 * (QWEN_INTER / 16) * (QWEN_HIDDEN / 64)];
        int32_t accum[QWEN_INTER] = {0};
        block16_coefficients(mat, work->inputs + row, units, coeffs);
        for (unsigned panel = 0; panel < mat->outputs / 16; ++panel)
            for (unsigned chunk = 0; chunk < chunks; ++chunk) {
                const unsigned idx = panel * chunks + chunk;
                const int32_t *dot = m->c12_partials + (idx * rows + row) * 16;
                for (unsigned i = 0; i < 16; ++i)
                    accum[panel * 16 + i] += (int32_t)round_even((int64_t)dot[i] * coeffs[idx], 256);
            }
        block16_finish(mat, accum, units, output + row * mat->outputs);
    }
    return 0;
}

static int project_prefill(qwen_reference *m, const matrix *mat, prefill_work *work,
                            int32_t *output, unsigned layer, unsigned rows)
{
    const unsigned n = mat->outputs, g = m->group, groups = mat->inputs / g;
    if (rows == 1 || !(m->c12_hardware_layers & (UINT32_C(1) << layer)) ||
        (mat->permutation && !m->c12_device.project_block16_rows)) {
        for (unsigned row = 0; row < rows; ++row)
            if (project_q16(m, mat, work->inputs + row, output + row * n, layer))
                return -1;
        return 0;
    }
    for (unsigned i = 0; i < groups; ++i)
        for (unsigned row = 0; row < rows; ++row)
            memcpy(work->grouped + (i * rows + row) * g, work->inputs[row].a8 + i * g, g);
    if (mat->permutation)
        return block16_prefill(m, mat, work, output, rows);
    if (m->c12_device.project_rows(m->c12_device.ctx, mat->packed, n, mat->inputs, g,
                                   mat->bits, rows, work->grouped, m->c12_partials))
        return -1;
    for (unsigned row = 0; row < rows; ++row) {
        const activation *input = work->inputs + row;
        for (unsigned c = 0; c < n; ++c) {
            const unsigned first = c / C16_GROUPED_COLS * C16_GROUPED_COLS;
            const unsigned width = n - first < C16_GROUPED_COLS ? n - first : C16_GROUPED_COLS;
            const unsigned at = first * rows + row * width + c - first;
            int64_t sum = 0;
            for (unsigned i = 0; i < groups; ++i) {
                const int64_t value = (int64_t)m->c12_partials[i * n * rows + at] *
                                      u32(mat->scale + 4u * (i * n + c));
                sum += input->grouped ? rescale_q24(value, input->groups[i]) : value;
            }
            if (!input->grouped)
                sum = rescale_q24(sum, input->scale);
            if (mat->bias)
                sum += s32(mat->bias + 4u * c);
            output[row * n + c] = bounded(sum);
        }
    }
    return 0;
}

static int prefill_tile(qwen_reference *m, prefill_work *work, const uint32_t *tokens, unsigned rows)
{
    const unsigned start = m->position;
    for (unsigned row = 0; row < rows; ++row) {
        const unsigned id = tokens[row];
        embed(&m->embedding, id, work->state[row]);
    }
    for (unsigned i = 0; i < QWEN_LAYERS; ++i) {
        const block *b = m->layer + i;
        for (unsigned row = 0; row < rows; ++row) {
            rms(work->state[row], b->rms_in, (work->norm + row * QWEN_HIDDEN));
            quantize_q16(m, work->inputs + row, (work->norm + row * QWEN_HIDDEN), QWEN_HIDDEN);
        }
        if (project_prefill(m, &b->q, work, work->q, i, rows) ||
            project_prefill(m, &b->k, work, work->k, i, rows) ||
            project_prefill(m, &b->v, work, work->v, i, rows))
            return -1;
        for (unsigned row = 0; row < rows; ++row) {
            m->position = start + row;
            const uint8_t *rope = m->rope + (size_t)m->position * QWEN_HEAD_DIM * 4;
            memcpy(m->q, (work->q + row * QWEN_HIDDEN), sizeof(m->q));
            memcpy(m->k, (work->k + row * KV_WIDTH), sizeof(m->k));
            memcpy(m->v, (work->v + row * KV_WIDTH), sizeof(m->v));
            rotate(m->q, QWEN_HEADS, rope);
            rotate(m->k, QWEN_KV_HEADS, rope);
            store_cache(m, i, row ? m->position : 0);
            causal_attention(m, m->keys, m->values);
            quantize_q16(m, work->inputs + row, m->mix, QWEN_HIDDEN);
        }
        if (project_prefill(m, &b->o, work, work->norm, i, rows))
            return -1;
        for (unsigned row = 0; row < rows; ++row) {
            for (unsigned d = 0; d < QWEN_HIDDEN; ++d)
                work->state[row][d] = bounded((int64_t)work->state[row][d] + (work->norm + row * QWEN_HIDDEN)[d]);
            rms(work->state[row], b->rms_ffn, (work->norm + row * QWEN_HIDDEN));
            quantize_q16(m, work->inputs + row, (work->norm + row * QWEN_HIDDEN), QWEN_HIDDEN);
        }
        if (project_prefill(m, &b->gate, work, work->gate, i, rows) ||
            project_prefill(m, &b->up, work, work->up, i, rows))
            return -1;
        for (unsigned row = 0; row < rows; ++row) {
            swiglu((work->gate + row * QWEN_INTER), (work->up + row * QWEN_INTER), m->exp);
            quantize_q16(m, work->inputs + row, (work->gate + row * QWEN_INTER), QWEN_INTER);
        }
        if (project_prefill(m, &b->down, work, work->norm, i, rows))
            return -1;
        for (unsigned row = 0; row < rows; ++row)
            for (unsigned d = 0; d < QWEN_HIDDEN; ++d)
                work->state[row][d] = bounded((int64_t)work->state[row][d] + (work->norm + row * QWEN_HIDDEN)[d]);
        if (m->c12_capture && m->c12_layer == i)
            memcpy(m->c12_capture->hidden, work->state[rows - 1], sizeof(m->x));
        for (unsigned row = 0; row < rows; ++row) {
            c12_qwen_capture *capture = prefill_capture(m, i, start + row);
            if (capture)
                memcpy(capture->hidden, work->state[row], sizeof(m->x));
        }
    }
    memcpy(m->x, work->state[rows - 1], sizeof(m->x));
    m->position = start + rows;
    return 0;
}

const int32_t *c16_qwen_prefill(qwen_reference *m, const uint32_t *tokens, unsigned count)
{
    if (!m || !tokens || !count || count > m->context - m->position)
        return NULL;
    if (!m->c12_device.project_rows || m->capture || m->vectors || count == 1) {
        for (unsigned i = 0; i + 1 < count; ++i)
#ifdef YSYX_C11B_NATIVE
            if (qwen_reference_advance(m, tokens[i]))
#else
            if (!process_q16(m, tokens[i], 0))
#endif
                return NULL;
        return qwen_reference_step(m, tokens[count - 1]);
    }
    for (unsigned i = 0; i < count; ++i)
        if (tokens[i] >= QWEN_VOCAB)
            return NULL;
    prefill_work *work = malloc(sizeof(*work));
    if (!work)
        return NULL;
    int failed = 0;
    for (unsigned at = 0; at < count && !failed; at += C16_PREFILL_ROWS) {
        const unsigned rows = count - at < C16_PREFILL_ROWS ? count - at : C16_PREFILL_ROWS;
        failed = prefill_tile(m, work, tokens + at, rows);
    }
    if (!failed)
        failed = emit_logits(m, work->inputs);
    free(work);
    return failed ? NULL : m->logits;
}

int qwen_reference_lm_input(const qwen_reference *m, int8_t out[QWEN_HIDDEN],
                            uint32_t *scale_q16)
{
    if (!m || !m->position || !out || !scale_q16 || m->grouped_a8)
        return -1;
    activation input;
    quantize(&input, m->norm, QWEN_HIDDEN);
    memcpy(out, input.a8, QWEN_HIDDEN);
    *scale_q16 = (uint32_t)input.scale;
    return 0;
}
