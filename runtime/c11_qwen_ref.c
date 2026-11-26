#define _POSIX_C_SOURCE 200809L
#include "c11_qwen_ref.h"

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
    const uint8_t *packed, *scale, *bias;
    unsigned inputs, outputs;
} matrix;

typedef struct {
    const uint8_t *rms_in, *rms_ffn;
    matrix q, k, v, o, gate, up, down;
} block;

struct qwen_reference {
    const uint8_t *image, *rms, *rope, *exp;
    size_t bytes, layout;
    unsigned context, position;
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
                       unsigned inputs, int bias)
{
    mat->inputs = inputs;
    mat->outputs = outputs;
    size_t weights = (size_t)outputs * inputs;
    mat->packed = extent(model, weights / 2);
    mat->scale = extent(model, weights / QWEN_GROUP * 4);
    mat->bias = bias ? extent(model, outputs * 4u) : NULL;
    return !mat->packed || !mat->scale || (bias && !mat->bias);
}

static int read_layout(qwen_reference *m)
{
    static const uint8_t magic[8] = {'Y','S','Q','W','4','A','8',0};
    static const uint32_t header[] = {
        1, QWEN_GROUP, QWEN_LAYERS, QWEN_HIDDEN, QWEN_INTER, QWEN_HEADS,
        QWEN_KV_HEADS, QWEN_HEAD_DIM, QWEN_VOCAB
    };
    if (m->bytes < 64 || memcmp(m->image, magic, sizeof(magic)))
        return -1;
    for (unsigned i = 0; i < sizeof(header) / sizeof(header[0]); ++i)
        if (u32(m->image + 8 + i * 4u) != header[i])
            return -1;
    m->context = u32(m->image + 44);
    if (m->context < 32 || m->context > 2048 || u32(m->image + 48) != 16 ||
        u32(m->image + 52) != 24 || u32(m->image + 56) != EXP_POINTS ||
        u32(m->image + 60) != RMS_EPS_Q32)
        return -1;

    m->layout = 64;
    if (read_matrix(m, &m->embedding, QWEN_VOCAB, QWEN_HIDDEN, 0))
        return -1;
    for (unsigned i = 0; i < QWEN_LAYERS; ++i) {
        block *b = m->layer + i;
        b->rms_in = extent(m, QWEN_HIDDEN * 4u);
        b->rms_ffn = extent(m, QWEN_HIDDEN * 4u);
        if (!b->rms_in || !b->rms_ffn ||
            read_matrix(m, &b->q, QWEN_HIDDEN, QWEN_HIDDEN, 1) ||
            read_matrix(m, &b->k, KV_WIDTH, QWEN_HIDDEN, 1) ||
            read_matrix(m, &b->v, KV_WIDTH, QWEN_HIDDEN, 1) ||
            read_matrix(m, &b->o, QWEN_HIDDEN, QWEN_HIDDEN, 0) ||
            read_matrix(m, &b->gate, QWEN_INTER, QWEN_HIDDEN, 0) ||
            read_matrix(m, &b->up, QWEN_INTER, QWEN_HIDDEN, 0) ||
            read_matrix(m, &b->down, QWEN_HIDDEN, QWEN_INTER, 0))
            return -1;
    }
    m->rms = extent(m, QWEN_HIDDEN * 4u);
    m->rope = extent(m, (size_t)m->context * QWEN_HEAD_DIM * 4);
    m->exp = extent(m, EXP_POINTS * 4u);
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
    if (fstat(fd, &info) || info.st_size <= 0 || info.st_size > 320 * 1024 * 1024) {
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
    const size_t kv = (size_t)QWEN_LAYERS * m->context * KV_WIDTH;
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
    free(m);
}

static int nibble(uint8_t byte)
{
    return (int)((byte & 15u) ^ 8u) - 8;
}

typedef struct {
    int8_t a8[QWEN_INTER];
    int sum[QWEN_INTER / QWEN_GROUP];
    int32_t scale;
} activation;

static void quantize(activation *out, const int32_t *src, unsigned width)
{
    int32_t peak = 0;
    for (unsigned i = 0; i < width; ++i) {
        int32_t a = src[i] < 0 ? -src[i] : src[i];
        if (a > peak)
            peak = a;
    }
    out->scale = (peak + 63) / 127;
    if (!out->scale)
        out->scale = 1;
    for (unsigned i = 0; i < width; i += QWEN_GROUP) {
        int sum = 0;
        for (unsigned j = 0; j < QWEN_GROUP; ++j) {
            int32_t a = (int32_t)rounded(src[i + j], out->scale);
            a = a < -127 ? -127 : a > 127 ? 127 : a;
            out->a8[i + j] = (int8_t)a;
            sum += a;
        }
        out->sum[i / QWEN_GROUP] = sum;
    }
}

static int group_dot(const uint8_t *w, const int8_t *a, int sum_a)
{
#if defined(__AVX2__)
    __m256i mask = _mm256_set1_epi8(15);
    __m256i offset = _mm256_set1_epi8(8);
    __m256i one = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    for (unsigned i = 0; i < QWEN_GROUP / 2; i += 32) {
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
    int32_t lanes[8];
    _mm256_storeu_si256((__m256i *)(void *)lanes, acc);
    int dot = -8 * sum_a;
    for (unsigned i = 0; i < 8; ++i)
        dot += lanes[i];
    return dot;
#else
    int dot = 0;
    (void)sum_a;
    for (unsigned i = 0; i < QWEN_GROUP / 2; ++i)
        dot += nibble(w[i]) * a[2 * i] + nibble(w[i] >> 4) * a[2 * i + 1];
    return dot;
#endif
}

static void linear(const matrix *mat, const activation *input, int32_t *out)
{
    const unsigned groups = mat->inputs / QWEN_GROUP;
    enum { ROWS = 16 };
    for (unsigned first = 0; first < mat->outputs; first += ROWS) {
        int64_t acc[ROWS] = {0};
        unsigned rows = mat->outputs - first;
        if (rows > ROWS)
            rows = ROWS;
        for (unsigned g = 0; g < groups; ++g) {
            const size_t panel = (size_t)g * mat->outputs + first;
            const uint8_t *weights = mat->packed + panel * (QWEN_GROUP / 2);
            const uint8_t *scale = mat->scale + panel * 4;
            for (unsigned i = 0; i < rows; ++i) {
                int dot = group_dot(weights + i * (QWEN_GROUP / 2),
                                    input->a8 + g * QWEN_GROUP, input->sum[g]);
                acc[i] += (int64_t)dot * u32(scale + 4 * i);
            }
        }
        for (unsigned i = 0; i < rows; ++i) {
            unsigned row = first + i;
            /* Split the product before rescaling to keep worst-case math in int64. */
            int64_t scaled = (acc[i] / Q24) * input->scale +
                             rounded((acc[i] % Q24) * input->scale, Q24);
            if (mat->bias)
                scaled += s32(mat->bias + 4 * row);
            out[row] = bounded(scaled);
        }
    }
}

static void embed(const matrix *mat, unsigned token, int32_t *out)
{
    for (unsigned g = 0; g < QWEN_HIDDEN / QWEN_GROUP; ++g) {
        size_t panel = (size_t)g * mat->outputs + token;
        const uint8_t *w = mat->packed + panel * (QWEN_GROUP / 2);
        uint32_t scale = u32(mat->scale + panel * 4);
        for (unsigned j = 0; j < QWEN_GROUP; j += 2) {
            out[g * QWEN_GROUP + j] = bounded(rounded((int64_t)nibble(w[j / 2]) * scale, 256));
            out[g * QWEN_GROUP + j + 1] = bounded(rounded((int64_t)nibble(w[j / 2] >> 4) * scale, 256));
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
    uint64_t sum = 0;
    for (unsigned i = 0; i < QWEN_HIDDEN; ++i) {
        int64_t v = x[i];
        sum += (uint64_t)(v * v);
    }
    uint64_t root = integer_sqrt(sum / QWEN_HIDDEN + RMS_EPS_Q32);
    for (unsigned i = 0; i < QWEN_HIDDEN; ++i)
        out[i] = bounded(rounded((int64_t)x[i] * s32(gamma + 4 * i), (int64_t)root));
}

static void rotate(int32_t *x, unsigned heads, const uint8_t *table)
{
    for (unsigned head = 0; head < heads; ++head, x += QWEN_HEAD_DIM)
        for (unsigned i = 0; i < QWEN_HEAD_DIM / 2; ++i) {
            int32_t u = x[i], v = x[i + QWEN_HEAD_DIM / 2];
            int64_t c = s32(table + 8 * i), s = s32(table + 8 * i + 4);
            x[i] = bounded(rounded(c * u - s * v, QWEN_Q16));
            x[i + QWEN_HEAD_DIM / 2] = bounded(rounded(s * u + c * v, QWEN_Q16));
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
            int64_t dot = 0;
            for (unsigned d = 0; d < QWEN_HEAD_DIM; ++d)
                dot += (int64_t)q[d] * k[d];
            m->scores[p] = bounded(rounded(dot, 8 * QWEN_Q16));
            if (m->scores[p] > maximum)
                maximum = m->scores[p];
        }
        int64_t total = 0, mix[QWEN_HEAD_DIM] = {0};
        for (unsigned p = 0; p <= m->position; ++p) {
            uint32_t weight = exp_negative(m->exp, (int64_t)maximum - m->scores[p]);
            const int32_t *v = values + (size_t)p * KV_WIDTH + group * QWEN_HEAD_DIM;
            total += weight;
            for (unsigned d = 0; d < QWEN_HEAD_DIM; ++d)
                mix[d] += (int64_t)weight * v[d];
        }
        for (unsigned d = 0; d < QWEN_HEAD_DIM; ++d)
            m->mix[head * QWEN_HEAD_DIM + d] = bounded(rounded(mix[d], total));
    }
}

static void swiglu(int32_t *gate, const int32_t *up, const uint8_t *exp)
{
    for (unsigned i = 0; i < QWEN_INTER; ++i) {
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

#ifdef YSYX_C11B_NATIVE
#include "c11b_reference.inc"
#endif

const int32_t *qwen_reference_step(qwen_reference *m, unsigned token)
{
#ifdef YSYX_C11B_NATIVE
    if (m && m->native)
        return step_native(m, token);
#endif
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
        quantize(&input, m->norm, QWEN_HIDDEN);
        linear(&b->q, &input, m->q);
        linear(&b->k, &input, m->k);
        linear(&b->v, &input, m->v);
        if (i == 0 && m->capture)
            captured_qk(m->capture->before, m->q, m->k);
        rotate(m->q, QWEN_HEADS, rope);
        rotate(m->k, QWEN_KV_HEADS, rope);
        if (i == 0 && m->capture) {
            captured_qk(m->capture->after, m->q, m->k);
            memcpy(m->capture->values, m->v, sizeof(m->v));
        }
        int32_t *keys = m->keys + (size_t)i * m->context * KV_WIDTH;
        int32_t *values = m->values + (size_t)i * m->context * KV_WIDTH;
        memcpy(keys + (size_t)m->position * KV_WIDTH, m->k, sizeof(m->k));
        memcpy(values + (size_t)m->position * KV_WIDTH, m->v, sizeof(m->v));
        causal_attention(m, keys, values);
        quantize(&input, m->mix, QWEN_HIDDEN);
        linear(&b->o, &input, m->norm);
        for (unsigned d = 0; d < QWEN_HIDDEN; ++d)
            m->x[d] = bounded((int64_t)m->x[d] + m->norm[d]);

        rms(m->x, b->rms_ffn, m->norm);
        if (i == 0 && m->vectors)
            captured_norm(m->vectors, 1, m->x, m->norm);
        quantize(&input, m->norm, QWEN_HIDDEN);
        linear(&b->gate, &input, m->gate);
        linear(&b->up, &input, m->up);
        if (i == 0 && m->vectors) {
            memcpy(m->vectors->gate, m->gate, sizeof(m->gate));
            memcpy(m->vectors->up, m->up, sizeof(m->up));
        }
        swiglu(m->gate, m->up, m->exp);
        if (i == 0 && m->vectors)
            memcpy(m->vectors->swiglu, m->gate, sizeof(m->gate));
        quantize(&input, m->gate, QWEN_INTER);
        linear(&b->down, &input, m->norm);
        for (unsigned d = 0; d < QWEN_HIDDEN; ++d)
            m->x[d] = bounded((int64_t)m->x[d] + m->norm[d]);
    }
    rms(m->x, m->rms, m->norm);
    quantize(&input, m->norm, QWEN_HIDDEN);
    linear(&m->embedding, &input, m->logits);
    ++m->position;
    return m->logits;
}

int qwen_reference_lm_input(const qwen_reference *m, int8_t out[QWEN_HIDDEN],
                            uint32_t *scale_q16)
{
    if (!m || !m->position || !out || !scale_q16)
        return -1;
    activation input;
    quantize(&input, m->norm, QWEN_HIDDEN);
    memcpy(out, input.a8, QWEN_HIDDEN);
    *scale_q16 = (uint32_t)input.scale;
    return 0;
}
