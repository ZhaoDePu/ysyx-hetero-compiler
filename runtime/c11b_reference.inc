/* Native Qwen layer policy, shared by the CPU oracle and the Step 2 transport. */
#include <math.h>

const uint8_t *qwen_reference_image(const qwen_reference *m, size_t *bytes)
{
    if (!m || !bytes)
        return NULL;
    *bytes = m->bytes;
    return m->image;
}

int qwen_reference_native(qwen_reference *m, const c11b_ops *ops)
{
    if (!m || m->position || (ops && (!ops->project || !ops->rms || !ops->rope ||
                                     !ops->attention || !ops->swiglu || !ops->faulted)))
        return -1;
    m->native = 1;
    m->device = ops;
    return 0;
}

int qwen_reference_projection(qwen_reference *m, const uint8_t *packed, unsigned outputs,
                              unsigned inputs, const int8_t *a8, int32_t *raw)
{
    if (!m || !m->device)
        return -1;
    return m->device->project(m->device->ctx, packed, outputs, inputs, a8, raw);
}

static uint32_t quant_scale(const int32_t *x, unsigned n)
{
    uint32_t peak = 0;
    for (unsigned i = 0; i < n; ++i) {
        uint32_t a = x[i] < 0 ? (uint32_t)(-(int64_t)x[i]) : (uint32_t)x[i];
        if (a > peak)
            peak = a;
    }
    uint32_t step = 1;
    while (step * UINT64_C(127) < peak)
        step *= 2;
    return step;
}

static void quant_bytes(const int32_t *x, unsigned n, uint32_t step, int8_t *out)
{
    for (unsigned i = 0; i < n; ++i) {
        const int64_t value = rounded(x[i], step);
        out[i] = (int8_t)(value > 127 ? 127 : value < -127 ? -127 : value);
    }
}

static int project_native(qwen_reference *m, const matrix *mat, const activation *input, int32_t *out)
{
    if (!m->device) {
        linear(mat, input, out);
        return 0;
    }
    const unsigned groups = mat->inputs / QWEN_GROUP;
    const size_t count = (size_t)groups * mat->outputs;
    int32_t *partials = malloc(count * sizeof(*partials));
    if (!partials)
        return -1;
    int rc = m->device->project(m->device->ctx, mat->packed, mat->outputs, mat->inputs,
                                input->a8, partials);
    if (!rc) {
        for (unsigned row = 0; row < mat->outputs; ++row) {
            int64_t sum = 0;
            for (unsigned g = 0; g < groups; ++g) {
                size_t at = (size_t)g * mat->outputs + row;
                sum += (int64_t)partials[at] * u32(mat->scale + 4 * at);
            }
            int64_t result = (sum / Q24) * input->scale + rounded((sum % Q24) * input->scale, Q24);
            if (mat->bias)
                result += s32(mat->bias + row * 4);
            out[row] = bounded(result);
        }
    }
    free(partials);
    return rc;
}

static int norm_native(qwen_reference *m, const int32_t *x, const uint8_t *gamma, int32_t *out)
{
    int32_t g32[QWEN_HIDDEN];
    int8_t a[QWEN_HIDDEN], g[QWEN_HIDDEN], result[QWEN_HIDDEN];
    const uint32_t xs = quant_scale(x, QWEN_HIDDEN);
    for (unsigned i = 0; i < QWEN_HIDDEN; ++i)
        g32[i] = s32(gamma + 4 * i);
    const uint32_t gs = quant_scale(g32, QWEN_HIDDEN);
    quant_bytes(x, QWEN_HIDDEN, xs, a);
    quant_bytes(g32, QWEN_HIDDEN, gs, g);
    /* Use the residual's ideal range to choose a byte-domain output scale. */
    rms(x, gamma, out);
    uint32_t ys = quant_scale(out, QWEN_HIDDEN);
    uint32_t shift = 16;
    uint64_t gain;
    for (;;) {
        /* sqrt(896) in Q16, and a fixed nearest integer rounding rule. */
        const uint64_t fixed = UINT64_C(1961709) * gs * (1u << (shift - 7));
        gain = (uint64_t)rounded((int64_t)fixed, (int64_t)ys * QWEN_Q16);
        if (gain <= 65535)
            break;
        if (shift == 7)
            ys *= 2;
        else
            --shift;
    }
    const unsigned epsilon = (unsigned)rounded((int64_t)RMS_EPS_Q32 * QWEN_HIDDEN,
                                               (int64_t)xs * xs);
    if (m->device) {
        if (m->device->rms(m->device->ctx, a, g, (unsigned)gain, epsilon, shift, result))
            return -1;
    } else {
        c11b_cpu_rms(a, g, (unsigned)gain, epsilon, shift, result);
    }
    for (unsigned i = 0; i < QWEN_HIDDEN; ++i)
        out[i] = bounded((int64_t)result[i] * ys);
    return 0;
}

static int native_ffn_activation(qwen_reference *m)
{
    const unsigned gs = quant_scale(m->gate, QWEN_INTER), us = quant_scale(m->up, QWEN_INTER);
    int8_t ga[QWEN_INTER], up[QWEN_INTER], out[QWEN_INTER], lut[256];
    int32_t values[256];
    quant_bytes(m->gate, QWEN_INTER, gs, ga);
    quant_bytes(m->up, QWEN_INTER, us, up);
    for (unsigned i = 0; i < 256; ++i) {
        int64_t x = (i < 128 ? (int)i : (int)i - 256) * (int64_t)gs;
        uint32_t e = exp_negative(m->exp, x < 0 ? -x : x);
        values[i] = bounded(rounded(x * (x < 0 ? e : Q24), Q24 + e));
    }
    const unsigned ys = quant_scale(values, 256);
    quant_bytes(values, 256, ys, lut);
    int peak = 0;
    for (unsigned i = 0; i < QWEN_INTER; ++i) {
        int prod = lut[(uint8_t)ga[i]] * up[i];
        if (prod < 0)
            prod = -prod;
        if (prod > peak)
            peak = prod;
    }
    unsigned shift = 0;
    while (rounded(peak, 1u << shift) > 127)
        ++shift;
    if (m->device) {
        if (m->device->swiglu(m->device->ctx, ga, up, lut, shift, out))
            return -1;
    } else {
        c11b_cpu_swiglu(ga, up, lut, shift, out);
    }
    for (unsigned i = 0; i < QWEN_INTER; ++i)
        m->gate[i] = bounded(rounded((int64_t)out[i] * ys * us * (1u << shift), QWEN_Q16));
    return 0;
}

static unsigned exp_shift(uint32_t qs, uint32_t ks)
{
    uint64_t product = (uint64_t)qs * ks;
    int exponent = 34;   /* round(log2(8 ln(2))) + 32 */
    for (; product > 1; product >>= 1)
        --exponent;
    return exponent > 31 ? 31u : exponent < 0 ? 0u : (unsigned)exponent;
}

static int attend_native(qwen_reference *m, unsigned layer, c11b_layer0 *capture)
{
    int8_t q[QWEN_HIDDEN], k[KV_WIDTH], v[KV_WIDTH], context[QWEN_HIDDEN];
    unsigned shift[QWEN_HEADS];
    const unsigned pos = m->position;
    if (!pos) {
        m->ks[layer] = 2 * quant_scale(m->k, KV_WIDTH);
        m->vs[layer] = 2 * quant_scale(m->v, KV_WIDTH);
    }
    for (unsigned h = 0; h < QWEN_HEADS; ++h) {
        const int32_t *piece = m->q + h * QWEN_HEAD_DIM;
        const uint32_t qs = quant_scale(piece, QWEN_HEAD_DIM);
        quant_bytes(piece, QWEN_HEAD_DIM, qs, q + h * QWEN_HEAD_DIM);
        shift[h] = exp_shift(qs, m->ks[layer]);
    }
    quant_bytes(m->k, KV_WIDTH, m->ks[layer], k);
    quant_bytes(m->v, KV_WIDTH, m->vs[layer], v);
    if (m->device) {
        if (m->device->attention(m->device->ctx, layer, pos, q, k, v, shift, context,
                                 capture ? capture->keys : NULL, capture ? capture->values : NULL))
            return -1;
    } else {
        const size_t extent = C11B_PITCH * KV_WIDTH;
        int8_t *keys = (int8_t *)m->keys + layer * extent;
        int8_t *values = (int8_t *)m->values + layer * extent;
        if (!pos) {
            memset(keys, 0, extent);
            memset(values, 0, extent);
        }
        memcpy(keys + pos * KV_WIDTH, k, KV_WIDTH);
        memcpy(values + pos * KV_WIDTH, v, KV_WIDTH);
        c11b_cpu_attention(q, keys, values, shift, pos, context);
        if (capture) {
            memcpy(capture->keys, keys, extent);
            memcpy(capture->values, values, extent);
        }
    }
    for (unsigned i = 0; i < QWEN_HIDDEN; ++i)
        m->mix[i] = bounded((int64_t)context[i] * m->vs[layer]);
    return 0;
}

static int layer_native(qwen_reference *m, unsigned layer, c11b_layer0 *capture)
{
    const block *b = m->layer + layer;
    const uint8_t *table = m->rope + (size_t)m->position * C11B_TRIG_BYTES;
    activation input;
    if (norm_native(m, m->x, b->rms_in, m->norm))
        return -1;
    quantize(&input, m->norm, QWEN_HIDDEN);
    if (project_native(m, &b->q, &input, m->q) ||
        project_native(m, &b->k, &input, m->k) ||
        project_native(m, &b->v, &input, m->v))
        return -1;
    if (m->device) {
        if (m->device->rope(m->device->ctx, m->q, m->k, table))
            return -1;
    } else {
        rotate(m->q, QWEN_HEADS, table);
        rotate(m->k, QWEN_KV_HEADS, table);
    }
    if (attend_native(m, layer, capture))
        return -1;
    quantize(&input, m->mix, QWEN_HIDDEN);
    if (project_native(m, &b->o, &input, m->norm))
        return -1;
    for (unsigned d = 0; d < QWEN_HIDDEN; ++d)
        m->x[d] = bounded((int64_t)m->x[d] + m->norm[d]);

    if (norm_native(m, m->x, b->rms_ffn, m->norm))
        return -1;
    quantize(&input, m->norm, QWEN_HIDDEN);
    if (project_native(m, &b->gate, &input, m->gate) ||
        project_native(m, &b->up, &input, m->up) ||
        native_ffn_activation(m))
        return -1;
    quantize(&input, m->gate, QWEN_INTER);
    if (project_native(m, &b->down, &input, m->norm))
        return -1;
    for (unsigned d = 0; d < QWEN_HIDDEN; ++d)
        m->x[d] = bounded((int64_t)m->x[d] + m->norm[d]);
    if (capture)
        memcpy(capture->hidden, m->x, sizeof(m->x));
    return 0;
}

static int valid_native_step(qwen_reference *m, unsigned token)
{
    return m && m->native && !m->native_failed && token < QWEN_VOCAB &&
           m->position < qwen_reference_context(m) &&
           (!m->device || !m->device->faulted(m->device->ctx));
}

int qwen_reference_step_layer0(qwen_reference *m, unsigned token, c11b_layer0 *capture)
{
    if (!valid_native_step(m, token) || !capture)
        return -1;
    embed(&m->embedding, token, m->x);
    if (layer_native(m, 0, capture)) {
        m->native_failed = 1;
        return -1;
    }
    ++m->position;
    return 0;
}

static int tied_head_native(qwen_reference *m, const activation *input)
{
    if (!m->device) {
        linear(&m->embedding, input, m->logits);
        return 0;
    }
    enum { PANEL_COLS = 2048, GROUPS = QWEN_HIDDEN / QWEN_GROUP };
    uint8_t *packed = malloc((size_t)PANEL_COLS * QWEN_HIDDEN / 2u);
    int32_t *partial = malloc((size_t)PANEL_COLS * GROUPS * sizeof(*partial));
    if (!packed || !partial) {
        free(packed);
        free(partial);
        return -1;
    }
    const matrix *mat = &m->embedding;
    int failure = 0;
    for (unsigned first = 0; first < mat->outputs && !failure; first += PANEL_COLS) {
        unsigned count = mat->outputs - first;
        if (count > PANEL_COLS)
            count = PANEL_COLS;
        for (unsigned g = 0; g < GROUPS; ++g)
            memcpy(packed + (size_t)g * count * (QWEN_GROUP / 2),
                   mat->packed + ((size_t)g * mat->outputs + first) * (QWEN_GROUP / 2),
                   (size_t)count * (QWEN_GROUP / 2));
        failure = m->device->project(m->device->ctx, packed, count, QWEN_HIDDEN, input->a8, partial);
        if (failure)
            break;
        for (unsigned col = 0; col < count; ++col) {
            int64_t sum = 0;
            for (unsigned g = 0; g < GROUPS; ++g) {
                const size_t at = (size_t)g * mat->outputs + first + col;
                sum += (int64_t)partial[(size_t)g * count + col] * u32(mat->scale + 4 * at);
            }
            m->logits[first + col] = bounded((sum / Q24) * input->scale +
                                             rounded((sum % Q24) * input->scale, Q24));
        }
    }
    free(packed);
    free(partial);
    return failure;
}

static int process_native(qwen_reference *m, unsigned token, int logits)
{
    if (!valid_native_step(m, token))
        return -1;
    embed(&m->embedding, token, m->x);
    for (unsigned i = 0; i < QWEN_LAYERS; ++i)
        if (layer_native(m, i, NULL)) {
            m->native_failed = 1;
            return -1;
        }
    if (logits) {
        rms(m->x, m->rms, m->norm);
        activation input;
        quantize(&input, m->norm, QWEN_HIDDEN);
        if (tied_head_native(m, &input)) {
            m->native_failed = 1;
            return -1;
        }
    }
    ++m->position;
    return 0;
}

int qwen_reference_advance(qwen_reference *m, unsigned token)
{
    return process_native(m, token, 0);
}

static const int32_t *step_native(qwen_reference *m, unsigned token)
{
    return process_native(m, token, 1) ? NULL : m->logits;
}
