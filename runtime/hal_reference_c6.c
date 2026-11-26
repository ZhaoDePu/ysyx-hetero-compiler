#include "hal_reference_c6.h"
#include "z15_regs.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint8_t *ddr;
    uint8_t scratch[Z15_SCRATCH_SIZE];
    uint32_t registers[64];
    uint32_t code[Z15_PROGRAM_BYTES / 4u];
    uint32_t scales[512];
    uint8_t activation[256];
} c6_reference_t;

#define REG(ref, address) ((ref)->registers[(address) / 4u])

static int extent(uint32_t at, size_t bytes, uint32_t capacity)
{
    return at <= capacity && bytes <= capacity - at;
}

static int scratch_span(uint32_t at, size_t bytes, unsigned half)
{
    return (at >> 20) == half && extent(at, bytes, (half + 1u) * Z15_HALF_BYTES);
}

static unsigned rounded_word(unsigned n)
{
    return (n + 3u) & ~3u;
}

static int32_t signed32(uint32_t bits)
{
    return bits <= INT32_MAX ? (int32_t)bits : -1 - (int32_t)(UINT32_MAX - bits);
}

static int32_t signed8(uint8_t bits)
{
    return bits < 128u ? (int32_t)bits : (int32_t)bits - 256;
}

static int64_t asr(int64_t value, unsigned shift)
{
    return value < 0 ? -1 - ((-1 - value) >> shift) : value >> shift;
}

static uint8_t clip8(int64_t value)
{
    return (uint8_t)(value < -128 ? -128 : value > 127 ? 127 : value);
}

static uint32_t read_word(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static void write_word(uint8_t *p, uint32_t word)
{
    for (unsigned i = 0; i < 4; ++i)
        p[i] = (uint8_t)(word >> (8u * i));
}

static int32_t dot(const uint8_t *a, const uint8_t *b, unsigned n, int unsigned_a)
{
    int32_t sum = 0;
    if (unsigned_a) {
        for (unsigned i = 0; i < n; ++i)
            sum += (int32_t)a[i] * signed8(b[i]);
    } else {
        for (unsigned i = 0; i < n; ++i)
            sum += signed8(a[i]) * signed8(b[i]);
    }
    return sum;
}

static int tensor(c6_reference_t *ref)
{
    const uint32_t dims = REG(ref, Z15_NPU_DIMS), depth = REG(ref, Z15_NPU_K);
    const unsigned m = dims & 511u, n = (dims >> 16) & 511u, k = depth & 511u;
    const unsigned kstride = rounded_word(k);
    const uint32_t a = REG(ref, Z15_NPU_A), b = REG(ref, Z15_NPU_B), c = REG(ref, Z15_NPU_C);
    const uint32_t bias = REG(ref, Z15_NPU_BIAS), flags = REG(ref, Z15_NPU_FLAGS);
    const unsigned half = a >> 20, packed = (flags >> 17) & 1u;
    const unsigned output_stride = packed ? rounded_word(n) : 4u * n;
    const unsigned shape = REG(ref, Z15_NPU_SHAPE), roi = REG(ref, Z15_NPU_SLICE);
    const unsigned conv = (flags >> 9) & 1u, unsigned_a = flags & 1u;
    const int height = (int)(shape & 511u), width = (int)((shape >> 9) & 511u);
    const unsigned channels = (shape >> 18) & 511u, cstride = rounded_word(channels);
    const int kernel = flags & (1u << 10) ? 3 : 1;
    const int stride = flags & (1u << 11) ? 2 : 1;
    const int pad = flags & (1u << 12) ? 1 : 0;
    unsigned ow = 0, first = 0;
    size_t input_size = (size_t)m * kstride;
    if (conv) {
        const int padded_h = height + 2 * pad - kernel, padded_w = width + 2 * pad - kernel;
        if (!height || !width || !channels || padded_h < 0 || padded_w < 0 ||
            cstride * (unsigned)(kernel * kernel) != k)
            return -1;
        const unsigned oh = (unsigned)(padded_h / stride + 1);
        ow = (unsigned)(padded_w / stride + 1);
        first = flags & (1u << 13) ? roi & 511u : 0;
        const unsigned rows = flags & (1u << 13) ? (roi >> 16) & 511u : oh;
        if (first + rows > oh || rows * ow != m)
            return -1;
        input_size = (size_t)height * (size_t)width * cstride;
    }
    if (!m || !n || !k || half > 1u || !scratch_span(a, input_size, half) ||
        !scratch_span(b, n * kstride, half) || !scratch_span(c, m * output_stride, half) ||
        ((flags & 2u) && !scratch_span(bias, 4u * n, half)))
        return -1;
    for (unsigned row = 0; row < m; ++row) {
        const int y = conv ? (int)(first + row / ow) * stride - pad : 0;
        const int x = conv ? (int)(row % ow) * stride - pad : 0;
        uint8_t *out = ref->scratch + c + row * output_stride;
        for (unsigned col = 0; col < n; ++col) {
            const uint8_t *weight = ref->scratch + b + col * kstride;
            uint32_t sum = flags & 2u ? read_word(ref->scratch + bias + 4u * col) : 0;
            if (conv) {
                for (int dy = 0; dy < kernel; ++dy) {
                    if (y + dy < 0 || y + dy >= height)
                        continue;
                    for (int dx = 0; dx < kernel; ++dx) {
                        if (x + dx < 0 || x + dx >= width)
                            continue;
                        const size_t pixel = (size_t)(y + dy) * (unsigned)width + (unsigned)(x + dx);
                        const uint8_t *data = ref->scratch + a + pixel * cstride;
                        const unsigned tap = (unsigned)(dy * kernel + dx) * cstride;
                        sum += (uint32_t)dot(data, weight + tap, channels, (int)unsigned_a);
                    }
                }
            } else {
                sum += (uint32_t)dot(ref->scratch + a + row * kstride, weight, k, (int)unsigned_a);
            }
            int64_t value = signed32(sum);
            if (flags & (1u << 15)) {
                const uint32_t scale = ref->scales[col];
                value = asr(value * (scale & 65535u), (scale >> 16) & 31u);
            } else if (flags & 4u) {
                value = asr(value, (flags >> 4) & 31u);
            }
            if ((flags & 8u) && value < 0)
                value = 0;
            if ((flags & (1u << 15)) || (flags & 4u))
                value = signed8(clip8(value));
            if (flags & (1u << 16))
                value = signed8(ref->activation[(uint8_t)value]);
            if (packed)
                out[col] = (uint8_t)value;
            else
                write_word(out + 4u * col, (uint32_t)value);
        }
    }
    return 0;
}

static int ew_linear(c6_reference_t *ref, unsigned op, uint32_t a, uint32_t b, uint32_t c,
                     unsigned half, uint32_t flags)
{
    const size_t count = REG(ref, Z15_EW_LENGTH);
    if (!count || !scratch_span(a, count, half) || !scratch_span(c, count, c >> 20) ||
        ((op == 0 || op == 6) && !scratch_span(b, count, half)))
        return -1;
    uint8_t *dst = ref->scratch + c;
    const uint8_t *first = ref->scratch + a;
    const uint8_t *second = ref->scratch + (op == 0 || op == 6 ? b : a);
    if (op == 8) {
        memmove(dst, first, count);
    } else if (op == 11) {
        for (size_t i = 0; i < count; ++i)
            dst[i] = ref->activation[first[i]];
    } else if (op == 0) {
        for (size_t i = 0; i < count; ++i)
            dst[i] = clip8(signed8(first[i]) + signed8(second[i]));
    } else {
        const int64_t mul = (flags >> 8) & 65535u;
        const unsigned shift = (flags >> 24) & 31u;
        const int64_t round = shift ? INT64_C(1) << (shift - 1u) : 0;
        for (size_t i = 0; i < count; ++i)
            dst[i] = clip8(asr((int64_t)signed8(first[i]) * signed8(second[i]) * mul + round, shift));
    }
    return 0;
}

static int ew_spatial(c6_reference_t *ref, unsigned op, uint32_t a, uint32_t b, uint32_t c, unsigned half)
{
    const unsigned shape = REG(ref, Z15_EW_SHAPE), chans = REG(ref, Z15_EW_CHANNELS);
    const unsigned h = shape & 1023u, w = (shape >> 16) & 1023u;
    const unsigned ca = chans & 1023u, cb = (chans >> 16) & 1023u;
    const size_t pixels = (size_t)h * w;
    const unsigned chow = ca + cb;
    if (!h || (!w && op != 9) || !ca)
        return -1;
    if (op == 9) {
        const size_t input_bytes = (size_t)h * ca, output_bytes = (size_t)(ca - 1u) * cb + h;
        if (cb < h || !scratch_span(a, input_bytes, half) || !scratch_span(c, output_bytes, half))
            return -1;
        for (unsigned row = 0; row < h; ++row)
            for (unsigned col = 0; col < ca; ++col)
                ref->scratch[c + col * cb + row] = ref->scratch[a + row * ca + col];
        return 0;
    }
    const size_t input_bytes = pixels * (op == 2 ? chow : ca);
    size_t output_bytes = pixels * (op == 1 ? chow : ca);
    if (op == 3)
        output_bytes *= 4;
    if (!scratch_span(a, input_bytes, half) || !scratch_span(c, output_bytes, half))
        return -1;
    uint8_t *dst = ref->scratch + c;
    const uint8_t *src = ref->scratch + a;
    if (op == 1 || op == 2) {
        const uint32_t other = op == 1 ? b : REG(ref, Z15_EW_D);
        if (!cb || !scratch_span(other, pixels * cb, half))
            return -1;
        for (size_t i = 0; i < pixels; ++i) {
            uint8_t *other_ptr = ref->scratch + other + i * cb;
            if (op == 1) {
                memcpy(dst + i * chow, src + i * ca, ca);
                memcpy(dst + i * chow + ca, other_ptr, cb);
            } else {
                memcpy(dst + i * ca, src + i * chow, ca);
                memcpy(other_ptr, src + i * chow + ca, cb);
            }
        }
    } else if (op == 3) {
        for (size_t y = 0; y < 2u * h; ++y)
            for (size_t x = 0; x < 2u * w; ++x)
                memcpy(dst + (y * 2u * w + x) * ca, src + (y / 2u * w + x / 2u) * ca, ca);
    } else if (op == 5) {
        for (unsigned y = 0; y < h; ++y) {
            for (unsigned x = 0; x < w; ++x) {
                for (unsigned channel = 0; channel < ca; ++channel) {
                    int32_t best = -128;
                    for (unsigned row = y > 2 ? y - 2u : 0; row <= y + 2u && row < h; ++row) {
                        for (unsigned col = x > 2 ? x - 2u : 0; col <= x + 2u && col < w; ++col) {
                            const int32_t at = signed8(src[(row * w + col) * ca + channel]);
                            if (at > best)
                                best = at;
                        }
                    }
                    dst[(y * w + x) * ca + channel] = (uint8_t)best;
                }
            }
        }
    } else {
        return -1;
    }
    return 0;
}

static unsigned softmax_weight(int32_t highest, int32_t value, unsigned shift)
{
    const uint64_t exponent = (uint64_t)((int64_t)highest - value) >> shift;
    return exponent > 8u ? 0u : 256u >> exponent;
}

static int ew_softmax(c6_reference_t *ref, uint32_t a, uint32_t c, unsigned half, uint32_t flags)
{
    const uint32_t shape = REG(ref, Z15_EW_SHAPE);
    const unsigned rows = shape & 1023u, cols = (shape >> 16) & 1023u;
    const unsigned shift = (flags >> 24) & 31u;
    if (!rows || rows > 511 || !cols || cols > 128 || (cols & 3u) || flags & 0xffff00u
            || !scratch_span(a, 4u * rows * cols, half) || !scratch_span(c, rows * cols, half))
        return -1;
    const uint8_t *input = ref->scratch + a;
    uint8_t *output = ref->scratch + c;
    for (unsigned row = 0; row < rows; ++row) {
        int32_t highest = INT32_MIN;
        unsigned weight[128], denominator = 0;
        for (unsigned col = 0; col < cols; ++col) {
            const int32_t value = signed32(read_word(input + col * 4u));
            if (value > highest)
                highest = value;
        }
        for (unsigned col = 0; col < cols; ++col) {
            weight[col] = softmax_weight(highest, signed32(read_word(input + col * 4u)), shift);
            denominator += weight[col];
        }
        for (unsigned col = 0; col < cols; ++col)
            output[col] = (uint8_t)(255u * weight[col] / denominator);
        input += 4u * cols;
        output += cols;
    }
    return 0;
}

static int elementwise(c6_reference_t *ref)
{
    const uint32_t flags = REG(ref, Z15_EW_FLAGS);
    const unsigned op = flags & 15u;
    const uint32_t a = REG(ref, Z15_EW_A), b = REG(ref, Z15_EW_B), c = REG(ref, Z15_EW_C);
    const unsigned half = a >> 20;
    if (half > 1u || c >= Z15_SCRATCH_SIZE || (op != 8 && (c >> 20) != half))
        return -1;
    if (op == 0 || op == 6 || op == 8 || op == 11)
        return ew_linear(ref, op, a, b, c, half, flags);
    if (op == 12)
        return ew_softmax(ref, a, c, half, flags);
    return ew_spatial(ref, op, a, b, c, half);
}

static uint32_t gpu_imm(uint32_t raw, unsigned bits)
{
    const uint32_t mask = UINT32_C(1) << (bits - 1u);
    return (raw ^ mask) - mask;
}

static int gpu_branch(unsigned op, uint32_t a, uint32_t b, int *take)
{
    switch (op) {
    case 0: *take = a == b; return 0;
    case 1: *take = a != b; return 0;
    case 4: *take = signed32(a) < signed32(b); return 0;
    case 5: *take = signed32(a) >= signed32(b); return 0;
    case 6: *take = a < b; return 0;
    case 7: *take = a >= b; return 0;
    default: return -1;
    }
}

static int gpu_arithmetic(unsigned op, unsigned f7, uint32_t a, uint32_t b, uint32_t *answer)
{
    const unsigned shift = b & 31u;
    if (f7 == 1u) {
        switch (op) {
        case 0: *answer = a * b; return 0;
        case 1: *answer = (uint32_t)asr((int64_t)signed32(a) * signed32(b), 32); return 0;
        /* Use a widened numerator so INT32_MIN / -1 wraps to RV32
           without causing signed host overflow. */
        case 4: *answer = b ? (uint32_t)((int64_t)signed32(a) / signed32(b)) : UINT32_MAX; return 0;
        case 5: *answer = b ? a / b : UINT32_MAX; return 0;
        case 6: *answer = b ? (uint32_t)((int64_t)signed32(a) % signed32(b)) : a; return 0;
        case 7: *answer = b ? a % b : a; return 0;
        default: return -1;
        }
    }
    if (f7 == 32u) {
        if (op == 0) {
            *answer = a - b;
            return 0;
        }
        if (op == 5) {
            *answer = (uint32_t)asr(signed32(a), shift);
            return 0;
        }
        return -1;
    }
    if (f7 != 0)
        return -1;
    switch (op) {
    case 0: *answer = a + b; return 0;
    case 1: *answer = a << shift; return 0;
    case 2: *answer = signed32(a) < signed32(b); return 0;
    case 3: *answer = a < b; return 0;
    case 4: *answer = a ^ b; return 0;
    case 5: *answer = a >> shift; return 0;
    case 6: *answer = a | b; return 0;
    case 7: *answer = a & b; return 0;
    default: return -1;
    }
}

static int gpu_memory(uint8_t *page, uint32_t bytes, uint32_t at, unsigned op, int store, uint32_t *value)
{
    const unsigned size = op == 2u ? 4u : op == 1u || op == 5u ? 2u : 1u;
    if (!extent(at, size, bytes) || (at & (size - 1u)) || op > 5u || op == 3u ||
        (store && op > 2u))
        return -1;
    if (store) {
        for (unsigned i = 0; i < size; ++i)
            page[at + i] = (uint8_t)(*value >> (8u * i));
        return 0;
    }
    *value = size == 4u ? read_word(page + at) : page[at];
    if (size == 2u)
        *value |= (uint32_t)page[at + 1u] << 8;
    if (op < 2u)
        *value = gpu_imm(*value, size * 8u);
    return 0;
}

static int gpu_lane(const uint32_t *code, uint8_t *page, uint32_t bytes, uint32_t pc, unsigned lane)
{
    uint32_t x[32] = {0};
    for (unsigned iterations = 0; iterations < 1000000u; ++iterations) {
        if (pc & 3u || pc >= Z15_PROGRAM_BYTES)
            return -1;
        const uint32_t word = code[pc / 4u], op = word & 127u, f3 = (word >> 12) & 7u;
        const unsigned rd = (word >> 7) & 31u, rs1 = (word >> 15) & 31u, rs2 = (word >> 20) & 31u;
        const unsigned f7 = word >> 25;
        const uint32_t a = x[rs1], b = x[rs2];
        uint32_t value = 0, next = pc + 4u;
        switch (op) {
        case 0x37: /* LUI */
            value = word & UINT32_C(0xfffff000);
            break;
        case 0x13: /* OP-IMM */
            if (gpu_arithmetic(f3, f3 == 1u || f3 == 5u ? f7 : 0, a, gpu_imm(word >> 20, 12), &value))
                return -1;
            break;
        case 0x33: /* OP / M */
            if (gpu_arithmetic(f3, f7, a, b, &value))
                return -1;
            break;
        case 0x2b: { /* G3 DOT4: signed B, signed or unsigned A. */
            if (f7 || (f3 != 1u && f3 != 2u))
                return -1;
            int32_t sum = 0;
            for (unsigned i = 0; i < 32; i += 8) {
                const uint8_t first = (uint8_t)(a >> i), second = (uint8_t)(b >> i);
                sum += (f3 == 1u ? signed8(first) : (int32_t)first) * signed8(second);
            }
            value = (uint32_t)sum;
            break;
        }
        case 0x03: /* LOAD */
            if (gpu_memory(page, bytes, a + gpu_imm(word >> 20, 12), f3, 0, &value))
                return -1;
            break;
        case 0x23: { /* STORE */
            uint32_t item = b;
            const uint32_t imm = ((word >> 7) & 31u) | (word >> 25) << 5;
            if (gpu_memory(page, bytes, a + gpu_imm(imm, 12), f3, 1, &item))
                return -1;
            break;
        }
        case 0x63: { /* BRANCH */
            const uint32_t imm = ((word >> 8) & 15u) << 1 | ((word >> 25) & 63u) << 5 |
                                 ((word >> 7) & 1u) << 11 | (word >> 31) << 12;
            int taken;
            if (gpu_branch(f3, a, b, &taken))
                return -1;
            pc = taken ? pc + gpu_imm(imm, 13) : next;
            continue;
        }
        case 0x6f: { /* JAL */
            const uint32_t imm = ((word >> 21) & 1023u) << 1 | ((word >> 20) & 1u) << 11 |
                                 ((word >> 12) & 255u) << 12 | (word >> 31) << 20;
            value = next;
            next = pc + gpu_imm(imm, 21);
            break;
        }
        case 0x73: /* C6 kernels use EBREAK and the logical GID CSR. */
            if (word == UINT32_C(0x00100073))
                return 0;
            if (word >> 20 != 0xcc3u || f3 != 2u || rs1)
                return -1;
            value = lane;
            break;
        default:
            return -1;
        }
        if (rd && op != 0x23u)
            x[rd] = value;
        pc = next;
    }
    return -1;
}

static int gpu(c6_reference_t *ref)
{
    const uint32_t cfg = REG(ref, Z15_GPU_JOB), pc = REG(ref, Z15_GPU_PC);
    const unsigned page = cfg & 511u, threads = (cfg >> 16) & 255u;
    if (!threads || threads > 16u || cfg & UINT32_C(0xfc00fe00))
        return -1;
    const uint32_t offset = 4096u * page;
    /* G3's 18-bit address is relative to the selected page. Leave the lane
       stack region unused and reject accesses outside the owned half. */
    const uint32_t window = (1u << 18) - 4u * 2u * 8u * 256u;
    const uint32_t remaining = Z15_HALF_BYTES - offset % Z15_HALF_BYTES;
    const uint32_t bytes = window < remaining ? window : remaining;
    uint8_t *data = ref->scratch + offset;
    for (unsigned lane = 0; lane < threads; ++lane)
        if (gpu_lane(ref->code, data, bytes, pc, lane))
            return -1;
    REG(ref, Z15_GPU_SLOTS) |= 1u << (8u + ((cfg >> 24) & 3u));
    return 0;
}

static int dma(c6_reference_t *ref)
{
    const uint32_t base = REG(ref, Z15_DMA_BASE), offset = REG(ref, Z15_DMA_OFFSET);
    const uint32_t bytes = REG(ref, Z15_DMA_BYTES), scratch = REG(ref, Z15_DMA_SCRATCH);
    const uint32_t flags = REG(ref, Z15_DMA_FLAGS);
    if (base < Z15_POOL_BASE || base >= Z15_POOL_BASE + Z15_POOL_SIZE ||
        offset >= Z15_POOL_SIZE - (base - Z15_POOL_BASE) || !bytes ||
        (base | offset | bytes | scratch) & 31u || flags & ~3u ||
        !extent(base - Z15_POOL_BASE + offset, bytes, Z15_POOL_SIZE) ||
        (scratch >> 20) > 1u || !scratch_span(scratch, bytes, scratch >> 20))
        return -1;
    uint8_t *dram = ref->ddr + (base - Z15_POOL_BASE) + offset;
    uint8_t *sram = ref->scratch + scratch;
    if (flags & 1u)
        memcpy(dram, sram, bytes);
    else
        memcpy(sram, dram, bytes);
    return 0;
}

static int fail_action(c6_reference_t *ref, uint32_t code)
{
    REG(ref, Z15_ERROR) |= code;
    REG(ref, Z15_STATUS) |= Z15_HAS_ERROR;
    return -1;
}

static int mmio_read(void *opaque, uint32_t offset, uint32_t *out)
{
    c6_reference_t *ref = opaque;
    if ((offset & 3u) || offset >= sizeof(ref->registers))
        return -1;
    *out = REG(ref, offset);
    return 0;
}

static int mmio_write(void *opaque, uint32_t offset, uint32_t value)
{
    c6_reference_t *ref = opaque;
    if (offset & 3u)
        return fail_action(ref, 1u << 4);
    if (offset >= Z15_PROGRAM && offset < Z15_PROGRAM + Z15_PROGRAM_BYTES) {
        ref->code[(offset - Z15_PROGRAM) / 4u] = value;
        return 0;
    }
    if (offset >= sizeof(ref->registers))
        return fail_action(ref, 1u << 4);
    if (offset != Z15_CTRL) {
        REG(ref, offset) = value;
        return 0;
    }
    switch (value) {
    case 2: /* clear sticky completion/error */
        REG(ref, Z15_STATUS) = Z15_LOCKED;
        REG(ref, Z15_ERROR) = 0;
        REG(ref, Z15_GPU_SLOTS) = 0;
        return 0;
    case Z15_START_DMA:
        if (dma(ref))
            return fail_action(ref, Z15_ERR_DMA_DESC);
        REG(ref, Z15_STATUS) |= Z15_DMA_DONE;
        return 0;
    case Z15_START_NPU:
    case Z15_START_EW:
        if (value == Z15_START_NPU ? tensor(ref) : elementwise(ref))
            return fail_action(ref, 1u << 2);
        REG(ref, Z15_STATUS) |= Z15_NPU_DONE;
        return 0;
    case Z15_START_GPU:
        if (gpu(ref))
            return fail_action(ref, 1u << 3);
        REG(ref, Z15_STATUS) |= Z15_GPU_DONE;
        return 0;
    case Z15_WRITE_SCALE:
        if (REG(ref, Z15_SCALE_INDEX) > 511u || REG(ref, Z15_SCALE_DATA) > 0x1fffffu)
            return fail_action(ref, 1u << 4);
        ref->scales[REG(ref, Z15_SCALE_INDEX)] = REG(ref, Z15_SCALE_DATA);
        return 0;
    case Z15_WRITE_ACT:
        if (REG(ref, Z15_ACTIVATION) > 65535u)
            return fail_action(ref, 1u << 4);
        ref->activation[REG(ref, Z15_ACTIVATION) & 255u] = (uint8_t)(REG(ref, Z15_ACTIVATION) >> 8);
        return 0;
    default:
        return fail_action(ref, 1u << 4);
    }
}

static int ddr_write(void *opaque, uint32_t offset, const void *data, size_t bytes)
{
    c6_reference_t *ref = opaque;
    if (!extent(offset, bytes, Z15_POOL_SIZE))
        return -1;
    memcpy(ref->ddr + offset, data, bytes);
    return 0;
}

static int ddr_read(void *opaque, uint32_t offset, void *data, size_t bytes)
{
    const c6_reference_t *ref = opaque;
    if (!extent(offset, bytes, Z15_POOL_SIZE))
        return -1;
    memcpy(data, ref->ddr + offset, bytes);
    return 0;
}

static int cache_sync(void *opaque, uint32_t offset, size_t bytes, int direction)
{
    (void)opaque;
    return extent(offset, bytes, Z15_POOL_SIZE) &&
           (direction == YSYX_SYNC_FOR_DEVICE || direction == YSYX_SYNC_FOR_HOST) ? 0 : -1;
}

static void close_reference(void *opaque)
{
    c6_reference_t *ref = opaque;
    free(ref->ddr);
    free(ref);
}

int ysyx_hal_reference_c6_open(ysyx_hal_t *hal, const char **error)
{
    if (!hal) {
        if (error)
            *error = "missing C6 software reference HAL output";
        return -1;
    }
    c6_reference_t *ref = calloc(1, sizeof(*ref));
    if (ref)
        ref->ddr = calloc(Z15_POOL_SIZE, 1);
    if (!ref || !ref->ddr) {
        free(ref);
        if (error)
            *error = "cannot allocate exact C6 reference HAL memory";
        return -1;
    }
    REG(ref, Z15_STATUS) = Z15_LOCKED;
    REG(ref, Z15_MAGIC) = Z15_MAGIC_VALUE;
    REG(ref, Z15_VERSION) = Z15_VERSION_VALUE;
    REG(ref, Z15_DMA_BASE) = Z15_POOL_BASE;
    *hal = (ysyx_hal_t){.ctx = ref, .mmio_read32 = mmio_read, .mmio_write32 = mmio_write,
                        .ddr_write = ddr_write, .ddr_read = ddr_read, .cache_sync = cache_sync,
                        .close = close_reference};
    if (error)
        *error = NULL;
    return 0;
}
