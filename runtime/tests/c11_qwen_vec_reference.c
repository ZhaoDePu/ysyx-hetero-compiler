/* Arithmetic reference for Step 2 VPU native byte-domain fixed point. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int8_t signed_byte(int64_t x)
{
    return (int8_t)(x < -128 ? -128 : x > 127 ? 127 : x);
}

static int64_t floor_shift(int64_t value, unsigned shift)
{
    int64_t denom = INT64_C(1) << shift;
    return value < 0 ? -((-value + denom - 1) / denom) : value / denom;
}

static unsigned parse(const char *text, unsigned max)
{
    char *end;
    unsigned long value = strtoul(text, &end, 10);
    return *end || value > max ? max + 1 : (unsigned)value;
}

static const uint16_t reciprocal_sqrt[] = {
    32768, 32268, 31790, 31332, 30894, 30474, 30070, 29682,
    29309, 28949, 28602, 28268, 27945, 27632, 27330, 27038,
    26755, 26481, 26214, 25956, 25705, 25462, 25225, 24994,
    24770, 24552, 24339, 24132, 23930, 23733, 23541, 23354,
    23170, 22992, 22817, 22646, 22479, 22315, 22155, 21999,
    21845, 21695, 21548, 21404, 21263, 21124, 20988, 20855,
    20724, 20596, 20470, 20346, 20225, 20106, 19988, 19873,
    19760, 19649, 19539, 19431, 19326, 19221, 19119, 19018,
    18919, 18821, 18725, 18630, 18536, 18444, 18354, 18264,
    18176, 18090, 18004, 17920, 17837, 17755, 17674, 17594,
    17515, 17438, 17361, 17285, 17211, 17137, 17064, 16992,
    16921, 16851, 16782, 16714, 16646, 16579, 16514, 16448,
    16384,
};

static void rms(int8_t *out, const int8_t *lhs, const int8_t *gamma, unsigned width,
                unsigned gain, unsigned epsilon, unsigned shift)
{
    uint64_t total = epsilon;
    for (unsigned i = 0; i < width; ++i)
        total += (unsigned)(lhs[i] * lhs[i]);
    if (!total)
        total = 1;
    unsigned high = 0;
    for (uint64_t t = total; t > 1; t >>= 1)
        ++high;
    unsigned exponent = high / 2;
    uint64_t m = exponent <= 4 ? total << (9 - 2 * exponent) : total >> (2 * exponent - 9);
    unsigned mantissa = (unsigned)m & 2047u;
    unsigned bin = (mantissa >> 4) - 32, frac = mantissa & 15u;
    unsigned a = reciprocal_sqrt[bin], b = reciprocal_sqrt[bin + 1];
    unsigned interp = a - (((a - b) * frac + 8) >> 4);
    int64_t inv = (((uint64_t)interp * gain) >> (8 + exponent)) & 0x1fffff;
    for (unsigned i = 0; i < width; ++i)
        out[i] = signed_byte(floor_shift((int64_t)lhs[i] * gamma[i] * inv, shift));
}

static void gated(int8_t *out, const int8_t *gate, const int8_t *up, const int8_t *lut,
                  unsigned width, unsigned shift)
{
    int64_t rounding = shift ? INT64_C(1) << (shift - 1) : 0;
    for (unsigned i = 0; i < width; ++i)
        out[i] = signed_byte(floor_shift(lut[(uint8_t)gate[i]] * up[i] + rounding, shift));
}

int main(int argc, char **argv)
{
    if (argc != 9)
        return 2;
    int norm = !strcmp(argv[1], "rms");
    if (!norm && strcmp(argv[1], "swiglu"))
        return 2;
    unsigned width = parse(argv[2], 4864), gain = parse(argv[3], 65535);
    unsigned eps = parse(argv[4], 0xffffff), shift = parse(argv[5], 31);
    if (!width || width > 4864 || gain > 65535 || eps > 0xffffff || shift > 31)
        return 2;
    int8_t left[4864], right[4864], result[4864], lut[256];
    FILE *file = fopen(argv[6], "rb");
    if (!file)
        return 1;
    int failed = fread(left, 1, width, file) != width || fread(right, 1, width, file) != width;
    failed |= fclose(file) != 0;
    if (!norm && !failed) {
        file = fopen(argv[7], "rb");
        if (!file)
            return 1;
        failed = fread(lut, 1, sizeof(lut), file) != sizeof(lut);
        failed |= fclose(file) != 0;
    }
    if (failed)
        return 1;

    if (norm)
        rms(result, left, right, width, gain, eps, shift);
    else
        gated(result, left, right, lut, width, shift);
    file = fopen(argv[8], "wb");
    if (!file)
        return 1;
    failed = fwrite(result, 1, width, file) != width;
    return fclose(file) || failed;
}
