/* Frozen byte-domain attention, independent from the RTL descriptor code. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { POSITIONS = 10, GROUPS = 2, QUERIES = 7, WIDTH = 64, PITCH = 12 };

static void attention(const int8_t *q, const int8_t k[PITCH][WIDTH],
                      const int8_t v[WIDTH][PITCH], unsigned pos, unsigned shift, int8_t *out)
{
    int score[POSITIONS], peak = -2147483647;
    for (unsigned p = 0; p <= pos; ++p) {
        int dot = 0;
        for (unsigned d = 0; d < WIDTH; ++d)
            dot += q[d] * k[p][d];
        score[p] = dot;
        if (dot > peak)
            peak = dot;
    }
    unsigned prob[POSITIONS], total = 0;
    for (unsigned p = 0; p <= pos; ++p) {
        unsigned distance = (unsigned)(peak - score[p]) >> shift;
        prob[p] = distance <= 8 ? 256u >> distance : 0;
        total += prob[p];
    }
    for (unsigned p = 0; p <= pos; ++p)
        prob[p] = 255u * prob[p] / total;
    for (unsigned d = 0; d < WIDTH; ++d) {
        int dot = 0;
        for (unsigned p = 0; p <= pos; ++p)
            dot += (int)prob[p] * v[d][p];
        int shifted = dot < 0 ? -((-dot + 255) / 256) : dot / 256;
        out[d] = (int8_t)(shifted < -128 ? -128 : shifted > 127 ? 127 : shifted);
    }
}

int main(int argc, char **argv)
{
    if (argc != 5)
        return 2;
    char *tail;
    unsigned long shift = strtoul(argv[4], &tail, 10);
    if (*tail || shift > 31)
        return 2;
    FILE *input = fopen(argv[1], "rb"), *output = fopen(argv[2], "wb"), *kv = fopen(argv[3], "wb");
    int rc = 1;
    if (input && output && kv) {
        int8_t keys[GROUPS][PITCH][WIDTH] = {{{0}}}, values[GROUPS][WIDTH][PITCH] = {{{0}}};
        rc = 0;
        for (unsigned pos = 0; pos < POSITIONS && !rc; ++pos)
            for (unsigned g = 0; g < GROUPS && !rc; ++g) {
                int8_t next[QUERIES + 2][WIDTH], result[QUERIES][WIDTH];
                if (fread(next, sizeof(next), 1, input) != 1) {
                    rc = 1;
                    break;
                }
                for (unsigned d = 0; d < WIDTH; ++d) {
                    keys[g][pos][d] = next[QUERIES][d];
                    values[g][d][pos] = next[QUERIES + 1][d];
                }
                for (unsigned head = 0; head < QUERIES; ++head)
                    attention(next[head], keys[g], values[g], pos, (unsigned)shift, result[head]);
                rc = fwrite(result, sizeof(result), 1, output) != 1;
                if (pos >= 7 && !rc)
                    rc = fwrite(keys[g], sizeof(keys[g]), 1, kv) != 1 ||
                         fwrite(values[g], sizeof(values[g]), 1, kv) != 1;
            }
        if (!rc && (fgetc(input) != EOF || ferror(input)))
            rc = 1;
    }
    if (input)
        fclose(input);
    if (output && fclose(output))
        rc = 1;
    if (kv && fclose(kv))
        rc = 1;
    return rc;
}
