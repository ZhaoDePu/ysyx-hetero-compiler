/* Exercise the full deterministic reference with externally tokenized input. */
#include "c11_qwen_ref.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned greedy(const int32_t *logits, int suppress_eos)
{
    unsigned best = 0;
    for (unsigned i = 1; i < QWEN_VOCAB; ++i)
        if ((!suppress_eos || (i != 151643 && i != 151645)) && logits[i] > logits[best])
            best = i;
    return best;
}

static int save_projection(const qwen_reference *model, const char *path)
{
    int8_t activation[QWEN_HIDDEN];
    uint32_t scale;
    if (qwen_reference_lm_input(model, activation, &scale))
        return -1;
    uint8_t header[4] = {(uint8_t)scale, (uint8_t)(scale >> 8),
                         (uint8_t)(scale >> 16), (uint8_t)(scale >> 24)};
    FILE *file = fopen(path, "wb");
    if (!file)
        return -1;
    int failed = fwrite(header, 1, 4, file) != 4 ||
                 fwrite(activation, 1, sizeof(activation), file) != sizeof(activation);
    return fclose(file) || failed;
}

static int generate(qwen_reference *model, FILE *input, FILE *gold, unsigned steps,
                    const char *projection, unsigned min_steps)
{
    const int32_t *logits = NULL;
    unsigned token;
    unsigned prompt = 0;
    for (int result; (result = fscanf(input, "%u", &token)) != EOF;) {
        if (result != 1 || ++prompt + steps > qwen_reference_context(model))
            return -1;
        logits = qwen_reference_step(model, token);
        if (!logits || fwrite(logits, sizeof(*logits), QWEN_VOCAB, gold) != QWEN_VOCAB)
            return -1;
    }
    if (!logits || ferror(input) || fflush(gold) || (projection && save_projection(model, projection)))
        return -1;
    for (unsigned i = 0; i < steps; ++i) {
        unsigned best = greedy(logits, i < min_steps);
        printf("%u\n", best);
        if (best == 151645 || best == 151643 || i + 1 == steps)
            break;
        logits = qwen_reference_step(model, best);
        if (!logits)
            return -1;
    }
    return fflush(stdout);
}

int main(int argc, char **argv)
{
    if (argc != 5 && argc != 6) {
        fputs("Usage: c11_qwen_reference WEIGHTS INPUT_IDS STEPS PREFILL_LOGITS [LM_INPUT]\n", stderr);
        return 2;
    }
    char *end;
    errno = 0;
    unsigned long steps = strtoul(argv[3], &end, 10);
    if (errno || *end || !steps || steps > 2048)
        return 2;
    const char *minimum = getenv("YSYX_QWEN_MIN_NEW_TOKENS");
    unsigned long min_steps = minimum ? strtoul(minimum, &end, 10) : 0;
    if (minimum && (*end || min_steps > steps))
        return 2;
    qwen_reference *model = qwen_reference_open(argv[1]);
    FILE *input = fopen(argv[2], "r");
    FILE *output = fopen(argv[4], "wb");
#ifdef YSYX_C11B_NATIVE
    int failed = qwen_reference_native(model, NULL);
#else
    int failed = 0;
#endif
    if (!failed)
        failed = !model || !input || !output || generate(model, input, output, (unsigned)steps,
                                                         argc == 6 ? argv[5] : NULL, (unsigned)min_steps);
    if (input)
        fclose(input);
    if (output && fclose(output))
        failed = 1;
    qwen_reference_close(model);
    if (failed)
        fputs("C11 reference could not read the weight image, tokens, or output\n", stderr);
    return failed;
}
