#include "demo_support.h"

#include <ctype.h>
#include <inttypes.h>
#include <string.h>

static int tokenize(const char *text, char vocab[DEMO_VOCAB_SIZE][DEMO_WORD_BYTES], uint8_t prompt[32])
{
    for (unsigned word = 0; word < 4; ++word) {
        while (isspace((unsigned char)*text))
            ++text;
        const char *start = text;
        while (*text && !isspace((unsigned char)*text))
            ++text;
        size_t size = (size_t)(text - start);
        if (!size || size >= DEMO_WORD_BYTES)
            return -1;
        unsigned id = 0;
        while (id < DEMO_VOCAB_SIZE && (strlen(vocab[id]) != size || strncmp(vocab[id], start, size)))
            ++id;
        if (id == DEMO_VOCAB_SIZE)
            return -1;
        prompt[word*8] = (uint8_t)id;
    }
    while (isspace((unsigned char)*text))
        ++text;
    return *text ? -1 : 0;
}

int main(int argc, char **argv)
{
    static const char usage[] = "z15_tinylm_demo CASE_DIR --prompt 'four known words' [--backend linux|verilator]\n";
    if (argc == 2 && !strcmp(argv[1], "--help")) {
        fputs(usage, stdout);
        return 0;
    }
    demo_options_t opt;
    if (demo_options(argc, argv, "--prompt", &opt)) {
        fputs(usage, stderr);
        return 2;
    }

    demo_t demo = {0};
    ysyx_kv_session_t *session = NULL;
    char path[DEMO_PATH_BYTES], vocab[DEMO_VOCAB_SIZE][DEMO_WORD_BYTES] = {{0}};
    uint8_t prompt[32] = {0};
    const char *error = "could not load the case vocabulary";
    int result = 1;
    if (demo_path(path, opt.directory, "vocab.tsv") || demo_dictionary(path, vocab))
        goto done;
    error = "prompt must contain exactly four words from this case vocabulary";
    if (tokenize(opt.input, vocab, prompt))
        goto done;
    if (demo_open(&demo, &opt, &error))
        goto done;
    const uint32_t count = ysyx_kv_step_count(demo.program);
    if (count < 9 || count > 17) {
        error = "expected one prefill and 8..16 incremental decode checkpoints";
        goto done;
    }
    const ysyx_input_t input = {"prompt_0", prompt, sizeof(prompt)};
    ysyx_stats_t total = {0};
    if (ysyx_kv_begin(demo.program, &demo.hal, &input, 1, &session, &error))
        goto done;
    printf("PROMPT %s\nCONTINUATION", opt.input);
    for (uint32_t step = 0; step < count; ++step) {
        ysyx_stats_t stats;
        uint8_t token[8];
        char name[36];
        if (ysyx_kv_step(session, step, &stats, NULL, &error))
            goto done;
        demo_accumulate(&total, &stats);
        snprintf(name, sizeof(name), "p0_token_%" PRIu32, step);
        error = "failed to read a checkpoint token";
        if (ysyx_kv_read_output(session, name, token, sizeof(token)))
            goto done;
        int64_t id = demo_integer(token, 4);
        error = "device token is outside the case vocabulary";
        if (id < 0 || id >= DEMO_VOCAB_SIZE || !vocab[id][0])
            goto done;
        printf(" %s", vocab[id]);
        fflush(stdout);
    }
    putchar('\n');
    demo_print_stats(&total, "token", count);
    result = 0;

done:
    ysyx_kv_close(session);
    demo_close(&demo);
    if (result)
        fprintf(stderr, "z15_tinylm_demo: %s\n", error);
    return result;
}
