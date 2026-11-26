#define _POSIX_C_SOURCE 200809L
#include "c11_qwen_bpe.h"
#include "c11_qwen_ref.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MAX_IDS = 2048, MAX_PROMPT = 4096, IM_END = 151645, END_OF_TEXT = 151643 };
static const char system_role[] =
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>";
static const char *const bench_prompts[] = {
    "请用两段通俗的中文解释什么是 FPGA、它和 CPU 的区别，并给出一个实际用途。",
    "Explain what a systolic array is in two complete sentences. Include how data moves and why it is efficient for matrix multiplication.",
    "用 Python 写一个计算第 n 个斐波那契数的迭代函数。请给出完整代码和一句中文解释。",
};

typedef struct {
    const char *weights, *tokenizer, *mode, *backend, *csv, *plain, *chat;
    unsigned steps;
} options;

typedef struct {
    qwen_reference *model;
    qwen_bpe *bpe;
    unsigned position;
    unsigned turn, records;
} session;

static double seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts))
        return 0.0;
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int chat_prompt(const qwen_bpe *bpe, const char *prompt, int first, uint32_t *ids)
{
    char text[MAX_PROMPT + 256];
    int length = snprintf(text, sizeof(text), "%s\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
                          first ? system_role : "", prompt);
    if (length < 0 || (size_t)length >= sizeof(text))
        return -1;
    return qwen_bpe_encode(bpe, text, ids, MAX_IDS);
}

static unsigned best(const int32_t *logits)
{
    unsigned id = 0;
    for (unsigned i = 1; i < QWEN_VOCAB; ++i)
        if (logits[i] > logits[id])
            id = i;
    return id;
}

static const int32_t *step(session *chat, unsigned id)
{
    const int32_t *result = qwen_reference_step(chat->model, id);
    if (result)
        ++chat->position;
    return result;
}

static void reset(session *chat)
{
    chat->position = 0;
    chat->turn = 0;
    qwen_reference_reset(chat->model);
}

static int turn(session *chat, const char *prompt, unsigned limit, FILE *csv)
{
    uint32_t ids[MAX_IDS];
    int count = chat_prompt(chat->bpe, prompt, !chat->turn, ids);
    if (count <= 0 || chat->position + (unsigned)count + limit + 1u >
        qwen_reference_context(chat->model)) {
        fputs("Invalid UTF-8 or prompt and decode budget exceed context; use /reset.\n", stderr);
        return 1;
    }
    const int32_t *logits = NULL;
    const double start = seconds();
    for (int i = 0; i < count; ++i)
        if (!(logits = step(chat, ids[i])))
            return -1;
    const double prefill = seconds() - start;

    const double begin = seconds();
    unsigned generated = 0;
    for (; generated < limit; ++generated) {
        const unsigned id = best(logits);
        if (id == IM_END || id == END_OF_TEXT)
            break;
        size_t bytes = 0;
        const char *piece = qwen_bpe_piece(chat->bpe, id, &bytes);
        if (!piece || fwrite(piece, 1, bytes, stdout) != bytes || fflush(stdout))
            return -1;
        logits = step(chat, id);
        if (!logits)
            return -1;
    }
    if (!step(chat, IM_END))
        return -1;
    const double elapsed = seconds() - begin;
    const double rate = elapsed > 0 ? generated / elapsed : 0;
    if (putchar('\n') == EOF)
        return -1;
    ++chat->turn;
    ++chat->records;
    fprintf(stderr, "prefill %.3f ms, decode %.3f tok/s (%u tokens, CPU C99)\n",
            prefill * 1000, rate, generated);
    if (csv && (fprintf(csv, "%u,%d,%u,%.6f,%.6f,%.6f,software_reference\n",
                        chat->records, count, generated, prefill * 1000, elapsed * 1000, rate) < 0 ||
                fflush(csv)))
        return -1;
    return 0;
}

static int parse(options *o, int argc, char **argv)
{
    *o = (options){.mode = "chat", .backend = "sim", .steps = 48};
    for (int i = 1; i < argc; ++i) {
        const char *key = argv[i];
        if (++i == argc)
            return -1;
        const char *value = argv[i];
        if (!strcmp(key, "--weights")) o->weights = value;
        else if (!strcmp(key, "--tokenizer")) o->tokenizer = value;
        else if (!strcmp(key, "--mode")) o->mode = value;
        else if (!strcmp(key, "--backend")) o->backend = value;
        else if (!strcmp(key, "--csv")) o->csv = value;
        else if (!strcmp(key, "--encode")) o->plain = value;
        else if (!strcmp(key, "--encode-chat")) o->chat = value;
        else if (!strcmp(key, "--max-new-tokens")) {
            char *end;
            errno = 0;
            unsigned long n = strtoul(value, &end, 10);
            if (errno || *end || !n || n > 512)
                return -1;
            o->steps = (unsigned)n;
        } else {
            return -1;
        }
    }
    const int encode = !!o->plain + !!o->chat;
    return !o->tokenizer || encode > 1 || (!encode && !o->weights) ||
           (strcmp(o->mode, "chat") && strcmp(o->mode, "bench")) ||
           (strcmp(o->backend, "sim") && strcmp(o->backend, "hw"));
}

static int run_console(session *chat, const options *o, FILE *csv)
{
    if (!strcmp(o->mode, "bench")) {
        for (unsigned i = 0; i < sizeof(bench_prompts) / sizeof(bench_prompts[0]); ++i) {
            reset(chat);
            printf("[%u] %s\n", i + 1, bench_prompts[i]);
            if (turn(chat, bench_prompts[i], o->steps, csv))
                return -1;
        }
        return 0;
    }
    char line[MAX_PROMPT];
    for (;;) {
        if (fputs("user> ", stdout) == EOF || fflush(stdout))
            return -1;
        if (!fgets(line, sizeof(line), stdin))
            return ferror(stdin) ? -1 : 0;
        size_t bytes = strlen(line);
        if (bytes == sizeof(line) - 1 && line[bytes - 1] != '\n') {
            int c;
            do {
                c = getchar();
            } while (c != '\n' && c != EOF);
            fputs("Input line is too long.\n", stderr);
            continue;
        }
        line[strcspn(line, "\r\n")] = '\0';
        if (!strcmp(line, "/quit"))
            return 0;
        if (!strcmp(line, "/reset")) {
            reset(chat);
            continue;
        }
        if (*line && turn(chat, line, o->steps, csv) < 0)
            return -1;
    }
}

int main(int argc, char **argv)
{
    static const char help[] =
        "ysyx_qwen_chat --weights FILE --tokenizer FILE [--backend sim] "
        "[--mode chat|bench] [--max-new-tokens N] [--csv FILE]\n"
        "ysyx_qwen_chat --tokenizer FILE --encode TEXT|--encode-chat PROMPT\n";
    options o;
    if (argc == 2 && !strcmp(argv[1], "--help")) {
        fputs(help, stdout);
        return 0;
    }
    if (parse(&o, argc, argv)) {
        fputs(help, stderr);
        return 2;
    }
    if (!strcmp(o.backend, "hw")) {
        fputs("Full-model Z15 hardware execution has not been signed off. Use --backend sim.\n", stderr);
        return 2;
    }
    session chat = {0};
    chat.bpe = qwen_bpe_open(o.tokenizer);
    if (!chat.bpe) {
        fputs("Invalid Qwen tokenizer asset.\n", stderr);
        return 1;
    }
    int rc = 0;
    if (o.chat || o.plain) {
        uint32_t ids[MAX_IDS];
        int count = o.chat ? chat_prompt(chat.bpe, o.chat, 1, ids) :
                            qwen_bpe_encode(chat.bpe, o.plain, ids, MAX_IDS);
        if (count < 0)
            rc = 1;
        for (int i = 0; i < count; ++i)
            printf("%u\n", ids[i]);
    } else {
        chat.model = qwen_reference_open(o.weights);
        FILE *csv = o.csv ? fopen(o.csv, "w") : NULL;
        if (!chat.model || (o.csv && !csv)) {
            fputs("Cannot open Qwen weights or CSV output.\n", stderr);
            rc = 1;
        } else {
            if (csv)
                fputs("turn,prefill_tokens,decode_tokens,prefill_ms,decode_ms,tokens_per_second,backend\n", csv);
            rc = run_console(&chat, &o, csv) != 0;
        }
        if (csv && fclose(csv))
            rc = 1;
    }
    qwen_reference_close(chat.model);
    qwen_bpe_close(chat.bpe);
    return rc;
}
