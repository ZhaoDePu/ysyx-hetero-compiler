#define _POSIX_C_SOURCE 200809L
#include "c11_qwen_bpe.h"
#include "c11_qwen_ref.h"

#include <errno.h>
#include <inttypes.h>
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
    const char *weights, *tokenizer, *mode, *backend, *csv, *plain, *chat, *prompt;
    unsigned steps;
#ifdef YSYX_C11B_NATIVE
    const char *golden;
    unsigned token;
    int token_given;
#endif
} options;

typedef struct {
    qwen_reference *model;
    qwen_bpe *bpe;
    unsigned position;
    unsigned turn, records;
#ifdef YSYX_C11B_NATIVE
    ysyx_hal_t hal;
    c11b_z15 *device;
#endif
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

static int advance(session *chat, unsigned id)
{
#ifdef YSYX_C11B_NATIVE
    if (qwen_reference_advance(chat->model, id))
        return -1;
    ++chat->position;
    return 0;
#else
    return step(chat, id) ? 0 : -1;
#endif
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
#ifdef YSYX_C11B_NATIVE
    ysyx_stats_t before;
    uint64_t vpu_before = 0;
    if (chat->device) {
        if (c11b_z15_stats(chat->device, &before))
            return -1;
        vpu_before = c11b_z15_vpu_ticks(chat->device);
    }
#endif
    const double start = seconds();
    for (int i = 0; i + 1 < count; ++i)
        if (advance(chat, ids[i]))
            return -1;
    logits = step(chat, ids[count - 1]);
    if (!logits)
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
        if (generated + 1 == limit) {
            if (advance(chat, id))
                return -1;
        } else if (!(logits = step(chat, id))) {
            return -1;
        }
    }
    const double elapsed = seconds() - begin;
    if (advance(chat, IM_END))
        return -1;
    const double rate = elapsed > 0 ? generated / elapsed : 0;
    if (putchar('\n') == EOF)
        return -1;
    ++chat->turn;
    ++chat->records;
    const char *backend = "software_reference";
#ifdef YSYX_C11B_NATIVE
    if (chat->device)
        backend = "z15";
#endif
    fprintf(stderr, "prefill %.3f ms, decode %.3f tok/s (%u tokens, %s)\n",
            prefill * 1000, rate, generated, backend);
    if (csv && fprintf(csv, "%u,%d,%u,%.6f,%.6f,%.6f,%s",
                       chat->records, count, generated, prefill * 1000, elapsed * 1000, rate, backend) < 0)
        return -1;
#ifdef YSYX_C11B_NATIVE
    if (chat->device) {
        ysyx_stats_t after;
        if (c11b_z15_stats(chat->device, &after))
            return -1;
        const uint64_t vpu = c11b_z15_vpu_ticks(chat->device) - vpu_before;
        const uint64_t npu = after.npu_ticks - before.npu_ticks - vpu;
        const uint64_t gpu = after.gpu_ticks - before.gpu_ticks;
        const uint64_t wall = after.wall_ticks - before.wall_ticks;
        const uint64_t dma = after.dma_bytes - before.dma_bytes;
        const uint64_t peak = c11b_z15_peak_bytes(chat->device);
        fprintf(stderr, "z15 wall=%" PRIu64 " npu_attn=%" PRIu64 " vpu=%" PRIu64
                " gpu=%" PRIu64 " dma_bytes=%" PRIu64 " ddr_high_water=%" PRIu64 " ERROR=%" PRIu32 "\n",
                wall, npu, vpu, gpu, dma, peak, after.abi_error);
        if (csv && fprintf(csv, ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
                           ",%" PRIu64 ",%" PRIu64 ",%" PRIu32,
                           wall, npu, vpu, gpu, dma, peak, after.abi_error) < 0)
            return -1;
    } else if (csv && fputs(",,,,,,,", csv) == EOF) {
        return -1;
    }
#endif
    return csv && (fputc('\n', csv) == EOF || fflush(csv)) ? -1 : 0;
}

static int parse(options *o, int argc, char **argv)
{
#ifdef YSYX_C11B_NATIVE
    *o = (options){.mode = "chat", .backend = "cpu", .steps = 48};
#else
    *o = (options){.mode = "chat", .backend = "sim", .steps = 48};
#endif
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
        else if (!strcmp(key, "--prompt")) o->prompt = value;
        else if (!strcmp(key, "--encode")) o->plain = value;
        else if (!strcmp(key, "--encode-chat")) o->chat = value;
#ifdef YSYX_C11B_NATIVE
        else if (!strcmp(key, "--golden")) o->golden = value;
        else if (!strcmp(key, "--token-id")) {
            char *end;
            errno = 0;
            unsigned long n = strtoul(value, &end, 10);
            if (!*value || errno || *end || n >= QWEN_VOCAB)
                return -1;
            o->token = (unsigned)n;
            o->token_given = 1;
        }
#endif
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
           (o->prompt && (encode || strcmp(o->mode, "chat"))) ||
#ifdef YSYX_C11B_NATIVE
           (strcmp(o->backend, "cpu") && strcmp(o->backend, "z15")) ||
           (encode && !strcmp(o->backend, "z15")) ||
           (!strcmp(o->mode, "verify") ? !o->golden || !o->token_given || encode || o->csv
             : (strcmp(o->mode, "chat") && strcmp(o->mode, "bench")) || o->golden || o->token_given);
#else
           (strcmp(o->mode, "chat") && strcmp(o->mode, "bench")) ||
           (strcmp(o->backend, "sim") && strcmp(o->backend, "hw"));
#endif
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
    if (o->prompt)
        return turn(chat, o->prompt, o->steps, csv);
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

#ifdef YSYX_C11B_NATIVE
static int verify_token(session *chat, const options *o)
{
    FILE *f = fopen(o->golden, "rb");
    if (!f)
        return 1;
    const int32_t *got = step(chat, o->token);
    int32_t diff = -1;
    if (got) {
        for (unsigned i = 0; i < QWEN_VOCAB; ++i) {
            uint8_t lane[4];
            if (fread(lane, 1, 4, f) != 4) {
                diff = (int32_t)i;
                break;
            }
            uint32_t gold = (uint32_t)lane[0] | (uint32_t)lane[1] << 8 |
                            (uint32_t)lane[2] << 16 | (uint32_t)lane[3] << 24;
            if ((uint32_t)got[i] != gold) {
                diff = (int32_t)i;
                break;
            }
        }
    }
    const int valid = got && diff < 0 && fgetc(f) == EOF && !ferror(f);
    fclose(f);
    if (!valid) {
        fprintf(stderr, "Golden verification failed (first mismatching logit %" PRId32 ").\n", diff);
        return 1;
    }
    if (chat->device) {
        ysyx_stats_t s;
        if (c11b_z15_stats(chat->device, &s))
            return 1;
        printf("z15 wall=%" PRIu64 " ddr_peak=%" PRIu64 " ERROR=%" PRIu32
               " invalid_dma=%" PRIu32 "\n", s.wall_ticks,
               c11b_z15_peak_bytes(chat->device), s.abi_error, s.invalid_dma_descriptors);
    }
    printf("C11B_VERIFY_PASS token=%u logits_bytes=%zu backend=%s\n",
           o->token, QWEN_VOCAB * sizeof(*got), o->backend);
    return 0;
}
#endif

#ifdef YSYX_C11B_NATIVE
static int native_start(session *chat, const options *o)
{
    if (!strcmp(o->backend, "cpu"))
        return qwen_reference_native(chat->model, NULL);
    FILE *state = fopen("/sys/class/fpga_manager/fpga0/state", "r");
    char line[32];
    const int operating = state && fgets(line, sizeof(line), state) &&
                          (!strcmp(line, "operating\n") || !strcmp(line, "operating"));
    if (state)
        fclose(state);
    if (!operating) {
        fputs("Z15 FPGA Manager must be operating before UIO open.\n", stderr);
        return -1;
    }
    const char *why = NULL;
    if (ysyx_hal_linux_open(&chat->hal, NULL, NULL, &why)) {
        fprintf(stderr, "%s\n", why ? why : "could not open Z15 UIO");
        return -1;
    }
    size_t bytes;
    const uint8_t *data = qwen_reference_image(chat->model, &bytes);
    chat->device = c11b_z15_open(&chat->hal, data, bytes, &why);
    if (!chat->device) {
        fprintf(stderr, "%s\n", why ? why : "could not open the Z15 Qwen runtime");
        return -1;
    }
    return qwen_reference_native(chat->model, c11b_z15_ops(chat->device));
}
#endif

int main(int argc, char **argv)
{
    static const char help[] =
#ifdef YSYX_C11B_NATIVE
         "z15_qwen25_chat --weights FILE --tokenizer FILE [--backend cpu|z15] "
        "[--mode verify --token-id ID --golden FILE]\n"
        "Chat and bench options: "
#else
        "ysyx_qwen_chat --weights FILE --tokenizer FILE [--backend sim] "
#endif
        "[--mode chat|bench] [--prompt TEXT] [--max-new-tokens N] [--csv FILE]\n"
        "--tokenizer FILE --encode TEXT|--encode-chat PROMPT\n";
    options o;
    if (argc == 2 && !strcmp(argv[1], "--help")) {
        fputs(help, stdout);
        return 0;
    }
    if (parse(&o, argc, argv)) {
        fputs(help, stderr);
        return 2;
    }
#ifndef YSYX_C11B_NATIVE
    if (!strcmp(o.backend, "hw")) {
        fputs("Full-model Z15 hardware execution has not been signed off. Use --backend sim.\n", stderr);
        return 2;
    }
#endif
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
#ifdef YSYX_C11B_NATIVE
        } else if (native_start(&chat, &o)) {
            rc = 1;
        } else if (!strcmp(o.mode, "verify")) {
            rc = verify_token(&chat, &o);
#endif
        } else {
            if (csv) {
                fputs("turn,prefill_tokens,decode_tokens,prefill_ms,decode_ms,tokens_per_second,backend", csv);
#ifdef YSYX_C11B_NATIVE
                fputs(",wall_ticks,npu_attn_ticks,vpu_ticks,gpu_ticks,dma_bytes,ddr_high_water,abi_error", csv);
#endif
                fputc('\n', csv);
            }
            rc = run_console(&chat, &o, csv) != 0;
        }
        if (csv && fclose(csv))
            rc = 1;
    }
#ifdef YSYX_C11B_NATIVE
    c11b_z15_close(chat.device);
    if (chat.hal.close)
        chat.hal.close(chat.hal.ctx);
#endif
    qwen_reference_close(chat.model);
    qwen_bpe_close(chat.bpe);
    return rc;
}
