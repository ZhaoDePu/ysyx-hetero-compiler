#define _POSIX_C_SOURCE 200809L
#include "c11_qwen_bpe.h"
#include "z15_exec_option.h"
#include "c12_qwen_ref.h"
#ifdef YSYX_C11B_NATIVE
#include "hal_reference_c6.h"
#endif

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
    const char *logits_path;
    unsigned steps, kv_bits;
    int turbo, trace_steps;
#ifdef YSYX_C11B_NATIVE
    const char *golden;
    valence_target target;
    unsigned token;
    int token_given;
#endif
} options;

typedef struct {
    qwen_reference *model;
    qwen_bpe *bpe;
    FILE *logits_dump;
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
    if (result) {
        ++chat->position;
        if (chat->logits_dump) {
            uint8_t block[4096];
            for (unsigned first = 0; first < QWEN_VOCAB; first += sizeof(block) / 4) {
                unsigned count = QWEN_VOCAB - first;
                if (count > sizeof(block) / 4)
                    count = sizeof(block) / 4;
                for (unsigned j = 0; j < count; ++j) {
                    const uint32_t value = (uint32_t)result[first + j];
                    for (unsigned i = 0; i < 4; ++i)
                        block[j * 4 + i] = (uint8_t)(value >> (8 * i));
                }
                if (fwrite(block, 4, count, chat->logits_dump) != count)
                    return NULL;
            }
        }
    }
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

static int turn(session *chat, const char *prompt, const options *o, FILE *csv)
{
    const unsigned limit = o->steps;
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
        const double step_start = o->trace_steps ? seconds() : 0;
#ifdef YSYX_C11B_NATIVE
        ysyx_stats_t token_before;
        const valence_submission_stats submit_before = c11b_z15_submission_stats(chat->device);
        if (o->trace_steps && chat->device && c11b_z15_stats(chat->device, &token_before))
            return -1;
#endif
        if (generated + 1 == limit) {
            if (advance(chat, id))
                return -1;
        } else if (!(logits = step(chat, id))) {
            return -1;
        }
        if (o->trace_steps) {
            double latency = seconds() - step_start;
            fprintf(stderr, "decode_step=%u token=%u ms=%.6f tok/s=%.3f",
                    generated + 1, id, latency * 1000, latency > 0 ? 1 / latency : 0);
#ifdef YSYX_C11B_NATIVE
            if (chat->device) {
                ysyx_stats_t token_after;
                if (c11b_z15_stats(chat->device, &token_after))
                    return -1;
                fprintf(stderr, " wall=%" PRIu64 " npu_vpu=%" PRIu64 " gpu=%" PRIu64
                        " dma_bytes=%" PRIu64 " overlap=%" PRIu64,
                        token_after.wall_ticks - token_before.wall_ticks,
                        token_after.npu_ticks - token_before.npu_ticks,
                        token_after.gpu_ticks - token_before.gpu_ticks,
                        token_after.dma_bytes - token_before.dma_bytes,
                        token_after.npu_gpu_overlap_ticks - token_before.npu_gpu_overlap_ticks);
                const valence_submission_stats submit_after = c11b_z15_submission_stats(chat->device);
                fprintf(stderr, " batches=%" PRIu64 " descriptor_words=%" PRIu64
                        " mmio_writes=%" PRIu64 " doorbells=%" PRIu64 " host_writes=%" PRIu64
                        " host_reads=%" PRIu64,
                        submit_after.batches - submit_before.batches,
                        submit_after.descriptor_words - submit_before.descriptor_words,
                        submit_after.mmio_writes - submit_before.mmio_writes,
                        submit_after.doorbells - submit_before.doorbells,
                        submit_after.host_writes - submit_before.host_writes,
                        submit_after.host_reads - submit_before.host_reads);
            }
#endif
            fputc('\n', stderr);
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
        backend = o->backend;
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
        fprintf(stderr, "%s wall=%" PRIu64 " npu_attn=%" PRIu64 " vpu=%" PRIu64
                " gpu=%" PRIu64 " dma_bytes=%" PRIu64 " ddr_high_water=%" PRIu64 " ERROR=%" PRIu32 "\n",
                backend, wall, npu, vpu, gpu, dma, peak, after.abi_error);
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
        else if (!strcmp(key, "--dump-logits")) o->logits_path = value;
        else if (!strcmp(key, "--prompt")) o->prompt = value;
        else if (!strcmp(key, "--encode")) o->plain = value;
        else if (!strcmp(key, "--encode-chat")) o->chat = value;
#ifdef YSYX_C11B_NATIVE
        else if (!strcmp(key, "--target")) {
            if (strcmp(value, "step2") && strcmp(value, "p17"))
                return -1;
            o->target = !strcmp(value, "p17") ? VALENCE_P17 : VALENCE_STEP2;
        }
        else if (!strcmp(key, "--mmio-profile")) {
            if (strcmp(value, "v1") && strcmp(value, "turbo"))
                return -1;
            o->turbo = !strcmp(value, "turbo");
        }
        else if (!strcmp(key, "--kv-bits")) {
            if ((value[0] != '4' && value[0] != '8') || value[1])
                return -1;
            o->kv_bits = (unsigned)(value[0] - '0');
        }
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
        else if (!strcmp(key, "--trace-steps")) {
            if (strcmp(value, "on") && strcmp(value, "off"))
                return -1;
            o->trace_steps = !strcmp(value, "on");
        }
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
           (strcmp(o->backend, "cpu") && strcmp(o->backend, "z15") &&
            strcmp(o->backend, "cmdq-reference")) ||
           (!strcmp(o->backend, "cmdq-reference") && (o->target != VALENCE_P17 || encode)) ||
           (encode && !strcmp(o->backend, "z15")) ||
           (o->turbo && strcmp(o->backend, "z15")) ||
           (o->target == VALENCE_P17 && (o->turbo || !strcmp(o->backend, "cpu"))) ||
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
            if (turn(chat, bench_prompts[i], o, csv))
                return -1;
        }
        return 0;
    }
    if (o->prompt)
        return turn(chat, o->prompt, o, csv);
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
        if (*line && turn(chat, line, o, csv) < 0)
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
    printf("%s_VERIFY_PASS token=%u logits_bytes=%zu backend=%s\n",
           c12_qwen_mixed(chat->model) ? "C12" : "C11B", o->token, QWEN_VOCAB * sizeof(*got), o->backend);
    return 0;
}
#endif

#ifdef YSYX_C11B_NATIVE
static int native_start(session *chat, const options *o)
{
    int mixed = c12_qwen_mixed(chat->model);
    if (o->kv_bits && c12_qwen_cache_bits(chat->model, o->kv_bits))
        return -1;
    if (!strcmp(o->backend, "cpu"))
        return mixed ? 0 : qwen_reference_native(chat->model, NULL);
    const char *why = NULL;
    if (!strcmp(o->backend, "cmdq-reference")) {
        const char *mode = getenv("YSYX_EXEC_MODE");
        if (!mixed || (mode && !strcmp(mode, "mmio")) ||
            ysyx_hal_reference_cmdq_open(&chat->hal, &why))
            return -1;
    } else {
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
        if (ysyx_hal_linux_open(&chat->hal, NULL, NULL, &why)) {
            fprintf(stderr, "%s\n", why ? why : "could not open Z15 UIO");
            return -1;
        }
    }
    size_t bytes;
    const uint8_t *data = qwen_reference_image(chat->model, &bytes);
    chat->device = c11b_z15_open_target(&chat->hal, data, bytes, o->target, &why);
    if (!chat->device) {
        fprintf(stderr, "%s\n", why ? why : "could not open the Z15 Qwen runtime");
        return -1;
    }
    if (o->turbo && c12_z15_enable_turbo(chat->device)) {
        fputs("The selected Turbo image did not report MMIO extension 0x0c0c0101.\n", stderr);
        return -1;
    }
    if (mixed) {
        fputs(o->target == VALENCE_P17
              ? "Valence P17: packed W4/INT8 TC projections, SC RoPE/residual, host Q16 nonlinearities.\n"
              : "Valence C12: NPU projections, GPU RoPE/residual and host Q16 nonlinearities.\n", stderr);
        return c12_z15_attach(chat->model, chat->device, (UINT32_C(1) << (QWEN_LAYERS + 1)) - 1);
    }
    return qwen_reference_native(chat->model, c11b_z15_ops(chat->device));
}
#endif

int main(int argc, char **argv)
{
    if (z15_exec_option(&argc, argv))
        return 2;
    static const char help[] =
#ifdef YSYX_C11B_NATIVE
         "z15_qwen25_chat --weights FILE --tokenizer FILE [--backend cpu|z15|cmdq-reference] "
        "[--kv-bits 4|8 (C12 default: 8)] [--target step2|p17] [--exec-mode auto|cmdq|mmio] [--exec-stats] [--mmio-profile v1|turbo] [--mode verify --token-id ID --golden FILE]\n"
        "Chat and bench options: "
#else
        "ysyx_qwen_chat --weights FILE --tokenizer FILE [--backend sim] "
#endif
        "[--mode chat|bench] [--prompt TEXT] [--max-new-tokens N] [--trace-steps on|off] [--csv FILE] [--dump-logits FILE]\n"
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
        chat.logits_dump = o.logits_path ? fopen(o.logits_path, "wb") : NULL;
        if (!chat.model || (o.csv && !csv) || (o.logits_path && !chat.logits_dump)) {
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
        if (chat.logits_dump && fclose(chat.logits_dump))
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
