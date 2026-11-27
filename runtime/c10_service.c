#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "z15_exec_option.h"
#include "c10_pixels.h"
#include "c10_session.h"
#include "c10_v4l2.h"
#include "yolo26n_target.h"

#include <arpa/inet.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

enum { Z15F_HEADER_BYTES = 152, Z15I_HEADER_BYTES = 16, Z15F_C10_OFFSET = 104 };
enum { Z15F_RGB = 1, Z15F_JPEG = 2, Z15F_FAULT = 2, Z15F_COUNTERS_UNAVAILABLE = 4 };
enum { MODE_VERIFY = 1, MODE_V4L2 = 2, MODE_SIM = 4, MODE_RECORDING = 8 };

typedef struct {
    const char *model, *device, *recording, *csv, *bind, *golden;
    unsigned width, height, fps, port, in_port, native_mhz, gpu_mhz;
    uint64_t frames;
    int mode, mode_explicit, phase17;
} settings;

typedef struct {
    unsigned width, height, codec;
    const uint8_t *payload;
    size_t bytes;
} wire_image;

typedef struct {
    uint64_t sequence;
    ysyx_c10_timing_t time;
} previous_frame;

typedef struct {
    uint8_t *rgb, *input, *rgb320, *heads, *golden;
    FILE *csv, *recording;
    c10_camera *camera;
    ysyx_yolo26n_ctx_t *model;
    int output_listener, input_listener;
} service;

static volatile sig_atomic_t stopping;

static void stop(int sig)
{
    (void)sig;
    stopping = 1;
}

static uint64_t now_ns(void)
{
    struct timespec t = {0};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000 + (uint64_t)t.tv_nsec;
}

static double elapsed_ms(uint64_t start, uint64_t end)
{
    return (double)(end - start) / 1000000.0;
}

static uint32_t get32(const uint8_t *ptr)
{
    return (uint32_t)ptr[0] | (uint32_t)ptr[1] << 8 |
           (uint32_t)ptr[2] << 16 | (uint32_t)ptr[3] << 24;
}

static void put16(uint8_t *ptr, unsigned value)
{
    ptr[0] = (uint8_t)value;
    ptr[1] = (uint8_t)(value >> 8);
}

static void put32(uint8_t *ptr, uint32_t value)
{
    put16(ptr, value);
    put16(ptr + 2, value >> 16);
}

static void put64(uint8_t *ptr, uint64_t value)
{
    put32(ptr, (uint32_t)value);
    put32(ptr + 4, (uint32_t)(value >> 32));
}

static void putfloat(uint8_t *ptr, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits)); /* IEEE-754 binary32 on x86-64 and AArch64. */
    put32(ptr, bits);
}

/* Nonblocking listener wait also gives SIGTERM a bounded shutdown time. */
static int accept_client(int listener, int waiting_viewer)
{
    while (!stopping) {
        struct pollfd pfd[] = { { .fd = listener, .events = POLLIN },
                                { .fd = waiting_viewer, .events = POLLIN } };
        int ready = poll(pfd, 2, 1000);
        if (ready == 0 || (ready < 0 && errno == EINTR))
            continue;
        if (ready < 0)
            return -1;
        if (pfd[1].revents)
            return -2; /* The viewer disconnected before the feeder arrived. */
        int fd = accept(listener, NULL, NULL);
        if (fd < 0 && errno == EINTR)
            continue;
        if (fd < 0)
            return -1;
        struct timeval timeout = { .tv_sec = 5 };
        int one = 1;
        if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) ||
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) ||
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one))) {
            close(fd);
            return -1;
        }
        return fd;
    }
    errno = EINTR;
    return -1;
}

static int listen_at(const char *ip, unsigned port)
{
    struct sockaddr_in addr = { .sin_family = AF_INET, .sin_port = htons((uint16_t)port) };
    if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1) {
        errno = EINVAL;
        return -1;
    }
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        return -1;
    int one = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) ||
        bind(fd, (const struct sockaddr *)&addr, sizeof(addr)) || listen(fd, 1)) {
        int saved = errno;
        close(fd);
        errno = saved;
        return -1;
    }
    return fd;
}

static int recv_exact(int fd, uint8_t *buffer, size_t size)
{
    while (!stopping && size) {
        ssize_t n = recv(fd, buffer, size, 0);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            return -1;
        buffer += n;
        size -= (size_t)n;
    }
    return size ? -1 : 0;
}

static int send_exact(int fd, const uint8_t *buffer, size_t size)
{
    while (!stopping && size) {
        ssize_t n = send(fd, buffer, size, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            return -1;
        buffer += n;
        size -= (size_t)n;
    }
    return size ? -1 : 0;
}

static int receive_frame(service *s, int fd, wire_image *image)
{
    uint8_t header[Z15I_HEADER_BYTES];
    if (recv_exact(fd, header, sizeof(header)))
        return 0;
    unsigned width = get32(header + 4), height = get32(header + 8);
    size_t bytes = get32(header + 12), rgb_bytes;
    int jpeg = memcmp(header, "Z15J", 4) == 0;
    if (c10_frame_size(width, height, &rgb_bytes) || !bytes || bytes > C10_MAX_RGB ||
        (!jpeg && (memcmp(header, "Z15I", 4) || bytes != rgb_bytes)) ||
        recv_exact(fd, jpeg ? s->input : s->rgb, bytes))
        return -1;
    if (jpeg && c10_pixels_rgb(C10_JPEG, width, height, 0, s->input, bytes, s->rgb, C10_MAX_RGB))
        return -1;
    *image = (wire_image){ width, height, jpeg ? Z15F_JPEG : Z15F_RGB,
                           jpeg ? s->input : s->rgb, bytes };
    return 1;
}

static int encode_detections(uint8_t *out, const ysyx_det_t *boxes, int count,
                             const c10_letterbox_geometry *g, int camera)
{
    unsigned left = camera ? 0 : g->left, top = camera ? 0 : g->top;
    unsigned w = camera ? YSYX_YOLO26N_SIZE : g->width;
    unsigned h = camera ? YSYX_YOLO26N_SIZE : g->height;
    double scale = camera ? 1 : g->scale;
    for (int i = 0; i < count; ++i, out += 24) {
        const ysyx_det_t *b = &boxes[i];
        if (!isfinite(b->x1) || !isfinite(b->y1) || !isfinite(b->x2) || !isfinite(b->y2) ||
            !isfinite(b->score) || b->x1 > b->x2 || b->y1 > b->y2 ||
            b->score < 0 || b->score > 1 || b->class_id < 0 || b->class_id >= 80)
            return -1;
        putfloat(out, c10_letterbox_unscale(b->x1, left, w, scale));
        putfloat(out + 4, c10_letterbox_unscale(b->y1, top, h, scale));
        putfloat(out + 8, c10_letterbox_unscale(b->x2, left, w, scale));
        putfloat(out + 12, c10_letterbox_unscale(b->y2, top, h, scale));
        putfloat(out + 16, b->score);
        putfloat(out + 20, (float)b->class_id);
    }
    return 0;
}

static uint32_t to_us(double ms)
{
    return !isfinite(ms) || ms < 0 || ms >= (UINT32_MAX - 1) / 1000.0
        ? UINT32_MAX : (uint32_t)(ms * 1000.0 + 0.5);
}

static void encode_timing(uint8_t *ptr, const settings *opts, const previous_frame *prev)
{
    /*
     * The original name is NUL-terminated at offset 98. Step 4.5 ignores the
     * remaining name bytes. C10T v1 follows at byte 104, still within Z15F v1.
     * The measured send interval is available after transmission. All six
     * timers therefore carry the previous frame sequence. Zero means no sample.
     */
    memcpy(ptr, "C10T", 4);
    put64(ptr + 4, prev->sequence);
    const ysyx_c10_timing_t *t = &prev->time;
    const double times[] = { t->t_capture_preproc_ms, t->t_h2d_ms,
        t->hardware_counters_valid ? t->t_pl_hw_ms : -1,
        t->t_d2h_post_ms, t->t_net_tx_ms, t->t_e2e_ms };
    for (size_t i = 0; i < sizeof(times) / sizeof(times[0]); ++i)
        put32(ptr + 12 + 4 * i, prev->sequence ? to_us(times[i]) : UINT32_MAX);
    put16(ptr + 36, opts->native_mhz);
    put16(ptr + 38, opts->gpu_mhz);
    ptr[40] = (uint8_t)opts->mode;
    ptr[41] = 1;
}

static int send_frame(int fd, const settings *opts, uint64_t seq, const wire_image *image,
                      unsigned count, const uint8_t *boxes, int fault, uint64_t infer_ns,
                      const ysyx_perf_t *perf, const previous_frame *prev)
{
    uint8_t header[Z15F_HEADER_BYTES] = {0};
    memcpy(header, "Z15F", 4);
    put16(header + 4, 1);
    put16(header + 6, sizeof(header));
    put64(header + 8, seq);
    put32(header + 16, image->width);
    put32(header + 20, image->height);
    put32(header + 24, image->codec);
    put32(header + 28, (uint32_t)image->bytes);
    put32(header + 32, count);
    put32(header + 36, fault != 0);
    put32(header + 40, perf->abi_error);
    put32(header + 44, (fault ? Z15F_FAULT : 0) |
                      (perf->hardware_counters_valid ? 0 : Z15F_COUNTERS_UNAVAILABLE));
    if (perf->hardware_counters_valid) {
        put64(header + 48, perf->npu_ticks);
        put64(header + 56, perf->gpu_ticks);
        put64(header + 64, perf->wall_ticks);
        put64(header + 72, perf->overlap_ticks);
    }
    put64(header + 80, infer_ns);
    memcpy(header + 88, "YOLO26n-C8", sizeof("YOLO26n-C8"));
    encode_timing(header + Z15F_C10_OFFSET, opts, prev);
    return send_exact(fd, header, sizeof(header)) ||
           send_exact(fd, image->payload, image->bytes) ||
           send_exact(fd, boxes, count * 24u);
}

static int verify_heads(const service *s, const settings *opts, uint64_t sequence)
{
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%" PRIu64 ".heads.bin", opts->golden, sequence);
    if (n < 0 || (size_t)n >= sizeof(path))
        return -1;
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return -1;
    }
    int ok = fread(s->golden, 1, YSYX_C10_YOLO_HEAD_BYTES, f) == YSYX_C10_YOLO_HEAD_BYTES &&
             fgetc(f) == EOF && !ferror(f);
    fclose(f);
    if (!ok || ysyx_c10_yolo_read_heads(s->model, s->heads, YSYX_C10_YOLO_HEAD_BYTES))
        return -1;
    for (size_t i = 0; i < YSYX_C10_YOLO_HEAD_BYTES; ++i) {
        if (s->heads[i] != s->golden[i]) {
            fprintf(stderr, "%s mismatch at byte %zu: got %u expected %u\n",
                    path, i, s->heads[i], s->golden[i]);
            return -1;
        }
    }
    return 0;
}

static void field_double(char *text, size_t size, int available, double value)
{
    if (available)
        snprintf(text, size, "%.6f", value);
    else
        text[0] = '\0';
}

static void field_ticks(char *text, size_t size, int available, uint64_t value)
{
    if (available)
        snprintf(text, size, "%" PRIu64, value);
    else
        text[0] = '\0';
}

static int csv_frame(FILE *f, const settings *o, uint64_t seq, const ysyx_c10_timing_t *t,
                     const char *verify, int sent, int fault)
{
    char hw[48], wall[32], native[32], gpu[32], overlap[32], dma[32], dma_overlap[32];
    int hw_valid = t->hardware_counters_valid;
    const ysyx_stats_t *s = &t->stats;
    field_double(hw, sizeof(hw), hw_valid, t->t_pl_hw_ms);
    field_ticks(wall, sizeof(wall), hw_valid, s->wall_ticks);
    field_ticks(native, sizeof(native), hw_valid, s->npu_ticks);
    field_ticks(gpu, sizeof(gpu), hw_valid, s->gpu_ticks);
    field_ticks(overlap, sizeof(overlap), hw_valid, s->npu_gpu_overlap_ticks);
    field_ticks(dma, sizeof(dma), hw_valid && s->has_observations, s->observed.dma_busy_ticks);
    field_ticks(dma_overlap, sizeof(dma_overlap), hw_valid && s->has_observations,
                s->observed.dma_compute_overlap_ticks);
    int status = fprintf(f, "%" PRIu64 ",%s,%s,%s,%u,%u,%.6f,%.6f,%s,%.6f,%.6f,%.6f,%.6f,"
        "%s,,,%s,%s,%s,%s,%s,%" PRIu64 ",%u,%" PRIu64 ",%" PRIu64 ",%u,%u,%s,%d,%d\n",
        seq, o->recording ? "recorded_yuyv" : (o->mode & MODE_V4L2) ? "v4l2" : "tcp",
        (o->mode & MODE_VERIFY) ? "verify" : "live",
        (o->mode & MODE_SIM) ? (hw_valid ? "verilator" : "fixed_point") : "uio",
        o->native_mhz, o->gpu_mhz, t->t_capture_preproc_ms, t->t_h2d_ms, hw,
        t->t_d2h_post_ms, t->t_net_tx_ms, t->t_e2e_ms, t->t_execute_ms, wall, native,
        gpu, dma, overlap, dma_overlap, s->dma_bytes, s->dma_operations,
        t->static_h2d_bytes, t->dynamic_h2d_bytes, s->abi_error, s->invalid_dma_descriptors,
        verify, sent, fault);
    return status < 0 || fflush(f);
}

static int recorded_frame(service *s, const settings *o)
{
    size_t bytes = (size_t)o->width * o->height * 2;
    size_t n = fread(s->input, 1, bytes, s->recording);
    if (n != bytes)
        return !n && feof(s->recording) && !ferror(s->recording) ? 0 : -1;
    return c10_pixels_rgb(C10_YUYV, o->width, o->height, 0, s->input, bytes,
                          s->rgb, C10_MAX_RGB) ? -1 : 1;
}

/* Returns 0 on an ordinary disconnect, 1 on finite completion, -1 on failure. */
static int stream(service *s, const settings *o, int viewer, int input, previous_frame *prev)
{
    int camera = (o->mode & MODE_V4L2) != 0;
    while (!stopping) {
        wire_image image;
        unsigned width = o->width, height = o->height;
        uint64_t start = now_ns();
        if (camera) {
            int n = s->recording ? recorded_frame(s, o)
                                : c10_camera_rgb(s->camera, s->rgb, C10_MAX_RGB, &width, &height);
            if (n < 0) {
                if (s->recording)
                    fprintf(stderr, "Recorded YUYV frame is truncated or unreadable\n");
                else
                    perror("V4L2 capture");
                return -1;
            }
            if (!n) {
                if (s->recording)
                    return 1;
                continue;
            }
            image = (wire_image){ YSYX_YOLO26N_SIZE, YSYX_YOLO26N_SIZE, Z15F_RGB,
                                  s->rgb320, YSYX_YOLO26N_RGB_BYTES };
        } else {
            int n = receive_frame(s, input, &image);
            if (n <= 0) {
                if (n < 0)
                    fprintf(stderr, "Rejected malformed or incomplete Z15I/Z15J frame\n");
                return 0;
            }
            width = image.width;
            height = image.height;
        }

        c10_rgb_frame raw = { width, height, s->rgb };
        c10_letterbox_geometry g;
        c10_letterbox_rgb320(&raw, s->rgb320, &g);
        uint64_t infer_start = now_ns();
        ysyx_c10_timing_t t = {0};
        ysyx_perf_t perf = {0};
        ysyx_det_t dets[YSYX_YOLO26N_MAX_DETECTIONS];
        int count = ysyx_c10_yolo_infer_rgb320(s->model, s->rgb320, 0.40f, dets,
                                              YSYX_YOLO26N_MAX_DETECTIONS, &perf, &t,
                                              o->native_mhz);
        uint64_t infer_end = now_ns();
        t.t_capture_preproc_ms += elapsed_ms(start, infer_start);
        uint64_t seq = prev->sequence + 1;
        uint8_t boxes[YSYX_YOLO26N_MAX_DETECTIONS * 24];
        int fault = count < 0 || perf.abi_error || perf.invalid_dma_descriptors;
        uint64_t post_start = now_ns();
        if (!fault && encode_detections(boxes, dets, count, &g, camera))
            fault = 1;
        const char *verify = "skipped";
        if (o->mode & MODE_VERIFY) {
            int ok = !fault && !verify_heads(s, o, seq);
            verify = ok ? "pass" : "fail";
            fault |= !ok;
        }
        uint64_t send_start = now_ns();
        t.t_d2h_post_ms += elapsed_ms(post_start, send_start);
        int sent = !send_frame(viewer, o, seq, &image, fault ? 0u : (unsigned)count,
                               boxes, fault, infer_end - infer_start, &perf, prev);
        uint64_t end = now_ns();
        t.t_net_tx_ms = elapsed_ms(send_start, end);
        t.t_e2e_ms = elapsed_ms(start, end);
        if (csv_frame(s->csv, o, seq, &t, verify, sent, fault)) {
            perror("CSV");
            return -1;
        }
        prev->sequence = seq;
        prev->time = t;
        if (fault) {
            fprintf(stderr, "Frame %" PRIu64 " failed: ABI ERROR=%u, invalid DMA=%u, verify=%s\n",
                    seq, perf.abi_error, perf.invalid_dma_descriptors, verify);
            return -1;
        }
        if (o->frames && seq >= o->frames)
            return sent ? 1 : -1;
        if (!sent)
            return 0;
    }
    return 0;
}

static int parse_number(const char *text, uint64_t limit, uint64_t *n)
{
    if (*text < '0' || *text > '9')
        return -1;
    char *end;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || *end || !value || value > limit)
        return -1;
    *n = (uint64_t)value;
    return 0;
}

static int parse_small(const char *text, unsigned limit, unsigned *value)
{
    uint64_t n;
    if (parse_number(text, limit, &n))
        return -1;
    *value = (unsigned)n;
    return 0;
}

static int parse_args(settings *o, int argc, char **argv)
{
    *o = (settings){ .device = "/dev/video0", .csv = "c10_live.csv", .bind = "0.0.0.0",
        .width = 640, .height = 480, .fps = 30, .port = 9000, .in_port = 9001,
        .native_mhz = 125, .gpu_mhz = 100 };
    for (int i = 1; i < argc; ++i) {
        const char *key = argv[i];
        if (!strcmp(key, "--sim")) {
            o->mode |= MODE_SIM;
            continue;
        }
        if (!strcmp(key, "--verify") || !strcmp(key, "--live")) {
            if (o->mode_explicit++)
                return -1;
            if (!strcmp(key, "--verify"))
                o->mode |= MODE_VERIFY;
            continue;
        }
        if (++i >= argc)
            return -1;
        const char *value = argv[i];
        if (!strcmp(key, "--source")) {
            if (!strcmp(value, "v4l2"))
                o->mode |= MODE_V4L2;
            else if (!strcmp(value, "tcp"))
                o->mode &= ~MODE_V4L2;
            else
                return -1;
        } else if (!strcmp(key, "--freq-profile")) {
            if (!strcmp(value, "125_100")) {
                o->native_mhz = 125;
                o->gpu_mhz = 100;
            } else if (!strcmp(value, "200_150")) {
                o->native_mhz = 200;
                o->gpu_mhz = 150;
            } else
                return -1;
        } else if (!strcmp(key, "--target")) {
            if (strcmp(value, "step2") && strcmp(value, "p17"))
                return -1;
            o->phase17 = !strcmp(value, "p17");
        } else if (!strcmp(key, "--model")) o->model = value;
        else if (!strcmp(key, "--device")) o->device = value;
        else if (!strcmp(key, "--recorded-yuyv")) {
            o->recording = value;
            o->mode |= MODE_RECORDING;
        } else if (!strcmp(key, "--csv")) o->csv = value;
        else if (!strcmp(key, "--bind")) o->bind = value;
        else if (!strcmp(key, "--golden")) o->golden = value;
        else if (!strcmp(key, "--width")) {
            if (parse_small(value, C10_MAX_WIDTH, &o->width)) return -1;
        } else if (!strcmp(key, "--height")) {
            if (parse_small(value, C10_MAX_HEIGHT, &o->height)) return -1;
        } else if (!strcmp(key, "--fps")) {
            if (parse_small(value, 240, &o->fps)) return -1;
        } else if (!strcmp(key, "--port")) {
            if (parse_small(value, 65535, &o->port)) return -1;
        } else if (!strcmp(key, "--in-port")) {
            if (parse_small(value, 65535, &o->in_port)) return -1;
        } else if (!strcmp(key, "--frames")) {
            if (parse_number(value, UINT64_MAX - 1, &o->frames)) return -1;
        } else return -1;
    }
    return !o->model || (!!o->golden != !!(o->mode & MODE_VERIFY)) ||
           (o->recording && (!(o->mode & MODE_V4L2) || (o->width & 1))) ||
           (!(o->mode & MODE_V4L2) && o->port == o->in_port) ? -1 : 0;
}

static int init_service(service *s, const settings *o)
{
    s->rgb = malloc(C10_MAX_RGB);
    s->input = malloc(C10_MAX_RGB);
    s->rgb320 = malloc(YSYX_YOLO26N_RGB_BYTES);
    if (!s->rgb || !s->input || !s->rgb320)
        return -1;
    if (o->mode & MODE_VERIFY) {
        s->heads = malloc(YSYX_C10_YOLO_HEAD_BYTES);
        s->golden = malloc(YSYX_C10_YOLO_HEAD_BYTES);
        if (!s->heads || !s->golden)
            return -1;
    }
    const int simulate = !!(o->mode & MODE_SIM);
    s->model = o->phase17 ? ysyx_yolo26n_open_p17(o->model, simulate) :
                           ysyx_yolo26n_open(o->model, simulate);
    if (!s->model)
        return -1;
    if (o->recording) {
        s->recording = fopen(o->recording, "rb");
        if (!s->recording)
            return -1;
    } else if (o->mode & MODE_V4L2) {
        s->camera = c10_camera_open(o->device, o->width, o->height, o->fps);
        if (!s->camera)
            return -1;
    }
    s->output_listener = listen_at(o->bind, o->port);
    if (s->output_listener < 0)
        return -1;
    if (!(o->mode & MODE_V4L2)) {
        s->input_listener = listen_at(o->bind, o->in_port);
        if (s->input_listener < 0)
            return -1;
    }
    s->csv = !strcmp(o->csv, "-") ? stdout : fopen(o->csv, "w");
    if (!s->csv)
        return -1;
    if (fputs("sequence,source,mode,backend,native_mhz,gpu_mhz,t_capture_preproc_ms,t_h2d_ms,"
        "t_pl_hw_ms,t_d2h_post_ms,t_net_tx_ms,t_e2e_ms,t_execute_ms,total_cycles,npu_busy,vpu_busy,"
        "native_npu_vpu_busy,gpu_busy,dma_busy,overlap,dma_compute_overlap,dma_bytes,dma_operations,"
        "static_h2d_bytes,dynamic_h2d_bytes,abi_error,invalid_dma,verify,network_complete,fault\n",
        s->csv) == EOF || fflush(s->csv))
        return -1;
    fprintf(stderr, "C10 %s: %s %u/%u MHz, Z15F %s:%u\n",
            (o->mode & MODE_VERIFY) ? "verify" : "live",
            (o->mode & MODE_SIM) ? "simulation" : "UIO",
            o->native_mhz, o->gpu_mhz, o->bind, o->port);
    return 0;
}

static void close_service(service *s)
{
    if (s->input_listener >= 0) close(s->input_listener);
    if (s->output_listener >= 0) close(s->output_listener);
    if (s->csv && s->csv != stdout) fclose(s->csv);
    if (s->recording) fclose(s->recording);
    c10_camera_close(s->camera);
    ysyx_yolo26n_close(s->model);
    free(s->golden);
    free(s->heads);
    free(s->rgb320);
    free(s->input);
    free(s->rgb);
}

int main(int argc, char **argv)
{
    if (z15_exec_option(&argc, argv))
        return 2;
    const char *usage = "z15_yolo_live_c10 --model FILE --source tcp|v4l2 "
        "[--device /dev/video0 --width 640 --height 480 --fps 30] "
        "[--freq-profile 125_100|200_150] [--live | --verify --golden DIRECTORY] "
        "[--csv PATH] [--frames N] [--bind IPv4 --port 9000 --in-port 9001] "
        "[--target step2|p17] [--exec-mode auto|cmdq|mmio] [--exec-stats] "
        "[--sim] [--recorded-yuyv FILE]\n"
        "--golden loads DIRECTORY/<sequence>.heads.bin (176400 raw bytes per frame).\n";
    settings o;
    service s = { .output_listener = -1, .input_listener = -1 };
    previous_frame prev = {0};
    if (argc == 2 && !strcmp(argv[1], "--help")) {
        fputs(usage, stdout);
        return 0;
    }
    if (parse_args(&o, argc, argv)) {
        fputs(usage, stderr);
        return 2;
    }
    struct sigaction action = { .sa_handler = stop };
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, NULL) || sigaction(SIGTERM, &action, NULL))
        return 1;
    if (init_service(&s, &o)) {
        perror("C10 service setup");
        close_service(&s);
        return 1;
    }
    int status = 0;
    while (!stopping && !status) {
        int viewer = accept_client(s.output_listener, -1);
        if (viewer < 0) {
            status = -1;
            break;
        }
        int input = s.input_listener >= 0 ? accept_client(s.input_listener, viewer) : -1;
        if (input >= 0 || s.input_listener < 0)
            status = stream(&s, &o, viewer, input, &prev);
        else if (input != -2)
            status = -1;
        if (input >= 0)
            close(input);
        close(viewer);
    }
    close_service(&s);
    return (!stopping && (status < 0 || (o.frames && prev.sequence < o.frames))) ? 1 : 0;
}
