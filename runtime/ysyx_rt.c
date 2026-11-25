#include "ysyx_rt.h"
#include "z15_regs.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

enum { HEADER = 32, NSECTION = 7, DIRECTORY = 16, POLL_LIMIT = 1000000 };
enum { STRINGS, SYMBOLS, DATA, INITIAL, STAGES, MOVES, COMMANDS };
enum { GPU = 1, NPU = 2, VPU = 3 };
enum { LAUNCH = 1, WAIT_STAGE = 2, DMA = 3, WAIT_DMA = 4, KV_YIELD = 5 };
enum { DMA_LOAD = 1, DMA_STORE = 2 };

typedef struct {
    const uint8_t *bytes;
    uint32_t count, length;
} section_t;

struct ysyx_program {
    uint32_t target, capacity, kv_steps;
    section_t sections[NSECTION];
};

static uint32_t u32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static uint32_t get(const uint8_t *p, unsigned i)
{
    return u32(p + 4u * i);
}

static int fail(const char **error, const char *message)
{
    if (error)
        *error = message;
    return -1;
}

static int region(uint32_t at, uint64_t n, uint32_t limit)
{
    return at <= limit && n <= limit - at;
}

static uint32_t crc32(const uint8_t *bytes, size_t size)
{
    uint32_t table[256], crc = UINT32_MAX;
    for (unsigned i = 0; i < 256; ++i) {
        uint32_t v = i;
        for (unsigned j = 0; j < 8; ++j)
            v = v & 1u ? (v >> 1) ^ UINT32_C(0xedb88320) : v >> 1;
        table[i] = v;
    }
    for (size_t i = 0; i < size; ++i) {
        uint8_t c = i >= 16 && i < 20 ? 0 : bytes[i];
        crc = (crc >> 8) ^ table[(crc ^ c) & 255u];
    }
    return ~crc;
}

static const uint8_t *record(const ysyx_program_t *p, unsigned section, size_t i)
{
    static const unsigned strides[NSECTION] = {1, 48, 1, 16, 32, 24, 16};
    return p->sections[section].bytes + strides[section] * i;
}

static int valid_name(const ysyx_program_t *p, uint32_t at, uint32_t n)
{
    const section_t *s = &p->sections[STRINGS];
    return n && n <= INT32_MAX && region(at, n, s->length) && !memchr(s->bytes + at, 0, n);
}

uint32_t ysyx_program_target(const ysyx_program_t *p)
{
    return p->target;
}

size_t ysyx_program_symbol_count(const ysyx_program_t *p)
{
    return p->sections[SYMBOLS].count;
}

int ysyx_program_symbol(const ysyx_program_t *p, size_t i, ysyx_symbol_t *v)
{
    if (i >= p->sections[SYMBOLS].count)
        return -1;
    const uint8_t *r = record(p, SYMBOLS, i);
    v->name = (const char *)(p->sections[STRINGS].bytes + get(r, 0));
    v->name_bytes = get(r, 1);
    v->flags = get(r, 2);
    v->dtype = get(r, 3);
    v->rank = get(r, 4);
    for (unsigned d = 0; d < 4; ++d)
        v->dims[d] = get(r, 5 + d);
    v->ddr_offset = get(r, 9);
    v->bytes = get(r, 10);
    return 0;
}

int ysyx_find_symbol(const ysyx_program_t *p, const char *name, uint32_t flags, ysyx_symbol_t *v)
{
    const size_t length = strlen(name);
    for (size_t i = 0; i < p->sections[SYMBOLS].count; ++i) {
        if (ysyx_program_symbol(p, i, v))
            return -1;
        if ((v->flags & flags) == flags && v->name_bytes == length && !memcmp(name, v->name, length))
            return 0;
    }
    return -1;
}

static int valid_symbols(const ysyx_program_t *p)
{
    static const uint8_t width[] = {0, 1, 1, 4, 8, 2, 2};
    for (size_t i = 0; i < p->sections[SYMBOLS].count; ++i) {
        const uint8_t *r = record(p, SYMBOLS, i);
        unsigned flags = get(r, 2), type = get(r, 3), rank = get(r, 4);
        if (!valid_name(p, get(r, 0), get(r, 1)) || flags > 7 || !(flags & 3) ||
            ((flags & YSYX_REQUIRED) && !(flags & YSYX_INPUT)) ||
            !rank || rank > 4 || !type || type >= sizeof(width) ||
            !get(r, 10) || get(r, 11) || !region(get(r, 9), get(r, 10), Z15_POOL_SIZE))
            return 0;
        uint32_t n = width[type];
        for (unsigned d = 0; d < 4; ++d) {
            uint32_t dim = get(r, 5 + d);
            if (d >= rank) {
                if (dim)
                    return 0;
            } else {
                if (!dim || n > get(r, 10) / dim)
                    return 0;
                n *= dim;
            }
        }
        if (n != get(r, 10))
            return 0;
        for (size_t j = 0; j < i; ++j) {
            const uint8_t *q = record(p, SYMBOLS, j);
            if (get(r, 1) == get(q, 1) &&
                !memcmp(p->sections[STRINGS].bytes + get(r, 0),
                        p->sections[STRINGS].bytes + get(q, 0), get(r, 1)))
                return 0;
        }
    }
    return 1;
}

static int allowed_tensor_register(uint32_t reg)
{
    return !(reg & 3u) && ((reg >= 0x60u && reg <= 0x98u) ||
                             (reg >= 0xa0u && reg <= 0xccu));
}

static int valid_tensor_bases(const uint32_t cfg[52], uint64_t present,
                              uint32_t action, uint32_t zone)
{
    unsigned bases[8];
    size_t count = 0;
    if (action == 64u) {
        bases[count++] = 0xa0u;
        bases[count++] = 0xa8u;
        const uint32_t op = cfg[0xbcu / 4u] & 15u;
        if (!(present & (UINT64_C(1) << (0xbcu / 4u))))
            return 0;
        if (op == 0u || op == 6u)
            bases[count++] = 0xa4u;
    } else {
        bases[count++] = 0x68u;
        bases[count++] = 0x6cu;
        bases[count++] = 0x70u;
        if (action == 32u) {
            bases[count++] = 0x8cu;
            bases[count++] = 0x90u;
            bases[count++] = 0x94u;
        } else {
            const uint32_t flags = cfg[0x78u / 4u];
            const uint32_t dimensions = cfg[0x64u / 4u];
            if (!(present & (UINT64_C(1) << (0x78u / 4u))) ||
                !(present & (UINT64_C(1) << (0x64u / 4u))))
                return 0;
            if (flags & 2u)
                bases[count++] = 0x74u;
            if ((flags & (1u << 17)) && (dimensions & 511u) > ((dimensions >> 16) & 511u))
                bases[count++] = 0x84u;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const unsigned index = bases[i] / 4u;
        if (!(present & (UINT64_C(1) << index)) || (cfg[index] & 3u) ||
            (cfg[index] >> 20) != zone)
            return 0;
    }
    return 1;
}

/* The software-only ASIC v3 target shares the YHB section format, but
 * uses explicit 64 KiB byte addresses rather than Phase 8 triplets. */
enum { ASIC_IP_V3_TARGET = 5, ASIC_IP_V3_BYTES = 65536, ASIC_IP_V3_WORDS = 32 };

static int valid_ip_span(const uint8_t *r, unsigned rank, unsigned type,
                         unsigned shape, uint32_t address, uint32_t *size)
{
    static const unsigned width[] = {0, 1, 1, 4};
    if (!rank || rank > 4 || !type || type > 3 || (address & 7u))
        return 0;
    uint32_t bytes = width[type];
    for (unsigned i = 0; i < 4; ++i) {
        const uint32_t dim = get(r, shape + i);
        if ((i >= rank && dim) || (i < rank && (!dim || bytes > ASIC_IP_V3_BYTES / dim)))
            return 0;
        if (i < rank)
            bytes *= dim;
    }
    *size = bytes;
    return region(address, bytes, ASIC_IP_V3_BYTES);
}

static int ip_overlap(uint32_t a, uint32_t n, uint32_t b, uint32_t m)
{
    return a < b + m && b < a + n;
}

static int valid_ip_stage(const uint8_t *r, uint32_t words, uint32_t engine)
{
    if (words != ASIC_IP_V3_WORDS || get(r, 0) != UINT32_C(0x33565049))
        return 0;
    const uint32_t op = get(r, 1);
    if (op < 1 || op > 10 || engine != (op <= 2 ? NPU : VPU))
        return 0;
    static const unsigned allowed_flags[] = {0, 3, 1, 17, 1, 9, 1, 5, 1, 3, 1};
    if (get(r, 2) & ~allowed_flags[op])
        return 0;
    const uint32_t a = get(r, 6), b = get(r, 7), c = get(r, 8), bias = get(r, 9);
    uint32_t alen = 0, blen = 0, clen = 0;
    if (!valid_ip_span(r, get(r, 3), get(r, 10), 13, a, &alen) ||
        !valid_ip_span(r, get(r, 5), get(r, 12), 21, c, &clen) ||
        ip_overlap(a, alen, c, clen))
        return 0;
    if (op <= 2 || op >= 9) {
        if (!valid_ip_span(r, get(r, 4), get(r, 11), 17, b, &blen) ||
            ip_overlap(b, blen, c, clen))
            return 0;
    } else if (b != UINT32_MAX || get(r, 4) || get(r, 11) ||
               get(r, 17) || get(r, 18) || get(r, 19) || get(r, 20)) {
        return 0;
    }
    if (bias != UINT32_MAX) {
        const uint32_t channels = get(r, 20 + get(r, 5));
        if (op > 2 || (bias & 7u) || channels > ASIC_IP_V3_BYTES / 4u ||
            !region(bias, channels * 4u, ASIC_IP_V3_BYTES) ||
            ip_overlap(bias, channels * 4u, c, clen))
            return 0;
    }
    return 1;
}

static int valid_stage(const ysyx_program_t *p, const uint8_t *s)
{
    uint32_t engine = get(s, 2), data = get(s, 3), words = get(s, 4);
    uint32_t aux = get(s, 5), jobs = get(s, 6);
    const section_t *payload = &p->sections[DATA];
    if (!valid_name(p, get(s, 0), get(s, 1)) || !words || get(s, 7) > 1 ||
        (data | aux) & 7u || !region(data, (uint64_t)words * 4u, payload->length) ||
        !region(aux, (uint64_t)jobs * 16u, payload->length))
        return 0;
    if (p->target == ASIC_IP_V3_TARGET && get(s, 7))
        return 0;
    if (engine == GPU) {
        if (!jobs || ((p->target <= 2u || p->target == ASIC_IP_V3_TARGET) &&
                      words > Z15_PROGRAM_BYTES / 4u))
            return 0;
        uint32_t page = get(payload->bytes + aux, 1) & 511u;
        for (uint32_t i = 0; i < jobs; ++i) {
            const uint8_t *j = payload->bytes + aux + 16u * i;
            uint32_t entry = get(j, 0), config = get(j, 1), threads = get(j, 2), warp = get(j, 3);
            if ((entry & 3u) || entry / 4u >= words || !threads)
                return 0;
            if ((p->target == YSYX_TARGET_Z15_BALANCED || p->target == 2u) &&
                (threads > (p->target == 2u ? 8u : 16u) || warp > 3 || config & UINT32_C(0xfc00fe00) ||
                 ((config >> 16) & 255u) != threads || (config >> 24) != warp ||
                 (config & 511u) != page || page / 256u != get(s, 7)))
                return 0;
            if (p->target == ASIC_IP_V3_TARGET && (threads > 255 || warp || config))
                return 0;
        }
        return 1;
    }
    if ((engine != NPU && engine != VPU) || jobs)
        return 0;
    if (p->target == ASIC_IP_V3_TARGET)
        return valid_ip_stage(payload->bytes + data, words, engine);
    if (p->target > 2)
        return words == 3;
    if (words & 1u)
        return 0;
    const uint8_t *regs = payload->bytes + data;
    uint32_t cfg[52] = {0};
    uint64_t present = 0;
    for (uint32_t i = 0; i < words; i += 2) {
        uint32_t reg = get(regs, i), value = get(regs, i + 1);
        if (i + 2 == words) {
            if (reg != Z15_CTRL || (engine == VPU ? value != 64u : value != 8u && value != 32u) ||
                !valid_tensor_bases(cfg, present, value, get(s, 7)))
                return 0;
        } else if (reg == Z15_CTRL) {
            if (value != 128u && value != 256u)
                return 0;
        } else if (!allowed_tensor_register(reg)) {
            return 0;
        } else {
            cfg[reg / 4u] = value;
            present |= UINT64_C(1) << (reg / 4u);
        }
    }
    return 1;
}

static int validate_schedule(ysyx_program_t *p, const char **error)
{
    uint32_t active[2] = {UINT32_MAX, UINT32_MAX}, active_dma = UINT32_MAX;
    uint32_t nstage = p->sections[STAGES].count;
    uint8_t *launched = calloc(nstage ? nstage : 1u, 1);
    if (!launched)
        return fail(error, "out of memory while validating schedule");
    const char *wrong = NULL;
    int work_since_yield = 0;
    uint32_t last_op = 0;
    for (uint32_t i = 0; i < p->sections[COMMANDS].count; ++i) {
        const uint8_t *c = record(p, COMMANDS, i);
        uint32_t op = get(c, 0), index = get(c, 1);
        if (get(c, 2) || get(c, 3) || op < LAUNCH || op > KV_YIELD ||
            ((op == LAUNCH || op == WAIT_STAGE) && index >= nstage) ||
            (op == DMA && index >= p->sections[MOVES].count) || (op == WAIT_DMA && index)) {
            wrong = "malformed YHB scheduling instruction";
            break;
        }
        last_op = op;
        if (op == LAUNCH || op == DMA)
            work_since_yield = 1;
        if (op == KV_YIELD) {
            if (p->target != YSYX_TARGET_Z15_BALANCED || index != p->kv_steps || !work_since_yield ||
                active[0] != UINT32_MAX || active[1] != UINT32_MAX || active_dma != UINT32_MAX) {
                wrong = "invalid or undrained YHB KV checkpoint";
                break;
            }
            ++p->kv_steps;
            work_since_yield = 0;
        } else if (op == WAIT_DMA) {
            active_dma = UINT32_MAX;
        } else if (op == DMA) {
            const uint8_t *move = record(p, MOVES, index);
            if (p->target == YSYX_TARGET_Z15_BALANCED) {
                for (unsigned engine = 0; engine < 2; ++engine)
                    if (active[engine] != UINT32_MAX &&
                        get(record(p, STAGES, active[engine]), 7) == get(move, 2) >> 20)
                        wrong = "DMA launch lacks compute ownership fence";
            }
            active_dma = index;
        } else {
            const uint8_t *stage = record(p, STAGES, index);
            unsigned engine = get(stage, 2) == GPU ? 0 : 1;
            if (op == WAIT_STAGE) {
                if (!launched[index])
                    wrong = "YHB waits on a stage not yet launched";
                else if (active[engine] == index)
                    active[engine] = UINT32_MAX;
            } else {
                if (launched[index] || active[engine] != UINT32_MAX)
                    wrong = "duplicate or unfenced YHB engine launch";
                if (p->target == YSYX_TARGET_Z15_BALANCED) {
                    if (active[1u - engine] != UINT32_MAX &&
                        get(record(p, STAGES, active[1u - engine]), 7) == get(stage, 7))
                        wrong = "compute launch lacks ownership fence";
                    if (active_dma != UINT32_MAX &&
                        get(record(p, MOVES, active_dma), 2) >> 20 == get(stage, 7))
                        wrong = "compute launch lacks DMA ownership fence";
                }
                launched[index] = 1;
                active[engine] = index;
            }
        }
        if (wrong)
            break;
    }
    if (!wrong && p->kv_steps && last_op != KV_YIELD)
        wrong = "YHB KV schedule must end at a checkpoint";
    for (uint32_t i = 0; !wrong && i < nstage; ++i)
        if (!launched[i])
            wrong = "YHB stage has no launch";
    free(launched);
    return wrong ? fail(error, wrong) : 0;
}

int ysyx_program_open(const void *raw, size_t size, ysyx_program_t **out, const char **error)
{
    static const unsigned strides[NSECTION] = {1, 48, 1, 16, 32, 24, 16};
    const uint8_t *bytes = raw;
    if (!out)
        return fail(error, "missing program handle");
    *out = NULL;
    if (!raw || size < HEADER + NSECTION * DIRECTORY || size > Z15_POOL_SIZE ||
        memcmp(bytes, "YHB1", 4) || get(bytes, 1) != (HEADER << 16 | 1) || get(bytes, 3) != size ||
        get(bytes, 5) != NSECTION || get(bytes, 7) || !get(bytes, 2) || get(bytes, 2) > ASIC_IP_V3_TARGET ||
        !get(bytes, 6) || get(bytes, 6) > Z15_SCRATCH_SIZE ||
        (get(bytes, 2) == ASIC_IP_V3_TARGET && get(bytes, 6) != ASIC_IP_V3_BYTES))
        return fail(error, "invalid YHB1 header");
    if (crc32(bytes, size) != get(bytes, 4))
        return fail(error, "YHB CRC32 mismatch");

    ysyx_program_t p = {0};
    p.target = get(bytes, 2);
    p.capacity = get(bytes, 6);
    uint32_t end = HEADER + NSECTION * DIRECTORY;
    for (unsigned i = 0; i < NSECTION; ++i) {
        const uint8_t *r = bytes + HEADER + DIRECTORY * i;
        uint32_t at = get(r, 1), n = get(r, 2), count = get(r, 3);
        if (get(r, 0) != i + 1 || at & 7u || at < end || !region(at, n, (uint32_t)size) ||
            (uint64_t)count * strides[i] != n)
            return fail(error, "malformed YHB section table");
        p.sections[i] = (section_t){bytes + at, count, n};
        end = at + n;
    }
    if (end != size || !valid_symbols(&p))
        return fail(error, "invalid YHB symbols or trailing bytes");
    if (p.target == ASIC_IP_V3_TARGET) {
        if (p.sections[MOVES].count)
            return fail(error, "ASIC IP v3 does not expose DMA transfers");
        for (uint32_t i = 0; i < p.sections[SYMBOLS].count; ++i) {
            const uint8_t *r = record(&p, SYMBOLS, i);
            if (!region(get(r, 9), get(r, 10), p.capacity))
                return fail(error, "ASIC IP v3 symbol exceeds scratchpad");
        }
    }

    uint32_t last = 0;
    for (uint32_t i = 0; i < p.sections[INITIAL].count; ++i) {
        const uint8_t *r = record(&p, INITIAL, i);
        uint32_t at = get(r, 0), n = get(r, 1), data = get(r, 2);
        if (!n || at < last ||
            !region(at, n, p.target == ASIC_IP_V3_TARGET ? p.capacity : Z15_POOL_SIZE) || data & 7u ||
            !region(data, n, p.sections[DATA].length) || get(r, 3))
            return fail(error, "invalid YHB initial span");
        last = at + n;
    }
    for (uint32_t i = 0; i < p.sections[MOVES].count; ++i) {
        const uint8_t *r = record(&p, MOVES, i);
        uint32_t dir = get(r, 0), ddr = get(r, 1), scratch = get(r, 2), n = get(r, 3);
        if ((dir != DMA_LOAD && dir != DMA_STORE) || !n || (ddr | scratch | n) & 31u ||
            !region(ddr, n, Z15_POOL_SIZE) || !region(scratch, n, p.capacity) ||
            scratch >> 20 != (scratch + n - 1) >> 20 || (get(r, 4) & ~2u) || get(r, 5))
            return fail(error, "invalid YHB DMA extent");
    }
    for (uint32_t i = 0; i < p.sections[STAGES].count; ++i)
        if (!valid_stage(&p, record(&p, STAGES, i)))
            return fail(error, "invalid YHB native stage");
    if (validate_schedule(&p, error))
        return -1;
    *out = malloc(sizeof(p));
    if (!*out)
        return fail(error, "out of memory while opening YHB");
    **out = p;
    return 0;
}

void ysyx_program_close(ysyx_program_t *p)
{
    free(p);
}

typedef struct {
    const ysyx_program_t *program;
    const ysyx_hal_t *hal;
    FILE *trace;
    uint32_t pending[2], dma_index;
    uint64_t before[2], dma_start;
    ysyx_stats_t *stats;
} run_t;

static int read32(const run_t *run, uint32_t reg, uint32_t *value)
{
    return run->hal->mmio_read32(run->hal->ctx, reg, value);
}

static int write32(const run_t *run, uint32_t reg, uint32_t value)
{
    return run->hal->mmio_write32(run->hal->ctx, reg, value);
}

static int read_counter(const run_t *run, uint32_t reg, uint64_t *value)
{
    for (unsigned i = 0; i < 8; ++i) {
        uint32_t hi, lo, second;
        if (read32(run, reg + 4, &hi) || read32(run, reg, &lo) || read32(run, reg + 4, &second))
            return -1;
        if (hi == second) {
            *value = (uint64_t)hi << 32 | lo;
            return 0;
        }
    }
    return -1;
}

static int poll_status(const run_t *run, uint32_t required, uint32_t busy)
{
    for (unsigned i = 0; i < POLL_LIMIT; ++i) {
        uint32_t value;
        if (read32(run, Z15_STATUS, &value) || value & Z15_HAS_ERROR || !(value & Z15_LOCKED))
            return -1;
        if ((value & required) == required && !(value & busy))
            return 0;
    }
    return -1;
}

static int wait_gpu_slot(const run_t *run, uint32_t warp)
{
    for (unsigned i = 0; i < POLL_LIMIT; ++i) {
        uint32_t value, status;
        if (read32(run, Z15_STATUS, &status) || status & Z15_HAS_ERROR ||
            read32(run, Z15_GPU_SLOTS, &value))
            return -1;
        if (!(value & (16u | (1u << warp))))
            return 0;
    }
    return -1;
}

static int launch_gpu(const run_t *run, const uint8_t *stage, const uint8_t *words)
{
    for (uint32_t i = 0; i < get(stage, 4); ++i)
        if (write32(run, Z15_PROGRAM + 4u * i, get(words, i)))
            return -1;
    const uint8_t *jobs = run->program->sections[DATA].bytes + get(stage, 5);
    uint32_t previous_page = UINT32_MAX;
    for (uint32_t i = 0; i < get(stage, 6); ++i) {
        const uint8_t *job = jobs + 16u * i;
        uint32_t config = get(job, 1), page = config & 511u;
        if (i && page != previous_page && poll_status(run, Z15_GPU_DONE, Z15_GPU_BUSY))
            return -1;
        if (wait_gpu_slot(run, get(job, 3)) || write32(run, Z15_GPU_PC, get(job, 0)) ||
            write32(run, Z15_GPU_JOB, config) || write32(run, Z15_CTRL, Z15_START_GPU))
            return -1;
        previous_page = page;
    }
    return 0;
}

static int launch_stage(run_t *run, uint32_t index)
{
    const uint8_t *stage = record(run->program, STAGES, index);
    const uint8_t *words = run->program->sections[DATA].bytes + get(stage, 3);
    unsigned engine = get(stage, 2) == GPU ? 0 : 1;
    if (run->pending[engine] != UINT32_MAX)
        return -1;
    if (run->trace && read_counter(run, engine ? Z15_NPU_CYCLES : Z15_GPU_CYCLES, &run->before[engine]))
        return -1;
    if (engine == 0) {
        if (launch_gpu(run, stage, words))
            return -1;
    } else {
        for (uint32_t i = 0; i < get(stage, 4); i += 2)
            if (write32(run, get(words, i), get(words, i + 1)))
                return -1;
    }
    run->pending[engine] = index;
    return 0;
}

static int finish_stage(run_t *run, uint32_t index)
{
    const uint8_t *stage = record(run->program, STAGES, index);
    uint32_t unit = get(stage, 2);
    unsigned engine = unit == GPU ? 0 : 1;
    if (run->pending[engine] != index)
        return 0;
    uint64_t start = 0;
    if (run->trace && read_counter(run, Z15_WALL_CYCLES, &start))
        return -1;
    if (poll_status(run, engine ? Z15_NPU_DONE : Z15_GPU_DONE, engine ? Z15_NPU_BUSY : Z15_GPU_BUSY))
        return -1;
    if (run->trace) {
        uint64_t end, busy;
        if (read_counter(run, Z15_WALL_CYCLES, &end) ||
            read_counter(run, engine ? Z15_NPU_CYCLES : Z15_GPU_CYCLES, &busy))
            return -1;
        const char *name = (const char *)run->program->sections[STRINGS].bytes + get(stage, 0);
        fprintf(run->trace, "STAGE id=%" PRIu32 " name=%.*s engine=%s npu_domain_busy_ticks=%" PRIu64
                " fence_ticks=%" PRIu64 "\n", index, (int)get(stage, 1), name,
                unit == GPU ? "GPU" : unit == VPU ? "VPU" : "NPU",
                busy - run->before[engine], end - start);
    }
    run->pending[engine] = UINT32_MAX;
    return 0;
}

static int finish_dma(run_t *run)
{
    if (run->dma_index == UINT32_MAX)
        return 0;
    if (poll_status(run, Z15_DMA_DONE, Z15_DMA_BUSY))
        return -1;
    const uint8_t *move = record(run->program, MOVES, run->dma_index);
    if (get(move, 0) == DMA_STORE &&
        run->hal->cache_sync(run->hal->ctx, get(move, 1), get(move, 3), YSYX_SYNC_FOR_HOST))
        return -1;
    if (run->trace) {
        uint64_t end;
        if (read_counter(run, Z15_WALL_CYCLES, &end))
            return -1;
        fprintf(run->trace, "DMA direction=%s ddr=%" PRIu32 " scratch=%" PRIu32
                " bytes=%" PRIu32 " issue_to_fence_ticks=%" PRIu64 "\n",
                get(move, 0) == DMA_STORE ? "out" : "in", get(move, 1), get(move, 2),
                get(move, 3), end - run->dma_start);
    }
    run->dma_index = UINT32_MAX;
    return 0;
}

static int launch_dma(run_t *run, uint32_t index)
{
    const uint8_t *move = record(run->program, MOVES, index);
    uint32_t out = get(move, 0) == DMA_STORE;
    if (finish_dma(run) ||
        run->hal->cache_sync(run->hal->ctx, get(move, 1), get(move, 3), YSYX_SYNC_FOR_DEVICE) ||
        (run->trace && read_counter(run, Z15_WALL_CYCLES, &run->dma_start)))
        return -1;
    if (write32(run, Z15_DMA_BASE, Z15_POOL_BASE) ||
        write32(run, Z15_DMA_OFFSET, get(move, 1)) ||
        write32(run, Z15_DMA_SCRATCH, get(move, 2)) ||
        write32(run, Z15_DMA_BYTES, get(move, 3)) ||
        write32(run, Z15_DMA_FLAGS, get(move, 4) | out) ||
        write32(run, Z15_CTRL, Z15_START_DMA))
        return -1;
    run->dma_index = index;
    ++run->stats->dma_operations;
    run->stats->dma_bytes += get(move, 3);
    return 0;
}

static int validate_inputs(const ysyx_program_t *program, const ysyx_input_t *inputs, size_t count,
                           const char **error)
{
    for (size_t i = 0; i < count; ++i) {
        ysyx_symbol_t symbol;
        if (!inputs[i].name || !inputs[i].data ||
            ysyx_find_symbol(program, inputs[i].name, YSYX_INPUT, &symbol) ||
            symbol.bytes != inputs[i].size)
            return fail(error, "unknown input name or input byte count does not match bundle");
        for (size_t j = 0; j < i; ++j)
            if (!strcmp(inputs[i].name, inputs[j].name))
                return fail(error, "duplicate runtime input");
    }
    for (size_t i = 0; i < ysyx_program_symbol_count(program); ++i) {
        ysyx_symbol_t symbol;
        if (ysyx_program_symbol(program, i, &symbol))
            return fail(error, "invalid program symbol index");
        if (!(symbol.flags & YSYX_REQUIRED))
            continue;
        size_t k;
        for (k = 0; k < count; ++k)
            if (strlen(inputs[k].name) == symbol.name_bytes &&
                !memcmp(inputs[k].name, symbol.name, symbol.name_bytes))
                break;
        if (k == count)
            return fail(error, "required input tensor was not supplied");
    }
    return 0;
}

static int write_initial(const ysyx_program_t *p, const ysyx_hal_t *hal,
                         const ysyx_input_t *inputs, size_t count)
{
    for (uint32_t i = 0; i < p->sections[INITIAL].count; ++i) {
        const uint8_t *r = record(p, INITIAL, i);
        if (hal->ddr_write(hal->ctx, get(r, 0), p->sections[DATA].bytes + get(r, 2), get(r, 1)))
            return -1;
    }
    for (size_t i = 0; i < count; ++i) {
        ysyx_symbol_t symbol;
        if (ysyx_find_symbol(p, inputs[i].name, YSYX_INPUT, &symbol) ||
            hal->ddr_write(hal->ctx, symbol.ddr_offset, inputs[i].data, inputs[i].size))
            return -1;
    }
    return 0;
}

static int sample_telemetry(const run_t *run, ysyx_stats_t *stats)
{
    return read_counter(run, Z15_WALL_CYCLES, &stats->wall_ticks) ||
           read_counter(run, Z15_NPU_CYCLES, &stats->npu_ticks) ||
           read_counter(run, Z15_GPU_CYCLES, &stats->gpu_ticks) ||
           read_counter(run, Z15_OVERLAP_CYCLES, &stats->npu_gpu_overlap_ticks);
}

static int attach(const ysyx_program_t *p, const ysyx_hal_t *hal, const ysyx_input_t *inputs,
                  size_t count, const char **error)
{
    if (!p || p->target != YSYX_TARGET_Z15_BALANCED || p->capacity != Z15_SCRATCH_SIZE)
        return fail(error, "this runtime executes only the frozen Z15 Balanced target");
    if (!hal || !hal->mmio_read32 || !hal->mmio_write32 || !hal->ddr_write || !hal->ddr_read ||
        !hal->cache_sync || (!inputs && count))
        return fail(error, "incomplete runtime HAL");
    if (validate_inputs(p, inputs, count, error))
        return -1;
    const run_t run = {.program = p, .hal = hal};
    uint32_t magic, version;
    if (read32(&run, Z15_MAGIC, &magic) || magic != Z15_MAGIC_VALUE ||
        read32(&run, Z15_VERSION, &version) || version != Z15_VERSION_VALUE ||
        poll_status(&run, 0, Z15_NPU_BUSY | Z15_GPU_BUSY | Z15_DMA_BUSY))
        return fail(error, "Z15 Balanced ABI absent, busy or reporting a sticky fault");
    if (write_initial(p, hal, inputs, count))
        return fail(error, "DDR initial image or runtime input write failed");
    return 0;
}

static int execute_range(const ysyx_program_t *p, const ysyx_hal_t *hal, uint32_t first,
                         uint32_t end, ysyx_stats_t *stats, FILE *trace, const char **error)
{
    memset(stats, 0, sizeof(*stats));
    run_t run = {p, hal, trace, {UINT32_MAX, UINT32_MAX}, UINT32_MAX, {0, 0}, 0, stats};
    ysyx_stats_t begin = {0};
    if (sample_telemetry(&run, &begin))
        return fail(error, "hardware counter read failed");
    if (hal->observe)
        hal->observe(hal->ctx, &begin.observed);

    for (uint32_t i = first; i < end; ++i) {
        const uint8_t *command = record(p, COMMANDS, i);
        uint32_t index = get(command, 1);
        switch (get(command, 0)) {
        case LAUNCH:
            if (launch_stage(&run, index))
                return fail(error, "compute descriptor or GPU program submission failed");
            break;
        case WAIT_STAGE:
            if (finish_stage(&run, index))
                return fail(error, "compute fence timed out or hardware reported a fault");
            break;
        case DMA:
            if (launch_dma(&run, index))
                return fail(error, "DMA submission failed");
            break;
        case WAIT_DMA:
            if (finish_dma(&run))
                return fail(error, "DMA fence timed out or hardware reported a fault");
            break;
        case KV_YIELD:
            /* The loader requires explicit fences before every checkpoint. */
            break;
        default:
            return fail(error, "invalid YHB opcode");
        }
    }
    for (unsigned i = 0; i < 2; ++i)
        if (run.pending[i] != UINT32_MAX && finish_stage(&run, run.pending[i]))
            return fail(error, "final compute fence failed");
    if (finish_dma(&run) || sample_telemetry(&run, stats))
        return fail(error, "final DMA fence or counter read failed");
    stats->wall_ticks -= begin.wall_ticks;
    stats->npu_ticks -= begin.npu_ticks;
    stats->gpu_ticks -= begin.gpu_ticks;
    stats->npu_gpu_overlap_ticks -= begin.npu_gpu_overlap_ticks;
    if (hal->observe) {
        stats->has_observations = 1;
        hal->observe(hal->ctx, &stats->observed);
        stats->observed.axi_transactions -= begin.observed.axi_transactions;
        stats->observed.dma_busy_ticks -= begin.observed.dma_busy_ticks;
        stats->observed.dma_compute_overlap_ticks -= begin.observed.dma_compute_overlap_ticks;
        stats->observed.triple_overlap_ticks -= begin.observed.triple_overlap_ticks;
        stats->observed.gpu_clock_busy_ticks -= begin.observed.gpu_clock_busy_ticks;
        stats->observed.dma_gpu_overlap_ticks -= begin.observed.dma_gpu_overlap_ticks;
        stats->observed.dma_tensor_overlap_ticks -= begin.observed.dma_tensor_overlap_ticks;
    }
    return 0;
}

int ysyx_execute(const ysyx_program_t *p, const ysyx_hal_t *hal, const ysyx_input_t *inputs, size_t count,
                 ysyx_stats_t *stats, FILE *trace, const char **error)
{
    if (!stats)
        return fail(error, "missing runtime statistics");
    if (attach(p, hal, inputs, count, error))
        return -1;
    return execute_range(p, hal, 0, p->sections[COMMANDS].count, stats, trace, error);
}

int ysyx_read_output(const ysyx_program_t *p, const ysyx_hal_t *hal, const char *name, void *data, size_t size)
{
    ysyx_symbol_t symbol;
    if (ysyx_find_symbol(p, name, YSYX_OUTPUT, &symbol) || size != symbol.bytes)
        return -1;
    return hal->cache_sync(hal->ctx, symbol.ddr_offset, size, YSYX_SYNC_FOR_HOST) ||
           hal->ddr_read(hal->ctx, symbol.ddr_offset, data, size);
}

struct ysyx_kv_session {
    const ysyx_program_t *program;
    const ysyx_hal_t *hal;
    uint32_t cursor, next_step;
    int failed;
};

uint32_t ysyx_kv_step_count(const ysyx_program_t *p)
{
    return p ? p->kv_steps : 0;
}

int ysyx_kv_begin(const ysyx_program_t *p, const ysyx_hal_t *hal, const ysyx_input_t *inputs,
                  size_t count, ysyx_kv_session_t **out, const char **error)
{
    if (!out)
        return fail(error, "missing KV session handle");
    *out = NULL;
    if (!ysyx_kv_step_count(p))
        return fail(error, "program has no KV checkpoints");
    ysyx_kv_session_t *session = calloc(1, sizeof(*session));
    if (!session)
        return fail(error, "out of memory opening KV session");
    if (attach(p, hal, inputs, count, error)) {
        free(session);
        return -1;
    }
    session->program = p;
    session->hal = hal;
    *out = session;
    return 0;
}

static int kv_appendable(const ysyx_kv_session_t *session)
{
    return session && !session->failed && session->next_step < session->program->kv_steps;
}

int ysyx_kv_step(ysyx_kv_session_t *s, uint32_t step, ysyx_stats_t *stats, FILE *trace,
                 const char **error)
{
    if (!stats || !kv_appendable(s) || s->next_step != step)
        return fail(error, "unexpected KV step or closed/failed KV session");
    const ysyx_program_t *p = s->program;
    uint32_t end = s->cursor;
    while (end < p->sections[COMMANDS].count && get(record(p, COMMANDS, end), 0) != KV_YIELD)
        ++end;
    if (end == p->sections[COMMANDS].count || execute_range(p, s->hal, s->cursor, end, stats, trace, error)) {
        s->failed = 1;
        return end == p->sections[COMMANDS].count ? fail(error, "missing KV checkpoint") : -1;
    }
    s->cursor = end + 1u;
    ++s->next_step;
    return 0;
}

int ysyx_kv_write_input(ysyx_kv_session_t *s, const ysyx_input_t *input, const char **error)
{
    ysyx_symbol_t symbol;
    if (!kv_appendable(s) || !input || !input->name || !input->data ||
        ysyx_find_symbol(s->program, input->name, YSYX_INPUT, &symbol) || symbol.bytes != input->size)
        return fail(error, "unknown KV input or invalid input size/session");
    if (s->hal->ddr_write(s->hal->ctx, symbol.ddr_offset, input->data, input->size)) {
        s->failed = 1;
        return fail(error, "KV runtime input write failed");
    }
    return 0;
}

int ysyx_kv_read_output(const ysyx_kv_session_t *s, const char *name, void *data, size_t size)
{
    ysyx_symbol_t symbol;
    if (!s || s->failed || !s->next_step || !name || !data ||
        ysyx_find_symbol(s->program, name, YSYX_OUTPUT, &symbol) || size != symbol.bytes)
        return -1;

    /* Commands before the cursor have drained. A tensor can span several stores. */
    uint64_t covered = symbol.ddr_offset;
    uint64_t end = covered + symbol.bytes;
    while (covered < end) {
        uint64_t next = covered;
        for (uint32_t i = 0; i < s->cursor; ++i) {
            const uint8_t *command = record(s->program, COMMANDS, i);
            if (get(command, 0) != DMA)
                continue;
            const uint8_t *move = record(s->program, MOVES, get(command, 1));
            uint64_t start = get(move, 1);
            uint64_t finish = start + get(move, 3);
            if (get(move, 0) == DMA_STORE && start <= covered && finish > next)
                next = finish;
        }
        if (next == covered)
            return -1;
        covered = next;
    }
    return s->hal->cache_sync(s->hal->ctx, symbol.ddr_offset, size, YSYX_SYNC_FOR_HOST) ||
           s->hal->ddr_read(s->hal->ctx, symbol.ddr_offset, data, size);
}

int ysyx_kv_feedback(ysyx_kv_session_t *s, const char *output, const char *input, const char **error)
{
    ysyx_symbol_t src, dst;
    if (!kv_appendable(s) || !s->next_step || !output || !input ||
        ysyx_find_symbol(s->program, output, YSYX_OUTPUT, &src) ||
        ysyx_find_symbol(s->program, input, YSYX_INPUT, &dst) ||
        src.bytes != dst.bytes || src.dtype != dst.dtype)
        return fail(error, "KV feedback requires matching output and input dtype and size");
    uint8_t *data = malloc(src.bytes);
    if (!data)
        return fail(error, "out of memory copying KV feedback");
    int rc;
    if (ysyx_kv_read_output(s, output, data, src.bytes))
        rc = fail(error, "KV feedback read failed");
    else {
        const ysyx_input_t binding = {input, data, src.bytes};
        rc = ysyx_kv_write_input(s, &binding, error);
    }
    free(data);
    return rc;
}

void ysyx_kv_close(ysyx_kv_session_t *s)
{
    free(s);
}
