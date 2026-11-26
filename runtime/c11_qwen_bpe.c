#define _POSIX_C_SOURCE 200809L
#include "c11_qwen_bpe.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum { LETTER = 1, NUMBER = 2, SPACE = 4, MAX_INPUT_BYTES = 8192 };

typedef struct {
    const uint8_t *data;
    uint32_t count;
    uint32_t width;
} table;

struct qwen_bpe {
    const uint8_t *map, *strings, *offset;
    size_t bytes, cursor;
    uint32_t base, total;
    table bytes_to_ids, merges, ranges, decompositions, nfd, compose, classes;
};

static uint32_t le32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static const uint8_t *section(qwen_bpe *m, uint64_t bytes)
{
    if (bytes > m->bytes - m->cursor)
        return NULL;
    const uint8_t *start = m->map + m->cursor;
    m->cursor += (size_t)bytes;
    return start;
}

static int read_table(qwen_bpe *m, table *t, uint32_t count, uint32_t fields)
{
    t->count = count;
    t->width = fields * 4;
    t->data = section(m, (uint64_t)count * t->width);
    return t->data ? 0 : -1;
}

static const uint8_t *row(const table *t, uint32_t index)
{
    return t->data + (size_t)index * t->width;
}

static int layout(qwen_bpe *m)
{
    if (m->bytes < 48 || memcmp(m->map, "YSQBPE1", 8))
        return -1;
    const uint8_t *p = m->map;
    m->base = le32(p + 8);
    m->total = le32(p + 12);
    if (m->base != 151643 || m->total < m->base + 3 || m->total > 151936)
        return -1;
    const uint32_t text_bytes = le32(p + 40);
    m->cursor = 48;
    m->offset = section(m, 4u * (m->total + 1));
    m->strings = section(m, text_bytes);
    if (!m->offset || !m->strings || !section(m, (4 - m->cursor % 4) % 4) ||
        read_table(m, &m->bytes_to_ids, 256, 1) ||
        read_table(m, &m->merges, le32(p + 16), 4) ||
        read_table(m, &m->ranges, le32(p + 20), 3) ||
        read_table(m, &m->decompositions, le32(p + 24), 3) ||
        read_table(m, &m->nfd, le32(p + 28), 1) ||
        read_table(m, &m->compose, le32(p + 32), 3) ||
        read_table(m, &m->classes, le32(p + 36), 2) ||
        m->cursor != m->bytes || le32(m->offset) != 0 ||
        le32(m->offset + 4 * m->total) != text_bytes)
        return -1;
    for (uint32_t i = 0; i < m->total; ++i)
        if (le32(m->offset + 4 * i) >= le32(m->offset + 4 * (i + 1)))
            return -1;
    for (uint32_t i = 0; i < m->bytes_to_ids.count; ++i)
        if (le32(row(&m->bytes_to_ids, i)) >= m->base)
            return -1;
    for (uint32_t i = 0; i < m->merges.count; ++i) {
        const uint8_t *r = row(&m->merges, i);
        if (le32(r) >= m->base || le32(r + 4) >= m->base || le32(r + 12) >= m->base)
            return -1;
    }
    return 0;
}

qwen_bpe *qwen_bpe_open(const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return NULL;
    struct stat st;
    if (fstat(fd, &st) || st.st_size <= 0 || st.st_size > 16 * 1024 * 1024) {
        close(fd);
        return NULL;
    }
    size_t size = (size_t)st.st_size;
    void *p = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED)
        return NULL;
    qwen_bpe *model = calloc(1, sizeof(*model));
    if (!model) {
        munmap(p, size);
        return NULL;
    }
    model->map = p;
    model->bytes = size;
    if (layout(model)) {
        qwen_bpe_close(model);
        return NULL;
    }
    return model;
}

void qwen_bpe_close(qwen_bpe *m)
{
    if (m) {
        munmap((void *)m->map, m->bytes);
        free(m);
    }
}

static const uint8_t *find(const table *t, uint32_t a, uint32_t b, int pair)
{
    uint32_t first = 0, end = t->count;
    while (first < end) {
        const uint32_t mid = first + (end - first) / 2;
        const uint8_t *item = row(t, mid);
        const uint32_t x = le32(item), y = pair ? le32(item + 4) : 0;
        if (a == x && b == y)
            return item;
        if (x < a || (x == a && y < b))
            first = mid + 1;
        else
            end = mid;
    }
    return NULL;
}

static uint32_t category(const qwen_bpe *m, uint32_t code)
{
    const table *t = &m->ranges;
    uint32_t first = 0, end = t->count;
    while (first < end) {
        const uint32_t mid = first + (end - first) / 2;
        const uint8_t *item = row(t, mid);
        if (code < le32(item))
            end = mid;
        else if (code > le32(item + 4))
            first = mid + 1;
        else
            return le32(item + 8);
    }
    return 0;
}

static unsigned combining(const qwen_bpe *m, uint32_t code)
{
    const uint8_t *r = find(&m->classes, code, 0, 0);
    return r ? le32(r + 4) : 0;
}

static int parse_utf8(const uint8_t **start, const uint8_t *end, uint32_t *code)
{
    const uint8_t *p = *start;
    uint8_t first = *p++;
    if (first < 0x80) {
        *start = p;
        *code = first;
        return 0;
    }
    unsigned extra;
    uint32_t value, minimum;
    if (first >= 0xc2 && first <= 0xdf) {
        extra = 1; value = first & 31u; minimum = 0x80;
    } else if (first >= 0xe0 && first <= 0xef) {
        extra = 2; value = first & 15u; minimum = 0x800;
    } else if (first >= 0xf0 && first <= 0xf4) {
        extra = 3; value = first & 7u; minimum = 0x10000;
    } else {
        return -1;
    }
    if ((size_t)(end - p) < extra)
        return -1;
    for (; extra; --extra) {
        if ((*p & 0xc0u) != 0x80u)
            return -1;
        value = (value << 6) | (*p++ & 63u);
    }
    if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff))
        return -1;
    *start = p;
    *code = value;
    return 0;
}

static unsigned to_utf8(uint32_t c, uint8_t bytes[4])
{
    if (c < 0x80) {
        bytes[0] = (uint8_t)c;
        return 1;
    }
    if (c < 0x800) {
        bytes[0] = (uint8_t)(0xc0 | (c >> 6));
        bytes[1] = (uint8_t)(0x80 | (c & 63));
        return 2;
    }
    if (c < 0x10000) {
        bytes[0] = (uint8_t)(0xe0 | (c >> 12));
        bytes[1] = (uint8_t)(0x80 | ((c >> 6) & 63));
        bytes[2] = (uint8_t)(0x80 | (c & 63));
        return 3;
    }
    bytes[0] = (uint8_t)(0xf0 | (c >> 18));
    bytes[1] = (uint8_t)(0x80 | ((c >> 12) & 63));
    bytes[2] = (uint8_t)(0x80 | ((c >> 6) & 63));
    bytes[3] = (uint8_t)(0x80 | (c & 63));
    return 4;
}

static size_t normalize(const qwen_bpe *m, const uint8_t *p, const uint8_t *end,
                        uint32_t *c, size_t cap)
{
    size_t n = 0;
    while (p < end) {
        uint32_t code;
        if (parse_utf8(&p, end, &code))
            return SIZE_MAX;
        const uint8_t *d = find(&m->decompositions, code, 0, 0);
        uint32_t count = d ? le32(d + 8) : 1;
        uint32_t off = d ? le32(d + 4) : 0;
        if (count > cap - n || (d && (off > m->nfd.count || count > m->nfd.count - off)))
            return SIZE_MAX;
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t pos = d ? le32(row(&m->nfd, off + i)) : code;
            const unsigned rank = combining(m, pos);
            size_t next = n++;
            if (rank)
                while (next && combining(m, c[next - 1]) > rank) {
                    c[next] = c[next - 1];
                    --next;
                }
            c[next] = pos;
        }
    }
    if (!n)
        return 0;
    size_t used = 1, starter = 0;
    unsigned last = combining(m, c[0]);
    for (size_t i = 1; i < n; ++i) {
        unsigned rank = combining(m, c[i]);
        const uint8_t *item = (last == 0 || last < rank)
            ? find(&m->compose, c[starter], c[i], 1) : NULL;
        if (item) {
            c[starter] = le32(item + 8);
        } else {
            if (!rank)
                starter = used;
            c[used++] = c[i];
            last = rank;
        }
    }
    return used;
}

static int crlf(uint32_t c)
{
    return c == '\r' || c == '\n';
}

static int ascii_fold(uint32_t code, int low)
{
    return code == (uint32_t)low || code == (uint32_t)(low - 'a' + 'A') ||
           (low == 's' && code == 0x017f);
}

/* The pinned Qwen Split regex, including the trailing-space lookahead. */
static size_t next_piece(const uint32_t *c, const uint8_t *tags, size_t size)
{
    const uint8_t first = tags[0];
    if (c[0] == '\'' && size > 1) {
        static const char *const endings[] = {"s", "t", "re", "ve", "m", "ll", "d"};
        for (unsigned i = 0; i < sizeof(endings) / sizeof(endings[0]); ++i) {
            const char *s = endings[i];
            if (ascii_fold(c[1], s[0]) && (!s[1] || (size > 2 && ascii_fold(c[2], s[1]))))
                return s[1] ? 3 : 2;
        }
    }
    size_t pos = 0;
    if (!(first & (LETTER | NUMBER)) && !crlf(c[0]) && size > 1 && (tags[1] & LETTER))
        pos = 1;
    if (tags[pos] & LETTER) {
        while (pos < size && (tags[pos] & LETTER))
            ++pos;
        return pos;
    }
    if (first & NUMBER)
        return 1;
    pos = c[0] == ' ' && size > 1 ? 1 : 0;
    if (!(tags[pos] & (SPACE | LETTER | NUMBER))) {
        while (pos < size && !(tags[pos] & (SPACE | LETTER | NUMBER)))
            ++pos;
        while (pos < size && crlf(c[pos]))
            ++pos;
        return pos;
    }

    pos = 0;
    size_t newline = 0;
    while (pos < size && (tags[pos] & SPACE)) {
        if (crlf(c[pos]))
            newline = pos + 1;
        ++pos;
    }
    if (newline)
        return newline;
    return pos > 1 && pos < size ? pos - 1 : pos ? pos : 1;
}

static size_t merge(const qwen_bpe *m, uint32_t *ids, size_t size)
{
    while (size > 1) {
        uint32_t rank = UINT32_MAX, a = 0, b = 0, joined = 0;
        for (size_t i = 0; i + 1 < size; ++i) {
            const uint8_t *r = find(&m->merges, ids[i], ids[i + 1], 1);
            if (r && le32(r + 8) < rank) {
                rank = le32(r + 8);
                a = ids[i];
                b = ids[i + 1];
                joined = le32(r + 12);
            }
        }
        if (rank == UINT32_MAX)
            break;
        size_t out = 0;
        for (size_t i = 0; i < size; ++i) {
            if (i + 1 < size && ids[i] == a && ids[i + 1] == b) {
                ids[out++] = joined;
                ++i;
            } else {
                ids[out++] = ids[i];
            }
        }
        size = out;
    }
    return size;
}

static int ordinary(const qwen_bpe *m, const uint8_t *first, const uint8_t *end,
                    uint32_t *ids, size_t cap)
{
    if (first == end)
        return 0;
    const size_t footprint = 4 * (size_t)(end - first);
    uint32_t *c = malloc(footprint * sizeof(*c));
    uint8_t *tags = malloc(footprint);
    uint32_t *work = malloc(4 * footprint * sizeof(*work));
    int status = -1;
    if (!c || !tags || !work)
        goto done;
    const size_t count = normalize(m, first, end, c, footprint);
    if (count == SIZE_MAX)
        goto done;
    for (size_t i = 0; i < count; ++i)
        tags[i] = (uint8_t)category(m, c[i]);
    size_t used = 0;
    for (size_t i = 0; i < count;) {
        const size_t width = next_piece(c + i, tags + i, count - i);
        size_t n = 0;
        for (size_t j = i; j < i + width; ++j) {
            uint8_t bytes[4];
            const unsigned length = to_utf8(c[j], bytes);
            for (unsigned k = 0; k < length; ++k)
                work[n++] = le32(row(&m->bytes_to_ids, bytes[k]));
        }
        n = merge(m, work, n);
        if (n > cap - used)
            goto done;
        memcpy(ids + used, work, n * sizeof(*ids));
        used += n;
        i += width;
    }
    status = (int)used;
done:
    free(work);
    free(tags);
    free(c);
    return status;
}

int qwen_bpe_encode(const qwen_bpe *m, const char *text, uint32_t *ids, size_t cap)
{
    if (!m || !text || !ids)
        return -1;
    const uint8_t *start = (const uint8_t *)text;
    const size_t n = strlen(text);
    if (n > MAX_INPUT_BYTES)
        return -1;
    const uint8_t *end = start + n;
    size_t used = 0;
    for (const uint8_t *p = start; p < end; ++p) {
        if (*p != '<')
            continue;
        uint32_t tag = m->total;
        size_t longest = 0;
        for (uint32_t i = m->base; i < m->total; ++i) {
            const uint32_t off = le32(m->offset + i * 4);
            const size_t bytes = le32(m->offset + (i + 1) * 4) - off;
            if (bytes > longest && bytes <= (size_t)(end - p) &&
                !memcmp(p, m->strings + off, bytes)) {
                tag = i;
                longest = bytes;
            }
        }
        if (tag == m->total)
            continue;
        int count = ordinary(m, start, p, ids + used, cap - used);
        if (count < 0 || (size_t)count >= cap - used)
            return -1;
        used += (size_t)count;
        ids[used++] = tag;
        p += longest - 1;
        start = p + 1;
    }
    int remaining = ordinary(m, start, end, ids + used, cap - used);
    return remaining < 0 ? -1 : (int)used + remaining;
}

const char *qwen_bpe_piece(const qwen_bpe *m, uint32_t id, size_t *bytes)
{
    if (!m || !bytes || id >= m->total)
        return NULL;
    const uint32_t offset = le32(m->offset + id * 4u);
    *bytes = id < m->base ? le32(m->offset + (id + 1u) * 4u) - offset : 0;
    return (const char *)m->strings + offset;
}
