#define _POSIX_C_SOURCE 200809L

#include "ysyx_rt.h"
#include "z15_regs.h"
#include "z15_cmdq.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

enum { MAX_PATH = 512 };

typedef struct {
    int control_fd, dma_fd;
    void *control, *ddr;
    size_t page;
    int noncached_ddr;
    int ext_probed;
    uint32_t ext_version, cap_flags;
} linux_hal_t;

static void ordered(void)
{
#if defined(__aarch64__)
    __asm__ __volatile__("dmb osh" ::: "memory");
#else
    __sync_synchronize();
#endif
}

static uint32_t little(uint32_t value)
{
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    return __builtin_bswap32(value);
#else
    return value;
#endif
}

static uint32_t mapped_read(const linux_hal_t *hal, uint32_t reg)
{
    ordered();
    const uint32_t value = little(((volatile uint32_t *)hal->control)[reg / 4u]);
    ordered();
    return value;
}

static void probe_fault(int signal_number)
{
    (void)signal_number;
    _exit(1);
}

static void probe_extension(linux_hal_t *hal)
{
    /* Some legacy shells return SLVERR for an unmapped optional register.
       Keep a synchronous external abort from killing the runtime process. */
    hal->ext_probed = 1;
    int channel[2];
    if (pipe(channel))
        return;
    const pid_t child = fork();
    if (!child) {
        close(channel[0]);
        if (signal(SIGBUS, probe_fault) == SIG_ERR)
            _exit(1);
        uint32_t found[2] = {mapped_read(hal, Z15_EXT_VERSION), 0};
        if (found[0] == Z15_P18_VERSION)
            found[1] = mapped_read(hal, Z15_CAP_FLAGS);
        _exit(write(channel[1], found, sizeof(found)) == sizeof(found) ? 0 : 1);
    }
    close(channel[1]);
    if (child > 0) {
        uint32_t found[2];
        ssize_t n;
        do {
            n = read(channel[0], found, sizeof(found));
        } while (n < 0 && errno == EINTR);
        int status;
        pid_t waited;
        do {
            waited = waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (n == sizeof(found) && waited == child && WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            hal->ext_version = found[0];
            hal->cap_flags = found[1];
        }
    }
    close(channel[0]);
}

static int mmio_read(void *ctx, uint32_t reg, uint32_t *value)
{
    linux_hal_t *hal = ctx;
    if (reg >= Z15_REGISTER_SIZE || (reg & 3u))
        return -1;
    if (reg == Z15_EXT_VERSION || reg == Z15_CAP_FLAGS) {
        if (!hal->ext_probed)
            probe_extension(hal);
        *value = reg == Z15_EXT_VERSION ? hal->ext_version : hal->cap_flags;
    } else {
        *value = mapped_read(hal, reg);
    }
    return 0;
}

static int mmio_write(void *ctx, uint32_t reg, uint32_t value)
{
    linux_hal_t *hal = ctx;
    if (reg >= Z15_REGISTER_SIZE || (reg & 3u))
        return -1;
    ordered();
    ((volatile uint32_t *)hal->control)[reg / 4u] = little(value);
    ordered();
    return 0;
}

static int allowed_ddr(uint32_t offset, size_t length)
{
    return offset <= Z15_POOL_SIZE && length <= Z15_POOL_SIZE - offset;
}

static int ddr_write(void *ctx, uint32_t offset, const void *input, size_t length)
{
    linux_hal_t *hal = ctx;
    const uint8_t *data = input;
    if (!allowed_ddr(offset, length))
        return -1;
    volatile uint8_t *ddr = hal->ddr;
    size_t i = 0;
    while (i < length && ((offset + i) & 3u)) {
        ddr[offset + i] = data[i];
        ++i;
    }
    /* Device mappings require aligned accesses. Keep the volatile writes
       scalar and use memcpy only on the ordinary process buffer. */
    while (length - i >= sizeof(uint32_t)) {
        uint32_t word;
        memcpy(&word, data + i, sizeof(word));
        *(volatile uint32_t *)(ddr + offset + i) = word;
        i += sizeof(word);
    }
    for (; i < length; ++i)
        ddr[offset + i] = data[i];
    ordered();
    return 0;
}

static int ddr_read(void *ctx, uint32_t offset, void *output, size_t length)
{
    linux_hal_t *hal = ctx;
    uint8_t *data = output;
    if (!allowed_ddr(offset, length))
        return -1;
    volatile const uint8_t *ddr = hal->ddr;
    ordered();
    size_t i = 0;
    while (i < length && ((offset + i) & 3u)) {
        data[i] = ddr[offset + i];
        ++i;
    }
    while (length - i >= sizeof(uint32_t)) {
        const uint32_t word = *(volatile const uint32_t *)(ddr + offset + i);
        memcpy(data + i, &word, sizeof(word));
        i += sizeof(word);
    }
    for (; i < length; ++i)
        data[i] = ddr[offset + i];
    return 0;
}

static int cache_sync(void *ctx, uint32_t offset, size_t length, int for_host)
{
    linux_hal_t *hal = ctx;
    if (!allowed_ddr(offset, length))
        return -1;
    if (!length)
        return 0;
    if (hal->noncached_ddr) {
        ordered();
        return 0;
    }
    const size_t start = offset - offset % hal->page;
    const size_t extent = offset - start + length;
    const int flags = MS_SYNC | (for_host ? MS_INVALIDATE : 0);
    ordered();
    int rc = msync((uint8_t *)hal->ddr + start, extent, flags);
    const int failure = errno;
    ordered();
    /* Physical UIO is mapped noncached. Some UIO drivers have no fsync. */
    if (rc && failure != EINVAL && failure != ENOSYS && failure != EOPNOTSUPP)
        return -1;
    return 0;
}

static void close_uio(void *ctx)
{
    linux_hal_t *hal = ctx;
    if (hal->ddr != MAP_FAILED)
        munmap(hal->ddr, Z15_POOL_SIZE);
    if (hal->control != MAP_FAILED)
        munmap(hal->control, Z15_REGISTER_SIZE);
    if (hal->dma_fd >= 0)
        close(hal->dma_fd);
    if (hal->control_fd >= 0)
        close(hal->control_fd);
    free(hal);
}

static int is_uio(const char *name)
{
    if (strncmp(name, "uio", 3) || !isdigit((unsigned char)name[3]))
        return 0;
    for (name += 4; *name; ++name)
        if (!isdigit((unsigned char)*name))
            return 0;
    return 1;
}

static int sysfs(const char *uio, const char *attribute, char output[MAX_PATH])
{
    char path[MAX_PATH];
    int n = snprintf(path, sizeof(path), "/sys/class/uio/%s/%s", uio, attribute);
    if (n < 0 || (size_t)n >= sizeof(path))
        return -1;
    FILE *file = fopen(path, "r");
    if (!file)
        return -1;
    int rc = fgets(output, MAX_PATH, file) ? 0 : -1;
    if (fclose(file))
        rc = -1;
    if (!rc)
        output[strcspn(output, "\r\n")] = 0;
    return rc;
}

static int sysfs_number(const char *uio, const char *attribute, uint64_t *value)
{
    char data[MAX_PATH], *end;
    if (sysfs(uio, attribute, data))
        return -1;
    errno = 0;
    unsigned long long v = strtoull(data, &end, 0);
    if (errno || end == data || *end)
        return -1;
    *value = v;
    return 0;
}

static int named_match(const char *name, int dma)
{
    if (!strstr(name, "ysyx") && !strstr(name, "z15"))
        return 0;
    return dma ? (strstr(name, "dma") || strstr(name, "pool")) :
                 (!strstr(name, "dma") && !strstr(name, "pool"));
}

static int valid_mapping(const char *uio, uint32_t base, uint32_t length)
{
    uint64_t addr, size, offset;
    return !sysfs_number(uio, "maps/map0/addr", &addr) && addr == base &&
           !sysfs_number(uio, "maps/map0/size", &size) && size >= length &&
           !sysfs_number(uio, "maps/map0/offset", &offset) && offset == 0;
}

static int physical_generic_uio(const char *device)
{
    char path[MAX_PATH], link[MAX_PATH];
    const int n = snprintf(path, sizeof(path), "/sys/class/uio/%s/device/driver", device + 5);
    if (n < 0 || (size_t)n >= sizeof(path))
        return 0;
    const ssize_t length = readlink(path, link, sizeof(link) - 1u);
    if (length < 1 || (size_t)length >= sizeof(link) - 1u)
        return 0;
    link[length] = 0;
    const char *name = strrchr(link, '/');
    /* uio_pdrv_genirq uses UIO_MEM_PHYS for a platform IORESOURCE_MEM
       map, which the UIO core maps with pgprot_noncached. */
    return !strcmp(name ? name + 1 : link, "uio_pdrv_genirq");
}

static int device_path(char output[MAX_PATH], const char *specified, int dma)
{
    const uint32_t base = dma ? Z15_POOL_BASE : Z15_REG_BASE;
    const uint32_t length = dma ? Z15_POOL_SIZE : Z15_REGISTER_SIZE;
    if (specified && specified[0] == '/') {
        if (strncmp(specified, "/dev/", 5))
            return -1;
        const char *base_name = specified + 5;
        if (!is_uio(base_name) || !valid_mapping(base_name, base, length) || strlen(specified) >= MAX_PATH)
            return -1;
        strcpy(output, specified);
        return 0;
    }
    DIR *dir = opendir("/sys/class/uio");
    if (!dir)
        return -1;
    struct dirent *entry;
    int best = 0, tied = 0;
    while ((entry = readdir(dir))) {
        char name[MAX_PATH];
        if (!is_uio(entry->d_name) || sysfs(entry->d_name, "name", name) ||
            !valid_mapping(entry->d_name, base, length))
            continue;
        int score = specified ? (!strcmp(specified, name) ? 2 : 0) : (named_match(name, dma) ? 2 : 1);
        if (score && score == best) {
            tied = 1;
        } else if (score > best) {
            int written = snprintf(output, MAX_PATH, "/dev/%s", entry->d_name);
            if (written < 0 || written >= MAX_PATH)
                continue;
            best = score;
            tied = 0;
        }
    }
    closedir(dir);
    return best && !tied ? 0 : -1;
}

int ysyx_hal_linux_open(ysyx_hal_t *out, const char *control_uio, const char *dma_uio, const char **error)
{
    char control[MAX_PATH], dma[MAX_PATH];
    const char *reason = "Z15 sysfs UIO name or map0 physical range does not match";
    if (device_path(control, control_uio, 0) || device_path(dma, dma_uio, 1))
        goto failed;
    long page = sysconf(_SC_PAGESIZE);
    if (page < 1) {
        reason = "cannot determine Linux page size";
        goto failed;
    }
    linux_hal_t *hal = malloc(sizeof(*hal));
    if (!hal) {
        reason = "out of memory opening UIO";
        goto failed;
    }
    *hal = (linux_hal_t){.control_fd = -1, .dma_fd = -1, .control = MAP_FAILED,
                       .ddr = MAP_FAILED, .page = (size_t)page, .noncached_ddr = physical_generic_uio(dma)};
    hal->control_fd = open(control, O_RDWR | O_SYNC | O_CLOEXEC);
    if (hal->control_fd < 0 || flock(hal->control_fd, LOCK_EX | LOCK_NB)) {
        reason = "cannot open or exclusively lock Z15 control UIO";
        goto close_failed;
    }
    hal->dma_fd = open(dma, O_RDWR | O_SYNC | O_CLOEXEC);
    if (hal->dma_fd < 0) {
        reason = "cannot open reserved Z15 DDR UIO";
        goto close_failed;
    }
    hal->control = mmap(NULL, Z15_REGISTER_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, hal->control_fd, 0);
    hal->ddr = mmap(NULL, Z15_POOL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, hal->dma_fd, 0);
    if (hal->control == MAP_FAILED || hal->ddr == MAP_FAILED) {
        reason = "cannot mmap the Z15 control aperture or DDR pool";
        goto close_failed;
    }
    *out = (ysyx_hal_t){hal, mmio_read, mmio_write, ddr_write, ddr_read, cache_sync, NULL, close_uio};
    return 0;

close_failed:
    close_uio(hal);
failed:
    if (error)
        *error = reason;
    return -1;
}
