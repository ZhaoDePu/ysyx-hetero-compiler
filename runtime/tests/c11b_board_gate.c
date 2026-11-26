/* Test sysfs parsing without reading host sysfs or opening any UIO device.
   Link with --wrap=fopen and --wrap=ysyx_hal_linux_open. */
#include "ysyx_rt.h"

#include <stdlib.h>
#include <string.h>

FILE *__real_fopen(const char *filename, const char *mode);

FILE *__wrap_fopen(const char *filename, const char *mode)
{
    if (!strcmp(filename, "/sys/class/fpga_manager/fpga0/state")) {
        const char *path = getenv("YSYX_C11B_TEST_STATE");
        return path ? __real_fopen(path, mode) : NULL;
    }
    return __real_fopen(filename, mode);
}

int __wrap_ysyx_hal_linux_open(ysyx_hal_t *hal, const char *control, const char *dma,
                               const char **error)
{
    (void)hal;
    (void)control;
    (void)dma;
    *error = "C11B_TEST_REACHED_UIO_GUARD";
    return -1;
}
