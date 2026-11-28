#ifndef YSYX_Z15_EXEC_OPTION_H
#define YSYX_Z15_EXEC_OPTION_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline int z15_exec_option(int *argc, char **argv)
{
    const char *mode = NULL;
    int out = 1;
    for (int i = 1; i < *argc; ++i) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--")) {
            while (i < *argc)
                argv[out++] = argv[i++];
            break;
        }
        if (!strcmp(arg, "--exec-stats")) {
            if (setenv("YSYX_CMDQ_STATS", "1", 1))
                return -1;
            continue;
        }
        if (!strcmp(arg, "--exec-mode")) {
            if (++i >= *argc)
                goto bad;
            mode = argv[i];
        } else if (!strncmp(arg, "--exec-mode=", 12)) {
            mode = arg + 12;
        } else {
            argv[out++] = argv[i];
            continue;
        }
        if (strcmp(mode, "auto") && strcmp(mode, "cmdq") && strcmp(mode, "mmio") &&
            strcmp(mode, "p18b") && strcmp(mode, "p19a"))
            goto bad;
    }
    if (mode && setenv("YSYX_EXEC_MODE", mode, 1))
        return -1;
    argv[out] = NULL;
    *argc = out;
    return 0;
bad:
    fputs("--exec-mode requires auto, p18b, p19a, cmdq or mmio\n", stderr);
    return -1;
}

#endif
