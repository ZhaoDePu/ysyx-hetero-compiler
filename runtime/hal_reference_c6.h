#ifndef YSYX_HAL_REFERENCE_C6_H
#define YSYX_HAL_REFERENCE_C6_H

#include "ysyx_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Exact C6 functional replay. Hardware cycle counters are unavailable and
 * read as zero. The software reference is synchronous and uses host RAM. */
int ysyx_hal_reference_p17_open(ysyx_hal_t *, const char **);
int ysyx_hal_reference_c6_open(ysyx_hal_t *, const char **error);

#ifdef __cplusplus
}
#endif

#endif
