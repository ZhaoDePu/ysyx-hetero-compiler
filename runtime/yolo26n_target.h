#ifndef YSYX_YOLO26N_TARGET_H
#define YSYX_YOLO26N_TARGET_H

#include "../include/ysyx_yolo26n_api.h"

/* Phase 17 deployment requires an explicitly selected, known PL image. */
ysyx_yolo26n_ctx_t *ysyx_yolo26n_open_p17(const char *path, int use_reference);

#endif
