#ifndef YSYX_C10_SESSION_CORE_H
#define YSYX_C10_SESSION_CORE_H

#include "ysyx_rt.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ysyx_c10_session ysyx_c10_session_t;

/*
 * All times are milliseconds. PL hardware time is the WALL counter delta
 * divided by the supplied MHz, including command issue gaps. Execute time is
 * the monotonic host interval for issuing commands and waiting for completion.
 * Capture and network time belong to the application. The YOLO extension
 * initially records quantization in t_capture_preproc_ms.
 */
typedef struct {
    double t_capture_preproc_ms;
    double t_h2d_ms;
    double t_pl_hw_ms;
    double t_d2h_post_ms;
    double t_net_tx_ms;
    double t_e2e_ms;
    double t_execute_ms;
    uint64_t static_h2d_bytes;
    uint64_t dynamic_h2d_bytes;
    int hardware_counters_valid;
    ysyx_stats_t stats;
} ysyx_c10_timing_t;

/*
 * The session borrows an immutable program and exclusive HAL until close.
 * Callers serialize frames, and retain the program and HAL for that lifetime.
 * The first successful run uploads the static DDR records. Every run accepts
 * fresh inputs, and outputs can be read only after a successful run.
 * Raw runs record wall ticks and host times. Convert ticks with the actual
 * fabric clock when reporting t_pl_hw_ms.
 */
int ysyx_c10_session_open(const ysyx_program_t *program, const ysyx_hal_t *hal,
                          ysyx_c10_session_t **out, const char **err);
int ysyx_c10_session_run(ysyx_c10_session_t *session,
                         const ysyx_input_t *inputs, size_t input_count,
                         ysyx_stats_t *stats, ysyx_c10_timing_t *timing,
                         FILE *trace, const char **err);
int ysyx_c10_session_read(ysyx_c10_session_t *session, const char *name,
                          void *data, size_t size);
/* Report whether a named output is completely backed by immutable INITIAL. */
int ysyx_c10_session_static_output(const ysyx_c10_session_t *session, const char *name);
void ysyx_c10_session_close(ysyx_c10_session_t *session);

#ifdef __cplusplus
}
#endif
#endif
