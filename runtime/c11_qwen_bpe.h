#ifndef YSYX_C11_QWEN_BPE_H
#define YSYX_C11_QWEN_BPE_H

#include <stddef.h>
#include <stdint.h>

typedef struct qwen_bpe qwen_bpe;

qwen_bpe *qwen_bpe_open(const char *path);
void qwen_bpe_close(qwen_bpe *bpe);
/* UTF-8/NFC with the checkpoint's split rules and ranked byte merges. */
int qwen_bpe_encode(const qwen_bpe *bpe, const char *text, uint32_t *ids, size_t cap);
/* Special IDs decode as zero bytes, and invalid IDs return NULL. */
const char *qwen_bpe_piece(const qwen_bpe *bpe, uint32_t id, size_t *bytes);

#endif
