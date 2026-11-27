#ifndef VALENCE_C13_QWEN_LAYOUT_H
#define VALENCE_C13_QWEN_LAYOUT_H

enum {
    C13_CONTROL = 0x100,
    C13_MAX = 0x140,
    C13_TOTAL = 0x180,
    C13_LEFT = 0x1000,
    C13_RIGHT = 0x6000,
    C13_OUT = 0xb000,
    /* The sums are transient and share the frozen NPU-C region while
     * the NPU is idle. The persistent LUT sits beyond the NPU-C footprint. */
    C13_SUMS = 0xfc00,
    C13_EXP = 0x12000,
    C13_CACHE = 0x16100,
    C13_SCORES = 0x1f100,
    C13_TILE = 128,
    C13_DEPTH = 2048,
    C13_HEAD = 64,
    C13_KV_HEADS = 2,
    C13_QUERIES = 14,
    C13_HIDDEN = 896,
    C13_INTER = 4864,
    C13_Q16 = 65536,
    C13_Q24 = 16777216,
    C13_LIMIT = 67108864,
    C13_EXP_COUNT = 4097,
    C13_KV_HEAD_BYTES = 68,
    C13_KV_BYTES = 272,
    C13_KV_SLOT_BYTES = 288
};

#endif
