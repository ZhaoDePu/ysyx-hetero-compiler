#ifndef YSYX_Z15_REGS_H
#define YSYX_Z15_REGS_H

#include <stdint.h>

/* Frozen Z15 MMIO ABI 1.0, 4d410d72508a5822bc5c8014981e5b55ef8faa56. */
#define Z15_MAGIC 0x00u
#define Z15_VERSION 0x04u
#define Z15_SCRATCH 0x08u
#define Z15_STATUS 0x0cu
#define Z15_CTRL 0x10u
#define Z15_ERROR 0x14u
#define Z15_GPU_SLOTS 0x18u
#define Z15_DMA_BASE 0x20u
#define Z15_DMA_OFFSET 0x24u
#define Z15_DMA_SCRATCH 0x28u
#define Z15_DMA_BYTES 0x2cu
#define Z15_DMA_FLAGS 0x30u
#define Z15_GPU_PC 0x40u
#define Z15_GPU_JOB 0x44u
#define Z15_NPU_DIMS 0x60u
#define Z15_NPU_K 0x64u
#define Z15_NPU_A 0x68u
#define Z15_NPU_B 0x6cu
#define Z15_NPU_C 0x70u
#define Z15_NPU_BIAS 0x74u
#define Z15_NPU_FLAGS 0x78u
#define Z15_NPU_SHAPE 0x7cu
#define Z15_NPU_SLICE 0x80u
#define Z15_NPU_PARTIAL 0x84u
#define Z15_ATTN_CONTROL 0x88u
#define Z15_ATTN_V 0x8cu
#define Z15_ATTN_SCORES 0x90u
#define Z15_ATTN_P 0x94u
#define Z15_ATTN_POSITION 0x98u
#define Z15_EW_A 0xa0u
#define Z15_EW_B 0xa4u
#define Z15_EW_C 0xa8u
#define Z15_EW_D 0xacu
#define Z15_EW_LENGTH 0xb0u
#define Z15_EW_SHAPE 0xb4u
#define Z15_EW_CHANNELS 0xb8u
#define Z15_EW_FLAGS 0xbcu
#define Z15_EW_EPS 0xc0u
#define Z15_SCALE_INDEX 0xc4u
#define Z15_SCALE_DATA 0xc8u
#define Z15_ACTIVATION 0xccu
#define Z15_NPU_CYCLES 0xe0u
#define Z15_GPU_CYCLES 0xe8u
#define Z15_WALL_CYCLES 0xf0u
#define Z15_OVERLAP_CYCLES 0xf8u
#define Z15_PROGRAM 0x4000u
#define Z15_PROGRAM_BYTES 0x4000u
#define Z15_REG_BASE UINT32_C(0x80000000)
#define Z15_REGISTER_SIZE UINT32_C(0x10000)
#define Z15_MAGIC_VALUE UINT32_C(0x5a313548)
#define Z15_VERSION_VALUE UINT32_C(0x0c0c0100)
#define Z15_VERSION_COMPAT_MASK UINT32_C(0xffffff00)
#define Z15_POOL_BASE UINT32_C(0x40000000)
#define Z15_POOL_SIZE UINT32_C(0x20000000)
#define Z15_HALF_BYTES UINT32_C(0x100000)
#define Z15_SCRATCH_SIZE UINT32_C(0x200000)
#define Z15_DMA_BUSY 2u
#define Z15_DMA_DONE 4u
#define Z15_NPU_BUSY 8u
#define Z15_NPU_DONE 16u
#define Z15_GPU_BUSY 32u
#define Z15_GPU_DONE 64u
#define Z15_HAS_ERROR 128u
#define Z15_LOCKED 1u
#define Z15_START_DMA 4u
#define Z15_START_NPU 8u
#define Z15_START_GPU 16u
#define Z15_START_ATTN 32u
#define Z15_START_EW 64u
#define Z15_WRITE_SCALE 128u
#define Z15_WRITE_ACT 256u
#define Z15_ACTIONS 0x1fcu
#define Z15_GPU_PENDING 0x10u
#define Z15_GPU_SLOT_FAULTS 0xf000u
#define Z15_ERR_DMA_DESC 1u

#endif
