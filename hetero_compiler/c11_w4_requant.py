"""Lower the Q24 group scale reduction for a 7-group Z15 LM projection."""

from pathlib import Path
import struct

from .gpu import Program, _EXIT
from .profile import load_profile
from .yhb import emit_yhb


GROUPS, TILE, CHANNELS = 7, 64, 2048
STRIDE = TILE * 4
SCALE = GROUPS * STRIDE
OUT = SCALE * 2
BIAS = OUT + STRIDE


def reduction_image(input_scale, has_bias=False):
    if not 0 < input_scale < 1 << 20:
        raise ValueError("invalid or overflowing Q16 activation scale")
    p = Program(None)
    jobs = []
    for warp in range(4):
        jobs.append({"entry": len(p.words) * 4, "threads": 16, "warp": warp,
                     "gpu_job_reg": (warp << 24) | (16 << 16)})
        p.li(15, 16 * warp)
        p.jump("reduce")
    p.label("reduce")
    p.emit(0xCC3022F3)
    p.arithmetic(5, 5, 15)
    p.addi(5, 5, 2, f3=1)
    p.addi(2, 5, 0)
    p.addi(3, 5, SCALE)
    p.li(1, GROUPS)
    p.li(9, 0)
    p.li(10, 0)
    p.label("group")
    p.load(4, 2, "int32")
    p.load(6, 3, "int32")
    p.arithmetic(7, 4, 6, f7=1)          # group dot * unsigned Q24 scale
    p.arithmetic(8, 4, 6, f3=2, f7=1)
    p.arithmetic(11, 9, 7)
    p.arithmetic(12, 11, 9, f3=3)
    p.arithmetic(10, 10, 8)
    p.arithmetic(10, 10, 12)
    p.addi(9, 11, 0)
    p.addi(2, 2, STRIDE)
    p.addi(3, 3, STRIDE)
    p.addi(1, 1, -1)
    p.branch(1, 1, 0, "group")

    p.li(14, input_scale)
    p.arithmetic(7, 9, 14, f7=1)       # low 64 bits of accumulator * input scale
    p.arithmetic(8, 9, 14, f3=3, f7=1)
    p.arithmetic(10, 10, 14, f7=1)
    p.arithmetic(10, 10, 8)
    p.li(1, 1 << 23)
    p.arithmetic(12, 10, 0, f3=2)
    p.arithmetic(1, 1, 12, f7=32)      # ties away from zero
    p.arithmetic(11, 7, 1)
    p.arithmetic(12, 11, 7, f3=3)
    p.arithmetic(10, 10, 12)
    p.addi(11, 11, 24, f3=5)
    p.addi(10, 10, 8, f3=1)
    p.arithmetic(11, 11, 10, f3=6)
    if has_bias:
        p.li(2, BIAS)
        p.arithmetic(2, 2, 5)
        p.load(4, 2, "int32")
        p.arithmetic(11, 11, 4)
    p.li(2, -(1 << 26))
    p.li(3, 1 << 26)
    p.branch(5, 11, 2, "upper")
    p.addi(11, 2, 0)
    p.label("upper")
    p.branch(5, 3, 11, "store")
    p.addi(11, 3, 0)
    p.label("store")
    p.li(2, OUT)
    p.arithmetic(2, 2, 5)
    p.store(11, 2, "int32")
    p.emit(_EXIT)
    return p.image(), jobs


def lower_reduction(output, scales, input_scale, *, channels=CHANNELS, bias=None):
    """Compile a real tied-head slice and retain the YHB1 transport format."""
    input_bytes = GROUPS * channels * 4
    if (channels <= 0 or channels % TILE or len(scales) != input_bytes or
            (bias is not None and len(bias) != channels * 4)):
        raise ValueError("expected an aligned group scale and bias matrix")
    result_base = 2 * input_bytes + (len(bias) if bias is not None else 0)
    multiplier = struct.unpack(f"<{GROUPS * channels}I", scales)
    bias_values = struct.unpack(f"<{channels}i", bias) if bias is not None else [0] * channels
    max_dot = 8 * 127 * 128
    # A signed nibble and a symmetric int8 activation give this absolute dot bound.
    for col, offset in enumerate(bias_values):
        total = sum(multiplier[col::channels])
        bound = ((max_dot * total * input_scale + (1 << 23)) >> 24) + abs(offset)
        if bound >= 1 << 31:
            raise ValueError("group scales or bias exceed the signed GPU rescale range")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    words, jobs = reduction_image(input_scale, bias is not None)
    (output / "kernel.hex").write_text("".join(f"{w:08x}\n" for w in words), encoding="ascii")
    (output / "scales.bin").write_bytes(scales)
    extents = [{"ddr_offset": input_bytes, "bytes": len(scales), "file": "scales.bin"}]
    if bias is not None:
        (output / "bias.bin").write_bytes(bias)
        extents.append({"ddr_offset": 2 * input_bytes, "bytes": len(bias), "file": "bias.bin"})
    stages, kernels, flow = [], [], []

    def dma(direction, ddr, scratch, size):
        flow.extend(({"type": "dma", "direction": direction, "ddr_offset": ddr,
                      "scratch_offset": scratch, "bytes": size}, {"type": "dma-fence"}))

    for tile in range(channels // TILE):
        for g in range(GROUPS):
            ddr = (g * channels + tile * TILE) * 4
            dma("load", ddr, g * STRIDE, STRIDE)
            dma("load", input_bytes + ddr, SCALE + g * STRIDE, STRIDE)
        if bias is not None:
            dma("load", 2 * input_bytes + tile * STRIDE, BIAS, STRIDE)
        stage = f"tile{tile}/requant"
        stages.append({"id": stage, "device": "GPU", "zone": 0})
        kernels.append({"id": stage, "hex": "kernel.hex", "jobs": jobs})
        flow.extend(({"type": "launch", "stage": stage}, {"type": "fence", "wait_for": stage}))
        dma("store", result_base + tile * STRIDE, OUT, STRIDE)

    manifest = {
        "profile": load_profile("fpga_zu15eg_balanced"),
        "plan": {"stages": stages, "allocations": {}},
        "graph": {"inputs": ["partial"], "outputs": ["logits"],
                  "tensors": {"partial": {"dtype": "int32", "shape": [GROUPS, channels]},
                              "logits": {"dtype": "int32", "shape": [channels]}}},
        "input_bindings": {"partial": {"ddr_offset": 0, "size": input_bytes, "needs_data": True}},
        "output_bindings": {"logits": {"ddr_offset": result_base, "size": channels * 4}},
        "backends": {"gpu": {"stages": kernels}, "npu": {"commands": []}},
        "initial_ddr_extents": extents,
        "execution": flow,
    }
    emit_yhb(output, manifest)
    return output / "program.yhb"
