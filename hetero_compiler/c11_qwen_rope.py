"""Lower Qwen half-split Q16 RoPE across seven queries and one key per group."""

from pathlib import Path
import struct

from .gpu import Program, _EXIT
from .profile import load_profile
from .yhb import emit_yhb


GROUP_BYTES = 2048
TRIG_BYTES = 256
TRIG_AT = 0x900
POSITIONS = 10


def make_rope_image():
    image = Program(None)
    jobs = []
    for warp in range(4):
        jobs.append({"entry": 4 * len(image.words), "threads": 16, "warp": warp})
        image.li(15, warp * 16)
        image.jump("kernel")
    image.label("kernel")
    image.emit(0xCC3022F3)
    image.arithmetic(5, 5, 15)
    image.li(1, 256)
    image.li(14, 32768)
    image.label("pair")
    image.addi(2, 5, -32, f3=7)
    image.addi(2, 2, 3, f3=1)
    image.addi(6, 5, 31, f3=7)
    image.addi(3, 6, 2, f3=1)
    image.arithmetic(2, 2, 3)
    image.addi(3, 2, 128)
    image.load(7, 2, "int32")
    image.load(8, 3, "int32")
    image.addi(6, 6, 3, f3=1)
    image.li(9, TRIG_AT)
    image.arithmetic(6, 6, 9)
    image.load(4, 6, "int32")
    image.load(6, 6, "int32", 4)

    def rotate(a, b, destination, subtract):
        # Exact signed 64-bit products, carry and rounded right shift by 16.
        image.arithmetic(9, a, 7, f7=1)
        image.arithmetic(10, a, 7, f3=1, f7=1)
        image.arithmetic(11, b, 8, f7=1)
        image.arithmetic(12, b, 8, f3=1, f7=1)
        if subtract:
            image.arithmetic(13, 9, 11, f3=3)
            image.arithmetic(15, 9, 11, f7=32)
            image.arithmetic(10, 10, 12, f7=32)
            image.arithmetic(10, 10, 13, f7=32)
        else:
            image.arithmetic(15, 9, 11)
            image.arithmetic(13, 15, 9, f3=3)
            image.arithmetic(10, 10, 12)
            image.arithmetic(10, 10, 13)
        image.arithmetic(13, 10, 0, f3=2)
        image.arithmetic(13, 14, 13, f7=32)
        image.arithmetic(11, 15, 13)
        image.arithmetic(13, 11, 15, f3=3)
        image.arithmetic(10, 10, 13)
        image.addi(11, 11, 16, f3=5)
        image.addi(10, 10, 16, f3=1)
        image.arithmetic(11, 11, 10, f3=6)
        image.li(12, -(1 << 26))
        image.branch(5, 11, 12, f"upper-{destination}")
        image.addi(11, 12, 0)
        image.label(f"upper-{destination}")
        image.li(12, 1 << 26)
        image.branch(5, 12, 11, f"store-{destination}")
        image.addi(11, 12, 0)
        image.label(f"store-{destination}")
        image.store(11, destination, "int32")

    rotate(4, 6, 2, True)
    rotate(6, 4, 3, False)
    image.addi(5, 5, 64)
    image.branch(6, 5, 1, "pair")
    image.emit(_EXIT)
    return image.image(), jobs


def lower_rope(output, trig):
    """Compile twenty group steps and return the hardware-ready YHB path."""
    if len(trig) != POSITIONS * TRIG_BYTES:
        raise ValueError("unexpected number of RoPE positions")
    if any(abs(value) > 65536 for (value,) in struct.iter_unpack("<i", trig)):
        raise ValueError("RoPE Q16 coefficient outside unit magnitude")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    words, jobs = make_rope_image()
    output.joinpath("rope.hex").write_text("".join(f"{w:08x}\n" for w in words), encoding="ascii")
    output.joinpath("trig.bin").write_bytes(trig)
    profile = load_profile("fpga_zu15eg_balanced")
    stages, kernels, program = [], [], []
    extent = POSITIONS * 2 * GROUP_BYTES
    trig_ddr = extent
    result = extent + POSITIONS * TRIG_BYTES

    def transfer(direction, ddr, scratch, size):
        program.extend(({"type": "dma", "direction": direction, "ddr_offset": ddr,
                         "scratch_offset": scratch, "bytes": size}, {"type": "dma-fence"}))

    for position in range(POSITIONS):
        for group in range(2):
            index = position * 2 + group
            stage = f"position{position}/gqa{group}/rope"
            # Q and K begin at scratch 0. Trig coefficients sit above the 2 KiB Q/K data.
            transfer("load", index * GROUP_BYTES, 0, GROUP_BYTES)
            transfer("load", trig_ddr + position * TRIG_BYTES, TRIG_AT, TRIG_BYTES)
            stages.append({"id": stage, "device": "GPU", "zone": 0})
            kernels.append({"id": stage, "hex": "rope.hex", "jobs": [
                {**job, "gpu_job_reg": (job["warp"] << 24) | (job["threads"] << 16)} for job in jobs]})
            program.extend(({"type": "launch", "stage": stage}, {"type": "fence", "wait_for": stage}))
            transfer("store", result + index * GROUP_BYTES, 0, GROUP_BYTES)

    tensor = {"dtype": "int32", "shape": [POSITIONS, 2, 8, 64]}
    manifest = {"profile": profile, "plan": {"stages": stages, "allocations": {}},
                "graph": {"inputs": ["qk"], "outputs": ["rotated"],
                          "tensors": {"qk": tensor, "rotated": tensor}},
                "input_bindings": {"qk": {"ddr_offset": 0, "size": extent, "needs_data": True}},
                "output_bindings": {"rotated": {"ddr_offset": result, "size": extent}},
                "backends": {"gpu": {"stages": kernels}, "npu": {"commands": []}},
                "initial_ddr_extents": [{"ddr_offset": trig_ddr, "bytes": len(trig), "file": "trig.bin"}],
                "execution": program}
    emit_yhb(output, manifest)
    return output / "program.yhb"
