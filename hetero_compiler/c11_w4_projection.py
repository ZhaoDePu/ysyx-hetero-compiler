"""Lower tiled, groupwise W4A8 projections into the frozen Z15 YHB ABI."""

from pathlib import Path

from .gpu import Program, _EXIT
from .npu import _z15_tensor
from .pipeline import _schedule_banks, schedule_tiles
from .profile import load_profile
from .yhb import emit_yhb


GROUP = 128
COLUMNS = 16
HALF = 1 << 20
GPU_PAGE = 4096
PACKED = 0x100
UNPACKED = 0x800
ACT = 0x10000
ACC = 0x11000


def unpack_program():
    """Four resident warps expand 32-bit packed loads into two word stores."""
    image = Program(None)
    jobs = []
    for warp in range(4):
        jobs.append({"entry": 4 * len(image.words), "threads": 16, "warp": warp})
        image.li(15, 16 * warp)
        image.jump("start")
    image.label("start")
    image.emit(0xCC3022F3)  # logical lane ID in x5
    image.arithmetic(5, 5, 15)
    image.li(2, UNPACKED)
    image.li(3, GROUP * COLUMNS // 8)
    image.li(4, PACKED)
    image.li(9, 0x00FF00FF)
    image.li(10, 0x0F0F0F0F)
    image.li(11, 0x08080808)
    image.li(12, 30)
    image.label("word")
    image.addi(6, 5, 2, f3=1)
    image.arithmetic(6, 6, 4)
    image.load(7, 6, "int32")
    image.addi(8, 7, 16, f3=5)
    image.addi(14, 5, 3, f3=1)
    image.arithmetic(14, 14, 2)

    def expand(word, displacement):
        # Ignore the other half of the loaded word while expanding these four nibbles.
        image.addi(13, word, 16, f3=1)
        image.addi(word, 13, 16, f3=5)
        image.addi(13, word, 8, f3=1)
        image.arithmetic(word, word, 13, f3=6)
        image.arithmetic(word, word, 9, f3=7)
        image.addi(13, word, 4, f3=1)
        image.arithmetic(word, word, 13, f3=6)
        image.arithmetic(word, word, 10, f3=7)
        image.arithmetic(13, word, 11, f3=7)
        image.arithmetic(13, 13, 12, f7=1)
        image.arithmetic(word, word, 13)
        image.store(word, 14, "int32", displacement)

    expand(7, 0)
    expand(8, 4)
    image.addi(5, 5, 64)
    image.branch(6, 5, 3, "word")  # unsigned less than
    image.emit(_EXIT)
    return image.image(), jobs


def lower_projection(output, weights, *, rows, groups, channels, variant):
    """Compile raw int32 dot products laid out as [G, N/16, rows, 16]."""
    if (variant not in ("B0", "B1", "B2") or not 1 <= rows <= 8 or
            not 1 <= groups <= 38 or channels <= 0 or channels > 4096 or channels % COLUMNS):
        raise ValueError("unsupported frozen Z15 group projection geometry")
    width = GROUP if variant == "B0" else GROUP // 2
    if len(weights) != groups * channels * width:
        raise ValueError("invalid group-major projection weight payload")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    output.joinpath("weight-panels.bin").write_bytes(weights)
    profile = load_profile("fpga_zu15eg_balanced")
    words, jobs = unpack_program()
    output.joinpath("unpack.hex").write_text("".join(f"{w:08x}\n" for w in words), encoding="ascii")
    stages, gpu, commands, tiles = [], [], [], []
    weight_at = 0x10000
    result_at = (weight_at + len(weights) + 31) & ~31
    counts = channels // COLUMNS

    def dma(direction, ddr, scratch, size):
        return {"type": "dma", "direction": direction, "ddr_offset": ddr,
                "scratch_offset": scratch, "bytes": size}

    def launch(stage, device, zone):
        stages.append({"id": stage, "device": device, "zone": zone})
        return {"type": "launch", "stage": stage, "device": device, "zone": zone}

    for group in range(groups):
        for tile in range(counts):
            index = group * counts + tile
            zone = index % 2 if variant == "B2" else 0
            base = zone * HALF
            group_weight = dma("load", weight_at + index * COLUMNS * width,
                               base + (UNPACKED if variant == "B0" else PACKED), COLUMNS * width)
            group_act = dma("load", group * rows * GROUP, base + ACT, rows * GROUP)
            result = dma("store", result_at + index * rows * COLUMNS * 4,
                         base + ACC, rows * COLUMNS * 4)
            chain = [("dma", group_weight, 96), ("dma", group_act, 40)]
            if variant != "B0":
                unpack = f"group{group}/tile{tile}/unpack"
                chain.append(("gpu", launch(unpack, "GPU", zone), 1000))
                page = base // GPU_PAGE
                gpu.append({"id": unpack, "hex": "unpack.hex",
                            "jobs": [{**job, "gpu_job_reg": (job["warp"] << 24) |
                                      (job["threads"] << 16) | page} for job in jobs]})
            dot = f"group{group}/tile{tile}/dot"
            spec = {"op": "Gemm", "m": rows, "n": COLUMNS, "k": GROUP, "tile_k": GROUP,
                    "unsigned_a": False, "packed": False, "shift": 0, "activation": None,
                    "activation_mode": None}
            spans = {"a": (base + ACT,), "b": (base + UNPACKED,), "c": (base + ACC,)}
            commands.append({"stage": dot, "device": "NPU", "writes": _z15_tensor(spec, spans, None)})
            chain.extend((("tensor", launch(dot, "NPU", zone), 600), ("dma", result, 32)))
            tiles.append(chain)

    act_shape = [groups, rows, GROUP]
    result_shape = [groups, counts, rows, COLUMNS]
    manifest = {
        "profile": profile,
        "plan": {"stages": stages, "allocations": {}},
        "graph": {"inputs": ["activation"], "outputs": ["partial"],
                  "tensors": {"activation": {"shape": act_shape, "dtype": "int8"},
                              "partial": {"shape": result_shape, "dtype": "int32"}}},
        "input_bindings": {"activation": {"ddr_offset": 0, "size": groups * rows * GROUP,
                                           "needs_data": True}},
        "output_bindings": {"partial": {"ddr_offset": result_at,
                                         "size": groups * rows * channels * 4}},
        "backends": {"gpu": {"stages": gpu}, "npu": {"commands": commands}},
        "initial_ddr_extents": [{"ddr_offset": weight_at, "bytes": len(weights),
                                 "file": "weight-panels.bin"}],
        "execution": _schedule_banks(tiles) if variant == "B2" else schedule_tiles(tiles, False),
    }
    emit_yhb(output, manifest)
    return {"variant": variant, "rows": rows, "groups": groups, "channels": channels,
            "weight_dma_bytes": len(weights), "activation_dma_bytes": groups * counts * rows * GROUP,
            "output_dma_bytes": groups * rows * channels * 4, "kernels": len(stages),
            "sram_live_payload_bytes": (COLUMNS * GROUP + (COLUMNS * width if variant != "B0" else 0) +
                                        rows * (GROUP + COLUMNS * 4)) * (2 if variant == "B2" else 1),
            "ddr_extent_bytes": result_at + groups * rows * channels * 4}
