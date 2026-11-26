"""Compile native GQA attention for real Qwen Layer-0 byte-domain operands."""

from pathlib import Path

from .npu import _z15_ewise
from .profile import load_profile
from .yhb import emit_yhb


POSITIONS, GROUPS, QUERIES, WIDTH, PITCH = 10, 2, 7, 64, 12
HALF = 1 << 20
Q, K, V, SCORE, PROB, OUTPUT, NEW_V = 0, 0x400, 0x800, 0xC00, 0xD00, 0xE00, 0x1200
ROW_BYTES = (QUERIES + 2) * WIDTH
STATES = (7, 8, 9)
CACHE_BYTES = PITCH * WIDTH
INPUT_BYTES = POSITIONS * GROUPS * ROW_BYTES
ATTN_BYTES = POSITIONS * GROUPS * QUERIES * WIDTH
KV_BYTES = len(STATES) * GROUPS * 2 * CACHE_BYTES


def compile_gqa(output, exp_shift):
    if not 0 <= exp_shift < 32:
        raise ValueError("invalid native attention exponent shift")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    output.joinpath("zero.bin").write_bytes(bytes(CACHE_BYTES))
    profile = load_profile("fpga_zu15eg_balanced")
    stages, native, flow = [], [], []
    zero, attention, cache = INPUT_BYTES, INPUT_BYTES + CACHE_BYTES, INPUT_BYTES + CACHE_BYTES + ATTN_BYTES

    def dma(direction, ddr, sram, length):
        flow.extend(({"type": "dma", "direction": direction, "ddr_offset": ddr,
                      "scratch_offset": sram, "bytes": length}, {"type": "dma-fence"}))

    def stage(name, group, device, writes):
        stages.append({"id": name, "zone": group, "device": device})
        native.append({"stage": name, "device": device, "writes": writes})
        flow.extend(({"type": "launch", "stage": name}, {"type": "fence", "wait_for": name}))

    for group in range(GROUPS):
        base = group * HALF
        dma("load", zero, base + K, CACHE_BYTES)
        dma("load", zero, base + V, CACHE_BYTES)

    for pos in range(POSITIONS):
        for group in range(GROUPS):
            base = group * HALF
            at = (pos * GROUPS + group) * ROW_BYTES
            name = f"pos{pos}/group{group}"
            dma("load", at, base + Q, QUERIES * WIDTH)
            dma("load", at + QUERIES * WIDTH, base + K + pos * WIDTH, WIDTH)
            dma("load", at + (QUERIES + 1) * WIDTH, base + NEW_V, WIDTH)
            transpose = {"op": 9, "h": 1, "ca": WIDTH, "cb": PITCH,
                         "num_bytes": 1 + (WIDTH - 1) * PITCH,
                         "activation": None, "activation_mode": None}
            stage(name + "/append-V", group, "VPU",
                  _z15_ewise(transpose, {"a": (base + NEW_V,), "c": (base + V + pos,)}))
            for head in range(QUERIES):
                x = head * WIDTH
                writes = [
                    (0x60, PITCH | WIDTH << 16), (0x64, WIDTH | WIDTH << 16),
                    (0x68, base + Q + x), (0x6C, base + K), (0x70, base + OUTPUT + x),
                    (0x88, 4 | 2 | 1 << 8 | exp_shift << 20), (0x8C, base + V),
                    (0x90, base + SCORE), (0x94, base + PROB), (0x98, pos),
                    (0x10, 1 << 5),
                ]
                stage(name + f"/head{head}", group, "NPU", writes)
            dma("store", attention + (pos * GROUPS + group) * QUERIES * WIDTH,
                base + OUTPUT, QUERIES * WIDTH)
            if pos in STATES:
                where = cache + (STATES.index(pos) * GROUPS + group) * 2 * CACHE_BYTES
                dma("store", where, base + K, CACHE_BYTES)
                dma("store", where + CACHE_BYTES, base + V, CACHE_BYTES)

    manifest = {
        "profile": profile, "plan": {"stages": stages, "allocations": {}},
        "graph": {"inputs": ["qkv"], "outputs": ["context", "kv"],
                  "tensors": {"qkv": {"dtype": "int8", "shape": [POSITIONS, GROUPS, QUERIES + 2, WIDTH]},
                              "context": {"dtype": "int8", "shape": [POSITIONS, GROUPS, QUERIES, WIDTH]},
                              "kv": {"dtype": "int8", "shape": [len(STATES), GROUPS, 2, CACHE_BYTES]}}},
        "input_bindings": {"qkv": {"ddr_offset": 0, "size": INPUT_BYTES, "needs_data": True}},
        "output_bindings": {"context": {"ddr_offset": attention, "size": ATTN_BYTES},
                            "kv": {"ddr_offset": cache, "size": KV_BYTES}},
        "backends": {"gpu": {"stages": []}, "npu": {"commands": native}},
        "initial_ddr_extents": [{"ddr_offset": zero, "bytes": CACHE_BYTES, "file": "zero.bin"}],
        "execution": flow,
    }
    emit_yhb(output, manifest)
    return output / "program.yhb"
