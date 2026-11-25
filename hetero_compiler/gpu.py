"""Lower row-major C1 tensors to YSYX GPU device images without a cross toolchain."""
from math import erf, exp, prod, sqrt
from pathlib import Path
from struct import pack
import json


_WIDTH = {"int8": 1, "uint8": 1, "int32": 4}
_LOAD = {"int8": 0, "uint8": 4, "int32": 2}
_VARIANTS = {"v6", "g3", "g4"}
_LEGACY_SILU_KNOTS = (0, 0, 1, 2, 5, 12, 31, 69, 128, 187, 225, 244, 251, 254, 255, 256, 256)
_EXIT = 0x00100073
_SCALAR_ENABLE = 0x5340000F


def supports(stage, graph, profile):
    """Check the mathematical contract; allocation and instruction limits are checked at emit time."""
    config = profile["gpu"]
    if config.get("variant") not in _VARIANTS:
        return False
    parts = stage.get("fused")
    if not parts:
        return _supports_node(stage, graph)
    if not (2 <= len(parts) <= 4 and parts[0]["op"] == "Gemm" and
            parts[-1]["output"] == stage["output"] and
            all(_supports_node(node, graph) for node in parts)):
        return False
    phases = {"Add": 0, "Requant": 1, "Relu": 2, "SiLU": 2, "GELU": 2}
    produced = {node["output"] for node in parts}
    tail, last_phase = parts[0]["output"], -1
    external = set(parts[0]["inputs"])
    for node in parts[1:]:
        phase = phases.get(node["op"], -1)
        if (phase <= last_phase or node["inputs"][0] != tail or
                any(name in produced for name in node["inputs"][1:]) or
                tail in graph.get("outputs", [])):
            return False
        external.update(node["inputs"][1:])
        tail, last_phase = node["output"], phase
    if external != set(stage["inputs"]) or external & produced:
        return False
    # Reject fusion when the external operands alone cannot fit in one GPU page.
    memory = profile.get("sram", {})
    alignment = memory.get("alignment", 1)
    total = sum((prod(graph["tensors"][name]["shape"]) * _WIDTH[graph["tensors"][name]["dtype"]]
                 + alignment - 1) // alignment * alignment for name in {*external, tail})
    return total <= config["window_bytes"] - memory.get("metadata_reserve", 0)


def _supports_node(stage, graph):
    tensors = graph["tensors"]
    op = stage["op"]
    inputs = [tensors[name] for name in stage["inputs"]]
    out = tensors[stage["output"]]
    attrs = stage.get("attrs", {})
    shape = out["shape"]
    if any(t["dtype"] not in _WIDTH for t in [out, *inputs]):
        return False
    if attrs.get("narrowing", "saturate") not in {"wrap", "saturate"}:
        return False
    if op in {"Relu", "Requant", "SiLU", "GELU", "GatedBranch"}:
        if len(inputs) != 1 or inputs[0]["shape"] != shape:
            return False
        if op in {"SiLU", "GELU"}:
            mode = attrs.get("mode")
            return (inputs[0]["dtype"] == out["dtype"] == "int8"
                    and (mode is None or (op == "SiLU" and mode == "legacy_q4_knots")))
        if op == "GatedBranch":
            dtype = inputs[0]["dtype"]
            lo, hi = (-128, 127) if dtype == "int8" else (0, 255)
            threshold, rounds = attrs.get("threshold", 0), attrs.get("rounds", 64)
            return (dtype in {"int8", "uint8"} and dtype == out["dtype"] and
                    type(threshold) is int and lo <= threshold <= hi and
                    type(rounds) is int and 64 <= rounds <= 4096)
        if op == "Requant":
            return (type(attrs.get("shift", 0)) is int and 0 <= attrs.get("shift", 0) <= 31
                    and all(type(attrs.get(a, default)) is int
                            and -(1 << 31) <= attrs.get(a, default) < 1 << 31
                            for a, default in (("multiplier", 1), ("zero_point", 0))))
        return True
    if op in {"Add", "Mul"}:
        if len(inputs) != 2 or inputs[0]["shape"] != shape or not shape:
            return False
        if attrs.get("broadcast") == "last":
            return op == "Add" and inputs[1]["shape"] == [shape[-1]]
        return attrs.get("broadcast", "none") == "none" and inputs[1]["shape"] == shape
    if op == "Gemm":
        if (len(inputs) not in {2, 3} or len(shape) != 2 or
                len(inputs[0]["shape"]) != 2 or len(inputs[1]["shape"]) != 2):
            return False
        a, b = (inp["shape"] for inp in inputs[:2])
        transpose = attrs.get("transpose_b", False)
        if type(transpose) is not bool:
            return False
        contract, columns = b[::-1] if transpose else b
        return (a[1] == contract and shape == [a[0], columns] and
                (len(inputs) == 2 or inputs[2]["shape"] == [columns]))
    return False


class Program:
    """RV32EM fixed-width instruction builder with local branch fixups."""

    def __init__(self, scalar_register):
        self.words = []
        self.labels = {}
        self.fixups = []
        self.scalar_register = scalar_register
        self.namespace = ""
        self.pair_candidates = 0

    def emit(self, word):
        self.words.append(word)

    def label(self, name):
        self.labels[self.namespace + name] = 4 * len(self.words)

    def addi(self, dst, a, imm, f3=0):
        self.emit(((imm & 0xFFF) << 20) | (a << 15) | (f3 << 12) | (dst << 7) | 0x13)

    def li(self, dst, value):
        lo = (value + 2048) % 4096 - 2048
        if value != lo:
            self.emit(((value - lo) & 0xFFFFF000) | (dst << 7) | 0x37)
            if lo:
                self.addi(dst, dst, lo)
        else:
            self.addi(dst, 0, lo)

    def arithmetic(self, dst, a, b, f3=0, f7=0):
        self.emit((f7 << 25) | (b << 20) | (a << 15) | (f3 << 12) | (dst << 7) | 0x33)

    def branch(self, f3, a, b, label):
        self.fixups.append((len(self.words), self.namespace + label))
        self.emit((b << 20) | (a << 15) | (f3 << 12) | 0x63)

    def jump(self, label):
        self.fixups.append((len(self.words), self.namespace + label))
        self.emit(0x6F)

    def load(self, dst, addr, dtype):
        self.emit((addr << 15) | (_LOAD[dtype] << 12) | (dst << 7) | 0x03)

    def store(self, src, addr, dtype):
        access = 2 if dtype == "int32" else 0
        self.emit((src << 20) | (addr << 15) | (access << 12) | 0x23)

    def address(self, dst, index, base, dtype, temp=9):
        self.li(temp, base)
        self.indexed(dst, index, temp, dtype)

    def indexed(self, dst, index, base_register, dtype):
        if dtype == "int32":
            self.addi(dst, index, 2, f3=1)
            index = dst
        self.arithmetic(dst, index, base_register)

    def independent_constant(self, dst, value):
        # Appears immediately after a vector load or ALU op that does not read dst.
        # The scalar constant has no incoming RAW dependency on that vector op.
        reg = self.scalar_register or dst
        if self.scalar_register:
            self.pair_candidates += 1
        self.li(reg, value)
        return reg

    def image(self):
        words = self.words.copy()
        for index, label in self.fixups:
            word = words[index]
            delta = self.labels[label] - 4 * index
            if word & 127 == 0x63:
                if not -4096 <= delta < 4096:
                    raise ValueError("GPU branch target outside signed 13-bit range")
                imm = delta & 0x1FFF
                encoded = (((imm >> 12) & 1) << 31) | (((imm >> 5) & 63) << 25)
                encoded |= (((imm >> 1) & 15) << 8) | (((imm >> 11) & 1) << 7)
            else:
                if not -(1 << 20) <= delta < (1 << 20):
                    raise ValueError("GPU jump target outside signed 21-bit range")
                imm = delta & 0x1FFFFF
                encoded = (((imm >> 20) & 1) << 31) | (((imm >> 1) & 1023) << 21)
                encoded |= (((imm >> 11) & 1) << 20) | (((imm >> 12) & 255) << 12)
            words[index] = word | encoded
        return words


def _clip(p, dtype, mode, materialized):
    if dtype == "int32":
        return
    if mode == "wrap":
        if not materialized:
            if dtype == "int8":
                p.addi(7, 7, 24, f3=1)
                p.addi(7, 7, 24 | 0x400, f3=5)
            else:
                p.addi(7, 7, 255, f3=7)
        return
    low, high = (-128, 127) if dtype == "int8" else (0, 255)
    p.li(8, low)
    p.branch(5, 7, 8, "clip_low_done")
    p.addi(7, 8, 0)
    p.label("clip_low_done")
    p.li(8, high)
    p.branch(5, 8, 7, "clip_high_done")
    p.addi(7, 8, 0)
    p.label("clip_high_done")


def _activation(p, op, mode):
    # Build a compact tree of comparisons over runs of equal Q4 outputs.
    runs = []
    for x in range(-128, 128):
        if mode == "legacy_q4_knots":
            knot = ((x + 8) >> 4) + 8
            scaled = abs(x) * _LEGACY_SILU_KNOTS[knot] // 256
            value = -scaled if x < 0 else scaled
        else:
            q = x / 16
            y = q / (1 + exp(-q)) if op == "SiLU" else q * (1 + erf(q / sqrt(2))) / 2
            value = min(127, max(-128, round(16 * y)))
        if not runs or runs[-1][1] != value:
            runs.append((x, value))

    def tree(first, end):
        if end - first == 1:
            p.li(7, runs[first][1])
            p.jump("activation_end")
            return
        mid = (first + end) // 2
        upper = f"activation_{first}_{mid}"
        p.li(8, runs[mid][0])
        p.branch(5, 7, 8, upper)
        tree(first, mid)
        p.label(upper)
        tree(mid, end)

    tree(0, len(runs))
    p.label("activation_end")


def _gemm(p, inputs, offsets, transpose):
    a, b = inputs[:2]
    da, db = a["dtype"], b["dtype"]
    contract, columns = b["shape"][::-1] if transpose else b["shape"]
    # x5 is the global output element. Quotient picks A's row, remainder B's column.
    p.li(14, columns)
    p.arithmetic(6, 5, 14, f3=5, f7=1)
    p.arithmetic(8, 5, 14, f3=7, f7=1)
    p.li(11, contract)
    p.arithmetic(6, 6, 11, f7=1)
    p.address(9, 6, offsets[0], da, temp=15)
    column_offset = 8
    if transpose:
        p.arithmetic(6, 8, 11, f7=1)
        column_offset = 6
    p.address(10, column_offset, offsets[1], db, temp=15)
    p.li(14, (1 if transpose else columns) * _WIDTH[db])
    if len(inputs) == 3:
        bias = inputs[2]["dtype"]
        p.address(15, 8, offsets[2], bias, temp=12)
        p.load(7, 15, bias)
    else:
        p.li(7, 0)
    p.label("dot")
    p.load(12, 9, da)
    p.load(13, 10, db)
    p.arithmetic(12, 12, 13, f7=1)
    p.arithmetic(7, 7, 12)
    p.addi(9, 9, _WIDTH[da])
    p.arithmetic(10, 10, 14)
    p.addi(11, 11, -1)
    p.branch(1, 11, 0, "dot")


def _gated(p, attrs):
    def rotate(dst, left):
        p.addi(13, dst, left, f3=1)
        p.addi(dst, dst, 32 - left, f3=5)
        p.arithmetic(dst, dst, 13, f3=6)

    p.addi(14, 7, 0)
    p.addi(9, 7, 255, f3=7)
    p.li(10, 0x01234567)
    p.arithmetic(7, 9, 10)
    p.li(10, 0x9E3779B9)
    p.arithmetic(8, 7, 10, f3=4)
    p.li(10, attrs.get("threshold", 0))
    p.li(11, 0)
    p.li(12, attrs.get("rounds", 64))
    p.branch(4, 14, 10, "gated_low_setup")

    p.li(10, 0xA5A5A5A5)
    p.addi(15, 9, 11, f3=1)
    p.label("gated_high")
    rotate(7, 5)
    p.arithmetic(14, 8, 9)
    p.arithmetic(14, 14, 11)
    p.arithmetic(7, 7, 14, f3=4)
    rotate(8, 29)
    p.arithmetic(14, 7, 10, f3=4)
    p.arithmetic(8, 8, 14)
    p.addi(14, 8, 7, f3=5)
    p.arithmetic(7, 7, 14)
    p.arithmetic(7, 7, 15, f3=4)
    p.addi(11, 11, 1)
    p.branch(1, 11, 12, "gated_high")
    p.jump("gated_done")

    p.label("gated_low_setup")
    p.li(10, 0x3C6EF372)
    p.addi(15, 9, 1, f3=5)
    p.label("gated_low")
    rotate(7, 25)
    p.arithmetic(14, 8, 9, f3=4)
    p.arithmetic(14, 14, 11, f3=4)
    p.arithmetic(7, 7, 14)
    rotate(8, 3)
    p.arithmetic(14, 7, 10)
    p.arithmetic(8, 8, 14, f3=4)
    p.addi(14, 8, 9, f3=1)
    p.arithmetic(7, 7, 14, f3=4)
    p.arithmetic(7, 7, 15)
    p.addi(11, 11, 1)
    p.branch(1, 11, 12, "gated_low")

    p.label("gated_done")
    p.arithmetic(7, 7, 8, f3=4)


def _elementwise(p, op, second, offset, attrs):
    # The previous IR result is in x7, either read from memory or held by this lane.
    if op in {"Add", "Mul"}:
        dtype = second["dtype"]
        base = p.independent_constant(10, offset)
        index = 5
        if attrs.get("broadcast") == "last":
            p.li(9, second["shape"][0])
            p.arithmetic(11, 5, 9, f3=7, f7=1)
            index = 11
        p.indexed(6, index, base, dtype)
        p.load(8, 6, dtype)
        p.arithmetic(7, 7, 8, f7=int(op == "Mul"))
    elif op == "Relu":
        p.branch(5, 7, 0, "relu_done")
        p.li(7, 0)
        p.label("relu_done")
    elif op == "Requant":
        multiplier = attrs.get("multiplier", 1)
        shift = attrs.get("shift", 0)
        zero = attrs.get("zero_point", 0)
        if multiplier != 1:
            scale = p.independent_constant(10, multiplier)
            p.arithmetic(7, 7, scale, f7=1)
        if shift:
            p.addi(7, 7, shift | 0x400, f3=5)
        if zero:
            p.li(9, zero)
            p.arithmetic(7, 7, 9)
    elif op == "GatedBranch":
        _gated(p, attrs)
    else:
        _activation(p, op, attrs.get("mode"))


def _elf(words, entry):
    """ELF32 little-endian ET_EXEC / EM_RISCV / EF_RISCV_RVE."""
    data_size = 4 * len(words)
    elf_header_size, program_header_size, section_header_size = 52, 32, 40
    text_offset = 4096
    shstrings = b"\0.text\0.shstrtab\0"
    string_offset = text_offset + data_size
    section_offset = (string_offset + len(shstrings) + 3) & ~3
    ident = b"\x7fELF\x01\x01\x01" + bytes(9)
    header = pack("<16sHHIIIIIHHHHHH", ident, 2, 243, 1, entry, elf_header_size,
                  section_offset, 8, elf_header_size, program_header_size, 1,
                  section_header_size, 3, 2)
    segment = pack("<8I", 1, text_offset, 0, 0, data_size, data_size, 5, text_offset)
    text = pack(f"<{len(words)}I", *words)
    pad = bytes(section_offset - string_offset - len(shstrings))
    sections = (bytes(section_header_size) +
                pack("<10I", 1, 1, 6, 0, text_offset, data_size, 0, 0, 4, 0) +
                pack("<10I", 7, 3, 0, 0, string_offset, len(shstrings), 0, 0, 1, 0))
    return header + segment + bytes(text_offset - len(header) - len(segment)) + text + shstrings + pad + sections


def _ranges(graph, plan, stage, window, variant):
    ranges = {}
    names = [*stage["inputs"], stage["output"]]
    for name in names:
        tensor = graph["tensors"][name]
        width = _WIDTH[tensor["dtype"]]
        size = width * prod(tensor["shape"])
        allocation = plan["allocations"][name]
        offset = allocation["offset"]
        if size <= 0 or size > allocation["size"] or offset < 0 or offset % width:
            raise ValueError(f"invalid GPU allocation for {name}")
        if offset // window != (offset + size - 1) // window:
            raise ValueError(f"GPU tensor {name} crosses a data page boundary")
        ranges[name] = {"offset": offset, "size": size}
    output = ranges[stage["output"]]
    for name in stage["inputs"]:
        operand = ranges[name]
        if max(output["offset"], operand["offset"]) < min(
                output["offset"] + output["size"], operand["offset"] + operand["size"]):
            raise ValueError(f"GPU output overlaps operand {name}")
    pages = {item["offset"] // window for item in ranges.values()}
    if len(pages) != 1:
        raise ValueError("GPU operands must share one data page")
    page = pages.pop()
    if page >= 512 or (variant == "v6" and page):
        raise ValueError("GPU data page is outside target's address window")
    return page, ranges


def _is_pairable(node):
    return (node["op"] in {"Add", "Mul"} or
            (node["op"] == "Requant" and node.get("attrs", {}).get("multiplier", 1) != 1))


def _stage_program(stage, graph, ranges, config, page):
    variant = config["variant"]
    lanes, warps = config["lanes"], config["warps"]
    window = config["window_bytes"]
    parts = stage.get("fused") or [stage]
    paired = config.get("issue_width", 1) > 1 and any(_is_pairable(node) for node in parts)
    public = (16 if variant == "v6" else 24) if paired else None
    p = Program(public)
    tensors = graph["tensors"]
    dst = tensors[stage["output"]]
    count = prod(dst["shape"])
    job_threads = 255 if variant == "v6" else lanes
    jobs = []
    for start in range(0, count, job_threads):
        warp = len(jobs) % warps
        threads = min(job_threads, count - start)
        entry = len(p.words) * 4
        jobs.append({"entry": entry, "threads": threads, "warp": warp, "page": page,
                     "first_element": start, "wave": len(jobs) // warps})
        if variant != "v6":
            jobs[-1]["gpu_job_reg"] = (warp << 24) | (threads << 16) | page
        p.li(15, start)
        p.jump("kernel")
    p.label("kernel")
    if paired:
        p.emit(_SCALAR_ENABLE)
    p.emit(0xCC3022F3)  # csrrs x5, logical GID CSR, x0
    p.arithmetic(5, 5, 15)
    offsets = {name: rec["offset"] % window for name, rec in ranges.items()}
    for i, node in enumerate(parts):
        p.namespace = f"part_{i}/"
        op, names = node["op"], node["inputs"]
        attrs = node.get("attrs", {})
        if op == "Gemm":
            _gemm(p, [tensors[name] for name in names], [offsets[name] for name in names],
                  attrs.get("transpose_b", False))
        else:
            if i == 0:
                dtype = tensors[names[0]]["dtype"]
                p.address(6, 5, offsets[names[0]], dtype)
                p.load(7, 6, dtype)
            second = tensors[names[1]] if len(names) == 2 else None
            offset = offsets[names[1]] if second else None
            _elementwise(p, op, second, offset, attrs)
        _clip(p, tensors[node["output"]]["dtype"], attrs.get("narrowing", "saturate"),
              materialized=i == len(parts) - 1)
    p.address(6, 5, ranges[stage["output"]]["offset"] % window, dst["dtype"])
    p.store(7, 6, dst["dtype"])
    p.emit(_EXIT)
    words = p.image()
    if len(words) * 4 > config["code_bytes"]:
        raise ValueError(f"GPU stage {stage['id']} exceeds instruction RAM budget")
    return words, jobs, p.pair_candidates


def lower(graph, plan, profile, output: Path):
    """Write per-stage linked device ELF, word HEX and FPGA job metadata."""
    config = profile["gpu"]
    variant, lanes, warps, window = (config[k] for k in ("variant", "lanes", "warps", "window_bytes"))
    if variant not in _VARIANTS or window != 4096:
        raise ValueError("unknown GPU ABI or data page size")
    if variant == "v6" and (lanes, warps) != (8, 1):
        raise ValueError("v6 profile requires 8 lanes and one launch slot")
    if variant in {"g3", "g4"} and (lanes, warps) not in {(16, 4), (8, 8)}:
        raise ValueError("G3/G4 requires 16 threads / 4 warps or 8 threads / 8 warps")
    abi = "v6-launch"
    if variant != "v6":
        abi = "z15-mmio-v1" if warps == 4 else "z15-mmio-8warp-projected"
    bundle = {"variant": variant, "lanes": lanes, "warps": warps,
              "window_bytes": window, "dispatch_abi": abi, "stages": []}
    artifacts = []
    for index, stage in enumerate(plan["stages"]):
        if stage["device"] != "GPU":
            continue
        if not supports(stage, graph, profile):
            raise ValueError(f"GPU does not support stage {stage['id']} ({stage['op']})")
        page, ranges = _ranges(graph, plan, stage, window, variant)
        words, jobs, pairs = _stage_program(stage, graph, ranges, config, page)
        base = f"gpu/{index:04d}"
        bundle["stages"].append({"id": stage["id"], "elf": base + ".elf", "hex": base + ".hex",
                                 "jobs": jobs, "tensor_ranges": ranges,
                                 "static_words": len(words), "scalar_pair_candidates": pairs})
        artifacts.append((base, words, jobs[0]["entry"]))
    output.mkdir(parents=True, exist_ok=True)
    for base, words, entry in artifacts:
        elf = output / (base + ".elf")
        elf.parent.mkdir(exist_ok=True)
        elf.write_bytes(_elf(words, entry))
        (output / (base + ".hex")).write_text("".join(f"{word:08x}\n" for word in words))
    (output / "gpu.json").write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle
