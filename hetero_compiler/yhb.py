"""Encode and structurally inspect a YHB1 program (see docs/yhb-format.md)."""

import struct
import zlib
from pathlib import Path

HEADER = struct.Struct("<4sHH6I")
DIRECTORY = struct.Struct("<4I")
SECTIONS = ("strings", "symbols", "data", "initial", "stages", "dma", "commands")
RECORD_BYTES = (1, 48, 1, 16, 32, 24, 16)
TARGET_IDS = {name: target for target, name in enumerate((
    "fpga_zu15eg_balanced", "fpga_zu15eg_full", "asic_nangate45_p8", "asic_nangate45_ip"), 1)}
DTYPES = {"int8": 1, "uint8": 2, "int32": 3, "int64": 4, "uint16": 5, "int16": 6}
WIDTHS = {1: 1, 2: 1, 3: 4, 4: 8, 5: 2, 6: 2}
ENGINE = {"GPU": 1, "NPU": 2, "VPU": 3}
LAUNCH, WAIT_STAGE, DMA, WAIT_DMA = range(1, 5)
LOAD, STORE = 1, 2
POOL_BYTES = 0x20000000
MAX_BYTES = 0x20000000


def _align(value):
    return (value + 7) & ~7


def _append(blob, data):
    offset = _align(len(blob))
    blob.extend(bytes(offset - len(blob)))
    blob.extend(data)
    return offset


def _words(values):
    return b"".join(struct.pack("<I", value) for value in values)


def _crc(blob):
    crc = zlib.crc32(blob[:16])
    crc = zlib.crc32(bytes(4), crc)
    return zlib.crc32(blob[20:], crc)


def _tensor_symbol(graph, plan, overrides, name):
    src = graph["tensors"][name]
    allocation = plan["allocations"].get(name, {})
    binding = overrides.get(name, {})
    shape = binding.get("shape", src["shape"])
    dtype = DTYPES[binding.get("dtype", src["dtype"])]
    offset = binding.get("ddr_offset", binding.get("offset", allocation.get("offset")))
    size = binding.get("size", allocation.get("size"))
    return dtype, shape, offset, size


def _execution(manifest):
    if "execution" in manifest:
        return manifest["execution"]
    program = []
    for item in manifest["dma"]:
        if item["direction"] == "load":
            program += [{**item, "type": "dma"}, {"type": "dma-fence"}]
    for command in manifest["plan"]["commands"]:
        if command["type"] == "fence":
            program.append(command)
        elif command["type"] == "launch":
            if pred := command.get("engine_predecessor"):
                program.append({"type": "fence", "wait_for": pred})
            program.append(command)
        else:
            raise ValueError(f"unknown planning command {command['type']}")
    for stage in manifest["plan"]["stages"]:
        program.append({"type": "fence", "wait_for": stage["id"]})
    for item in manifest["dma"]:
        if item["direction"] == "store":
            program += [{**item, "type": "dma"}, {"type": "dma-fence"}]
    return program


def _merged_ranges(ranges):
    merged = []
    for lo, hi in sorted(ranges):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return merged


def emit_yhb(directory, manifest):
    """Write a compact program.yhb from the emitted compiler bundle."""
    directory = Path(directory)
    profile, plan, graph, backends = (manifest[k] for k in ("profile", "plan", "graph", "backends"))
    target = TARGET_IDS.get(profile["name"])
    if target is None:
        raise ValueError(f"cannot encode unknown YHB target {profile['name']}")
    strings, symbols, data, initial, stages, moves, commands = (bytearray() for _ in SECTIONS)
    interned = {}

    def text(name):
        if name not in interned:
            raw = name.encode("utf-8")
            if not raw or b"\0" in raw:
                raise ValueError("invalid YHB string")
            interned[name] = (len(strings), len(raw))
            strings.extend(raw)
        return interned[name]

    inputs = manifest["input_bindings"]
    outputs = manifest.get("output_bindings", {})
    initial_ranges = []
    for name in dict.fromkeys(graph["inputs"] + graph["outputs"]):
        flags = int(name in graph["inputs"]) | (2 * int(name in graph["outputs"]))
        if name in inputs and inputs[name].get("needs_data", "data" not in graph["tensors"][name]):
            flags |= 4
        dtype, shape, at, nbytes = _tensor_symbol(graph, plan, inputs if flags & 1 else outputs, name)
        count = WIDTHS[dtype]
        for d in shape:
            count *= d
        if not 1 <= len(shape) <= 4 or nbytes != count or not 0 <= at < POOL_BYTES or at + nbytes > POOL_BYTES:
            raise ValueError(f"invalid YHB tensor shape or DDR range for {name!r}")
        symbols.extend(_words((*text(name), flags, dtype, len(shape),
                               *shape, *([0] * (4 - len(shape))), at, nbytes, 0)))
        if flags & 1:
            initial_ranges.append((at, at + nbytes))

    programs = {record["id"]: record for record in backends["gpu"]["stages"]}
    tensors = {record["stage"]: record for record in backends["npu"]["commands"]}
    slots = {}
    for idx, stage in enumerate(plan["stages"]):
        stage_id = stage["id"]
        if stage_id in slots:
            raise ValueError(f"duplicate YHB stage {stage_id!r}")
        slots[stage_id] = idx
        engine = ENGINE[stage["device"]]
        jobs = []
        if engine == ENGINE["GPU"]:
            gpu = programs[stage_id]
            words = [int(line, 16) for line in (directory / gpu["hex"]).read_text("ascii").splitlines()]
            for job in gpu["jobs"]:
                jobs.extend((job["entry"], job.get("gpu_job_reg", 0), job["threads"], job["warp"]))
        else:
            native = tensors[stage_id]
            if "writes" in native:
                words = [value for pair in native["writes"] for value in pair]
            else:
                words = [native["a_base"], native["descriptor"], native["program_slots"]]
        word_offset = _append(data, _words(words))
        job_offset = _append(data, _words(jobs))
        stages.extend(_words((*text(stage_id), engine, word_offset, len(words),
                              job_offset, len(jobs) // 4, stage.get("zone", 0))))

    move_ids = {}
    for item in _execution(manifest):
        kind = item["type"]
        if kind == "launch":
            opcode, operand = LAUNCH, slots[item["stage"]]
        elif kind == "fence":
            opcode, operand = WAIT_STAGE, slots[item["wait_for"]]
        elif kind == "dma-fence":
            opcode, operand = WAIT_DMA, 0
        elif kind == "dma":
            direction = {"load": LOAD, "store": STORE}[item["direction"]]
            ddr, scratch, length = (item[k] for k in ("ddr_offset", "scratch_offset", "bytes"))
            if (length <= 0 or min(ddr, scratch) < 0 or (length | ddr | scratch) & 31 or
                    ddr + length > POOL_BYTES or scratch + length > profile["sram"]["capacity"] or
                    (scratch >> 20) != ((scratch + length - 1) >> 20)):
                raise ValueError(f"cannot encode invalid YHB DMA extent {item}")
            key = (direction, ddr, scratch, length, item.get("flags", 2) & 2, 0)
            if key not in move_ids:
                move_ids[key] = len(move_ids)
                moves.extend(_words(key))
            if direction == LOAD:
                initial_ranges.append((ddr, ddr + length))
            opcode, operand = DMA, move_ids[key]
        else:
            raise ValueError(f"cannot encode YHB instruction {kind!r}")
        commands.extend(_words((opcode, operand, 0, 0)))

    image = (directory / manifest.get("initial_ddr", manifest["initial_image"])).read_bytes()
    if len(image) > POOL_BYTES:
        raise ValueError("YHB initial image exceeds DDR pool")
    if target >= 3:
        initial_ranges.append((0, min(len(image), 4096)))
    for lo, hi in _merged_ranges(initial_ranges):
        if hi > lo:
            chunk = image[lo:hi]
            extent = _append(data, chunk + bytes(hi - lo - len(chunk)))
            initial.extend(_words((lo, hi - lo, extent, 0)))

    output = bytearray(HEADER.size + len(SECTIONS) * DIRECTORY.size)
    for index, (section, stride) in enumerate(zip(
            (strings, symbols, data, initial, stages, moves, commands), RECORD_BYTES), 1):
        pos = _append(output, section)
        DIRECTORY.pack_into(output, HEADER.size + (index - 1) * DIRECTORY.size,
                            index, pos, len(section), len(section) // stride)
    if len(output) > MAX_BYTES:
        raise ValueError("encoded YHB exceeds the loader file limit")
    HEADER.pack_into(output, 0, b"YHB1", 1, HEADER.size, target, len(output),
                     0, len(SECTIONS), profile["sram"]["capacity"], 0)
    struct.pack_into("<I", output, 16, _crc(output))
    dest = directory / "program.yhb"
    dest.write_bytes(output)
    return dest


def inspect_yhb(blob):
    """Verify checksums and the section table, returning record extents."""
    if len(blob) < HEADER.size:
        raise ValueError("truncated YHB header")
    magic, version, header, target, total, crc, nsections, capacity, flags = HEADER.unpack_from(blob)
    if ((magic, version, header, nsections, flags) != (b"YHB1", 1, HEADER.size, len(SECTIONS), 0)
            or target not in TARGET_IDS.values() or total != len(blob) or total > MAX_BYTES
            or not 0 < capacity <= 2 << 20):
        raise ValueError("invalid YHB header")
    if _crc(blob) != crc:
        raise ValueError("YHB CRC32 mismatch")
    result, end = {}, HEADER.size + DIRECTORY.size * nsections
    if end > total:
        raise ValueError("truncated YHB section directory")
    for index, (name, stride) in enumerate(zip(SECTIONS, RECORD_BYTES), 1):
        ident, start, length, count = DIRECTORY.unpack_from(blob, HEADER.size + (index - 1) * DIRECTORY.size)
        if ident != index or start & 7 or start < end or start + length > total or count * stride != length:
            raise ValueError(f"invalid YHB {name} section")
        result[name] = {"offset": start, "bytes": length, "count": count}
        end = start + length
    if end != total:
        raise ValueError("trailing bytes after YHB directory payloads")
    return {"target": target, "total": total, "scratch_bytes": capacity,
            "crc32": crc, "sections": result}
