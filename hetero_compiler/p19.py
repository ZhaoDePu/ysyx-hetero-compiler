"""Offline Phase 19A rectilinear-transfer and barrier planner."""

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import struct

from . import yhb

ABI = json.loads(Path(__file__).with_name("p19_abi.json").read_text())
DDR = ABI["ddr_bytes"]
SRAM = ABI["sram_bytes"]
BANK = SRAM // ABI["logical_banks"]
OPS = ABI["cmdq_opcodes"]
REGS = ABI["registers"]


@dataclass(frozen=True)
class Rectangle:
    ddr: int
    sram: int
    row_bytes: int
    rows: int
    ddr_stride: int
    sram_stride: int
    store: bool = False
    swizzle: bool = False

    def validate(self):
        dims = (self.ddr, self.sram, self.row_bytes, self.ddr_stride, self.sram_stride)
        if any(type(x) is not int or x < 0 or x % 32 for x in dims):
            raise ValueError("TMA fields require nonnegative 32-byte alignment")
        if type(self.rows) is not int or not 1 <= self.rows <= 65535:
            raise ValueError("TMA row count outside 1..65535")
        if min(self.ddr_stride, self.sram_stride) < self.row_bytes or self.row_bytes == 0:
            raise ValueError("TMA rows must have nonzero, nonoverlapping extents")
        if self.end(self.ddr, self.ddr_stride) > DDR or self.end(self.sram, self.sram_stride) > SRAM:
            raise ValueError("TMA rectangle exceeds DDR or SRAM")
        if self.sram // BANK != (self.end(self.sram, self.sram_stride) - 1) // BANK:
            raise ValueError("TMA rectangle crosses the planner's SRAM bank boundary")
        return self

    def end(self, base, stride):
        return base + (self.rows - 1) * stride + self.row_bytes

    def moves(self):
        return [(self.ddr + i * self.ddr_stride, self.sram + i * self.sram_stride,
                 self.row_bytes, self.store, self.swizzle) for i in range(self.rows)]

    def commands(self):
        flags = ABI["tma_flags"]
        control = flags["START"] | (flags["STORE"] if self.store else 0) | (flags["XOR"] if self.swizzle else 0)
        fields = (("DDR_BASE", ABI["ddr_base"] + self.ddr), ("SRAM_ADDR", self.sram),
                  ("ROW_BYTES", self.row_bytes), ("NUM_ROWS", self.rows),
                  ("DDR_STRIDE", self.ddr_stride), ("SRAM_STRIDE", self.sram_stride),
                  ("CTRL_FLAGS", control))
        return [word("WRITE", REGS[k], v) for k, v in fields] + [word("WAIT_ONE", REGS["STATUS"], 2)]


def word(op, register=0, value=0):
    return OPS[op] << 60 | register << 32 | value


def fold_1d(moves):
    """Combine only consecutive, order-equivalent rectangular rows."""
    result = []
    first = 0
    while first < len(moves):
        ddr, sram, size, store, swizzle = moves[first]
        rect = Rectangle(ddr, sram, size, 1, size, size, store, swizzle).validate()
        if first + 1 < len(moves):
            nd, ns, nb, nw, nx = moves[first + 1]
            ds, ss = nd - ddr, ns - sram
            if (nb, nw, nx) == (size, store, swizzle) and ds >= size and ss >= size:
                stop = first + 1
                while stop < len(moves) and stop - first < 65535:
                    index = stop - first
                    if moves[stop] != (ddr + index * ds, sram + index * ss, size, store, swizzle):
                        break
                    candidate = Rectangle(ddr, sram, size, index + 1, ds, ss, store, swizzle)
                    try:
                        candidate.validate()
                    except ValueError:
                        break
                    rect = candidate
                    stop += 1
        result.append(rect)
        first += rect.rows
    return result


def pipeline(tiles, rows, row_bytes, ddr_stride, sram_stride, *, load_base=0, store_base=None):
    """Assign two Ping/Pong banks per stage and emit a token dependency schedule."""
    if type(tiles) is not int or not 1 <= tiles <= 4096:
        raise ValueError("tile count outside 1..4096")
    first = Rectangle(load_base, 0, row_bytes, rows, ddr_stride, sram_stride).validate()
    extent = first.end(0, ddr_stride)
    store_base = load_base + extent * tiles if store_base is None else store_base
    if not (store_base >= load_base + tiles * extent or load_base >= store_base + tiles * extent):
        raise ValueError("DDR input and output regions overlap")
    events = []
    for tile in range(tiles):
        half = tile % 2
        a, b = half, 2 + half
        src, dst = load_base + tile * extent, store_base + tile * extent
        ingress = Rectangle(src, a * BANK, row_bytes, rows, ddr_stride, sram_stride).validate()
        egress = Rectangle(dst, b * BANK, row_bytes, rows, ddr_stride, sram_stride, True).validate()
        events.extend((
            {"slot": 2 * tile, "tile": tile, "engine": "TMA", "op": "load", "banks": [a],
             "wait": [2 + half] if tile >= 2 else [], "signal": [half], "extent": asdict(ingress)},
            {"slot": 2 * (tile + 1), "tile": tile, "engine": "NPU", "op": "compute", "banks": [a, b],
             "wait": [half] + ([6 + half] if tile >= 2 else []), "signal": [2 + half, 4 + half]},
            {"slot": 2 * (tile + 2), "tile": tile, "engine": "GPU", "op": "post", "banks": [b],
             "wait": [4 + half], "signal": [4 + half]},
            {"slot": 2 * (tile + 2) + 1, "tile": tile, "engine": "TMA", "op": "store", "banks": [b],
             "wait": [4 + half], "signal": [6 + half], "extent": asdict(egress)},
        ))
    events.sort(key=lambda item: (item["slot"], item["tile"]))
    summary = audit(events)
    moves = 2 * rows * tiles
    return {"target": "p19a", "events": events, "summary": {
        **summary, "dma_1d": moves, "tma_2d": 2 * tiles,
        "phase18_dma_command_estimate": moves * 7,
        "phase19_dma_barrier_command_estimate":
            2 * tiles * 8 + sum(len(x["wait"]) + len(x["signal"]) for x in events),
    }}


def audit(events):
    """Validate token consumption, logical-bank ownership and transfer ranges."""
    if not events or any(e["slot"] < 0 for e in events):
        raise ValueError("missing or invalid schedule")
    by_slot = {}
    for event in events:
        by_slot.setdefault(event["slot"], []).append(event)
    tokens = [0] * ABI["barrier_slots"]
    triple = 0
    for cycle, active in sorted(by_slot.items()):
        used = set()
        engines = set()
        for op in active:
            bank_list = op["banks"]
            if not bank_list or len(set(bank_list)) != len(bank_list) or any(not 0 <= b < 4 for b in bank_list):
                raise ValueError("invalid SRAM bank set")
            if used.intersection(bank_list) or op["engine"] in engines:
                raise ValueError(f"SRAM bank or engine conflict in slot {cycle}")
            used.update(bank_list)
            engines.add(op["engine"])
            if "extent" in op:
                rect = Rectangle(**op["extent"]).validate()
                if rect.sram // BANK not in bank_list or rect.store != (op["op"] == "store"):
                    raise ValueError("TMA transfer disagrees with its assigned bank or direction")
            for k in op["wait"]:
                if not 0 <= k < len(tokens) or not tokens[k]:
                    raise ValueError(f"unsatisfied barrier in slot {cycle}")
                tokens[k] -= 1
        for op in active:
            for k in op["signal"]:
                if not 0 <= k < len(tokens) or tokens[k] >= 1:
                    raise ValueError("invalid or unconsumed barrier signal")
                tokens[k] += 1
        triple += engines == {"TMA", "NPU", "GPU"}
    slots = max(by_slot) + 1
    return {"slots": slots, "triple_overlap_slots": triple,
            "triple_overlap_fraction": triple / slots}


def transfer_audit(binary):
    """Analyze exact fold opportunities within existing fenced YHB transfers."""
    if binary.startswith(b"YHP1"):
        binary = b"YHB1" + binary[4:]
    info = yhb.inspect_yhb(binary)["sections"]
    def records(section, fields):
        extent = info[section]
        start = extent["offset"]
        return list(struct.iter_unpack("<" + "I" * fields, binary[start:start + extent["bytes"]]))
    dma, commands = records("dma", 6), records("commands", 4)
    old, new, run = 0, 0, []
    by_rows = Counter()

    def drain():
        nonlocal new
        current = fold_1d(run)
        new += len(current)
        by_rows.update(item.rows for item in current)
        assert [row for item in current for row in item.moves()] == run
        run.clear()

    for opcode, operand, _, _ in commands:
        if opcode == yhb.DMA:
            direction, ddr, sram, size, _, _ = dma[operand]
            row = (ddr, sram, size, direction == yhb.STORE, False)
            old += 1
            # Existing 1D streams can span two logical 512 KiB banks
            # within a supported physical half. Retain those transactions.
            if sram // BANK != (sram + size - 1) // BANK:
                drain()
                new += 1
                by_rows[1] += 1
            else:
                run.append(row)
        elif opcode != yhb.WAIT_DMA:
            drain()
    drain()
    return {"target": "p19a", "mode": "transfer-only", "dma_1d": old, "tma_or_existing": new,
            "rows_histogram": dict(sorted(by_rows.items())),
            "legacy_graph_commands": len(commands), "retained_stage_dependencies": True}


def c_header():
    macros = {"DDR_BASE": ABI["ddr_base"], "DDR_BYTES": DDR, "SRAM_BYTES": SRAM,
              "LOGICAL_BANK_BYTES": BANK, "BARRIER_SLOTS": ABI["barrier_slots"],
              "G11_OPCODE": ABI["g11_opcode"]}
    for prefix, group in (("CAP", "cap_flags"), ("REG", "registers"), ("CTRL", "tma_flags"),
                          ("CMDQ", "cmdq_opcodes")):
        macros.update({f"{prefix}_{k}": v for k, v in ABI[group].items()})
    for name, (f7, f3) in ABI["g11_rform"].items():
        macros[f"G11_{name}_FUNCT7"] = f7
        macros[f"G11_{name}_FUNCT3"] = f3
    lines = ["/* Generated from hetero_compiler/p19_abi.json. Offline preview. */",
             "/* SYNC_POINT: " + ABI["sync_point"] + " */",
             "#ifndef VALENCE_P19_ABI_DRAFT_H", "#define VALENCE_P19_ABI_DRAFT_H", ""]
    lines.extend(f"#define VALENCE_P19_{key} 0x{value:x}u" for key, value in macros.items())
    return "\n".join(lines + ["", "#endif", ""])


def swizzled_address(address):
    beat, byte = address & ~31, address & 3
    lane = (address >> 2) & 7
    return beat + ((lane ^ ((beat >> 5) & 7)) << 2) + byte


def reference(plan):
    """Replay a concrete pipeline fixture using deterministic byte tensors."""
    import hashlib
    events = plan["events"]
    audit(events)
    memory = {}
    loaded = {}
    result = hashlib.sha256()
    stored = 0
    words = []
    for op in events:
        tile, banks = op["tile"], op["banks"]
        words.extend(word("WAIT_BARRIER", k) for k in op["wait"])
        if op["op"] in ("load", "store"):
            rect = Rectangle(**op["extent"])
            words.extend(rect.commands())
            if op["op"] == "load":
                blob = bytes((ddr + j * 19 + tile) & 255 for ddr, _, size, _, _ in rect.moves()
                             for j in range(size))
                if tile in loaded:
                    raise ValueError("tile was loaded twice")
                loaded[tile] = blob
                memory[banks[0]] = (tile, blob)
            else:
                at, blob = memory.pop(banks[0])
                expected = bytes((((x << 1) + 3) & 255) ^ 90 for x in loaded.pop(tile))
                if at != tile or blob != expected or len(blob) != rect.row_bytes * rect.rows:
                    raise ValueError("output differs from scalar golden fixture")
                result.update(blob)
                stored += 1
        else:
            src = memory[banks[0]]
            if src[0] != tile:
                raise ValueError("read observed a different tile in its bank")
            if op["op"] == "compute":
                memory[banks[1]] = (tile, bytes((2 * x + 3) & 255 for x in src[1]))
                del memory[banks[0]]
            elif op["op"] == "post":
                memory[banks[0]] = (tile, bytes(x ^ 90 for x in src[1]))
            else:
                raise ValueError("unsupported offline fixture operation")
        words.extend(word("SIGNAL_BARRIER", k) for k in op["signal"])
    if not stored or loaded or memory:
        raise ValueError("unfinished tile or missing store")
    return {"stored_tiles": stored, "golden_sha256": result.hexdigest(),
            "fixture_byte_exact": True, "transfer_and_barrier_words": len(words)}, words
