"""Lower physical Z15 tile schedules to legacy YHB1 and RTL replay."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
from pathlib import Path

from .c6_gpu import GpuStage
from .c6_memory import HALF_BYTES
from .c6_tensor import NativeStage, Transfer
from .npu import Z15_CTRL, Z15_NPU_START, Z15_VPU_START
from .yhb import emit_yhb


def _append_moves(program, moves):
    for move in moves:
        program.extend((move.command(), {"type": "dma-fence"}))


def _record_stage(stage, zone, stages, native, gpu, directory):
    stages.append({"id": stage.id, "device": stage.device, "zone": zone})
    if isinstance(stage, GpuStage):
        path = directory / f"stage-{len(stages)}.hex"
        path.write_text("".join(f"{word:08x}\n" for word in stage.words), encoding="ascii")
        gpu.append({"id": stage.id, "hex": path.name, "jobs": stage.jobs})
    else:
        native.append({"stage": stage.id, "device": stage.device, "writes": stage.writes})


def emit_layer(tiles, program, stages, native, gpu, directory):
    """Flush each layer at its boundary and overlap DMA in the other half."""
    if not tiles:
        raise ValueError("a native layer must issue at least one tile")
    _append_moves(program, tiles[0].loads)
    for index, tile in enumerate(tiles):
        if not tile.stages:
            raise ValueError("a native tile must contain a stage")
        before = tiles[index - 1] if index else None
        after = tiles[index + 1] if index + 1 < len(tiles) else None
        if before is not None and before.zone == tile.zone:
            raise ValueError("streaming tiles must alternate Z15 ownership halves")
        for stage_index, stage in enumerate(tile.stages):
            _record_stage(stage, tile.zone, stages, native, gpu, directory)
            program.append({"type": "launch", "stage": stage.id})
            if stage_index == 0 and before is not None:
                _append_moves(program, before.stores)
            if stage_index == len(tile.stages) - 1 and after is not None:
                _append_moves(program, after.loads)
            program.append({"type": "fence", "wait_for": stage.id})
    _append_moves(program, tiles[-1].stores)


@dataclass(frozen=True)
class _CrossK:
    gpu: int
    epilogues: range


@dataclass
class _Operation:
    zone: int
    resource: str
    payload: Transfer | GpuStage | NativeStage
    dependencies: set[int] = field(default_factory=set)
    cross_k: _CrossK | None = None


def _raw_conv(stage):
    """Identify Conv split or INT32 stages that never touch shared tables."""
    if not isinstance(stage, NativeStage) or not stage.writes:
        return False
    regs = dict(stage.writes)
    if len(regs) != len(stage.writes):
        return False
    if stage.device == "VPU":
        return (stage.writes[-1] == (Z15_CTRL, Z15_VPU_START) and regs.get(0xBC) == 2
                and set(regs) <= {Z15_CTRL, 0xA0, 0xA4, 0xA8, 0xAC, 0xB0, 0xB4, 0xB8, 0xBC, 0xC0})
    flags = regs.get(0x78, 0)
    return (stage.device == "NPU" and stage.writes[-1] == (Z15_CTRL, Z15_NPU_START)
            and flags & (1 << 9 | 1 << 13 | 1 << 14) == (1 << 9 | 1 << 13 | 1 << 14)
            and not flags & (1 << 15 | 1 << 16)
            and set(regs) <= {Z15_CTRL, 0x60, 0x64, 0x68, 0x6C, 0x70, 0x74, 0x78, 0x7C, 0x80, 0x84})


def _requant_epilogue(stage):
    if not isinstance(stage, NativeStage) or stage.device != "NPU" or not stage.writes:
        return False
    regs = dict(stage.writes)
    flags = regs.get(0x78, 0)
    required = 1 << 1 | 1 << 2 | 1 << 15 | 1 << 17
    return (stage.writes[-1] == (Z15_CTRL, Z15_NPU_START)
            and regs.get(0x60, 0) & 0xFFFF == 1 and regs.get(0x64) == (4 | 16 << 16)
            and flags & ~(1 << 16) == required)


def _fused_reduction(tile, first):
    """Find the GPU and native tail of a single-page raw/requant tile."""
    gpu = next((i for i, stage in enumerate(tile.stages) if isinstance(stage, GpuStage)), 0)
    tail = tile.stages[gpu + 1:]
    if (gpu and tail and all(map(_raw_conv, tile.stages[:gpu]))
            and all(map(_requant_epilogue, tail))):
        return _CrossK(first + gpu, range(first + gpu + 1, first + len(tile.stages)))
    return None


def _lanes(layers, *, stream_overlap=False):
    """Keep tile, scratch operand and global native-table order."""
    operations, pending, native, ddr = [], [deque(), deque()], deque(), []
    for layer in layers:
        if not layer:
            raise ValueError("a native layer must issue at least one tile")
        if any(a.zone == b.zone for a, b in zip(layer, layer[1:])):
            raise ValueError("streaming tiles must alternate Z15 ownership halves")
        tiles = []
        previous = None
        for tile in layer:
            if not tile.stages or tile.zone not in (0, 1) or not 0 < tile.scratch_bytes <= HALF_BYTES:
                raise ValueError("invalid overlapping Z15 tile")
            first = len(operations)
            for item in (*tile.loads, *tile.stages, *tile.stores):
                index = len(operations)
                unit = "gpu" if isinstance(item, GpuStage) else "native"
                if isinstance(item, Transfer):
                    if item.scratch_offset // HALF_BYTES != tile.zone:
                        raise ValueError("DMA ownership half differs from its tile")
                    unit = "dma"
                operations.append(_Operation(tile.zone, unit, item))
                pending[tile.zone].append(index)
                if unit == "native":
                    native.append(index)
            stage_first = first + len(tile.loads)
            reduction = _fused_reduction(tile, stage_first) if stream_overlap else None
            if previous is not None and reduction is not None:
                for index in range(stage_first, reduction.gpu):
                    operations[index].cross_k = previous
            previous = reduction
            tiles.append((range(first, stage_first),
                          range(len(operations) - len(tile.stores), len(operations))))
        # Retain the chronological order of all conflicting DDR accesses.
        # C6 prefetches a tile before the previous tile in the same layer is stored.
        ddr.extend(tiles[0][0])
        for index, _ in enumerate(tiles):
            if index > 0:
                ddr.extend(tiles[index - 1][1])
            if index + 1 < len(tiles):
                ddr.extend(tiles[index + 1][0])
        ddr.extend(tiles[-1][1])
    return operations, pending, native, ddr


def _ddr_dependencies(operations, ddr_order):
    """Constrain actual (including padded) DMA extents across DDR alias reuse."""
    history = defaultdict(list)
    for index in ddr_order:
        operation = operations[index]
        move = operation.payload
        start, end = move.ddr_offset, move.ddr_offset + move.size
        pages = range(start // 4096, (end - 1) // 4096 + 1)
        previous = {item for page in pages for item in history[page]}
        for old in previous:
            other = operations[old].payload
            if (start < other.ddr_offset + other.size and other.ddr_offset < end
                    and (move.direction == "store" or other.direction == "store")):
                operation.dependencies.add(old)
        for page in pages:
            history[page].append(index)


def _emit_overlap(layers, program, stages, native, gpu, directory, *, stream_overlap=False):
    operations, pending, native_order, ddr_order = _lanes(layers, stream_overlap=stream_overlap)
    _ddr_dependencies(operations, ddr_order)
    finished, running = set(), {}
    # Start native compute before an immediately available GPU launch. The GPU
    # host submission can wait for several warp waves to drain.
    issue_order = ("native", "dma", "gpu")
    while any(pending):
        busy_halves = {operations[index].zone for index in running.values()}
        for unit in issue_order:
            if unit in running:
                continue
            for zone, chain in enumerate(pending):
                if zone in busy_halves or not chain:
                    continue
                index = chain[0]
                op = operations[index]
                if op.resource != unit or not op.dependencies <= finished:
                    continue
                if unit == "native" and native_order[0] != index:
                    # Only let the next tile's raw stages pass an epilogue
                    # blocked on its GPU producer. Both half queues and the
                    # global order of scale/LUT writers remain intact.
                    window = op.cross_k
                    other = pending[1 - zone]
                    if (window is None or native_order[0] not in window.epilogues
                            or not other or other[0] != window.gpu):
                        continue
                command = op.payload.command() if unit == "dma" else {"type": "launch", "stage": op.payload.id}
                program.append(command)
                running[unit] = index
                busy_halves.add(zone)
                break
        if not running:
            raise ValueError("circular dependency in overlapping Z15 schedule")
        # A short DMA wait releases its half to the next stage while the
        # other half computes. Native generally finishes before the GPU tail.
        unit = next(unit for unit in ("dma", "native", "gpu") if unit in running)
        index = running.pop(unit)
        op = operations[index]
        if unit == "dma":
            program.append({"type": "dma-fence"})
        else:
            program.append({"type": "fence", "wait_for": op.payload.id})
            # Index the trace by drain order, matching the C99 RTL ledger.
            _record_stage(op.payload, op.zone, stages, native, gpu, directory)
            if unit == "native":
                if native_order[0] == index:
                    native_order.popleft()
                else:
                    native_order.remove(index)
        pending[op.zone].popleft()
        finished.add(index)


def emit_program(directory, layers, tensors, inputs, outputs, slots, initial, *,
                 external_inputs=True, c6=None, overlap=False, stream_overlap=False):
    """Write a bundle and let the existing serializer validate every move."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / "ddr_initial.bin"
    peak = max((slot.offset + slot.capacity for slot in slots.values()), default=0)
    with image.open("wb") as stream:
        stream.truncate(peak)
        for name, data in initial.items():
            slot = slots[name]
            if len(data) != slot.size:
                raise ValueError(f"{name}: invalid initializer size")
            stream.seek(slot.offset)
            stream.write(data)
    stages, native, gpu, program = [], [], [], []
    layers = tuple(layers)
    peaks = [max((tile.scratch_bytes for layer in layers for tile in layer if tile.zone == half),
                 default=0) for half in (0, 1)]
    if overlap or stream_overlap:
        _emit_overlap(layers, program, stages, native, gpu, directory, stream_overlap=stream_overlap)
        # The sum conservatively includes resident constants in preloaded halves.
        scratch_peak = sum(peaks)
    else:
        scratch_peak = 0
        for layer in layers:
            emit_layer(layer, program, stages, native, gpu, directory)
            scratch_peak = max(scratch_peak, *(tile.scratch_bytes for tile in layer),
                               *(a.scratch_bytes + b.scratch_bytes for a, b in zip(layer, layer[1:])))
    graph = {"tensors": tensors, "inputs": inputs, "outputs": outputs}

    def bindings(names, needs_data):
        return {name: {"ddr_offset": slots[name].offset, "size": slots[name].size,
                       "needs_data": needs_data} for name in names}

    manifest = {
        "profile": {"name": "fpga_zu15eg_balanced", "sram": {"capacity": 2 * HALF_BYTES}},
        "plan": {"stages": stages, "allocations": {
            name: {"offset": slot.offset, "size": slot.size} for name, slot in slots.items()},
            "ddr_peak": peak, "scratch_peak": scratch_peak, "scratch_peak_by_half": peaks},
        "graph": graph,
        "backends": {"gpu": {"stages": gpu}, "npu": {"commands": native}},
        "execution": program,
        "initial_image": image.name,
        "initial_ddr_extents": [
            {"ddr_offset": slots[name].offset, "file": image.name,
             "file_offset": slots[name].offset, "bytes": slots[name].capacity}
            for name in initial],
        "input_bindings": bindings(inputs, external_inputs),
        "output_bindings": bindings(outputs, False),
    }
    manifest["plan"]["dma_bytes"] = sum(move["bytes"] for move in program if move["type"] == "dma")
    if overlap or stream_overlap:
        manifest["plan"]["schedule"] = "stream_overlap" if stream_overlap else "overlap"
    if c6 is not None:
        manifest["c6"] = c6
    emit_yhb(directory, manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def emit_rtl_control(directory, manifest, initial, expected):
    """Emit the frozen shell runner's timed path with independent DDR checks."""
    directory = Path(directory).resolve()
    slots = manifest["plan"]["allocations"]
    lines = ["- 0 0"]
    for directive, seed in (("ddr_put", initial), ("compare_ddr", expected)):
        if directive == "compare_ddr":
            stages = {stage["stage"]: stage for stage in manifest["backends"]["npu"]["commands"]}
            gpu = {stage["id"]: stage for stage in manifest["backends"]["gpu"]["stages"]}
            for command in manifest["execution"]:
                kind = command["type"]
                if kind == "launch":
                    name = command["stage"]
                    if name in gpu:
                        stage = gpu[name]
                        for index, word in enumerate((directory / stage["hex"]).read_text().splitlines()):
                            lines.append(f"write {0x4000 + 4 * index} {int(word, 16)}")
                        lines.extend(f"submit_gpu {job['entry']} {job['gpu_job_reg']}" for job in stage["jobs"])
                    else:
                        stage = stages[name]
                        lines.extend(f"write {reg} {value}" for reg, value in stage["writes"])
                elif kind == "fence":
                    name = command["wait_for"]
                    lines.append(f"{'wait_gpu' if name in gpu else 'wait_npu'} {name}")
                elif kind == "dma":
                    action = "dma_in" if command["direction"] == "load" else "dma_out"
                    lines.append(f"{action} {command['ddr_offset']} "
                                 f"{command['scratch_offset']} {command['bytes']}")
                elif kind == "dma-fence":
                    lines.append("dma_wait")
                else:
                    raise ValueError(f"unsupported C6 RTL command: {kind}")
        for index, (name, data) in enumerate(seed.items()):
            path = directory / f"{directive}-{index}.bin"
            path.write_bytes(data)
            lines.append(f"{directive} {path} {slots[name]['offset']} {len(data)}")
    control = directory / "replay.txt"
    control.write_text("\n".join(lines) + "\n")
    return control
