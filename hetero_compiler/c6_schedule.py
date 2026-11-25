"""Lower physical Z15 tile schedules to legacy YHB1 and RTL replay."""

import json
from pathlib import Path

from .c6_gpu import GpuStage
from .c6_memory import HALF_BYTES
from .yhb import emit_yhb


def _append_moves(program, moves):
    for move in moves:
        program.extend((move.command(), {"type": "dma-fence"}))


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
            record = {"id": stage.id, "device": stage.device, "zone": tile.zone}
            stages.append(record)
            if isinstance(stage, GpuStage):
                path = directory / f"stage-{len(stages)}.hex"
                path.write_text("".join(f"{word:08x}\n" for word in stage.words), encoding="ascii")
                gpu.append({"id": stage.id, "hex": path.name, "jobs": stage.jobs})
            else:
                native.append({"stage": stage.id, "device": stage.device, "writes": stage.writes})
            program.append({"type": "launch", "stage": stage.id})
            if stage_index == 0 and before is not None:
                _append_moves(program, before.stores)
            if stage_index == len(tile.stages) - 1 and after is not None:
                _append_moves(program, after.loads)
            program.append({"type": "fence", "wait_for": stage.id})
    _append_moves(program, tiles[-1].stores)


def emit_program(directory, layers, tensors, inputs, outputs, slots, initial, *, external_inputs=True, c6=None):
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
    all_tiles = []
    scratch_peak = 0
    for layer in layers:
        emit_layer(layer, program, stages, native, gpu, directory)
        all_tiles.extend(layer)
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
            "ddr_peak": peak, "scratch_peak": scratch_peak, "scratch_peak_by_half": [
                max((tile.scratch_bytes for tile in all_tiles if tile.zone == half), default=0)
                for half in (0, 1)]},
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
