"""Load hardware topology, address limits, and ABI capabilities from JSON."""

import json
from pathlib import Path

from .ir import GraphError

PROFILE_DIR = Path(__file__).resolve().parent.parent / "profiles"


def load_profile(name):
    filename = Path(name)
    if filename.parent == Path(".") and filename.suffix == "":
        filename = PROFILE_DIR / (filename.name + ".json")
    with filename.open(encoding="utf-8") as stream:
        p = json.load(stream)
    validate_profile(p)
    return p


def validate_profile(p):
    if not isinstance(p, dict) or p.get("schema") != "ysyx-target-v1" or not p.get("name"):
        raise GraphError("invalid target profile header")
    try:
        npu, gpu, vpu, dma, sram = (p[x] for x in ("npu", "gpu", "vpu", "dma", "sram"))
        values = [npu[x] for x in ("rows", "cols", "window_bytes")]
        values += [gpu[x] for x in ("lanes", "warps", "code_bytes", "window_bytes", "issue_width")]
        values += [sram[x] for x in ("capacity", "alignment", "zone_bytes")]
        if any(type(x) is not int or x <= 0 for x in values):
            raise GraphError("profile topology and positive sizes must be integers")
        reserve, beat = sram["metadata_reserve"], dma["beat_bytes"]
        if type(beat) is not int or type(reserve) is not int or reserve < 0 or beat < 0:
            raise GraphError("invalid metadata reservation or DMA beat")
        if not isinstance(vpu["enabled"], bool) or not isinstance(vpu["verified"], bool):
            raise GraphError("invalid VPU capability")
        if vpu["verified"] and not vpu["enabled"]:
            raise GraphError("verified VPU requires an enabled VPU")
        cap, align, zone = (sram[x] for x in ("capacity", "alignment", "zone_bytes"))
        if (cap % zone or zone % align or align & (align - 1) or
                (beat and (beat & (beat - 1) or zone % beat or align % beat)) or
                npu["window_bytes"] > cap or gpu["window_bytes"] > cap or
                min(npu["window_bytes"], gpu["window_bytes"]) <= reserve or
                reserve % align or not 1 <= gpu["lanes"] <= 255 or gpu["warps"] > 8 or
                gpu["issue_width"] not in (1, 2)):
            raise GraphError("inconsistent target memory or GPU limits")
        if npu["variant"] not in ("p8", "z15") or gpu["variant"] not in ("v6", "g3", "g4"):
            raise GraphError("unsupported hardware ABI")
        if npu["variant"] == "z15" and (cap, zone, beat, npu["window_bytes"]) != (
                2 << 20, 1 << 20, 32, 2 << 20):
            raise GraphError("Z15 requires two 1 MiB ownership halves and 32-byte DMA beats")
        if npu["variant"] == "p8" and (npu["window_bytes"] != 4096 or beat or zone != cap or align < 8):
            raise GraphError("Phase 8 requires a 4096-byte native window and 8-byte alignment")
    except (KeyError, TypeError) as exc:
        raise GraphError(f"incomplete target profile: {exc}") from exc
    return p
