"""Explicit offline-only target dispatch for the P19 drafts."""

import json
from pathlib import Path

from .p19 import c_header, pipeline, reference, transfer_audit
from .p19_g11 import encode_binary, kernel


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compile_preview(target, graph, out):
    if target == "p19a" and graph.suffix.lower() in (".yhp", ".yhb"):
        data = transfer_audit(graph.read_bytes())
    else:
        spec = json.loads(graph.read_text(encoding="utf-8"))
        if target == "p19a":
            data = pipeline(**spec)
            result, commands = reference(data)
            data["reference"] = result
            data["transfer_barrier_words_hex"] = [f"{command:016x}" for command in commands]
        else:
            words = kernel(**spec)
            data = {"target": target, "register_window": spec,
                    "instruction_count": len(words), "word_endianness": "little"}
    out.mkdir()
    (out / "valence_p19_abi_draft.h").write_text(c_header(), encoding="ascii")
    write_json(out / "plan.json", data)
    if target == "p19b-g11":
        (out / "kernel.bin").write_bytes(encode_binary(words))
        (out / "kernel.S").write_text(".text\n.balign 4\n" + "".join(f"    .word 0x{w:08x}\n" for w in words))
    return data
