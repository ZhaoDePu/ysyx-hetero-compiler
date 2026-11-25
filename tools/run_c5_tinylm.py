"""Replay the offline TinyLM ONNX variants and export a checkpointed board case."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from examples.c5_tinylm.export import Q_LOGIT
from examples.c5_tinylm.train import generate_fp32
from hetero_compiler.compiler import tensor_bytes
from hetero_compiler.reference import evaluate
from hetero_compiler.tinylm import PITCH, PROMPT, VOCAB, compile_tinylm, read_onnx
from hetero_compiler.vpu_lowering import lower
from tools.c4a_imem import ReplayPrograms
from tools.verify_bundle import _z15_launch

RAM = Path("/dev/shm/ysyx_compiler_c5")
RUNNER = RAM / "npu/z15-build/run-npu-z15"


def fixed_reference(graph, prompts, native):
    """Evaluate the full causal graph per appended token, independent of SRAM layout."""
    graph = lower(graph) if native else graph
    expected = {}
    for lane, prompt in enumerate(prompts):
        vals = list(prompt)
        for index in range(PITCH - PROMPT + 1):
            pos = len(vals)-1
            values = evaluate(graph,{"tokens":vals+[0]*(PITCH-len(vals))})
            logits = values["lm.out"][pos*VOCAB:(pos+1)*VOCAB]
            token = values["token"][pos]
            expected[f"p{lane}_logits_{index}"] = logits
            expected[f"p{lane}_token_{index}"] = [token]
            if index == PITCH-PROMPT:
                for layer in range(2):
                    for head in range(2):
                        for label in ("k","v"):
                            expected[f"p{lane}_l{layer}_h{head}_{label.upper()}"] = values[
                                f"l{layer}.h{head}.{label}.out"]
            vals.append(token)
    return expected


def replay(bundle, desc, expected, root):
    if expected.keys() != desc["output_bindings"].keys():
        raise ValueError("TinyLM oracle symbols differ from compiled outputs")
    root.mkdir()
    gpu = {r["id"]:r for r in desc["backends"]["gpu"]["stages"]}
    npu = {r["stage"]:r for r in desc["backends"]["npu"]["commands"]}
    images = {key:[int(line,16) for line in (bundle/r["hex"]).read_text().splitlines()] for key,r in gpu.items()}
    cache = ReplayPrograms(images)
    src = bundle/"initial_ddr.bin"
    commands = ["- 0 0",f"ddr_put {src} 0 {src.stat().st_size}", *cache.preload]
    verified = {}
    pending = []
    for row in desc["execution"]:
        kind = row["type"]
        if kind == "launch":
            commands.extend(_z15_launch(bundle,row["stage"],gpu,npu,cache))
        elif kind == "fence":
            stage = row["wait_for"]
            commands.append(f"wait_{'gpu' if stage in gpu else 'npu'} {stage}")
        elif kind == "dma":
            direction = "dma_in" if row["direction"] == "load" else "dma_out"
            commands.append(f"{direction} {row['ddr_offset']} {row['scratch_offset']} {row['bytes']}")
            if row["direction"] == "store":
                pending.append(row["binding"])
        elif kind == "dma-fence":
            commands.append("dma_wait")
        elif kind == "yield":
            for name in pending:
                data = desc["output_bindings"][name]
                dst = root / f"{name}.bin"
                blob = tensor_bytes(expected[name],data["dtype"])
                if len(blob) != data["size"]:
                    raise ValueError(f"{name} proof size mismatch")
                dst.write_bytes(blob)
                commands.append(f"compare_ddr {dst} {data['ddr_offset']} {len(blob)}")
                if name in verified:
                    raise ValueError(f"duplicate TinyLM output DMA for {name}")
                verified[name] = len(blob)
            pending.clear()
        else:
            raise ValueError(f"unhandled TinyLM command {kind}")
    if pending or verified.keys() != expected.keys():
        raise ValueError("missing TinyLM output stores or undrained checkpoints")
    control = root/"mmio.txt"
    control.write_text("\n".join(commands)+"\n")
    log = root/"rtl.log"
    nice_adjustment = max(0, 10 - os.nice(0))
    with log.open("w") as stream:
        proc = subprocess.run(["nice","-n",str(nice_adjustment),str(RUNNER),str(control)], stdout=stream,
                              stderr=subprocess.STDOUT, timeout=1800, check=False)
    raw = log.read_text()
    if proc.returncode or raw.count("Z15_DDR_PASS ") != len(verified):
        detail = "\n".join(raw.splitlines()[-10:])
        raise RuntimeError(f"TinyLM frozen comparison failed: {log}\n{detail}")
    rows = [json.loads(line.split(" ",1)[1]) for line in raw.splitlines() if line.startswith("Z15_METRICS ")]
    assert len(rows)==1
    stats = rows[0]
    if stats["rtl_dma_bytes"] != desc["estimates"]["external_transfer_bytes"]:
        raise RuntimeError("TinyLM schedule and physical DMA byte counters differ")
    return dict(stats, status="byte_exact", compared_outputs=len(verified), compared_bytes=sum(verified.values()),
                imem_unique=len(cache.address), imem_reloads=cache.reloads)


def accuracy(expected, prompts, fp, vocab):
    result = []
    for lane,prompt in enumerate(prompts):
        tokens, fp_logits = generate_fp32(fp,prompt)
        got = [expected[f"p{lane}_token_{i}"][0] for i in range(len(tokens))]
        logits = np.array([expected[f"p{lane}_logits_{i}"] for i in range(len(tokens))]) / 2**Q_LOGIT
        fp_logits = fp_logits.astype(np.float64)
        norms = np.linalg.norm(logits,axis=1)*np.linalg.norm(fp_logits,axis=1)
        cosine = np.divide((logits*fp_logits).sum(axis=1),norms,out=np.zeros_like(norms),where=norms!=0)
        result.append({"prompt":" ".join(vocab[t] for t in prompt),
                       "fp32":" ".join(vocab[t] for t in tokens).replace(" .","."),
                       "rtl":" ".join(vocab[t] for t in got).replace(" .","."),
                       "fp32_ids":tokens,"rtl_ids":got,"tokens":len(tokens),
                       "top1":sum(a==b for a,b in zip(tokens,got,strict=True)),
                       "logit_cosine_mean":float(cosine.mean()),
                       "logit_cosine_by_token":cosine.tolist(),
                       "logit_mae":float(np.abs(logits-fp_logits).mean())})
    return result


def board_case(bundle, desc, expected, prompt, words, root):
    root.mkdir()
    shutil.copyfile(bundle/"program.yhb",root/"program.yhb")
    (root/"input-0.bin").write_bytes(tensor_bytes(prompt,"int64"))
    lines = ["input\tprompt_0\tinput-0.bin\n"]
    for i,(name,spec) in enumerate(desc["output_bindings"].items()):
        file = f"output-{i}.bin"
        (root/file).write_bytes(tensor_bytes(expected[name],spec["dtype"]))
        lines.append(f"output\t{name}\t{file}\n")
    (root/"io.tsv").write_text("".join(lines))
    (root/"samples.txt").write_text("1\n")
    (root/"kv.tsv").write_text("".join(f"step\t{i}\n" for i in range(PITCH-PROMPT+1)))
    (root/"vocab.tsv").write_text("".join(f"{i}\t{word}\n" for i,word in enumerate(words)))


def run(root, learned, variants, kit):
    model = learned/"TinyLM-Chat.onnx"
    graph = read_onnx(model)[0]
    training = json.loads((learned/"training.json").read_text())
    prompts,words = (training[k] for k in ("prompts","vocab"))
    root.mkdir()
    records, results = [], {}
    for phase,vpu in variants:
        slug = phase+"-"+vpu
        bundle = root/slug
        desc = compile_tinylm(model,bundle,prompts,phase=phase,vpu=vpu=="on")
        native = desc["tinylm"]["effective_native_attention"]
        ref = fixed_reference(graph,prompts,native)
        with np.load(learned/"weights.fp32.npz",allow_pickle=False) as w:
            proof = accuracy(ref,prompts,w,words)
        (root/f"{slug}-reference.json").write_text(json.dumps(proof,indent=2)+"\n")
        for r in proof:
            print(f"{slug} {r['prompt']} -> {r['rtl']} ({r['top1']}/{r['tokens']})",flush=True)
        stats = replay(bundle,desc,ref,root/(slug+"-rtl"))
        tokens = len(prompts) * (PITCH-PROMPT+1)
        ticks = stats["wall_npu_ticks"]
        extra = dict(generated_tokens=tokens, latency_164_555_ms=ticks/164555, latency_125_ms=ticks/125000,
                     ms_per_token_164_555=ticks/164555/tokens,ms_per_token_125=ticks/125000/tokens,
                     toks_s_164_555=tokens*164555000/ticks,toks_s_125=tokens*125000000/ticks,
                     total_top1=sum(row["top1"] for row in proof),
                     mean_cosine=float(np.mean([row["logit_cosine_mean"] for row in proof])))
        records.append(dict(phase=phase,requested_vpu=vpu,native_vpu=native,**desc["estimates"],**stats,**extra))
        results[slug]=proof
        print(f"TINYLM_RTL_PASS {slug} ticks={ticks} bytes={stats['compared_bytes']}",flush=True)
    if kit:
        kit_bundle=root/"kit-bundle"
        desc=compile_tinylm(model,kit_bundle,prompts[:1],phase="A3",vpu=True)
        expected=fixed_reference(graph,prompts[:1],True)
        kstats=replay(kit_bundle,desc,expected,root/"kit-rtl")
        (root/"kit-stats.json").write_text(json.dumps(kstats,indent=2)+"\n")
        board_case(kit_bundle,desc,expected,prompts[0],words,root/"kit-case")
    baseline = next((r["wall_npu_ticks"] for r in records if r["phase"]=="A0"), None)
    if baseline:
        for row in records:
            row["speedup_vs_A0"] = baseline / row["wall_npu_ticks"]
    (root/"performance.json").write_text(json.dumps(records,indent=2)+"\n")
    (root/"generation.json").write_text(json.dumps(results,indent=2)+"\n")
    with (root/"performance.csv").open("w",newline="") as out:
        w=csv.DictWriter(out,fieldnames=list(records[0]) if records else ["phase"])
        w.writeheader()
        w.writerows(records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--learned",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--variant",action="append", choices=[f"A{i}-{s}" for i in range(4) for s in ("on","off")])
    p.add_argument("--kit",action="store_true")
    args=p.parse_args()
    if RAM not in args.out.resolve().parents or RAM.resolve()!=RAM:
        p.error("output must live in the C5 scratch directory")
    variants=[key.split("-") for key in args.variant] if args.variant else ([] if args.kit else
                            [(f"A{i}", vpu) for i in range(4) for vpu in ("off","on")])
    run(args.out,args.learned,variants,args.kit)


if __name__=="__main__":
    main()
