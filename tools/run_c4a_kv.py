"""Build and measure a cached and a recompute causal session in frozen RTL."""

import argparse
import csv
import json
from pathlib import Path
import subprocess

from hetero_compiler.compiler import tensor_bytes
from hetero_compiler.kv_reference import evaluate_kv
from hetero_compiler.kv_schedule import compile_kv_bundle
from tools.verify_bundle import _z15_launch

_ROOT = Path("/dev/shm/ysyx_compiler_c4a")
_FROZEN = _ROOT/"npu/z15-build/run-npu-z15"


def _blob(case, name, data):
    path=case/name
    path.write_bytes(data)
    return path


def replay(bundle,manifest,case,runner=_FROZEN):
    case.mkdir()
    weights=bundle/manifest["initial_ddr"]
    commands=["- 0 0",f"ddr_put {weights} 0 {weights.stat().st_size}"]
    backend=manifest["backends"]
    gpu={stage["id"]:stage for stage in backend["gpu"]["stages"]}
    npu={stage["stage"]:stage for stage in backend["npu"]["commands"]}
    expected=evaluate_kv(manifest["kv"]["model"])
    pending=[]
    output_bindings=manifest["output_bindings"]
    command_labels={stage["id"]:f"s{i}" for i,stage in enumerate(manifest["plan"]["stages"])}
    compares=0

    def verify_pending():
        nonlocal compares
        for name in pending:
            spec=output_bindings[name]
            path=_blob(case,name+".golden.bin",tensor_bytes(expected[name],spec["dtype"]))
            if path.stat().st_size != spec["size"]:
                raise ValueError(f"{name}: reference disagrees with output shape")
            commands.append(f"compare_ddr {path} {spec['ddr_offset']} {spec['size']}")
            compares+=1
        pending.clear()

    for cmd in manifest["execution"]:
        kind=cmd["type"]
        if kind=="launch":
            commands.extend(_z15_launch(bundle,cmd["stage"],gpu,npu))
        elif kind=="fence":
            stage=cmd["wait_for"]
            engine="gpu" if stage in gpu else "npu"
            commands.append(f"wait_{engine} {command_labels[stage]}")
        elif kind=="dma":
            rw="dma_in" if cmd["direction"]=="load" else "dma_out"
            commands.append(f"{rw} {cmd['ddr_offset']} {cmd['scratch_offset']} {cmd['bytes']}")
            if rw=="dma_out" and cmd["binding"] in output_bindings:
                pending.append(cmd["binding"])
        elif kind=="dma-fence":
            commands.append("dma_wait")
        elif kind=="yield":
            verify_pending()
        else:
            raise ValueError(f"unknown command {kind}")
    verify_pending()
    control=case/"mmio.txt"
    control.write_text("\n".join(commands)+"\n")
    log=case/"rtl.log"
    with log.open("w") as fout:
        proc=subprocess.run(["nice","-n","10",str(runner),str(control)],
                            stdout=fout,stderr=subprocess.STDOUT,timeout=900,check=False)
    text=log.read_text()
    if proc.returncode or text.count("Z15_DDR_PASS ")!=compares or compares!=len(output_bindings):
        raise RuntimeError(f"RTL output verification failed: {log}\n"+"\n".join(text.splitlines()[-8:]))
    raw,=[line.split(" ",1)[1] for line in text.splitlines() if line.startswith("Z15_METRICS ")]
    result=json.loads(raw)
    if result.get("rtl_dma_bytes")!=manifest["estimates"]["external_transfer_bytes"]:
        raise RuntimeError("hardware AXI transfer counter differs from scheduled bytes")
    result.update(mode=manifest["kv"]["mode"],status="byte_exact",
                  compared_outputs=compares,compared_bytes=sum(o["size"] for o in output_bindings.values()),
                  **{k:manifest["estimates"][k] for k in ("projected_kv_rows","scratch_peak_live_bytes",
                                                         "scratch_reserved_bytes","prefill_tiles","decode_steps")})
    result["wall_us_npu_fmax"]=result["wall_npu_ticks"]/164.555
    result["tokens"]=[expected[f"decode_{step}_token"][0] for step in range(1,result["decode_steps"]+1)]
    return result


def export_case(bundle,manifest,case):
    case.mkdir()
    (case/"program.yhb").write_bytes((bundle/"program.yhb").read_bytes())
    model=manifest["kv"]["model"]
    _blob(case,"input-0.bin",tensor_bytes(model["prompt"],"int8"))
    lines=["input\tprompt\tinput-0.bin"]
    expected=evaluate_kv(model)
    for idx,(name,info) in enumerate(manifest["output_bindings"].items()):
        file=f"output-{idx}.bin"
        _blob(case,file,tensor_bytes(expected[name],info["dtype"]))
        lines.append(f"output\t{name}\t{file}")
    (case/"io.tsv").write_text("\n".join(lines)+"\n")
    (case/"samples.txt").write_text("1\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,default=Path("examples/c4a_kv/causal_kv32.json"))
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    source=json.loads(args.model.read_bytes())
    root=args.out.resolve()
    if _ROOT not in root.parents or root.exists():
        raise SystemExit("expected a new directory inside C4-A compiler RAM")
    root.mkdir()
    rows=[]
    for mode in ("recompute","cached"):
        bundle=root/mode
        manifest=compile_kv_bundle(source,bundle,mode=mode)
        row=replay(bundle,manifest,root/(mode+"-rtl"))
        print(mode,row["wall_npu_ticks"],row["rtl_dma_bytes"],row["tokens"],flush=True)
        rows.append(row)
        if mode=="cached":
            export_case(bundle,manifest,root/"kit-case")
    if rows[0]["tokens"]!=rows[1]["tokens"]:
        raise RuntimeError("cached and baseline produced different token sequences")
    result={"measurement":"frozen Z15 Balanced Verilator MMIO","npu_fmax_mhz":164.555,
            "speedup":rows[0]["wall_npu_ticks"]/rows[1]["wall_npu_ticks"],"results":rows}
    (root/"kv-compare.json").write_text(json.dumps(result,indent=2)+"\n")
    with (root/"kv-compare.csv").open("w",newline="") as out:
        w=csv.DictWriter(out,fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
