"""Train and verify the full YOLO26-Tiny-Style detector on frozen Z15 MMIO."""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from examples.c5_yolo.export import encode
from examples.c5_yolo.report import decode_fp32, decode_rtl, ledger, sample_report
from examples.c5_yolo.scenes import BOX_SCALE, CLASS_NAMES, SCORE_SCALE, THRESHOLD, held_out, quantized_image
from examples.c5_yolo.train import forward, load_parameters, save_parameters, train
from hetero_compiler.compiler import compile_bundle, tensor_bytes
from hetero_compiler.onnx import parse_model
from hetero_compiler.profile import load_profile
from hetero_compiler.reference import evaluate as fixed_reference

ROOT=Path("/dev/shm/ysyx_compiler_c5")
CLOCKS={"Phase_12C":164.555,"Step_2":125.0}
ABLATIONS={
    "A0":dict(device_policy="gpu-only",fuse=False,reuse=False,overlap=False),
    "A1":dict(device_policy="npu-greedy",fuse=False,reuse=False,overlap=False),
    "A2":dict(device_policy="npu-greedy",fuse=True,reuse=True,overlap=False),
    "A3":dict(device_policy="npu-greedy",fuse=True,reuse=True,overlap=True),
}


def _dump(where,data):
    where.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def _csv(where,rows):
    fields=sorted({k for r in rows for k,v in r.items() if not isinstance(v,(dict,list))})
    with where.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _tagged(log,key):
    rows=[json.loads(line[len(key)+1:]) for line in log.splitlines() if line.startswith(key+" ")]
    if len(rows)!=1:
        raise RuntimeError(f"expected exactly one {key} record")
    return rows[0]


def _priority(args):
    return ["nice","-n",str(max(0,10-os.nice(0))),"ionice","-c","2","-n","7",*map(str,args)]


def verify_variant(out,raw,level,mode,frames):
    profile=load_profile("fpga_zu15eg_balanced")
    profile["pipeline_scheduler"]="bank-aware"
    if mode == "off":
        profile["vpu"].update(enabled=False, verified=False)
    variant=f"yolo-{mode}-{level}"
    bundle=out/variant
    manifest=compile_bundle(raw,profile,bundle,pipeline_tiles=True,vpu_lowering=mode,**ABLATIONS[level])
    run=out/(variant+".rtl")
    log=out/(variant+".log")
    with log.open("w") as fd:
        subprocess.run(_priority([sys.executable,"-B","-m","tools.verify_bundle","--bundle",bundle,
                                 "--case",run,"--execute-plan","--imem-cache"]),
                       stdout=fd,stderr=subprocess.STDOUT,check=True,timeout=300)
    transcript=log.read_text()
    metrics=_tagged(transcript,"Z15_METRICS")
    outputs=_tagged(transcript,"BUNDLE_RTL_PASS")
    if set(outputs)!=set(manifest["output_bindings"]) or len(outputs)!=frames*4:
        raise RuntimeError("RTL replay did not validate every published detection tensor")
    if transcript.count("Z15_DDR_PASS ")!=len(outputs):
        raise RuntimeError("published output count differs from actual DDR comparison count")
    stats=manifest["estimates"]
    if stats["external_transfer_bytes"]!=metrics["rtl_dma_bytes"]:
        raise RuntimeError("actual DMA bytes do not match planned transfer")
    stages=manifest["plan"]["stages"]
    if level == "A0" and any(s["device"]!="GPU" for s in stages):
        raise RuntimeError("pure GPU baseline contains a native stage")
    if mode == "off" and any(s["device"]=="VPU" for s in stages):
        raise RuntimeError("VPU-off baseline contains a vector stage")
    if level in ("A2","A3") and not any(s["device"]=="NPU" and s.get("fused") for s in stages):
        raise RuntimeError("optimized variant did not fuse any NPU operator")
    if mode=="on" and level in ("A2","A3") and not any(s["op"]=="Z15RMSNorm" for s in stages):
        raise RuntimeError("hardware variant did not lower RMSNorm to the VPU")
    _dump(out/(variant+".outputs.json"),outputs)
    row={"variant":variant,"vpu_lowering":mode,"ablation":level,"frames":frames,"status":"byte_exact",
         "verified_outputs":len(outputs),"verified_bytes":sum(x["size"] for x in manifest["output_bindings"].values()),
         "scratch_peak_live_bytes":stats["scratch_peak_live_bytes"],"scratch_reserved_bytes":stats["scratch_reserved_bytes"],
         "device_stages":stats["device_stage_counts"],"fused_stage_count":sum(bool(s.get("fused")) for s in stages),
         "vpu_ops":dict(Counter(s["op"] for s in stages if s["device"]=="VPU")),
         **metrics}
    for clock,mhz in CLOCKS.items():
        row["fps_"+clock]=frames*mhz*1e6/metrics["wall_npu_ticks"]
    return row,outputs


def _accuracy_summary(rows):
    selected = [r for r in rows if r["variant"]=="yolo-on-A3"]
    matched = sum(r["fp32_matches"] for r in selected)
    truthful = sum(r["gt_matches"] for r in selected)

    def mean(metric,count_key,total):
        if not total:
            return None
        return sum((r[metric] or 0)*r[count_key] for r in selected)/total

    errors = [r["fp32_score_max_abs_err"] for r in selected if r["fp32_score_max_abs_err"] is not None]
    return {"images":len(selected),"detections":sum(r["detected"] for r in selected),
            "fp32_matches":matched,"fp32_classes":sum(r["fp32_class_correct"] for r in selected),
            "weighted_mean_iou_vs_fp32":mean("mean_iou_vs_fp32","fp32_matches",matched),
            "weighted_mean_iou_vs_gt":mean("mean_iou_vs_gt","gt_matches",truthful),
            "weighted_mean_confidence_mae_vs_fp32":mean("fp32_score_mae","fp32_matches",matched),
            "max_confidence_abs_error_vs_fp32":max(errors,default=None),
            "gt_total":sum(r["ground_truth"] for r in selected),
            "gt_iou50_class_correct":sum(r["gt_class_correct"] for r in selected)}


def export_kit(out,single,cases,compare,runner):
    profile=load_profile("fpga_zu15eg_balanced")
    manifest=compile_bundle(single,profile,out/"kit-bundle",vpu_lowering="on",device_policy="npu-greedy")
    case=out/"kit-case"
    case.mkdir()
    images=case/"images"
    images.mkdir()
    shutil.copy2(out/"kit-bundle"/"program.yhb",case/"program.yhb")
    (case/"detection.tsv").write_text(f"score_scale\t{SCORE_SCALE}\nscore_threshold\t{THRESHOLD}\n"
                                       f"box_codes_per_pixel\t{round(1/BOX_SCALE)}\n",encoding="ascii")
    (case/"labels.tsv").write_text("".join(f"{i}\t{name}\n" for i,name in enumerate(CLASS_NAMES)),encoding="ascii")
    bindings=[]
    result={}
    verify_root=out/"kit-c99"
    verify_root.mkdir()
    for idx,(name,(image,_)) in enumerate(cases.items()):
        src=images/(name+".bin")
        values=quantized_image(image).ravel()
        src.write_bytes(values.tobytes())
        if idx == 0:
            shutil.copy2(src,case/"input-0.bin")
            bindings.append(("input","image","input-0.bin"))
        graph=manifest["graph"]
        evaluated=fixed_reference(graph,{"image":values.tolist()})
        reference={key:evaluated[key] for key in graph["outputs"]}
        if any(reference[k]!=compare[f"im{idx:02d}_"+k] for k in reference):
            raise RuntimeError(f"{name}: stand-alone YHB does not match batched frozen outputs")
        probe=verify_root/name
        probe.mkdir()
        cmd=[runner,case/"program.yhb","--backend","verilator","--input",f"image={src}","--dump-stats"]
        for number,key in enumerate(manifest["graph"]["outputs"]):
            output=probe/(key+".bin")
            cmd.extend(["--output",f"{key}={output}"])
            if idx == 0:
                file=f"output-{number}.bin"
                (case/file).write_bytes(tensor_bytes(reference[key],graph["tensors"][key]["dtype"]))
                bindings.append(("output",key,file))
        with (probe/"runtime.log").open("w") as log:
            subprocess.run(_priority(cmd),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        for key,expected in reference.items():
            wanted=tensor_bytes(expected,graph["tensors"][key]["dtype"])
            if wanted!=(probe/(key+".bin")).read_bytes():
                raise RuntimeError(f"{name} {key}: C99 output differs")
        stats=[ln for ln in (probe/"runtime.log").read_text().splitlines() if ln.startswith("STATS ")]
        if (len(stats)!=1 or "abi_error=0x00000000" not in stats[0] or
                "invalid_dma_descriptors=0 " not in stats[0]+" "):
            raise RuntimeError(f"{name}: C99 replay did not demonstrate a clean ABI")
        result[name]={"status":"byte_exact","frames":decode_rtl(reference),"stats":stats[0]}
    (case/"io.tsv").write_text("".join("\t".join(row)+"\n" for row in bindings),encoding="ascii")
    (case/"samples.txt").write_text("1\n",encoding="ascii")
    _dump(verify_root/"result.json",result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path,help="Fresh directory below Phase C5 RAM")
    parser.add_argument("--parameters",type=Path)
    parser.add_argument("--runner",type=Path,default=ROOT/"runtime/verilator/ysyx_run")
    args=parser.parse_args()
    out=args.out.resolve()
    if ROOT not in out.parents or ROOT.resolve()!=ROOT:
        parser.error("output must be below the dedicated Phase C5 RAM directory")
    out.mkdir()
    if args.parameters:
        params=load_parameters(args.parameters)
        report={"external_parameters":str(args.parameters)}
    else:
        params,report=train()
    save_parameters(out/"parameters.json",params)
    _dump(out/"training.json",report)
    cases=held_out()
    samples=[image for image,_ in cases.values()]
    encoded=encode(params,samples,independent=True)
    (out/"YOLO26_Tiny_Style_4screen_INT8.onnx").write_bytes(encoded)
    packaged=encode(params,samples[:1])
    (out/"YOLO26_Tiny_Style_INT8.onnx").write_bytes(packaged)
    raw=parse_model(encoded)
    fp32=forward(np.stack(samples),params)[0]
    goldens={name:decode_fp32(pred) for name,pred in zip(cases,fp32,strict=True)}
    observed,rows,accuracy={},[],[]
    expected=None
    for mode in ("off","on"):
        base=None
        for level in ABLATIONS:
            row,outputs=verify_variant(out,raw,level,mode,len(samples))
            if expected is None:
                expected=outputs
            if outputs!=expected:
                raise RuntimeError(f"{mode} {level}: detection bytes changed across equivalent ablations")
            if base is None:
                base=row["wall_npu_ticks"]
            row["speedup_vs_A0"]=base/row["wall_npu_ticks"]
            print(row["variant"],row["status"],row["wall_npu_ticks"],flush=True)
            observed[row["variant"]]=outputs
            rows.append(row)
            for n,(name,(_,truth)) in enumerate(cases.items()):
                detections=decode_rtl(outputs,f"im{n:02d}_")
                accuracy.append({"variant":row["variant"],**ledger(name,detections,goldens[name],truth)})
    _dump(out/"performance.json",{"measurement":"frozen Step 2 Balanced Verilator MMIO","clocks_mhz":CLOCKS,
                                  "case_names":list(cases),"strict_vpu_off":True,"results":rows})
    _csv(out/"performance.csv",rows)
    _dump(out/"accuracy.json",accuracy)
    _csv(out/"accuracy.csv",accuracy)
    _dump(out/"accuracy_summary.json",_accuracy_summary(accuracy))
    final=observed["yolo-on-A3"]
    published={}
    sample={}
    for n,(name,(image,truth)) in enumerate(cases.items()):
        actual=decode_rtl(final,f"im{n:02d}_")
        published[name]={"rtl":actual,"fp32":goldens[name],"ground_truth":truth}
        sample[name]=(image,actual,goldens[name],truth)
    _dump(out/"detections.json",published)
    (out/"detection_samples.md").write_text(sample_report(sample),encoding="utf-8")
    export_kit(out,parse_model(packaged),cases,final,args.runner)


if __name__=="__main__":
    main()
