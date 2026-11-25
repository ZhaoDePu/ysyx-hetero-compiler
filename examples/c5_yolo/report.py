"""Independent floating-point decode and held-out detection comparison."""

import math

import numpy as np

from .scenes import BOX_SCALE, CLASS_NAMES, GRID, SCORE_SCALE, SIZE, THRESHOLD, TOP_K


def _confidence(logit):
    return 1 / (1 + math.exp(-min(50,max(-50,float(logit)))))


def decode_fp32(raw):
    rank = sorted(range(len(raw)), key=lambda i:(-float(raw[i,0]),i))[:TOP_K]
    return [{"index":i, "class":int(np.argmax(raw[i,1:3])),
             "score":_confidence(raw[i,0]),
             "box":[float(v) for v in GRID[i]+raw[i,4:8]]}
            for i in rank if _confidence(raw[i,0]) > THRESHOLD]


def decode_rtl(symbols, prefix=""):
    corners = np.asarray(symbols[prefix+"boxes"]).reshape(-1,4)*BOX_SCALE
    records = zip(symbols[prefix+"indices"],symbols[prefix+"classes"],symbols[prefix+"scores"],corners,strict=True)
    return [{"index":i, "class":cls, "score":_confidence(code*SCORE_SCALE), "score_code":code,
             "box":box.tolist()} for i,cls,code,box in records if _confidence(code*SCORE_SCALE) > THRESHOLD]


def iou(a,b):
    left,top,right,bottom = [max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])]
    intersection = max(0.,right-left)*max(0.,bottom-top)
    union = (max(0.,a[2]-a[0])*max(0.,a[3]-a[1])+
             max(0.,b[2]-b[0])*max(0.,b[3]-b[1])-intersection)
    return intersection/union if union else 0.


def match(predictions, reference, min_iou=.50):
    """Greedy geometric matching, class assessed after matching."""
    candidates = sorted(((iou(p["box"],r["box"]),i,j)
                         for i,p in enumerate(predictions) for j,r in enumerate(reference)),reverse=True)
    used_left,used_right,out = set(),set(),[]
    for overlap,i,j in candidates:
        if overlap < min_iou:
            break
        if i not in used_left and j not in used_right:
            used_left.add(i)
            used_right.add(j)
            record = {"detection":i,"reference":j,"iou":overlap,
                      "class_agrees":predictions[i]["class"]==reference[j]["class"]}
            if "score" in reference[j]:
                record["confidence_abs_error"] = abs(predictions[i]["score"]-reference[j]["score"])
            out.append(record)
    return out


def ledger(name, actual, fp32, truth):
    peers, labels = match(actual,fp32),match(actual,truth)
    correct = sum(r["class_agrees"] for r in labels)
    values = [r["confidence_abs_error"] for r in peers]
    return {"image":name,"detected":len(actual),"fp32_detected":len(fp32),"ground_truth":len(truth),
            "fp32_matches":len(peers),"fp32_class_correct":sum(r["class_agrees"] for r in peers),
            "mean_iou_vs_fp32":float(np.mean([r["iou"] for r in peers])) if peers else None,
            "fp32_score_mae":float(np.mean(values)) if values else None,
            "fp32_score_max_abs_err":max(values) if values else None,
            "gt_matches":len(labels),"gt_class_correct":correct,
            "mean_iou_vs_gt":float(np.mean([r["iou"] for r in labels])) if labels else None,
            "precision_iou50":correct/max(1,len(actual)),"recall_iou50":correct/max(1,len(truth)),
            "matches_fp32":peers,"matches_gt":labels}


def ascii_detection(image, detections):
    canvas = [["#" if float(px.max()) > .10 else "." for px in row] for row in image]
    yy, xx = np.meshgrid(np.arange(SIZE)+.5,np.arange(SIZE)+.5,indexing="ij")
    ink = [[[] for _ in range(SIZE)] for _ in range(SIZE)]
    for record in detections:
        x0,y0,x1,y1=record["box"]
        inside=(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)
        marker = "S" if record["class"]==0 else "D"
        for y,x in np.argwhere(inside):
            ink[y][x].append(marker)
    for y in range(SIZE):
        for x in range(SIZE):
            if ink[y][x]:
                canvas[y][x] = "*" if len(ink[y][x])>1 else ink[y][x][0]
    return "\n".join("".join(row) for row in canvas)


def sample_report(cases):
    lines=["# YOLO26-Tiny-Style 冻结 RTL 检测样本","",
           "S 为 square，D 为 diamond，* 为多个预测框重叠，# 为未覆盖的前景像素。",
           "坐标为原始图像像素单位。标注框允许越过图像边缘，ASCII 只显示可见区域。",""]
    for name,(image, actual, fp32, truth) in cases.items():
        accuracy=ledger(name,actual,fp32,truth)
        overlap=accuracy["mean_iou_vs_fp32"]
        overlap_text=f"{overlap:.4f}" if overlap is not None else "N/A"
        lines.extend([f"## {name}","",f"标注 {len(truth)}，FP32 {len(fp32)}，RTL {len(actual)}。",
                      f"RTL ↔ FP32 平均 IoU {overlap_text}，类别一致 "
                      f"{accuracy['fp32_class_correct']}/{accuracy['fp32_matches']}。","",
                      "```text",ascii_detection(image,actual),"```","",
                      "| 路径 | 类别 | 分数 | x1 | y1 | x2 | y2 |","|---|---|---:|---:|---:|---:|---:|"])
        for label,rows in (("RTL",actual),("FP32",fp32)):
            for r in rows:
                corners=" | ".join(f"{v:.3f}" for v in r["box"])
                cls=CLASS_NAMES[r["class"]]
                lines.append(f"| {label} | {cls} | {r['score']:.6f} | {corners} |")
        lines.append("")
    return "\n".join(lines)
