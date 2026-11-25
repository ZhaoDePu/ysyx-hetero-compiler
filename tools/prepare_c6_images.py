from pathlib import Path
import collections, hashlib, json, zipfile

root=Path.cwd()
release="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
audit={"eval":{},"calibration":{}}
ev=root/"eval"; cal=root/"calibration"
ev.mkdir(exist_ok=True); cal.mkdir(exist_ok=True)
found=json.loads((root/"downloads.json").read_text(encoding="utf-8"))
for src in ("bus.jpg","zidane.jpg"):
  (ev/src).write_bytes((root/src).read_bytes())
  audit["eval"][src]=found[src]
eval_ids=("000000000241","000000000384")
with zipfile.ZipFile(root/"coco128.zip") as z:
  prefix="coco128/images/train2017/"
  labels="coco128/labels/train2017/"
  files={Path(n).stem:n for n in z.namelist() if n.startswith(prefix) and n.endswith(".jpg")}
  def export(kind,dest,idx):
    src=files[idx]; data=z.read(src); name=f"coco{idx}.jpg"
    (dest/name).write_bytes(data)
    audit[kind][name]=dict(source=release,member=src,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
  for idx in eval_ids: export("eval",ev,idx)
  remain=set(files)-set(eval_ids)
  def cats(idx):
    try: return set(int(row.split()[0]) for row in z.read(labels+idx+".txt").splitlines() if row.strip())
    except KeyError: return set()
  classes={idx:cats(idx) for idx in remain}
  frequency=collections.Counter()
  for i in range(32):
    idx=min(remain,key=lambda k:(-sum(1/(frequency[x]+1)**2 for x in classes[k]),k))
    remain.remove(idx); frequency.update(classes[idx]); export("calibration",cal,idx)
  audit["calibration_source"]={"suite":"Ultralytics COCO128 subset of COCO train2017","source":release,"separate_image_ids":True,"model_training_overlap":"COCO train2017; do not treat these four scenes as a held-out COCO benchmark","represented_label_count":len(frequency)}
(root/"images.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
print("eval",sorted(audit['eval']),"calibration",len(audit['calibration']),"labels",len(frequency))
