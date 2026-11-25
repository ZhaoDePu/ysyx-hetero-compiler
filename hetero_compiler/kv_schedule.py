"""Lower a bounded static causal decoder to persistent KV storage and YHB commands."""

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from . import gpu, npu
from .compiler import tensor_bytes
from .ir import DTYPES, GraphError
from .planner import _align, _allocate
from .profile import load_profile
from .yhb import emit_yhb

_SCHEMA = "ysyx-causal-kv-v1"
_PROJECTIONS = ("q", "k", "v", "out", "ff1", "ff2", "lm")
_MAX_PAGE = 4096
_BEAT = 32
_WIDTH = 4
_SCORE_SHIFT = 3
# Raw dot products represented approximately logit = raw / 64 in C4-A.
# The native base-two weights use the nearest integral exp shift to that scale.
_NATIVE_EXP_SHIFT = 5


def normalize_model(raw):
    """Validate the static integer transformer and its runtime prompt binding."""
    if not isinstance(raw, dict) or raw.get("schema") != _SCHEMA:
        raise GraphError("unsupported causal KV model schema")
    sequence = raw.get("prefill_rows")
    tile = raw.get("tile_rows")
    steps = raw.get("decode_steps")
    if (type(sequence) is not int or sequence not in (32, 64) or type(tile) is not int or
            tile not in (4, 8) or sequence % tile or type(steps) is not int or not 4 <= steps <= 8):
        raise GraphError("KV geometry requires a 32/64-row prompt, 4/8-row tiles and 4..8 decode steps")
    if raw.get("width") != _WIDTH:
        raise GraphError("the frozen KV lowering supports width four")

    def integers(location, values, count):
        if (not isinstance(values, list) or len(values) != count or
                any(type(x) is not int or not -128 <= x <= 127 for x in values)):
            raise GraphError(f"{location} must contain {count} signed bytes")
        return list(values)

    weights = raw.get("weights")
    if not isinstance(weights, dict) or set(weights) != set(_PROJECTIONS):
        raise GraphError("KV weights must contain exactly " + ", ".join(_PROJECTIONS))
    name = raw.get("name", "causal_kv")
    if not isinstance(name, str) or not name:
        raise GraphError("KV model name must be a nonempty string")
    gain = integers("token_gain", raw.get("token_gain"), _WIDTH)
    bias = integers("token_bias", raw.get("token_bias"), _WIDTH)
    if any(not -128 <= g*i+b <= 127 for g,b in zip(gain,bias,strict=True) for i in range(_WIDTH)):
        raise GraphError("affine token embedding would saturate")
    return {"schema": _SCHEMA, "name": name, "prefill_rows": sequence, "tile_rows": tile,
            "decode_steps": steps, "width": _WIDTH,
            "weights": {k: integers(k, weights[k], _WIDTH**2) for k in _PROJECTIONS},
            "prompt": integers("prompt", raw.get("prompt"), sequence*_WIDTH),
            "token_gain": gain, "token_bias": bias}


def _nbytes(spec):
    return DTYPES[spec["dtype"]][0] * math.prod(spec["shape"])


class _Schedule:
    def __init__(self, model, mode, checkpoints, vpu_lowering, vpu_attention):
        self.model, self.mode, self.checkpoints = model, mode, checkpoints
        self.vpu_lowering, self.vpu_attention = vpu_lowering, vpu_attention
        self.native_attention = vpu_lowering == "on"
        self.profile = load_profile("fpga_zu15eg_balanced")
        self.tensors, self.storage, self.views = {}, {}, {}
        self.stages, self.execution, self.tiles = [], [], []
        self.inputs, self.outputs = {}, {}
        self.packed, self.loads = {}, []
        self.bank = 0

    def tensor(self, name, shape, dtype, kind="temp", data=None):
        if name in self.tensors:
            raise GraphError(f"duplicate physical tensor {name}")
        self.tensors[name] = {"shape": list(shape), "dtype": dtype, "kind": kind}
        if data is not None:
            self.tensors[name]["data"] = list(data)
        if kind not in ("ssa", "alias"):
            self.storage[name] = {"size": _nbytes(self.tensors[name]), "bank": self.bank,
                                  "kind": kind}
        return name

    def view(self, name, parent, start, shape):
        self.tensor(name, shape, self.tensors[parent]["dtype"], "alias")
        if start < 0 or start + _nbytes(self.tensors[name]) > self.storage[parent]["size"]:
            raise GraphError(f"{name} exceeds backing allocation {parent}")
        self.views[name] = (parent, start)
        return name

    def move(self, direction, tensor, binding, at_stage=True):
        target = self.execution if at_stage else self.loads
        target.extend([{"type": "dma", "direction": direction, "tensor": tensor, "binding": binding},
                       {"type": "dma-fence"}])

    def input(self, tensor, binding, default):
        self.inputs[binding] = {"shape": self.tensors[tensor]["shape"], "dtype": self.tensors[tensor]["dtype"],
                                "data": default, "kind": "input"}

    def output(self, tensor, binding):
        self.outputs[binding] = self.tensors[tensor].copy()
        self.move("store", tensor, binding)

    def stage(self, name, op, ins, out, attrs=None, device="GPU", fused=None):
        obj = {"id": name, "op": op, "inputs": ins, "output": out, "attrs": attrs or {},
               "device": device, "zone": 0, "bank": self.bank}
        if fused:
            obj["fused"] = fused
        self.stages.append(obj)
        self.execution.extend([{"type": "launch", "stage": name, "device": device, "zone": 0},
                               {"type": "fence", "wait_for": name, "zone": 0}])
        return obj

    def project(self, name, inp, projection, out, shift, relu=False):
        mid = self.tensor(name+"-acc", self.tensors[out]["shape"], "int32", "ssa")
        req = {"id": name+"-requant", "op": "Requant", "inputs": [mid],
               "output": out, "attrs": {"shift": shift}}
        ins = [inp, "weight-"+projection]
        chain = [{"id": name, "op": "Gemm", "inputs": ins, "output": mid, "attrs": {}}, req]
        if relu:
            mid = self.tensor(name+"-rq-value", self.tensors[out]["shape"], "int8", "ssa")
            req["output"] = mid
            chain.append({"id": name+"-relu", "op": "Relu", "inputs": [mid],
                          "output": out, "attrs": {}})
        self.stage(name, "Gemm", ins, out, device="NPU", fused=chain)

    def scores(self, name, q, kp, out):
        mid = self.tensor(name+"-acc", self.tensors[out]["shape"], "int32", "ssa")
        ins = [q,kp]
        chain = [{"id": name, "op": "Gemm", "inputs": ins, "output": mid,
                  "attrs": {"transpose_b": True}},
                 {"id": name+"-requant", "op": "Requant", "inputs": [mid],
                  "output": out, "attrs": {"shift": _SCORE_SHIFT}}]
        self.stage(name, "Gemm", ins, out, fused=chain)

    def add(self, name, a, b, out):
        node = {"op": "Add", "inputs": [a,b], "output": out, "attrs": {}}
        graph = {"tensors": self.tensors}
        self.stage(name, "Add", [a,b], out, device="VPU" if npu.supports(node,graph,self.profile) else "GPU")

    def attend_native(self, name, q, start, rows, end, new_start):
        # ATTN uses the full fixed pitch. Its causal mask excludes all slots
        # beyond the current query, so future rows need no initialization.
        pitch = self.tensors["k-state"]["shape"][0]
        vp = self.view(name+"-values-append-packed", "value-pack-buffer", new_start,
                       (end - new_start + (_WIDTH - 1) * pitch,))
        self.stage(name+"-pack-new-values", "Z15Transpose", [name+"-append-v"], vp,
                   {"stride":pitch}, device="VPU")
        kp = self.view(name+"-keys-visible", "k-state", 0, (pitch, _WIDTH))
        packed = self.view(name+"-values-packed", "value-pack-buffer", 0, (_WIDTH, pitch))
        pv = self.view(name+"-pv", "native-pv-workspace", 0, (rows, _WIDTH))
        stage = self.stage(name+"-native-attention", "Attention", [q, kp, packed], pv,
                           {"causal":True, "position":start, "packed_v":True,
                            "exp_shift":_NATIVE_EXP_SHIFT, "fast":self.vpu_attention=="fast"},
                           device="NPU")
        self.view(npu._scores(stage), "native-scores-workspace", 0, (rows, pitch))
        self.view(npu._probabilities(stage), "native-prob-workspace", 0, (rows, pitch))
        return pv

    def tile(self, start, rows, index, prefill):
        end, d = start+rows, _WIDTH
        self.bank = index % 2
        name = f"p{index}" if prefill else f"d{index-self.model['prefill_rows']//self.model['tile_rows']+1}"
        shape = (rows,d)
        x = self.view(name+"-x", "x-history", start*d, shape)
        if not prefill:
            feedback64 = self.tensor(name+"-feedback64", (1,), "int64")
            self.move("load", feedback64, "token_feedback")
            feedback8 = self.tensor(name+"-feedback8", (1,), "int8")
            self.stage(name+"-cast", "Cast", [feedback64], feedback8, {"narrowing": "wrap"})
            claimed = self.tensor(name+"-feedback-vector", shape, "int8")
            self.stage(name+"-expand", "Expand", [feedback8], claimed)
            emb_prod = self.tensor(name+"-feedback-product", shape, "int32")
            self.stage(name+"-embedding-mul", "Mul", [claimed,"token-gain"], emb_prod)
            self.stage(name+"-embedding-add", "Add", [emb_prod,"token-bias"], x)
        q = (self.view(name+"-q", "query-stage-buffer", 0, shape) if self.native_attention
             else self.tensor(name+"-q", shape, "int8"))
        self.project(name+"-query", x, "q", q, 1)
        # The no-reuse baseline differs only in the extent of these two physical stores.
        new_start = 0 if not prefill and self.mode == "recompute" else start
        kv_rows = end-new_start
        source = self.view(name+"-project-source", "x-history", new_start*d, (kv_rows,d))
        for key in ("k", "v"):
            wrote = self.view(name+f"-append-{key}", f"{key}-state", new_start*d, (kv_rows,d))
            self.project(name+f"-{key}-projection", source, key, wrote, 1)
        if self.native_attention:
            pv = self.attend_native(name, q, start, rows, end, new_start)
        else:
            kp = self.view(name+"-keys-visible", "k-state", 0, (end,d))
            vp = self.view(name+"-values-visible", "v-state", 0, (end,d))
            logits_qk = self.tensor(name+"-scores-q", (rows,end), "int8")
            self.scores(name+"-qk", q, kp, logits_qk)
            probability = self.tensor(name+"-prob", (rows,end), "uint8")
            self.stage(name+"-causal-softmax", "QSoftmax", [logits_qk], probability,
                       {"axis": 1, "algorithm": "q23", "causal_offset": start,
                        "in_scale": [1,8], "out_scale": [1,128], "zero_point": 0})
            pv = self.tensor(name+"-pv", shape, "int8")
            mid = self.tensor(name+"-pv-acc", shape, "int32", "ssa")
            ins = [probability,vp]
            self.stage(name+"-weighted-value", "Gemm", ins, pv, fused=[
                {"id": name+"-pv-dot", "op": "Gemm", "inputs": ins, "output": mid, "attrs": {}},
                {"id": name+"-pv-scale", "op": "Requant", "inputs": [mid],
                 "output": pv, "attrs": {"shift": 7}},
            ])
        attention = self.tensor(name+"-attention-out", shape, "int8")
        self.project(name+"-attention-project", pv, "out", attention, 1)
        residual = self.tensor(name+"-residual", shape, "int8")
        self.add(name+"-attention-residual", x, attention, residual)
        ff_hidden = self.tensor(name+"-ff-hidden", shape, "int8")
        self.project(name+"-ff-expand", residual, "ff1", ff_hidden, 2, relu=True)
        ff_out = self.tensor(name+"-ff-out", shape, "int8")
        self.project(name+"-ff-contract", ff_hidden, "ff2", ff_out, 2)
        result = self.tensor(name+"-block", shape, "int8")
        self.add(name+"-ff-residual", residual, ff_out, result)
        if prefill:
            lm = self.view(name+"-logits", "prefill-logits-buffer", start*d, shape)
            ids = self.view(name+"-ids", "prefill-tokens-buffer", start*8, (rows,))
        else:
            lm = self.tensor(name+"-logits", shape, "int8")
            ids = self.tensor(name+"-ids", (rows,), "int64")
        self.project(name+"-lm-project", result, "lm", lm, 2)
        self.stage(name+"-greedy", "ArgMax", [lm], ids, {"axis": 1, "keepdims": False})
        self.tiles.append({"phase": "prefill" if prefill else "decode", "tile": index, "bank": self.bank,
                           "first_query": start, "queries": rows, "key_prefix_rows": end,
                           "first_written_kv_row": new_start, "projected_kv_rows_per_tensor": kv_rows,
                           "stages_end": len(self.stages)})
        return lm,ids

    def layout(self):
        """Allocate loop invariants first and independently color the two transient banks."""
        fixed = 0
        alloc = {}
        for name, desc in self.storage.items():
            if desc["kind"] in ("temp", "native_attention_workspace"):
                continue
            used = _align(desc["size"], _BEAT)
            alloc[name] = {"offset": fixed, "size": desc["size"], "stored_size": used,
                           "region": desc["kind"], "zone": 0}
            fixed += used
        live = [{}, {}]

        def touch(name, index):
            parent = self.views.get(name,(name,0))[0]
            item = self.storage.get(parent)
            if item is None or item["kind"] != "temp":
                return
            bank = item["bank"]
            spec = live[bank].setdefault(parent, {"size": item["size"], "kind": "temp",
                  "stored_size": _align(item["size"], _BEAT), "alignment": _BEAT,
                  "first": index, "last": index, "limit": _MAX_PAGE, "zone": 0, "gpu": True})
            spec["last"] = index

        by_stage = {s["id"]: s for s in self.stages}
        for index, cmd in enumerate(self.execution):
            if cmd["type"] == "launch":
                stage = by_stage[cmd["stage"]]
                for name in [*stage["inputs"], stage["output"]]:
                    touch(name,index)
            if cmd["type"] == "dma":
                touch(cmd["tensor"],index)
        banks = []
        cursor = fixed
        for bank, specs in enumerate(live):
            local, ranges = _allocate(specs, [], _MAX_PAGE, True)
            width = max((end for _,end,_,_ in ranges[0]), default=0)
            for name, desc in local.items():
                alloc[name] = {**desc,"offset": cursor+desc["offset"], "region": f"transient{bank}"}
            banks.append({"bank": bank,"offset": cursor,"size": width, "half": 0})
            cursor += width
        if cursor > _MAX_PAGE:
            raise GraphError("resident KV and temporary banks exceed the selected GPU page")
        native_workspace = 0
        for name, desc in self.storage.items():
            if desc["kind"] != "native_attention_workspace":
                continue
            used = _align(desc["size"], _BEAT)
            alloc[name] = {"offset": cursor, "size": desc["size"], "stored_size": used,
                           "region": desc["kind"], "zone": 0}
            cursor += used
            native_workspace += used
        for name,(parent,offset) in self.views.items():
            backing = alloc[parent]
            alloc[name] = {"offset": backing["offset"] + offset,"size": _nbytes(self.tensors[name]),
                           "region": backing["region"], "view_of": parent,"byte_offset": offset,"zone": 0}
        peak = max(fixed+native_workspace+sum(rec["stored_size"] for group in live for rec in group.values()
                             if rec["first"] <= i <= rec["last"])
                   for i in range(len(self.execution)))
        return alloc,banks,peak,cursor

    def bundle(self,directory):
        seq, tile, steps, d = (self.model[k] for k in ("prefill_rows","tile_rows","decode_steps","width"))
        total = seq+steps
        self.tensor("x-history",(total,d),"int8","resident_history")
        hardware_rows = _align(total, 4) if self.native_attention else total
        if self.native_attention:
            self.tensor("query-stage-buffer", (tile, d), "int8", "resident_attention_staging")
        for key in ("k", "v"):
            self.tensor(key+"-state", (hardware_rows,d), "int8",
                        "resident_kv" if self.mode=="cached" else "full_prefix_workspace")
        if self.native_attention:
            self.tensor("value-pack-buffer", (d, hardware_rows), "int8",
                        "resident_kv" if self.mode=="cached" else "full_prefix_workspace")
            self.tensor("native-scores-workspace", (tile, hardware_rows), "int32", "native_attention_workspace")
            self.tensor("native-prob-workspace", (tile, hardware_rows), "uint8", "native_attention_workspace")
            self.tensor("native-pv-workspace", (tile, d), "int8", "native_attention_workspace")
        self.tensor("prefill-logits-buffer",(seq,d),"int8","resident_output")
        self.tensor("prefill-tokens-buffer",(seq,),"int64","resident_output")
        prompt = self.view("prompt-input","x-history",0,(seq,d))
        self.input(prompt,"prompt",self.model["prompt"])
        self.move("load",prompt,"prompt",at_stage=False)
        self.inputs["token_feedback"] = {"shape":[1],"dtype":"int64","data":[0],"kind":"input"}
        for name in ("gain","bias"):
            weight = self.tensor("token-"+name,(1,d),"int8","constant",self.model["token_"+name])
            self.move("load",weight,weight,at_stage=False)
        for key in _PROJECTIONS:
            weight = self.tensor("weight-"+key,(d,d),"int8","ssa",self.model["weights"][key])
            packed = f"{weight}.__npu_colmajor"
            self.tensor(packed,(d,d),"int8","constant")
            self.move("load",packed,packed,at_stage=False)
        for index,start in enumerate(range(0,seq,tile)):
            self.tile(start,tile,index,True)
        self.output("prefill-logits-buffer","prefill_logits")
        self.output("prefill-tokens-buffer","prefill_tokens")
        self.bank = (seq//tile-1)%2
        last = self.view("last-prefill-token","prefill-tokens-buffer",(seq-1)*8,(1,))
        feedback = self.tensor("prefill-feedback-aligned",(1,),"int64")
        self.stage("first-token-feedback","Reshape",[last],feedback,{"shape":[1]})
        self.move("store",feedback,"token_feedback")
        if self.checkpoints:
            self.execution.append({"type":"yield","step":0})
        for step in range(steps):
            lm,ids = self.tile(seq+step,1,seq//tile+step,False)
            self.output(lm,f"decode_{step+1}_logits")
            self.output(ids,f"decode_{step+1}_token")
            self.move("store",ids,"token_feedback")
            if step == steps-1:
                for key in ("k", "v"):
                    state = key+"-state"
                    exported = (self.view(key+"-export", state, 0, (total,d))
                                if hardware_rows != total else state)
                    self.output(exported, key+"_cache")
            if self.checkpoints:
                self.execution.append({"type":"yield","step":step+1})

        alloc,banks,peak,high = self.layout()
        graph = {"name":self.model["name"],"tensors":self.tensors,"inputs":list(self.inputs),
                 "outputs":list(self.outputs),"nodes":[]}
        plan = {"schema":"hetero-plan-v1","profile":self.profile["name"],"stages":self.stages,
                "allocations":alloc,"peak_live_bytes":peak,"high_water_bytes":high,
                "arena_end":self.profile["sram"]["capacity"],"commands":self.execution,
                "streaming":True,
                "resident_kv":{name:alloc[name] for name in ("k-state","v-state")},
                "transient_banks":banks,"kv_tiles":self.tiles}
        if self.native_attention:
            plan["resident_kv"]["value-pack-buffer"] = alloc["value-pack-buffer"]
            plan["native_attention_pitch"] = hardware_rows
        devices = {"gpu":gpu.lower(graph,plan,self.profile,directory),
                   "npu":npu.lower(graph,plan,self.profile,directory)}
        constants = {entry["tensor"]:bytes(entry["data"]) for entry in devices["npu"]["constants"]}
        ddr = bytearray()
        in_bind,out_bind = {},{}
        for dtype_table, bindings in ((self.inputs,in_bind),(self.outputs,out_bind)):
            for name,info in dtype_table.items():
                length = _nbytes(info)
                where = len(ddr)
                raw = tensor_bytes(info["data"],info["dtype"]) if "data" in info else bytes(length)
                ddr.extend(raw+bytes(_align(length,_BEAT)-length))
                bindings[name] = {"dtype":info["dtype"],"shape":info["shape"],"size":length,
                                  "offset":where,"ddr_offset":where,"needs_data":False}
        data_offsets = {}
        for cmd in self.loads:
            if cmd["type"] != "dma" or cmd["binding"] in in_bind:
                continue
            name = cmd["binding"]
            raw = constants[name] if name in constants else tensor_bytes(
                self.tensors[name]["data"],self.tensors[name]["dtype"])
            data_offsets[name] = len(ddr)
            ddr.extend(raw+bytes(_align(len(raw),_BEAT)-len(raw)))

        def physical(cmd):
            if cmd["type"] != "dma":
                return cmd
            obj = alloc[cmd["tensor"]]
            name = cmd["binding"]
            binding = in_bind.get(name) or out_bind.get(name)
            ddr_offset = binding["ddr_offset"] if binding else data_offsets[name]
            if obj["offset"]%_BEAT:
                raise GraphError(f"unaligned DMA-visible buffer {cmd['tensor']}")
            return {"type":"dma","direction":cmd["direction"],"ddr_offset":ddr_offset,
                    "scratch_offset":obj["offset"],"bytes":_align(obj["size"],_BEAT),
                    "zone":0,"binding":name}

        execution = [physical(cmd) for cmd in self.loads+self.execution]
        dma = [entry for entry in execution if entry["type"]=="dma"]
        graph["tensors"].update(self.inputs)
        graph["tensors"].update(self.outputs)
        (directory/"initial_ddr.bin").write_bytes(ddr)
        (directory/"initial.bin").write_bytes(bytes(self.profile["sram"]["capacity"]))
        estimates={"scope":"static footprint only","scratch_peak_live_bytes":peak,
                   "scratch_reserved_bytes":high,"external_transfer_bytes":sum(m["bytes"] for m in dma),
                   "projected_kv_rows":2*sum(t["projected_kv_rows_per_tensor"] for t in self.tiles),
                   "prefill_tiles":seq//tile,"decode_steps":steps}
        manifest = {"schema":"ysyx-heterogeneous-bundle-v1","graph":graph,"profile":self.profile,
                    "plan":plan,"initial_image":"initial.bin","initial_ddr":"initial_ddr.bin",
                    "execution":execution,"dma":dma,"input_bindings":in_bind,"output_bindings":out_bind,
                    "backends":devices,"estimates":estimates,
                    "kv":{"model":self.model,"mode":self.mode,"checkpoints":self.checkpoints},
                    "program":"program.yhb"}
        if self.native_attention:
            manifest["kv"].update(vpu_lowering=self.vpu_lowering, vpu_attention=self.vpu_attention)
            estimates["native_attention_commands"] = sum(stage["op"] == "Attention" for stage in self.stages)
        (directory/"bundle.json").write_text(json.dumps(manifest,indent=2)+"\n")
        (directory/"memory_plan.json").write_text(json.dumps(plan,indent=2)+"\n")
        (directory/"analysis.json").write_text(json.dumps(estimates,indent=2)+"\n")
        emit_yhb(directory,manifest)
        return manifest


def compile_kv_bundle(model, destination, *, mode="cached", checkpoints=True,
                      vpu_lowering="off", vpu_attention="exact"):
    """Produce the normal Z15 bundle with a statically tiled, stateful execution plan."""
    if mode not in ("cached","recompute"):
        raise GraphError("KV mode must be cached or recompute")
    if vpu_lowering not in ("off", "on") or vpu_attention not in ("exact", "fast"):
        raise GraphError("invalid KV VPU lowering or attention mode")
    dest = Path(destination)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(dest)
    model = normalize_model(model)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix=".kv-",dir=dest.parent) as name:
        staging=Path(name)/"bundle"
        staging.mkdir()
        result=_Schedule(model,mode,checkpoints,vpu_lowering,vpu_attention).bundle(staging)
        staging.rename(dest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path, help="Static causal KV model JSON")
    parser.add_argument("--out", required=True, type=Path, help="New bundle directory")
    parser.add_argument("--mode", choices=("cached", "recompute"), default="cached")
    parser.add_argument("--vpu-lowering", choices=("off", "on"), default="off")
    parser.add_argument("--vpu-attention", choices=("exact", "fast"), default="exact")
    args = parser.parse_args()
    try:
        model = json.loads(args.graph.read_text(encoding="utf-8"))
        result = compile_kv_bundle(model, args.out, mode=args.mode,
                                   vpu_lowering=args.vpu_lowering, vpu_attention=args.vpu_attention)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"compiler error: {exc}\n")
    print(f"{result['profile']['name']}: {1 + result['estimates']['decode_steps']} checkpoints, bundle {args.out}")


if __name__ == "__main__":
    main()
