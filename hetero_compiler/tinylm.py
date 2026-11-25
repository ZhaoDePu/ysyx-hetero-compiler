"""Import TinyLM ONNX and schedule one-word causal tiles with persistent KV."""

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from . import gpu, npu
from .compiler import tensor_bytes
from .ir import DTYPES, GraphError, normalize
from .onnx import parse_model
from .pipeline import _schedule_banks, schedule_tiles
from .planner import _align, _allocate, _estimate
from .profile import load_profile
from .vpu_lowering import _exponent_shift
from .yhb import emit_yhb

PITCH, PROMPT, VOCAB, WIDTH, HEAD_WIDTH = 12, 4, 64, 8, 4
HALF, PAGE, BEAT = 1 << 20, 4096, 32
PHASES = ("A0", "A1", "A2", "A3")


def _bytes(t):
    return DTYPES[t["dtype"]][0] * math.prod(t["shape"])


def read_onnx(path):
    data = Path(path).read_bytes()
    graph = normalize(parse_model(data), dialect="canonical")
    nodes = {n["output"]: n for n in graph["nodes"]}
    if (graph["inputs"] != ["tokens"] or set(graph["outputs"]) != {"lm.out", "token"} or
            (graph["tensors"]["tokens"]["shape"], graph["tensors"]["tokens"]["dtype"]) != ([PITCH], "int64") or
            (graph["tensors"]["lm.out"]["shape"], graph["tensors"]["lm.out"]["dtype"]) != ([PITCH,VOCAB], "int8") or
            (graph["tensors"]["token"]["shape"], graph["tensors"]["token"]["dtype"]) != ([PITCH], "int64")):
        raise GraphError("TinyLM expects the static 12-position causal ONNX interface")
    return graph, nodes, hashlib.sha256(data).hexdigest()


class _TileLane:
    def __init__(self, model, index, prompt, phase, native, vectors):
        self.source, self.nodes, _ = model
        self.prompt, self.native = prompt, native
        self.vector_add = vectors and phase != "A0"
        self.gpu_only = phase == "A0"
        self.reuse = phase in ("A2", "A3")
        self.scope, self.zone = f"p{index}", index % 2
        self.base = index % 2 * HALF + index // 2 * PAGE
        self.tensors, self.fixed, self.views = {}, {}, {}
        self.constants, self.allocations = {}, {}
        self.stages, self.phases = [], []
        self.used_nodes = set()

    def tensor(self, name, shape, dtype, *, fixed=False, data=None, align=4):
        name = f"{self.scope}/{name}"
        if name in self.tensors:
            raise GraphError(f"duplicate TinyLM value {name}")
        self.tensors[name] = {"shape": list(shape), "dtype": dtype,
                              "kind": "constant" if data is not None else "value"}
        if data is not None:
            self.tensors[name]["data"] = data
        if fixed:
            self.fixed[name] = align
        return name

    def const(self, name, *, packed=False):
        cache = self.constants.get(name)
        if cache:
            return cache
        spec = self.source["tensors"][name]
        if spec["kind"] != "constant":
            raise GraphError(f"TinyLM operand {name} is not an initializer")
        target = self.tensor(name, spec["shape"], spec["dtype"], data=spec["data"], fixed=not packed)
        if packed:
            self.tensor(name + ".__npu_colmajor", spec["shape"], spec["dtype"], fixed=True)
        self.constants[name] = target
        return target

    def alias(self, name, parent, shape, offset=0):
        target = self.tensor(name, shape, self.tensors[parent]["dtype"])
        if offset + _bytes(self.tensors[target]) > _bytes(self.tensors[parent]):
            raise GraphError(f"TinyLM alias {name} exceeds resident allocation")
        self.views[target] = (parent, offset)
        return target

    def source_node(self, name, op, inputs=None):
        value = self.nodes.get(name)
        if not value or value["op"] != op or (inputs is not None and value["inputs"] != inputs):
            raise GraphError(f"TinyLM imported {name} does not match {op} topology")
        self.used_nodes.add(name)
        return value

    def stage(self, name, op, inps, out, attrs=None, device="GPU"):
        obj = {"id": f"{self.scope}/{name}", "op": op, "inputs": list(inps), "output": out,
               "attrs": attrs or {}, "device": device, "zone": self.zone}
        self.stages.append(obj)
        return obj

    def _linear_nodes(self, name, source_name, width, rhs_name=None):
        acc_name, out_name = name + ".acc", name + ".out"
        original = self.source_node(acc_name, "Gemm")
        out = self.source["tensors"][out_name]
        a = original["attrs"]
        if (len(original["inputs"]) != 2 or original["inputs"][0] != source_name or
                (rhs_name is not None and original["inputs"][1] != rhs_name) or
                a.get("transpose_b") or a.get("a_zero_point") or a.get("b_zero_point") or
                (out["shape"],out["dtype"]) != ([PITCH,width], "int8")):
            raise GraphError(f"TinyLM {name} does not match two-input symmetric signed matmul")
        result = self.source_node(out_name, "Requant", [acc_name])
        return original, result["attrs"]

    def linear(self, name, source_name, source, out, *, contract=None, transpose=False, rhs_name=None):
        acc_name = name + ".acc"
        canonical, requant = self._linear_nodes(name,source_name,
                                               self.tensors[out]["shape"][-1] if contract is None or not transpose
                                               else PITCH,rhs_name)
        native = contract is None and not self.gpu_only
        rhs = self.const(canonical["inputs"][1], packed=native) if contract is None else contract
        attrs = dict(canonical["attrs"], transpose_b=transpose)
        mid = self.tensor(f"{self.step}/{acc_name}", self.tensors[out]["shape"], "int32")
        launch = self.stage(f"{self.step}/{name}/matmul", "Gemm", [source, rhs], mid, attrs,
                            device="NPU" if native else "GPU")
        if self.reuse and not native:
            launch["fused"] = [{"id": launch["id"], "op": "Gemm", "inputs": launch["inputs"],
                                "output": mid, "attrs": attrs},
                               {"id": launch["id"] + "/requant", "op": "Requant", "inputs": [mid],
                                "output": out, "attrs": requant}]
            launch["output"] = out
        else:
            self.stage(f"{self.step}/{name}/requant", "Requant", [mid], out, requant)

    def project(self, name, source_name, source, size, out=None):
        out = out or self.tensor(f"{self.step}/{name}", [1, size], "int8")
        self.linear(name, source_name, source, out)
        return out

    def rms(self, name, source_name, source):
        raw = self.source_node(name + ".out", "QRMSNorm", [source_name, name + ".weight"])
        out = self.tensor(f"{self.step}/{name}", [1, WIDTH], "int8")
        self.stage(f"{self.step}/{name}", "Z15RMSNorm" if self.native else "QRMSNorm",
                   [source, self.const(name + ".weight")], out, raw["attrs"],
                   device="VPU" if self.native else "GPU")
        return out

    def residual(self, name, a_name, b_name, a, b):
        raw = self.source_node(name + ".out", "Add", [a_name, b_name])
        out = self.tensor(f"{self.step}/{name}", [1, WIDTH], "int8")
        self.stage(f"{self.step}/{name}", "Add", [a, b], out, raw["attrs"],
                   device="VPU" if self.vector_add else "GPU")
        return out

    def _attend(self, sub, normal_name, normal, pos):
        k_state, v_state = self.kv[sub]
        row = self.step + "/" + sub
        keys = self.alias(row + "new-k", k_state, [1, HEAD_WIDTH], pos * HEAD_WIDTH)
        values = self.alias(row + "new-v", v_state, [1, HEAD_WIDTH], pos * HEAD_WIDTH)
        query = self.alias(row + "query", self.q_stage, [1, HEAD_WIDTH]) if self.native else None
        q = self.project(sub + "q", normal_name, normal, HEAD_WIDTH, query)
        self.project(sub + "k", normal_name, normal, HEAD_WIDTH, keys)
        self.project(sub + "v", normal_name, normal, HEAD_WIDTH, values)
        kt = self.source_node(sub + "kt", "Transpose", [sub + "k.out"])
        soft = self.source_node(sub + "p", "QSoftmax", [sub + "score.out"])
        if kt["attrs"].get("perm") != [1,0] or (
                soft["attrs"].get("axis"),soft["attrs"].get("causal_offset"),
                soft["attrs"].get("dtype"),soft["attrs"].get("out_scale"),
                soft["attrs"].get("zero_point")) != (1,0,"uint8",[1,256],0):
            raise GraphError("TinyLM requires K transpose and zero-centered causal last-axis Q/DQ Softmax")
        context = self.tensor(row + "pv", [1, HEAD_WIDTH], "int8")
        if self.native:
            _, req = self._linear_nodes(sub + "score",sub + "q.out",PITCH,sub + "kt")
            _, pv = self._linear_nodes(sub + "pv",sub + "p",HEAD_WIDTH,sub + "v.out")
            for attrs in (req,pv):
                if (attrs.get("multiplier",1)!=1 or attrs.get("input_zero_point",0) or
                        attrs.get("zero_point",0) or attrs.get("narrowing","saturate")!="saturate" or
                        attrs.get("divisor",1 << attrs.get("shift",0))!=1 << attrs.get("shift",0) or
                        attrs.get("rounding","floor") not in ("floor","nearest_even")):
                    raise GraphError("TinyLM native attention needs symmetric saturating power-of-two requantization")
            if pv.get("shift")!=8:
                raise GraphError("TinyLM native attention needs a PV shift of eight")
            k_view = self.alias(row + "full-k", k_state, [PITCH, HEAD_WIDTH])
            packed = self.packed[sub]
            slot = self.alias(row + "packed-column", packed, [1 + (HEAD_WIDTH - 1) * PITCH], pos)
            self.stage(row + "append-packed-v", "Z15Transpose", [values], slot, {"stride": PITCH}, "VPU")
            raw_scale = Fraction(*soft["attrs"]["in_scale"]) / (1 << req["shift"])
            launch = self.stage(row + "attention", "Attention", [q, k_view, packed], context,
                                {"causal": True, "position": pos, "packed_v": True,
                                 "exp_shift": _exponent_shift(raw_scale)}, "NPU")
            self.alias(npu._scores(launch).split("/", 1)[1], self.scores, [1, PITCH])
            self.alias(npu._probabilities(launch).split("/", 1)[1], self.probabilities, [1, PITCH])
        else:
            active = pos + 1
            k_view = self.alias(row + "active-k", k_state, [active, HEAD_WIDTH])
            v_view = self.alias(row + "active-v", v_state, [active, HEAD_WIDTH])
            scores = self.tensor(row + "score", [1, active], "int8")
            self.linear(sub + "score", sub + "q.out", q, scores, contract=k_view, transpose=True,
                        rhs_name=sub + "kt")
            prob = self.tensor(row + "probability", [1, active], "uint8")
            self.stage(row + "softmax", "QSoftmax", [scores], prob,
                       dict(soft["attrs"], causal_offset=pos))
            self.linear(sub + "pv", sub + "p", prob, context, contract=v_view, rhs_name=sub + "v.out")
        return context

    def _feedforward(self, sub, y_name, y):
        rms = self.rms(sub + "gamma2", y_name, y)
        gate = self.project(sub + "gate", sub + "gamma2.out", rms, WIDTH)
        up = self.project(sub + "up", sub + "gamma2.out", rms, WIDTH)
        act = self.tensor(self.step + "/" + sub + "silu", [1, WIDTH], "int8")
        lut = self.source_node(sub + "silu.out", "Lookup", [sub + "gate.out"])
        self.stage(self.step + "/" + sub + "silu", "Lookup", [gate], act, lut["attrs"],
                   "VPU" if self.native else "GPU")
        mod = self.source_node(sub + "mod.out", "Requant")
        mult = self.source_node(mod["inputs"][0], "Mul", [sub + "silu.out", sub + "up.out"])
        glu = self.tensor(self.step + "/" + sub + "mod", [1, WIDTH], "int8")
        if self.native:
            quant = mod["attrs"]
            shift = quant.get("shift",0)
            if (quant.get("input_zero_point",0) or quant.get("zero_point",0) or
                    quant.get("narrowing","saturate") != "saturate" or
                    quant.get("divisor",1 << shift) != 1 << shift):
                raise GraphError("TinyLM native SwiGLU requires symmetric power-of-two scaling")
            self.stage(self.step + "/" + sub + "swiglu", "Z15Mul", [act, up], glu,
                       {"scale": quant["multiplier"], "shift": shift}, "VPU")
        else:
            tmp = self.tensor(self.step + "/" + sub + "product", [1, WIDTH], "int32")
            self.stage(self.step + "/" + sub + "product", "Mul", [act, up], tmp, mult["attrs"])
            self.stage(self.step + "/" + sub + "mod", "Requant", [tmp], glu, mod["attrs"])
        down = self.project(sub + "down", sub + "mod.out", glu, WIDTH)
        return self.residual(sub + "res2", y_name, sub + "down.out", y, down)

    def make(self):
        self.token_prompt = self.tensor("prompt", [PROMPT], "int64", fixed=True, align=BEAT)
        self.feedback = self.tensor("feedback", [1], "int64", fixed=True, align=BEAT)
        self.logits = self.tensor("logits", [1, VOCAB], "int8", fixed=True, align=BEAT)
        # Keep the native Q/K/Packed-V/score/P/output extents strictly ordered.
        self.q_stage = self.tensor("attention-q", [1, HEAD_WIDTH], "int8", fixed=True) if self.native else None
        self.kv = {}
        for layer in range(2):
            for head in range(2):
                key = f"l{layer}.h{head}."
                self.kv[key] = tuple(self.tensor(key + x + "-state", [PITCH, HEAD_WIDTH], "int8",
                                                  fixed=True, align=BEAT) for x in ("k", "v"))
        self.packed = {
            key: self.tensor(key + "packed-v", [HEAD_WIDTH, PITCH], "int8", fixed=True)
            for key in self.kv
        } if self.native else {}
        if self.native:
            self.scores = self.tensor("attention-scores", [1, PITCH], "int32", fixed=True)
            self.probabilities = self.tensor("attention-probability", [1, PITCH], "uint8", fixed=True)
        embed = self.source_node("embedded", "Gather", ["embedding.weight", "tokens"])
        weights = self.const(embed["inputs"][0])
        if (self.tensors[weights]["shape"], self.tensors[weights]["dtype"], embed["attrs"].get("axis")) != (
                [VOCAB,WIDTH], "int8", 0):
            raise GraphError("TinyLM requires a signed int8 vocabulary×width embedding")
        for pos in range(PITCH):
            self.step = f"row{pos}"
            begin = len(self.stages)
            ids = self.alias(self.step + "/token-in", self.token_prompt, [1], pos * 8) if pos < PROMPT else self.feedback
            x = self.tensor(self.step + "/embedded", [1, WIDTH], "int8")
            self.stage(self.step + "/gather", "Gather", [weights, ids], x, embed["attrs"])
            x_name = "embedded"
            for layer in range(2):
                base = f"l{layer}."
                norma = self.rms(base + "gamma1", x_name, x)
                context = [self._attend(base + f"h{head}.", base + "gamma1.out", norma, pos) for head in range(2)]
                h = self.source_node(base + "heads", "Concat", [base + f"h{i}.pv.out" for i in range(2)])
                if h["attrs"].get("axis") != 1:
                    raise GraphError("TinyLM heads concatenate on the channel axis")
                merged = self.tensor(self.step + "/" + base + "heads", [1, WIDTH], "int8")
                self.stage(self.step + "/" + base + "heads", "Concat", context, merged, h["attrs"])
                attended = self.project(base + "out", base + "heads", merged, WIDTH)
                y = self.residual(base + "res1", x_name, base + "out.out", x, attended)
                x = self._feedforward(base, base + "res1.out", y)
                x_name = base + "res2.out"
            self.project("lm", x_name, x, VOCAB, self.logits)
            argmax = self.source_node("token", "ArgMax", ["lm.out"])
            if (argmax["attrs"].get("axis"),argmax["attrs"].get("keepdims",True),
                    argmax["attrs"].get("select_last_index",False)) != (1,False,False):
                raise GraphError("TinyLM expects first-match ArgMax across the vocabulary")
            self.stage(self.step + "/greedy", "ArgMax", [self.logits], self.feedback, argmax["attrs"])
            self.phases.append((begin, len(self.stages)))
        if set(self.nodes) != self.used_nodes:
            raise GraphError(f"TinyLM unsupported ONNX nodes: {set(self.nodes) - self.used_nodes}")

    def layout(self):
        offset = 0
        for name, boundary in self.fixed.items():
            offset = _align(offset, boundary)
            length = _bytes(self.tensors[name])
            self.allocations[name] = {"offset": self.base + offset, "size": length, "zone": self.zone}
            offset += _align(length, boundary)
        self.used = _align(offset, BEAT)
        maximum, cursor = offset, offset
        self.transient_peak = 0
        for first, last in self.phases:
            lifetimes = {}
            for index in range(first, last):
                stage = self.stages[index]
                for name in [*stage["inputs"], stage["output"]]:
                    if name in self.fixed or name in self.views or self.tensors[name]["kind"] == "constant":
                        continue
                    size = _bytes(self.tensors[name])
                    bound = max(4, DTYPES[self.tensors[name]["dtype"]][0])
                    spec = lifetimes.setdefault(name, {"size": size, "kind":"temp", "stored_size": _align(size,bound),
                        "first": index, "last": index, "alignment": bound, "zone": 0, "gpu": True,
                        "limit": PAGE-offset})
                    spec["last"] = index
            if cursor >= PAGE:
                raise GraphError("TinyLM constants exceed one physical GPU page")
            space, ranges = _allocate(lifetimes, [], PAGE, self.reuse)
            for name, entry in space.items():
                self.allocations[name] = {"offset": self.base + cursor + entry["offset"],
                                          "size": entry["size"], "zone": self.zone}
            maximum = max(maximum, cursor + max((r[1] for r in ranges[0]), default=0))
            for here in range(first,last):
                live_bytes = sum(r["stored_size"] for r in lifetimes.values() if r["first"]<=here<=r["last"])
                self.transient_peak = max(self.transient_peak,live_bytes)
        for name,(parent, at) in self.views.items():
            self.allocations[name] = {"offset": self.allocations[parent]["offset"]+at,
                                      "size": _bytes(self.tensors[name]), "zone": self.zone,
                                      "view_of": parent, "byte_offset": at}
        self.peak = maximum
        if maximum > PAGE:
            raise GraphError(f"TinyLM needs {maximum} B in a {PAGE}-byte GPU page")
        self.cache_bytes = sum(self.allocations[x]["size"] for pair in self.kv.values() for x in pair)
        self.cache_bytes += sum(self.allocations[x]["size"] for x in self.packed.values())
        self.reserved = _align(maximum, BEAT)


def compile_tinylm(model_path, destination, prompts, *, phase="A3", vpu=True):
    """Compile three independent sequence lanes or a single reusable input lane."""
    if (phase not in PHASES or type(vpu) is not bool or not 1 <= len(prompts) <= 3 or
            any(len(p)!=PROMPT or any(type(x) is not int or not 0<=x<VOCAB for x in p) for p in prompts)):
        raise GraphError("TinyLM expects 1..3 four-word prompts and a valid A0..A3 phase")
    model = read_onnx(model_path)
    profile = load_profile("fpga_zu15eg_balanced")
    native = bool(vpu and phase in ("A2", "A3"))
    if native:
        profile["vpu_fused_lowering"] = True
    lanes = [_TileLane(model, i, prompt, phase, native, vpu) for i,prompt in enumerate(prompts)]
    for lane in lanes:
        lane.make()
        lane.layout()
    dest = Path(destination)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".tinylm-",dir=dest.parent) as tmp:
        target = Path(tmp)
        manifest = _bundle(target, lanes, model[2], profile, phase, vpu, native)
        target.rename(dest)
    return manifest


def _bundle(folder, lanes, digest, profile, phase, requested_vpu, native):
    tensors = {n:t for lane in lanes for n,t in lane.tensors.items()}
    alloc = {n:r for lane in lanes for n,r in lane.allocations.items()}
    stages = [s for lane in lanes for s in lane.stages]
    graph = {"name":"TinyLM-Chat", "tensors":tensors,"inputs":[],"outputs":[],"nodes":[]}
    plan = {"schema":"hetero-plan-v1", "profile":profile["name"], "stages":stages, "allocations":alloc}
    devices = {"gpu":gpu.lower(graph,plan,profile,folder),"npu":npu.lower(graph,plan,profile,folder)}
    packed = {x["tensor"]:bytes(x["data"]) for x in devices["npu"]["constants"]}
    ddr, inputs, outputs = bytearray(), {}, {}

    def extent(blob):
        where = len(ddr)
        ddr.extend(blob)
        ddr.extend(bytes(_align(len(blob),BEAT)-len(blob)))
        return where

    def symbol(table, name, spec, payload):
        table[name] = {k:spec[k] for k in ("dtype","shape")}
        table[name].update(ddr_offset=extent(payload), size=len(payload), needs_data=False)
        graph["tensors"][name] = {**spec, "kind":"input" if table is inputs else "value"}
        graph["inputs" if table is inputs else "outputs"].append(name)
        return table[name]["ddr_offset"]

    def dma(lane, addr, lo, length, direction, name=None):
        return {"type":"dma","direction":direction,"ddr_offset":addr,"scratch_offset":lo,
                "bytes":_align(length,BEAT),"zone":lane.zone,"binding":name}

    def out(lane, key, physical):
        name = lane.scope + "_" + key
        spec = tensors[physical]
        addr = symbol(outputs,name,spec,bytes(_bytes(spec)))
        return dma(lane,addr,alloc[physical]["offset"],_bytes(spec),"store",name)

    prologue, phases = [], []
    for lane in lanes:
        initial = bytearray(lane.used)
        for name in lane.fixed:
            value = packed.get(name)
            spec = tensors[name]
            if value is None and "data" in spec:
                value = tensor_bytes(spec["data"],spec["dtype"])
            if value is not None:
                where = alloc[name]["offset"] - lane.base
                initial[where:where+len(value)] = value
        prologue.append(dma(lane,extent(initial),lane.base,len(initial),"load"))
        prompt_name = "prompt_" + lane.scope[1:]
        where = symbol(inputs,prompt_name,tensors[lane.token_prompt],tensor_bytes(lane.prompt,"int64"))
        prologue.append(dma(lane,where,alloc[lane.token_prompt]["offset"],PROMPT*8,"load",prompt_name))
    execution = []
    for c in prologue:
        execution.extend([c,{"type":"dma-fence"}])
    for pos in range(PITCH):
        groups = []
        for lane in lanes:
            first,last = lane.phases[pos]
            group = []
            for s in lane.stages[first:last]:
                engine = "gpu" if s["device"]=="GPU" else "tensor"
                estimate = _estimate(s,graph,profile)["compute_cycles"]
                if s["op"]=="QSoftmax":
                    estimate *= 40
                rec = {"type":"launch","stage":s["id"],"device":s["device"],"zone":s["zone"]}
                group.append((engine,rec,estimate+30))
            if pos >= PROMPT-1:
                idx = pos-PROMPT+1
                for name, phys in ((f"logits_{idx}",lane.logits),(f"token_{idx}",lane.feedback)):
                    group.append(("dma",out(lane,name,phys),25))
            if pos==PITCH-1:
                for key, pair in lane.kv.items():
                    for name,phys in zip(("K","V"),pair,strict=True):
                        group.append(("dma",out(lane,key.replace(".","_")+name,phys),28))
            groups.append(group)
        execution.extend(_schedule_banks(groups) if phase=="A3" else schedule_tiles(groups,False))
        if pos >= PROMPT-1:
            execution.append({"type":"yield","step":pos-PROMPT+1})
    transfers = [c for c in execution if c["type"]=="dma"]
    estimates = {"scratch_peak_live_bytes":sum(lane.peak for lane in lanes),
                 "scratch_reserved_bytes":sum(lane.reserved for lane in lanes),
                 "resident_kv_bytes":sum(lane.cache_bytes for lane in lanes),
                 "external_transfer_bytes":sum(t["bytes"] for t in transfers),
                 "device_stage_counts":{dev:sum(s["device"]==dev for s in stages) for dev in ("NPU","VPU","GPU")},
                 "checkpoint_count":PITCH-PROMPT+1,"native_attention_commands":sum(s["op"]=="Attention" for s in stages)}
    transient = (sum(max((l.transient_peak for l in lanes if l.zone==z),default=0) for z in range(2))
                 if phase=="A3" else max(l.transient_peak for l in lanes))
    estimates["scratch_peak_live_bytes"] = sum(l.used for l in lanes)+transient
    (folder/"initial.bin").write_bytes(bytes(profile["sram"]["capacity"]))
    (folder/"initial_ddr.bin").write_bytes(ddr)
    plan.update(streaming=True, high_water_bytes=max(l.base+l.peak for l in lanes),
                peak_live_bytes=estimates["scratch_peak_live_bytes"], commands=execution)
    manifest = {"schema":"ysyx-heterogeneous-bundle-v1","graph":graph,"profile":profile,
                "plan":plan,"backends":devices,"execution":execution,"dma":transfers,
                "initial_image":"initial.bin","initial_ddr":"initial_ddr.bin",
                "input_bindings":inputs,"output_bindings":outputs,"program":"program.yhb",
                "estimates":estimates,"tinylm":{"onnx_sha256":digest,"phase":phase,"requested_vpu":requested_vpu,
                   "effective_native_attention":native,"prompts":[l.prompt for l in lanes],"prefill_rows":PROMPT,
                   "decode_steps":PITCH-PROMPT}}
    emit_yhb(folder,manifest)
    for name,item in (("bundle",manifest),("memory_plan",plan),("analysis",estimates)):
        (folder/f"{name}.json").write_text(json.dumps(item,indent=2)+"\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph",type=Path,required=True)
    p.add_argument("--prompts",type=Path,required=True,help="JSON array of 1..3 four-token ID arrays")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--phase",choices=PHASES,default="A3")
    p.add_argument("--vpu-lowering",choices=("on","off"),default="on")
    a = p.parse_args()
    compile_tinylm(a.graph,a.out,json.loads(a.prompts.read_text()),phase=a.phase,vpu=a.vpu_lowering=="on")


if __name__=="__main__":
    main()
