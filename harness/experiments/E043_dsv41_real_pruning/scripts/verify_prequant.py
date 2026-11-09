#!/usr/bin/env python3
"""E043: build inputs for, and verify, a pre-quantization (native MXFP4) export of a pruned DeepSeek-V4.1-Flash-1017.

Guide: experiments/E043_dsv41_real_pruning/PREQUANT_E128.md. numpy + requests + huggingface_hub only (no torch).
A SOURCE is a local checkpoint directory or `hf:<repo>@<revision>` (HTTP range reads; set HF_TOKEN for private repos).

  verify_prequant.py keep  --manifest e043_manifest.json [--config config.json] --out keep.npy
      bool [40, 384] keep mask from the manifest's kept original ids; checks its sha256 against
      config.json e043_pruning.expert_keep_sha256 when --config is given.

  verify_prequant.py check --export OUT --base SOURCE [--gptq SOURCE] [--layers 0,13,27,39] [--per-layer 4 | --all]
      1. config: n_routed_experts, keep sha, no e043_lowbit (GPTQ) block;
      2. index: the export holds exactly layers x E x {w1,w2,w3} x {weight,scale} routed-expert tensors;
      3. router: ffn.gate.weight / bias / bias_vl equal the base rows at the kept ids;
      4. experts: export expert j of layer L is byte-equal to base expert keep_original_ids[L][j];
      5. (--gptq: the 2-bit model, e.g. hf:chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit@main) the GPTQ
         expert dequantized vs this expert decoded from MXFP4: cosine ~0.9 (2-bit error) vs ~0 for the next expert.
"""
import argparse
import hashlib
import json
import os
import random
import struct
import sys
from pathlib import Path

import numpy as np

EXPERT_PARTS = [f"{w}.{p}" for w in ("w1", "w2", "w3") for p in ("weight", "scale")]
DT = {"BF16": (np.uint16, 2), "F32": (np.float32, 4), "I32": (np.int32, 4), "U8": (np.uint8, 1), "I8": (np.uint8, 1),
      "F8_E4M3": (np.uint8, 1), "F8_E8M0": (np.uint8, 1)}
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)


class Source:
    """Tensor bytes from a local safetensors checkpoint dir or a HF repo (range reads)."""

    def __init__(self, spec):
        self.spec = spec
        self.remote = spec.startswith("hf:")
        if self.remote:
            import requests
            from huggingface_hub import HfApi
            self.repo, _, self.rev = spec[3:].partition("@")
            self.rev = self.rev or "main"
            self.http = requests.Session()
            tok = os.environ.get("HF_TOKEN")
            self.hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
            self.files = [f for f in HfApi().list_repo_files(self.repo, revision=self.rev, token=tok or None)
                          if f.endswith(".safetensors")]
        else:
            self.root = Path(spec)
            self.files = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*.safetensors"))
        self._hdr, self._loc = {}, None

    def _get(self, f, a, b):
        if not self.remote:
            with open(self.root / f, "rb") as fh:
                fh.seek(a)
                return fh.read(b - a)
        from huggingface_hub import hf_hub_url
        url = hf_hub_url(self.repo, f, revision=self.rev)
        for _ in range(5):
            try:
                r = self.http.get(url, headers={**self.hdr, "Range": f"bytes={a}-{b - 1}"}, timeout=300)
                if len(r.content) == b - a:
                    return r.content
            except Exception:
                pass
        raise RuntimeError(f"range read failed: {self.spec} {f} {a}-{b}")

    def header(self, f):
        if f not in self._hdr:
            n = struct.unpack("<Q", self._get(f, 0, 8))[0]
            self._hdr[f] = (8 + n, json.loads(self._get(f, 8, 8 + n)))
        return self._hdr[f]

    def loc(self):
        """tensor name -> (file, header entry); reads every file's header once."""
        if self._loc is None:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(16) as ex:
                hs = dict(zip(self.files, ex.map(self.header, self.files)))
            self._loc = {k: (f, v) for f, (_, h) in hs.items() for k, v in h.items() if k != "__metadata__"}
        return self._loc

    def raw(self, name, row0=None, nrows=None):
        f, v = self.loc()[name]
        base = self.header(f)[0]
        a, b = v["data_offsets"]
        if row0 is not None:
            rb = int(np.prod(v["shape"][1:])) * DT[v["dtype"]][1]
            a, b = a + row0 * rb, a + (row0 + nrows) * rb
        return self._get(f, base + a, base + b)

    def array(self, name):
        f, v = self.loc()[name]
        return np.frombuffer(self.raw(name), dtype=DT[v["dtype"]][0]).reshape(v["shape"])

    def json(self, name):
        if not self.remote:
            return json.load(open(self.root / name))
        from huggingface_hub import hf_hub_download
        return json.load(open(hf_hub_download(self.repo, name, revision=self.rev, token=os.environ.get("HF_TOKEN"))))


def mxfp4_to_f32(q, s):
    """Native V4.1 expert layout: weight I8 [out, in/2], two E2M1 codes per byte, low nibble = even column; scale
    F8_E8M0 [out, in/32], value 2^(s-127) per 32 consecutive inputs. Every value is exact in bf16."""
    c = np.stack([q & 15, q >> 4], -1).reshape(q.shape[0], -1)
    v = np.where((c & 8) != 0, -E2M1[c & 7], E2M1[c & 7])
    return v * np.repeat(np.exp2(s.astype(np.float32) - 127), 32, -1)


def gptq_to_f32(src, layer, e, w, group):
    """Humming uint2 GPTQ tensors of the 2-bit export: gptq/layer_LL.safetensors e{E}.{w}.{q,s,z}."""
    f = f"gptq/layer_{layer:02d}.safetensors"
    _, h = src.header(f)

    def get(p):
        v = h[f"e{e}.{w}.{p}"]
        a, b = v["data_offsets"]
        return np.frombuffer(src._get(f, src.header(f)[0] + a, src.header(f)[0] + b),
                             dtype=DT[v["dtype"]][0]).reshape(v["shape"])

    def unpack(p):
        sh = np.arange(0, 32, 2).astype(np.uint32)  # lsb first (probed in hf_verify_content.py)
        return ((p.view(np.uint32)[..., None] >> sh) & 3).reshape(p.shape[0], -1)
    q = unpack(get("q")).astype(np.float32)
    z = unpack(np.ascontiguousarray(get("z").T)).T.astype(np.float32)
    s = (get("s").astype(np.uint32) << 16).view(np.float32)
    return (q - np.repeat(z, group, -1)) * np.repeat(s, group, -1)


def keep_mask(manifest):
    ids = {int(l): v for l, v in manifest["keep_original_ids"].items()}
    k = np.zeros((len(ids), 384), bool)
    for l, v in ids.items():
        k[l, v] = True
    return k, ids


def cmd_keep(a):
    k, ids = keep_mask(json.load(open(a.manifest)))
    sha = hashlib.sha256(k.tobytes()).hexdigest()
    print(f"keep mask {k.shape}, experts per layer {sorted(set(k.sum(1).tolist()))}, sha256 {sha}")
    if a.config:
        want = json.load(open(a.config))["e043_pruning"]["expert_keep_sha256"]
        print("matches config e043_pruning.expert_keep_sha256:", sha == want)
        if sha != want:
            sys.exit(1)
    np.save(a.out, k)
    print("wrote", a.out)


def cmd_check(a):
    exp, base = Source(a.export), Source(a.base)
    man = exp.json("e043_manifest.json")
    keep, ids = keep_mask(man)
    E = int(keep.sum(1)[0])
    cfg = exp.json("config.json")
    tc = cfg.get("text_config", cfg)
    fails = []

    def check(ok, what):
        print(("  ok   " if ok else "  FAIL ") + what)
        if not ok:
            fails.append(what)
    print("1. config")
    check(tc["n_routed_experts"] == E, f"n_routed_experts == {E}")
    check(cfg.get("e043_pruning", {}).get("expert_keep_sha256") == hashlib.sha256(keep.tobytes()).hexdigest(),
          "e043_pruning.expert_keep_sha256 matches the manifest")
    check("e043_lowbit" not in tc, "no e043_lowbit (GPTQ) block")
    check("routed_experts" not in cfg.get("e043_pruning", {}), "e043_pruning does not say 'GPTQ-only base'")
    layers = sorted(ids) if not a.layers else [int(x) for x in a.layers.split(",")]
    print("2. index")
    wm = exp.json("model.safetensors.index.json")["weight_map"]
    ek = {k for k in wm if ".ffn.experts." in k and ".shared" not in k}
    want = {f"layers.{l}.ffn.experts.{j}.{p}" for l in layers for j in range(E) for p in EXPERT_PARTS}
    have = {k for k in ek if int(k.split(".")[1]) in layers}
    present = {l: sorted({int(k.split(".")[4]) for k in have if int(k.split(".")[1]) == l}) for l in layers}
    if a.only_present:  # partial test exports: only the experts the export holds (all six parts each)
        want = {f"layers.{l}.ffn.experts.{j}.{p}" for l in layers for j in present[l] for p in EXPERT_PARTS}
    check(have == want, f"routed-expert tensors of layers {layers[0]}..{layers[-1]}: {len(have)} == {len(want)}"
          + ("" if have == want else f" (missing {len(want - have)}, extra {len(have - want)})"))
    if not a.layers and not a.only_present:
        check(len(ek) == len(ids) * E * 6, f"all routed-expert tensors: {len(ek)} == {len(ids) * E * 6}")
    print("3. router rows")
    for l in layers:
        bad = [nm for nm in ("weight", "bias", "bias_vl")
               if not np.array_equal(exp.array(f"layers.{l}.ffn.gate.{nm}"),
                                     base.array(f"layers.{l}.ffn.gate.{nm}")[ids[l]])]
        check(not bad, f"layer {l}: gate weight/bias/bias_vl == base rows at kept ids" + (f" ({bad})" if bad else ""))
    print("4. expert bytes")
    rng = random.Random(0)
    for l in layers:
        pool = present[l] if a.only_present else list(range(E))
        js = pool if a.all else sorted({pool[0], pool[-1], *rng.sample(pool, min(len(pool), a.per_layer))})
        bad = [j for j in js for p in EXPERT_PARTS
               if exp.raw(f"layers.{l}.ffn.experts.{j}.{p}") != base.raw(f"layers.{l}.ffn.experts.{ids[l][j]}.{p}")]
        check(not bad, f"layer {l}: {len(js)} experts byte-equal to base at their original ids"
              + (f" (bad compact ids {sorted(set(bad))[:8]})" if bad else ""))
    if a.gptq:
        print("5. vs the 2-bit GPTQ experts (cosine to the right expert / to the next one)")
        g = Source(a.gptq)
        for l in layers[:4]:
            for j in (0, E - 1):
                for w, grp in (("w1", 128), ("w2", 64)):
                    wq = gptq_to_f32(g, l, j, w, grp)
                    f32 = lambda jj: mxfp4_to_f32(exp.array(f"layers.{l}.ffn.experts.{jj}.{w}.weight"),
                                                  exp.array(f"layers.{l}.ffn.experts.{jj}.{w}.scale"))
                    cos = lambda x, y: float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y)))
                    right, nxt = cos(wq, f32(j)), cos(wq, f32((j + 1) % E))
                    check(right > 0.8 and abs(nxt) < 0.05, f"layer {l} expert {j} {w}: {right:.3f} / {nxt:+.3f}")
    print("PASS" if not fails else f"FAILED: {len(fails)} check(s)")
    sys.exit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keep")
    k.add_argument("--manifest", required=True)
    k.add_argument("--config", default=None)
    k.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--export", required=True)
    c.add_argument("--base", required=True)
    c.add_argument("--gptq", default=None)
    c.add_argument("--layers", default=None)
    c.add_argument("--per-layer", type=int, default=4)
    c.add_argument("--all", action="store_true")
    c.add_argument("--only-present", action="store_true", help="partial test exports: check the experts present")
    a = ap.parse_args()
    cmd_keep(a) if a.cmd == "keep" else cmd_check(a)


if __name__ == "__main__":
    main()
