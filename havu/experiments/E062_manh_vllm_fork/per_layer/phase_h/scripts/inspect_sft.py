"""Inspect the SFT repo on the Hub WITHOUT downloading weights: file list (size, LFS sha256), config.json, the index,
every safetensors header (range requests). Compare with E128 (local dumps): config diff, tensor names / dtypes /
shapes per tensor, Engram tensors, byte-identical files. Token: HF_TOKEN env. Writes sft_meta/ (untrusted data).
usage: inspect_sft.py REPO OUTDIR E128_HEADERS E128_INDEX E128_CONFIG E128_INTEGRITY"""
import json, os, re, sys
from huggingface_hub import HfApi, hf_hub_download
repo, out, eh, ei, ec, eint = sys.argv[1:7]
os.makedirs(out, exist_ok=True)
api = HfApi()
info = api.model_info(repo, revision=os.environ.get("SFT_REV"), files_metadata=True)
files = {s.rfilename: (s.size, s.lfs.sha256 if s.lfs else None) for s in info.siblings}
print(f"repo {repo} private={info.private} sha={info.sha} files={len(files)} bytes={sum(v[0] or 0 for v in files.values()) / 1e9:.2f} GB")
for f, (sz, h) in sorted(files.items()):
    print(f"  {f}\t{sz}\t{(h or '')[:16]}")
json.dump({"sha": info.sha, "files": files}, open(f"{out}/files.json", "w"), indent=1)
E_int = json.load(open(eint))["model"]
same = [f for f, (sz, h) in files.items() if f in E_int and h and E_int[f][1] == h]
same_by_hash = {f: [g for g, v in E_int.items() if v[1] == h] for f, (sz, h) in files.items() if h and any(v[1] == h for v in E_int.values())}
print("files byte-identical to E128 (same sha256):", same_by_hash)
small = [f for f, (sz, h) in files.items() if f.endswith((".json", ".jinja", ".txt", ".md", ".py", ".model", ".npy")) and (sz or 0) < 50_000_000]
for f in small:
    hf_hub_download(repo, f, revision=info.sha, local_dir=out)
E_hdr = json.load(open(eh)); E_idx = json.load(open(ei)); E_cfg = json.load(open(ec))
# config diff
def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a: print(f"  config + {path}{k} = {json.dumps(b[k])[:200]}")
            elif k not in b: print(f"  config - {path}{k} (E128: {json.dumps(a[k])[:200]})")
            else: diff(a[k], b[k], f"{path}{k}.")
    elif a != b:
        print(f"  config ~ {path[:-1]}: E128 {json.dumps(a)[:200]} -> SFT {json.dumps(b)[:200]}")
if os.path.exists(f"{out}/config.json"):
    print("config.json diff (E128 -> SFT):"); diff(E_cfg, json.load(open(f"{out}/config.json")))
else:
    print("NO config.json in the SFT repo")
# headers
S_hdr = {}
for f in files:
    if f.endswith(".safetensors"):
        m = api.parse_safetensors_file_metadata(repo, f, revision=info.sha)
        S_hdr[f] = {"__metadata__": m.metadata, "tensors": {k: [v.dtype, list(v.shape)] for k, v in m.tensors.items()}}
json.dump(S_hdr, open(f"{out}/headers.json", "w"))
def flat(H, skip_gptq=False):
    d = {}
    for f, v in H.items():
        for k, t in v["tensors"].items():
            key = (f"{f}::{k}" if f.startswith("gptq/") else k)
            d[key] = (t[0], tuple(t[1]), f)
    return d
E, S = flat(E_hdr), flat(S_hdr)
eng_tab = lambda k: re.search(r"\.engram\.(embed\.|embed_tokens\.)", k) is not None
E_non = {k: v for k, v in E.items() if not eng_tab(k)}
print(f"E128 tensors {len(E)} (non-Engram-table {len(E_non)}); SFT tensors {len(S)}")
print("SFT Engram-table tensors:", sorted(k for k in S if eng_tab(k)))
print("SFT engram projection tensors:", sorted(k for k in S if ".engram." in k and not eng_tab(k)))
miss = sorted(set(E_non) - set(S)); extra = sorted(set(S) - set(E_non))
print(f"E128 non-Engram tensors missing in SFT: {len(miss)} {miss[:20]}")
print(f"SFT tensors not in E128's non-Engram set: {len(extra)} {extra[:20]}")
mism = [(k, E_non[k][:2], S[k][:2]) for k in set(E_non) & set(S) if E_non[k][:2] != S[k][:2]]
print(f"dtype/shape mismatches: {len(mism)} {mism[:20]}")
if os.path.exists(f"{out}/model.safetensors.index.json"):
    si = json.load(open(f"{out}/model.safetensors.index.json"))
    print("SFT index metadata:", si.get("metadata"), "| entries", len(si["weight_map"]), "| files", sorted(set(si["weight_map"].values())))
    hdr_keys = {k for f, v in S_hdr.items() if not f.startswith("gptq/") for k in v["tensors"]}
    print("index keys == non-gptq header keys:", set(si["weight_map"]) == hdr_keys)
else:
    print("NO model.safetensors.index.json in the SFT repo")
