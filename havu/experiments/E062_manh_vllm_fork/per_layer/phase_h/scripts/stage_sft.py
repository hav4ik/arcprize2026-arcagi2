"""Zero-copy model dir for the SFT checkpoint: real directories, symlinked files.
- every file of SFT_DIR (except its model.safetensors.index.json) -> OUT/<same rel path>;
- E128's Engram-table shards (model-engram-*.safetensors: only layers.N.engram.embed.* / embed_tokens.* tensors) ->
  OUT/<same name>;
- E128's non-weight files the SFT lacks (FROM_E128 list) -> OUT/<same rel path>;
- model.safetensors.index.json: the SFT's unchanged when its Engram-table entries equal E128's exactly (an E043 export
  names the Kaggle Engram shards already); else a merged one: the Engram-table entries from E128's index, everything
  else from the SFT's; metadata = E128's (e043_reshard: the row-chunk suffix) updated with the SFT's.
Checks: no file both in the SFT and among the Engram shards; every index path relative without '..'; each indexed
file's safetensors header holds exactly the tensors the merged index maps to it (vLLM loads whole files); no tensor in
two files; nothing of E128's backbone (model-0000N-of-00002, gptq/) is linked.
usage: stage_sft.py SFT_DIR E128_DIR OUT_DIR"""
import json, os, re, struct, sys
sft, e128, out = (os.path.abspath(x) for x in sys.argv[1:4])
assert not os.path.exists(out), f"{out} exists"
ENG = re.compile(r"\.engram\.(embed\.|embed_tokens\.)")
FROM_E128 = ["config.json", "tokenizer.json", "tokenizer_config.json", "e043_manifest.json", "LICENSE", "README.md",
             "encoding", "inference", "serving"]  # used only where the SFT has no file of that name


def header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def walk(d):
    for root, dirs, names in os.walk(d, followlinks=True):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        for n in names:
            if not n.startswith("."):
                yield os.path.relpath(os.path.join(root, n), d)


def link(src, rel):
    dst = os.path.join(out, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)
    plan.append((rel, src))


plan = []
os.makedirs(out)
s_files = sorted(walk(sft))
e_idx = json.load(open(f"{e128}/model.safetensors.index.json"))
s_idx = json.load(open(f"{sft}/model.safetensors.index.json"))
eng_map = {k: f for k, f in e_idx["weight_map"].items() if ENG.search(k)}
eng_files = sorted(set(eng_map.values()))
for f in eng_files:
    hk = {k for k in header(f"{e128}/{f}") if k != "__metadata__"}
    assert all(ENG.search(k) for k in hk), f"{f} holds non-Engram-table tensors"
    assert f not in s_files, f"name clash: {f} is also an SFT file"
s_eng = {k: f for k, f in s_idx["weight_map"].items() if ENG.search(k)}
# An E043 export's index may already name E128's Kaggle Engram shards: then they must match E128's index exactly, and
# the SFT index is used unchanged (linked); otherwise a merged index is written.
keep_index = bool(s_eng)
assert not s_eng or s_eng == eng_map, f"the SFT index's Engram entries differ from E128's: {sorted(set(s_eng.items()) ^ set(eng_map.items()))[:5]}"
for rel in s_files:
    if rel != "model.safetensors.index.json" or keep_index:
        link(os.path.join(sft, rel), rel)
for f in eng_files:
    link(os.path.join(e128, f), f)
s_top = {r.split("/")[0] for r in s_files}
for name in FROM_E128:
    if name in s_top or not os.path.exists(os.path.join(e128, name)):
        continue
    if os.path.isdir(os.path.join(e128, name)):
        for rel in sorted(walk(os.path.join(e128, name))):
            link(os.path.join(e128, name, rel), os.path.join(name, rel))
    else:
        link(os.path.join(e128, name), name)
wm = dict(s_idx["weight_map"]); wm.update(eng_map)
meta = dict(e_idx.get("metadata") or {}); meta.update(s_idx.get("metadata") or {})
if "total_size" in meta:  # the SFT's figure leaves out the Engram shards: recompute over the merged file set
    meta["total_size"] = sum(max([v["data_offsets"][1] for k, v in header(os.path.join(out, f)).items() if k != "__metadata__"] or [0])
                             for f in set(wm.values()))
if keep_index:
    meta = s_idx.get("metadata")
for f in set(wm.values()):
    assert not os.path.isabs(f) and ".." not in f.split("/"), f"bad index path {f}"
    hk = {k for k in header(os.path.join(out, f)) if k != "__metadata__"}
    ik = {k for k, v in wm.items() if v == f}
    assert hk == ik, f"{f}: header keys != merged index keys ({len(hk ^ ik)} differ, e.g. {sorted(hk ^ ik)[:3]})"
for rel, src in plan:
    assert not (src.startswith(e128 + "/") and re.match(r"gptq/|model-0000\d-of-00002", os.path.relpath(src, e128))), f"E128 backbone linked: {src}"
if keep_index:
    assert wm == s_idx["weight_map"]
else:
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": meta, "weight_map": dict(sorted(wm.items()))}, f, indent=2)
json.dump({"sft": sft, "e128": e128, "index": "the SFT's, unchanged" if keep_index else "merged", "links": plan, "index_entries": len(wm), "engram_entries": len(eng_map),
           "engram_files": eng_files}, open(os.path.join(os.path.dirname(out), os.path.basename(out) + ".stage.json"), "w"), indent=1)
print(f"staged {out}: {len(plan)} symlinks ({sum(1 for _, s in plan if s.startswith(sft + '/'))} SFT, "
      f"{len(eng_files)} Engram shards, {sum(1 for r, s in plan if s.startswith(e128 + '/') and r not in eng_files)} other E128 files); "
      f"index {'SFT unchanged' if keep_index else 'merged'}: {len(wm)} entries ({len(eng_map)} Engram-table entries = E128's); metadata {meta}")
