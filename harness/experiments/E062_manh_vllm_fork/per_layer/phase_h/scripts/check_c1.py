"""Checks of the C1 image model dir before the V12 + images launch:
1. every file of the manifest exists with its size and sha256, and nothing else is in the dir (hidden dirs skipped);
2. config.json pairs with E128's Engram table: engram_layer_ids / engram_num_embeddings / engram_compact_kept /
   engram_* sizes equal E128's; e043_manifest's engram section (rows, kept, mask sha256, engram_from) equals E128's;
3. the index's Engram-table entries equal E128's index exactly; each indexed file's header holds exactly the tensors
   the index maps to it; no tensor in two files; index paths relative without '..';
4. the vision tower is present: vision_config in config.json and vision tensors in the index/headers (count, files);
   the gptq dir holds one layer file per text layer.
usage: check_c1.py C1_DIR E128_DIR MANIFEST.json"""
import hashlib, json, os, re, struct, sys
from concurrent.futures import ThreadPoolExecutor
d, e128, manp = sys.argv[1:4]
fails = []
def check(ok, msg):
    print(("OK   " if ok else "FAIL ") + msg); (None if ok else fails.append(msg))
man = {e["path"]: e for e in json.load(open(manp))["entries"]}
have = set()
for root, dirs, names in os.walk(d, followlinks=True):
    dirs[:] = [x for x in dirs if not x.startswith(".")]
    have |= {os.path.relpath(os.path.join(root, n), d) for n in names if not n.startswith(".")}
check(have == set(man), f"file set equals the manifest ({len(have)} files; missing {sorted(set(man) - have)[:5]}, extra {sorted(have - set(man))[:5]})")
def sha(rel):
    h = hashlib.sha256()
    with open(os.path.join(d, rel), "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""): h.update(b)
    return h.hexdigest()
rels = sorted(set(man) & have, key=lambda r: -man[r]["size"])
with ThreadPoolExecutor(12) as ex: hs = dict(zip(rels, ex.map(sha, rels)))
bad = [r for r in rels if os.path.getsize(os.path.join(d, r)) != man[r]["size"] or hs[r] != man[r]["sha256"]]
check(not bad, f"size + sha256 of {len(rels)} files vs the C1 manifest (bad: {bad[:5]})")
c, ce = json.load(open(f"{d}/config.json")), json.load(open(f"{e128}/config.json"))
t, te = c.get("text_config", c), ce.get("text_config", ce)
for k in sorted(k for k in te if k.startswith("engram")):
    check(t.get(k) == te.get(k), f"config text_config.{k} equals E128's")
diff = sorted(k for k in set(t) | set(te) if t.get(k) != te.get(k))
print("     (text_config keys that differ from E128's: %s)" % diff)
m, me = json.load(open(f"{d}/e043_manifest.json")), json.load(open(f"{e128}/e043_manifest.json"))
check(m.get("engram") == me.get("engram"), f"e043_manifest engram section equals E128's ({ {k: (v['rows'], v['kept']) for k, v in m.get('engram', {}).items()} })")
check(m.get("engram_from") == me.get("engram_from"), f"e043_manifest engram_from equals E128's ({m.get('engram_from')})")
print("     (expert keep set equals E128's: %s; experts_per_layer %s)" % (m.get("keep_original_ids") == me.get("keep_original_ids"), m.get("experts_per_layer")))
ENG = re.compile(r"\.engram\.(embed\.|embed_tokens\.)")
idx = json.load(open(f"{d}/model.safetensors.index.json")); wm = idx["weight_map"]
ie = json.load(open(f"{e128}/model.safetensors.index.json"))["weight_map"]
check({k: v for k, v in wm.items() if ENG.search(k)} == {k: v for k, v in ie.items() if ENG.search(k)}, "index Engram-table entries equal E128's exactly (18)")
def header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; return json.loads(f.read(n))
seen, hdr_bad, nt = {}, [], 0
for f in sorted(set(wm.values())):
    if os.path.isabs(f) or ".." in f.split("/"): hdr_bad.append(f"bad path {f}"); continue
    hk = {k for k in header(os.path.join(d, f)) if k != "__metadata__"}; ik = {k for k, v in wm.items() if v == f}
    nt += len(hk)
    if hk != ik: hdr_bad.append(f"{f}: {len(hk ^ ik)} keys differ")
    for k in hk:
        if k in seen: hdr_bad.append(f"{k} in {seen[k]} and {f}")
        seen[k] = f
check(not hdr_bad, f"each indexed file's header == its index keys ({len(set(wm.values()))} files, {nt} tensors): {hdr_bad[:3]}")
vis = sorted(k for k in wm if re.search(r"vision|visual|vit\b|mm_proj|projector", k, re.I))
check(bool(c.get("vision_config")) and len(vis) > 0, f"vision tower: vision_config {bool(c.get('vision_config'))}, {len(vis)} vision tensors in {sorted({wm[k] for k in vis})} (e.g. {vis[:2]})")
layers = t.get("num_hidden_layers")
g = sorted(x for x in have if x.startswith("gptq/") and x.endswith(".safetensors"))
check(len(g) == layers, f"gptq layer files {len(g)} == num_hidden_layers {layers}")
print("C1_CHECK", "PASS" if not fails else f"FAIL ({len(fails)})")
