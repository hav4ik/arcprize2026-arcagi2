"""Hard-link E128's 16 Engram shards into the C1 model dir after checking each one's size and sha256 against the C1
manifest (the E066 build's c1_gptq_sha256_manifest.json). usage: link_engram.py MANIFEST.json E128_DIR C1_DIR"""
import hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor
man, src, dst = sys.argv[1:4]
ents = {e["path"]: e for e in json.load(open(man))["entries"] if e["path"].startswith("model-engram-")}
assert len(ents) == 16, len(ents)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""): h.update(b)
    return h.hexdigest()
names = sorted(ents)
with ThreadPoolExecutor(8) as ex:
    hs = dict(zip(names, ex.map(lambda n: sha(os.path.join(src, n)), names)))
bad = [n for n in names if os.path.getsize(os.path.join(src, n)) != ents[n]["size"] or hs[n] != ents[n]["sha256"]]
print("engram shards checked", len(names), "mismatch", bad)
if bad: sys.exit(1)
os.makedirs(dst, exist_ok=True)
for n in names:
    t = os.path.join(dst, n)
    if os.path.exists(t):
        assert os.path.samefile(t, os.path.join(src, n)), f"{t} exists and is another file"
        continue
    os.link(os.path.join(src, n), t)
print("LINK_OK: 16 hard links in", dst)
