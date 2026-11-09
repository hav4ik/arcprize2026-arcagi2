"""Manifest of a model dir in cell 1b's INTEGRITY format: {"model": {rel path: [size, sha256]}}, every file under the
dir (symlinks followed; hidden dirs/files skipped, as stage_model.py's walk), hashed with N threads.
usage: make_manifest.py MODEL_DIR OUT.json [threads]"""
import hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
d, outp = sys.argv[1], sys.argv[2]; th = int(sys.argv[3]) if len(sys.argv) > 3 else 12
rels = []
for root, dirs, names in os.walk(d, followlinks=True):
    dirs[:] = [x for x in dirs if not x.startswith(".")]
    rels += [os.path.relpath(os.path.join(root, n), d) for n in names if not n.startswith(".")]
def sha(rel):
    h = hashlib.sha256()
    with open(os.path.join(d, rel), "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()
t0 = time.time()
rels.sort(key=lambda r: -os.path.getsize(os.path.join(d, r)))
with ThreadPoolExecutor(th) as ex:
    hs = list(ex.map(sha, rels))
man = {"model": {r: [os.path.getsize(os.path.join(d, r)), h] for r, h in sorted(zip(rels, hs))}}
json.dump(man, open(outp, "w"))
print(f"{len(rels)} files, {sum(v[0] for v in man['model'].values()) / 1e9:.2f} GB hashed in {time.time() - t0:.0f}s -> {outp}")
