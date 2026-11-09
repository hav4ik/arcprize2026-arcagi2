"""Compare a new wheel set with an old one and (with --link) replace each new file that is byte-identical (same name,
size, sha256) to an old one by a hard link to the old file, to save disk. Prints a summary: identical, changed (same
name, different bytes), new-only, old-only, and every vllm wheel.
usage: wheel_dedupe.py NEW_DIR OLD_DIR [--link]"""
import hashlib, os, sys
from concurrent.futures import ThreadPoolExecutor
new, old = sys.argv[1], sys.argv[2]; link = "--link" in sys.argv
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()
N = {f: os.path.join(new, f) for f in os.listdir(new) if os.path.isfile(os.path.join(new, f))}
O = {f: os.path.join(old, f) for f in os.listdir(old) if os.path.isfile(os.path.join(old, f))}
common = sorted(f for f in N if f in O and os.path.getsize(N[f]) == os.path.getsize(O[f]))
with ThreadPoolExecutor(8) as ex:
    hn = dict(zip(common, ex.map(lambda f: sha(N[f]), common)))
    ho = dict(zip(common, ex.map(lambda f: sha(O[f]), common)))
same = [f for f in common if hn[f] == ho[f]]
changed = sorted(f for f in N if f in O and f not in same)
print(f"new {len(N)} files, old {len(O)}; identical {len(same)}; same name but different bytes {changed}")
print("only in new:", sorted(set(N) - set(O)))
print("only in old:", sorted(set(O) - set(N)))
print("vllm wheels in new:", [f for f in N if f.startswith("vllm-")], "| old:", [f for f in O if f.startswith("vllm-")])
saved = 0
if link:
    for f in same:
        if os.stat(N[f]).st_ino == os.stat(O[f]).st_ino:
            continue
        tmp = N[f] + ".lnk"
        os.link(O[f], tmp); os.replace(tmp, N[f]); saved += os.path.getsize(O[f])
    print(f"hard-linked {len(same)} identical files, freed {saved / 1e9:.2f} GB")
