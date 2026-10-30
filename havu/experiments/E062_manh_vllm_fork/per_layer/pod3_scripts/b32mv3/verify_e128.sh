#!/bin/bash
# sha256 of every E128 file against the B32-M v3 notebook's cell 1b manifest (136 files). Log: verify_e128.log
B=/root/e066/b32mv3; D=/root/models/e128
/usr/bin/python3 - $D $B/e128_cell1b_manifest.json > $B/verify_e128.log 2>&1 <<'PY'
import hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor
d, m = sys.argv[1], json.load(open(sys.argv[2]))
def h(rel):
    p = os.path.join(d, rel)
    if not os.path.exists(p): return rel, "missing"
    if os.path.getsize(p) != m[rel][0]: return rel, "size"
    x = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): x.update(b)
    return rel, "ok" if x.hexdigest() == m[rel][1] else "sha"
with ThreadPoolExecutor(8) as ex: res = list(ex.map(h, sorted(m)))
bad = [r for r in res if r[1] != "ok"]
print(f"VERIFY {'OK' if not bad else 'FAIL'}: {len(res) - len(bad)}/{len(res)} files match cell 1b's manifest; bad: {bad}")
PY
cat $B/verify_e128.log
