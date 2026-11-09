"""E064b (pod): assemble the Kaggle Model copy of E128 (row-chunked Engram tables) next to the pod's HF copy.

    python make_kaggle_copy.py plan <hf dir> <kaggle manifest.json> <dst dir> <todo.tsv>
    python make_kaggle_copy.py verify <dst dir> <kaggle manifest.json>

plan: every manifest file whose HF-copy counterpart (same path) has the same size and sha256 is hardlinked into <dst>;
the rest is written to <todo.tsv> (path<TAB>size) for dl_kaggle_files.py (model). verify: every manifest file exists in
<dst> with its size and sha256; prints VERIFY_OK or the mismatches.
"""
import concurrent.futures as cf
import hashlib
import json
import os
import sys


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(64 << 20):
            h.update(chunk)
    return h.hexdigest()


cmd = sys.argv[1]
if cmd == "plan":
    hf, man, dst, todo = sys.argv[2:6]
    entries = json.load(open(man))["entries"]
    def check(e):
        src = os.path.join(hf, e["path"])
        if os.path.isfile(src) and os.path.getsize(src) == e["size"] and sha(src) == e["sha256"]:
            return e, True
        return e, False
    link, need = [], []
    with cf.ThreadPoolExecutor(16) as ex:
        for e, same in ex.map(check, entries):
            (link if same else need).append(e)
    for e in link:
        out = os.path.join(dst, e["path"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if not os.path.exists(out):
            os.link(os.path.join(hf, e["path"]), out)
    with open(todo, "w") as f:
        for e in need:
            f.write(f"{e['path']}\t{e['size']}\n")
    print(f"PLAN linked={len(link)} ({sum(e['size'] for e in link) / 1e9:.1f} GB) "
          f"download={len(need)} ({sum(e['size'] for e in need) / 1e9:.1f} GB)")
elif cmd == "verify":
    dst, man = sys.argv[2:4]
    entries = json.load(open(man))["entries"]
    def ok(e):
        p = os.path.join(dst, e["path"])
        return e["path"], os.path.isfile(p) and os.path.getsize(p) == e["size"] and sha(p) == e["sha256"]
    with cf.ThreadPoolExecutor(16) as ex:
        bad = [p for p, good in ex.map(ok, entries) if not good]
    extra = []
    for root, _, files in os.walk(dst):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), dst)
            if not rel.startswith(".khcache") and rel not in {e["path"] for e in entries}:
                extra.append(rel)
    print("VERIFY_OK" if not bad else f"VERIFY_BAD {bad[:20]}", f"files={len(entries)} extra={extra[:10]}")
