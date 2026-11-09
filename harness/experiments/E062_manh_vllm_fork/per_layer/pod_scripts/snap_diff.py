"""Group live CUDA blocks of two torch memory snapshots (end of model loading) by allocating call site, and print the
sites whose live bytes differ most. Usage: snap_diff.py A.pkl B.pkl"""
import collections
import pickle
import sys


def site(frames):
    """First frame inside vllm/ or humming/ (skipping torch internals), as file:line func."""
    for fr in frames or []:
        fn = fr.get("filename", "")
        if ("/vllm/" in fn or "/humming/" in fn) and "/torch/" not in fn:
            return f"{fn.split('site-packages/')[-1]}:{fr.get('line')} {fr.get('name')}"
    return "(no vllm frame)"


def live_by_site(path):
    snap = pickle.load(open(path, "rb"))
    out = collections.Counter()
    for seg in snap["segments"]:
        for blk in seg["blocks"]:
            if blk["state"] == "active_allocated":
                out[site(blk.get("frames"))] += blk["size"]
    return out


a, b = live_by_site(sys.argv[1]), live_by_site(sys.argv[2])
print(f"total live A {sum(a.values()) / 2**30:.3f} GiB   B {sum(b.values()) / 2**30:.3f} GiB")
rows = sorted(set(a) | set(b), key=lambda k: -abs(a[k] - b[k]))
for k in rows[:25]:
    print(f"{(a[k] - b[k]) / 2**20:+9.1f} MiB  A {a[k] / 2**20:9.1f}  B {b[k] / 2**20:9.1f}  {k[:150]}")
