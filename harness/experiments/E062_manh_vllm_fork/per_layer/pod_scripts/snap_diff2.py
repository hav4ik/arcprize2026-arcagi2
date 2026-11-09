"""Like snap_diff.py, but keyed by the first 4 vllm/humming frames, and only for _prepare_output allocations; also
reports block-size histograms of those allocations."""
import collections, pickle, sys

def chain(frames, n=4):
    out = []
    for fr in frames or []:
        fn = fr.get("filename", "")
        if ("/vllm/" in fn or "/humming/" in fn) and "/torch/" not in fn:
            out.append(f"{fn.split('site-packages/')[-1].split('/')[-1]}:{fr.get('line')}:{fr.get('name')}")
            if len(out) == n:
                break
    return " <- ".join(out)

def load(path):
    snap = pickle.load(open(path, "rb"))
    by, sizes = collections.Counter(), collections.Counter()
    for seg in snap["segments"]:
        for blk in seg["blocks"]:
            if blk["state"] != "active_allocated":
                continue
            c = chain(blk.get("frames"))
            if "_prepare_output" in c:
                by[c] += blk["size"]
                sizes[(c.split(" <- ")[1] if " <- " in c else c, blk["size"])] += 1
    return by, sizes

(a, sa), (b, sb) = load(sys.argv[1]), load(sys.argv[2])
for k in sorted(set(a) | set(b), key=lambda k: -abs(a[k] - b[k]))[:12]:
    print(f"{(a[k] - b[k]) / 2**20:+9.1f} MiB  A {a[k] / 2**20:9.1f}  B {b[k] / 2**20:9.1f}  {k[:220]}")
