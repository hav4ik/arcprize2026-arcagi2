"""Summarise E067 memory-lab hook logs (<name>.<pid>.jsonl).

  analyze.py LOG.jsonl [LOG2.jsonl ...]            one summary block per run
  analyze.py --timeline N LOG.jsonl                 N-row timeline (max per slice)
  analyze.py --by ntok|maxctx|nreq LOG.jsonl        transient peak (ap - a) by bucket
"""

import argparse
import collections
import json


def load(p):
    return [json.loads(line) for line in open(p)]


def summary(p):
    rs = load(p)
    ev = {r["ev"]: r for r in rs if r["ev"] not in ("step", "frag")}
    st = [r for r in rs if r["ev"] == "step" and "a" in r]
    busy = [r for r in st if r.get("ntok", 0) > 0]
    fr = [r for r in rs if r["ev"] == "frag"]
    out = [f"== {p.rsplit('/', 1)[-1]}"]
    if "profiled" in ev:
        e = ev["profiled"]
        out.append(
            f"startup: total {e['total']:.2f}  init_used {e['init_used']:.3f}  requested {e['requested']:.2f}"
            f"  weights {ev.get('loaded', {}).get('weights', 0):.2f}  peak_act {e['peak_act']:.3f}"
            f"  non_torch {e['non_torch']:.3f}  cg_est {e['cg_est']:.3f}  KV {e['kv_budget']:.2f}")
    if "ready" in ev:
        e = ev["ready"]
        out.append(f"ready:   alloc {e['a']:.3f}  reserved {e['r']:.3f}  free {e['free']:.3f}"
                   f"  inactive_split {e['isp']:.3f}  segs {e['seg']}")
    if "cap" in ev:
        e = ev["cap"]
        out.append(f"cap:     {e['cap']:.3f} (reserved {e['reserved']:.3f} + free {e['free']:.3f}"
                   f" - {e['keep_mb']} MiB)")
    if busy:
        a0 = min(r["a"] for r in busy)
        out.append(
            f"run:     {len(st)} steps, {busy[-1]['t'] - busy[0]['t']:.0f} s busy; alloc min {a0:.3f}"
            f" max {max(r['a'] for r in busy):.3f}; peak max {max(r['ap'] for r in busy):.3f}"
            f" (transient {max(r['ap'] - r['a'] for r in busy):.3f})")
        out.append(
            f"         reserved ready->end {ev.get('ready', busy[0])['r']:.3f} -> {st[-1]['r']:.3f}"
            f" (max {max(r['rp'] for r in st):.3f}); free min {min(r['free'] for r in st):.3f};"
            f" inactive_split max {max(r['isp'] for r in st):.3f}; retries {st[-1]['ret']}; ooms {st[-1]['oom']};"
            f" cudaMalloc {ev.get('ready', st[0])['dalloc']}->{st[-1]['dalloc']}")
        out.append(f"         max ntok {max(r['ntok'] for r in busy)}  max nreq {max(r['nreq'] for r in busy)}"
                   f"  max maxctx {max(r['maxctx'] for r in busy)}  max sumctx {max(r['sumctx'] for r in busy)}"
                   f"  preempted {sum(r.get('pre', 0) for r in busy)}")
    if fr:
        f = fr[-1]
        out.append(f"frag(last): {f['nseg']} segs, {f['nseg_split_with_free']} split w/ free, cached free"
                   f" {f['cached_free']:.3f}, largest free {f['largest_free']:.3f}")
    for r in rs:
        if r["ev"] == "oom":
            out.append(f"OOM at step {r['s']} ({r['where']}): ntok {r.get('ntok')} maxctx {r.get('maxctx')}"
                       f" alloc {r['a']:.3f} reserved {r['r']:.3f} free {r['free']:.3f}: {r['msg'][:220]}")
    return "\n".join(out)


def timeline(p, n):
    st = [r for r in load(p) if r["ev"] == "step" and "a" in r]
    if not st:
        return
    t0, t1 = st[0]["t"], st[-1]["t"]
    w = (t1 - t0) / n or 1
    rows = collections.defaultdict(list)
    for r in st:
        rows[min(int((r["t"] - t0) / w), n - 1)].append(r)
    print(f"{'t_s':>6} {'steps':>5} {'ntok':>5} {'nreq':>4} {'maxctx':>6} {'alloc':>7} {'peak':>7}"
          f" {'resv':>7} {'isp':>6} {'free':>6} {'ret':>4} {'mallocs':>7}")
    for k in sorted(rows):
        v = rows[k]
        print(f"{v[0]['t']:6.0f} {len(v):5d} {max(x['ntok'] for x in v):5d} {max(x['nreq'] for x in v):4d}"
              f" {max(x['maxctx'] for x in v):6d} {max(x['a'] for x in v):7.3f} {max(x['ap'] for x in v):7.3f}"
              f" {max(x['r'] for x in v):7.3f} {max(x['isp'] for x in v):6.3f} {min(x['free'] for x in v):6.3f}"
              f" {v[-1]['ret']:4d} {v[-1]['dalloc']:7d}")


def by(p, key):
    st = [r for r in load(p) if r["ev"] == "step" and r.get("ntok", 0) > 0]
    edges = {"ntok": [1, 2, 64, 256, 1024, 2048, 4096, 8192, 1 << 30],
             "maxctx": [0, 4096, 8192, 16384, 32768, 49152, 1 << 30],
             "nreq": [0, 1, 4, 16, 32, 64, 1 << 30]}[key]
    rows = collections.defaultdict(list)
    for r in st:
        for i in range(len(edges) - 1):
            if edges[i] <= r[key] < edges[i + 1] or (key == "ntok" and r[key] == edges[i + 1] - 1):
                rows[i].append(r)
                break
    print(f"{key:>16} {'steps':>6} {'transient p50':>13} {'max':>7}")
    for i in sorted(rows):
        v = sorted(x["ap"] - x["a"] for x in rows[i])
        print(f"{edges[i]:>7}-{edges[i + 1]:<8} {len(v):6d} {v[len(v) // 2]:13.3f} {v[-1]:7.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--timeline", type=int, default=0)
    ap.add_argument("--by", default="")
    a = ap.parse_args()
    for p in a.logs:
        if a.timeline:
            timeline(p, a.timeline)
        elif a.by:
            by(p, a.by)
        else:
            print(summary(p))
