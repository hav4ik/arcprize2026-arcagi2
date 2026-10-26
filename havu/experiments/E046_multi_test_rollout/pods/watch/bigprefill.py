"""Find turns whose prompt was mostly uncached (big re-prefill) in solver traces; show what preceded them.
Usage: bigprefill.py <attempts.jsonl> <traces dir> [min_uncached]"""
import gzip, json, os, sys, collections
att, tdir = sys.argv[1], sys.argv[2]
MIN = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
recs = {}
for r in map(json.loads, open(att)):
    if "task_id" in r:
        recs[(r["task_id"], r.get("pair", 0), r["attempt"])] = r
tot = collections.Counter()
rows = []
for f in sorted(os.listdir(tdir)):
    try:
        d = json.load(gzip.open(os.path.join(tdir, f)))
    except Exception:
        tot["bad"] += 1
        continue
    tid, p, a = f.replace(".json.gz", "").split("_")
    r = recs.get((tid, int(p[1:]), int(a[1:])), {})
    ev = d.get("events", [])
    turns = [e for e in ev if "usage" in e]
    prev = None
    for i, e in enumerate(ev):
        if "usage" not in e:
            continue
        u = e["usage"] or {}
        pt = u.get("prompt_tokens") or 0
        ct = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
        unc = pt - ct
        tot["turns"] += 1; tot["uncached"] += unc; tot["prompt"] += pt
        if unc >= MIN and e.get("turn", 0) > 1:
            between = [x for x in ev[ev.index(prev) + 1:i]] if prev is not None else []
            rows.append((f, e.get("turn"), pt, ct, unc, prev.get("finish_reason") if prev else None,
                         prev.get("max_tokens") if prev else None, (prev.get("usage") or {}).get("completion_tokens") if prev else None,
                         [{k: v for k, v in x.items() if k not in ("tool_call_id",)} for x in between][:3],
                         r.get("token_exact"), r.get("t_end_h"), r.get("wall_s")))
        prev = e
print(f"{tdir}: {tot['bad']} unreadable, {tot['turns']} turns, prompt {tot['prompt']/1e6:.1f}M, uncached {tot['uncached']/1e6:.2f}M; turns (>1) with uncached >= {MIN}: {len(rows)}")
for x in rows:
    f, t, pt, ct, unc, pfin, pmax, pcomp, between, te, tend, wall = x
    print(f"  {f} turn {t}: prompt {pt}, cached {ct}, uncached {unc}; prev turn finish={pfin} max_tokens={pmax} completion={pcomp}; between={json.dumps(between)[:200]}; token_exact={te}; ended {tend}h after {wall}s")
