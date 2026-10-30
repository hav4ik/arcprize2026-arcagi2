#!/usr/bin/env python3
"""First-attempt accuracy of a (running) 12 h replica vs a no-limit benchmark of the same model and harness, paired
on the same original tasks.

The replica runs a workload of transformed copies (evalx2t_s2026: the 120 public eval tasks plus one D8 + colour-
permuted copy of each; the idmap maps every workload id to its original). The benchmark ran the original tasks with no
deadline and no KV pressure. Finished replica first attempts are biased towards easy tasks (the solver queues the
smallest tasks first, and short attempts finish first), so overall numbers are not comparable; this script compares
each finished replica first attempt with the benchmark's pass@1 on the same original task.

Scoring (as the replicas and ARC Prize do): a task scores the fraction of its test inputs answered exactly; "fully
correct" = every test input right (the pod reports' first_correct). Benchmark pass@1 per task = mean over its finished
attempts (all benchmark files pooled). Benchmark pairs are matched to the original tasks by test-input content; a
benchmark task whose prompt differs from the original (the GitHub eval set has 6 such tasks) is left out.

Two selection effects, and what controls each:
- task-level (easy tasks first): removed by the pairing;
- attempt-level (on any task, a short attempt finishes earlier and is more often right): not removed by the pairing,
  so a paired difference of a running replica is biased upwards. Two controls: (1) a length-censored benchmark
  (benchmark attempts no longer than the replica attempt could have run by the cut, at its own rate), and (2) the same
  table for finished replicas of the same workload, cut at the same hours (pass them as extra replica dirs).

Usage:
  replica_vs_bench.py REPLICA [REPLICA ...] --bench [LABEL=]FILE_OR_DIR [--bench ...] --idmap IDMAP
      [--cut H] [--at-n N,N] [--prefix-m M,M] [--boot 10000] [--losses N] [--json OUT]
REPLICA: a run dir (attempts.jsonl in it, dsv41_logs/, out/dsv41_logs/, data/ or data/dsv41_logs/) or the file;
LABEL=PATH names it. Benchmark files: solver attempts.jsonl (one record per task, Kaggle eval set) or harness results
(evals/results/*.jsonl, per-pair records with config.challenges).
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REF_CH = ROOT / "data/arc-agi_evaluation_challenges.json"
REF_SOL = ROOT / "data/arc-agi_evaluation_solutions.json"
BIG = 10 ** 9  # completion tokens the solver adds per cut stream (loop guard / force deadline)
D8 = {
    "id": lambda g: [list(r) for r in g],
    "r90": lambda g: [list(r) for r in zip(*g[::-1])],
    "r180": lambda g: [list(r[::-1]) for r in g[::-1]],
    "r270": lambda g: [list(r) for r in zip(*g)][::-1],
    "fh": lambda g: [list(r[::-1]) for r in g],
    "fv": lambda g: [list(r) for r in g[::-1]],
    "t": lambda g: [list(r) for r in zip(*g)],
    "at": lambda g: [list(r[::-1]) for r in [list(x) for x in zip(*g)][::-1]],
}


def H(g):
    return hashlib.sha1(json.dumps(g, separators=(",", ":")).encode()).hexdigest()


def load_challenges(path):
    p = Path(path)
    if p.is_dir():
        return {Path(f).stem: json.load(open(f)) for f in glob.glob(str(p / "*.json"))}
    return json.load(open(p))


def find_attempts(path):
    p = Path(path)
    if p.is_file():
        return p
    for sub in ("", "dsv41_logs", "out/dsv41_logs", "data", "data/dsv41_logs"):
        q = p / sub / "attempts.jsonl"
        if q.exists():
            return q
    raise SystemExit(f"no attempts.jsonl under {p}")


def labelled(arg):
    if "=" in arg and not Path(arg).exists():
        lab, path = arg.split("=", 1)
        return lab, path
    return Path(arg).stem if Path(arg).is_file() else Path(arg).name, arg


def tokens(rec):
    """Output tokens; a cut stream reports BIG per cut instead of that turn's count (lower bound, flagged)."""
    c = (rec.get("usage") or {}).get("completion") or 0
    ncut = sum((rec.get("cuts") or {}).values())
    if ncut and c >= BIG:
        return c - ncut * BIG, True
    return c, False


def last_records(path):
    last = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "task_id" in r:
            last[(r["task_id"], r.get("pair", 0), r["attempt"])] = r
    return last


def attempts_of(last):
    """{(task, attempt): {"preds": {local pair: grid}, "meta": record of the lowest pair, "error": bool}}."""
    out = {}
    for (t, p, a), r in sorted(last.items(), key=lambda kv: kv[0][1]):
        d = out.setdefault((t, a), {"preds": {}, "meta": r, "error": False})
        if "error" in r:
            d["error"] = True
        n, pred = r.get("multi_test"), r.get("pred")
        if n and is_grid_list(pred, n):  # solver record: one per task, pred = one grid per test input
            for q in range(n):
                d["preds"].setdefault(q, pred[q])
        else:  # one test input (harness results: one record per pair); a missing pair counts as wrong
            d["preds"][p] = pred
    return out


def is_grid_list(x, n):
    """A list of n grids (not one grid of n rows: a grid's rows hold ints)."""
    return (isinstance(x, list) and len(x) == n
            and all(g is None or (isinstance(g, list) and g and isinstance(g[0], list)) for g in x))


def meta_row(r):
    tok, cut = tokens(r)
    return {"stop": r.get("stop_reason"), "turns": r.get("turns") or 0, "tokens": tok, "tok_lb": cut,
            "cuts": r.get("cuts") or {}, "train_pass": r.get("train_pass"), "tool_errors": r.get("tool_errors") or 0,
            "wall_s": r.get("wall_s") or r.get("seconds") or 0, "t_end_h": r.get("t_end_h"),
            "answer_source": r.get("answer_source"), "auto_repairs": (r.get("auto_repairs") or {}).get("applied", 0),
            "te_breaks": (r.get("token_exact") or {}).get("breaks", 0),
            "te_resyncs": (r.get("token_exact") or {}).get("resyncs", 0)}


def load_bench(label, path, ref_ch, ref_sol):
    """-> {orig task: [attempt rows]}, excluded tasks. Row: right = {ref pair q: bool}, plus meta."""
    path = find_attempts(path) if Path(path).is_dir() else Path(path)
    last = last_records(path)
    any_rec = next(iter(last.values()))
    cfg = any_rec.get("config") or {}
    ch = load_challenges(cfg["challenges"]) if cfg.get("challenges") else ref_ch
    ref_key = {(t, H(x["input"])): q for t, v in ref_ch.items() for q, x in enumerate(v["test"])}
    out, excluded = collections.defaultdict(list), set()
    for (t, a), d in attempts_of(last).items():
        if d["error"] or t not in ref_ch or t not in ch:
            excluded.add(t)
            continue
        same = (json.dumps(ch[t]["train"]) == json.dumps(ref_ch[t]["train"])
                and [H(x["input"]) for x in ch[t]["test"]] == [H(x["input"]) for x in ref_ch[t]["test"]])
        if not same:
            excluded.add(t)
            continue
        right = {}
        for ql, x in enumerate(ch[t]["test"]):
            q = ref_key.get((t, H(x["input"])))
            if q is None:
                continue
            g = d["preds"].get(ql)
            right[q] = g is not None and g == ref_sol[t][q]
        out[t].append(dict(meta_row(d["meta"]), right=right, src=label, attempt=a))
    return dict(out), sorted(excluded - set(out))


def check_workload(ch, idmap, ref_ch):
    """Every copy's test inputs are its original's under the idmap's D8 + colour permutation, in the same order."""
    bad = []
    for t, m in idmap.items():
        o = ref_ch[m["orig"]]
        if len(o["test"]) != len(ch[t]["test"]):
            bad.append(t)
            continue
        perm = m.get("perm") or list(range(10))
        for q, x in enumerate(o["test"]):
            g = [[perm[v] for v in r] for r in D8[m.get("d8", "id")](x["input"])]
            if g != ch[t]["test"][q]["input"]:
                bad.append(t)
                break
    return bad


def load_replica(path, ch, sol, idmap, cut=None):
    last = last_records(find_attempts(path))
    rows, errored = {}, []
    for (t, a), d in attempts_of(last).items():
        if a != 0:
            continue
        m = meta_row(d["meta"])
        if cut is not None and (m["t_end_h"] or 0) > cut:
            continue
        if d["error"]:
            errored.append(t)
            continue
        n = len(ch[t]["test"])
        right = {q: d["preds"].get(q) is not None and d["preds"].get(q) == sol[t][q] for q in range(n)}
        im = idmap[t]
        rows[t] = dict(m, right=right, orig=im["orig"], kind=im["kind"], n=n)
    return rows, errored


def queue_order(ch, universe):
    """The solver's first-attempt queue: tasks by prompt size (train JSON + largest test input), stable in file order."""
    size = {t: len(json.dumps(ch[t]["train"])) + max(len(json.dumps(y["input"])) for y in ch[t]["test"]) for t in universe}
    return sorted(universe, key=lambda t: size[t])


def score(right, qs=None):
    qs = sorted(right) if qs is None else qs
    if not qs:
        return None, None
    v = [bool(right.get(q)) for q in qs]
    return sum(v) / len(v), float(all(v))


def censor_cap(r, at_h):
    """Tokens this replica attempt could have produced by `at_h` at its own average rate (inf if unknown)."""
    if not r["wall_s"] or not r["tokens"] or r["t_end_h"] is None or at_h is None:
        return math.inf
    start_h = r["t_end_h"] - r["wall_s"] / 3600
    return r["tokens"] * max(at_h - start_h, 0) * 3600 / r["wall_s"]


def paired_items(rep, bench, at_h=None, censored=False):
    items = []
    for t, r in rep.items():
        b = bench.get(r["orig"])
        if not b:
            continue
        if censored:
            cap = censor_cap(r, at_h)
            b = [x for x in b if x["tokens"] <= cap]
            if not b:
                continue
        qs = sorted(set(r["right"]) & set.intersection(*(set(x["right"]) for x in b)))
        if not qs:
            continue
        rs, rf = score(r["right"], qs)
        bs = statistics.mean(score(x["right"], qs)[0] for x in b)
        bf = statistics.mean(score(x["right"], qs)[1] for x in b)
        items.append({"task": t, "orig": r["orig"], "kind": r["kind"], "r": rs, "b": bs, "rf": rf, "bf": bf,
                      "nb": len(b), "t_end_h": r["t_end_h"], "rep": r, "bench": b})
    return items


def boot_ci(items, key_r, key_b, reps, seed=0):
    groups = collections.defaultdict(list)
    for it in items:
        groups[it["orig"]].append(it[key_r] - it[key_b])
    keys = sorted(groups)
    if len(keys) < 3:
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        tot = n = 0
        for _ in keys:
            g = groups[keys[rng.randrange(len(keys))]]
            tot += sum(g)
            n += len(g)
        vals.append(tot / n)
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]


def summarise(items, reps, full=False):
    kr, kb = ("rf", "bf") if full else ("r", "b")
    if not items:
        return None
    n = len(items)
    r = sum(it[kr] for it in items) / n
    b = sum(it[kb] for it in items) / n
    lo, hi = boot_ci(items, kr, kb, reps)
    better = sum(it[kr] > it[kb] + 1e-9 for it in items)
    worse = sum(it[kr] < it[kb] - 1e-9 for it in items)
    return {"n": n, "orig_tasks": len({it["orig"] for it in items}), "replica": 100 * r, "bench": 100 * b,
            "diff": 100 * (r - b), "ci": (None if lo is None else 100 * lo, None if hi is None else 100 * hi),
            "better": better, "worse": worse}


def fmt_s(s):
    if not s:
        return "| - | - | - | - | - | - |"
    ci = "n/a" if s["ci"][0] is None else f"[{s['ci'][0]:+.1f}, {s['ci'][1]:+.1f}]"
    return (f"| {s['n']} ({s['orig_tasks']} orig) | {s['replica']:.1f} | {s['bench']:.1f} | {s['diff']:+.1f} | {ci} | "
            f"{s['better']} / {s['worse']} |")


HEAD = "| n (orig tasks) | replica pass@1 | bench pass@1, same tasks | diff (pp) | 95% CI | tasks better / worse |"
SEP = "|---|---|---|---|---|---|"


def stop_counts(rows):
    c = collections.Counter(r["stop"] for r in rows)
    return ", ".join(f"{k} {v}" for k, v in c.most_common())


def report(label, path, ch, sol, idmap, bench, cut, reps, n_losses, at_n=(), prefix_m=()):
    rep, errored = load_replica(path, ch, sol, idmap, cut)
    if not rep:
        print(f"\n## {label}: no finished first attempts")
        return {}
    t_max = max(r["t_end_h"] or 0 for r in rep.values())
    at_h = cut if cut is not None else t_max
    items = paired_items(rep, bench)
    print(f"\n## {label}")
    print(f"`{find_attempts(path)}`: {len(rep)} finished first attempts (latest t_end_h {t_max:.2f}), "
          f"{len(errored)} errored; kinds {dict(collections.Counter(r['kind'] for r in rep.values()))}; "
          f"paired with the benchmark: {len(items)}")
    out = {"label": label, "finished": len(rep), "t_max": t_max, "errored": len(errored)}
    print("\n### Paired on the same original tasks (as of the latest record)\n")
    print("| scoring | " + HEAD[2:])
    print("|---|" + SEP[1:])
    for name, full, sel in (("ARC task score", False, items), ("fully correct", True, items),
                            ("ARC, originals only", False, [i for i in items if i["kind"] == "orig"]),
                            ("ARC, transformed copies only", False, [i for i in items if i["kind"] == "copy"])):
        s = summarise(sel, reps, full)
        out[name] = s
        print(f"| {name} " + fmt_s(s))
    cens = paired_items(rep, bench, at_h, censored=True)
    s = summarise(cens, reps)
    out["censored"] = s
    print(f"| ARC, length-censored benchmark (attempts <= what the replica attempt could run by {at_h:.2f} h) " + fmt_s(s))
    srcs = sorted({x["src"] for v in bench.values() for x in v})
    if len(srcs) > 1:
        out["by_bench"] = {}
        for src in srcs:
            one = {t: [x for x in v if x["src"] == src] for t, v in bench.items()}
            s = summarise(paired_items(rep, {t: v for t, v in one.items() if v}), reps)
            out["by_bench"][src] = s
            print(f"| ARC, benchmark {src} only " + fmt_s(s))

    # progress profile: the first N finished first attempts (aligns runs of different speed by progress)
    order = sorted(items, key=lambda i: (i["t_end_h"] or 0))
    print("\n### Progress profile (ARC task score; the first N finished first attempts, by t_end_h)\n")
    print("| first N | up to t_end_h | replica | bench, same tasks | diff [95% CI] |")
    print("|---|---|---|---|---|")
    out["progress"] = []
    for n in sorted(set(list(range(25, len(order), 25)) + [len(order)] + [k for k in at_n if k <= len(order)])):
        s = summarise(order[:n], reps)
        out["progress"].append(dict(s, first_n=n, t_end_h=order[n - 1]["t_end_h"]))
        ci = "" if s["ci"][0] is None else f" [{s['ci'][0]:+.1f}, {s['ci'][1]:+.1f}]"
        print(f"| {n} | {order[n - 1]['t_end_h']:.2f} | {s['replica']:.1f} | {s['bench']:.1f} | {s['diff']:+.1f}{ci} |")

    # queue order: the solver starts first attempts smallest task first, whatever their outcome, so a queue prefix whose
    # first attempts have all finished carries no attempt-level selection (only task-level, which the pairing removes)
    universe = [t for t in ch if any(r["kind"] == "copy" for r in rep.values()) or idmap[t]["kind"] == "orig"]
    qpos = {t: i for i, t in enumerate(queue_order(ch, universe))}
    at_pos = sorted(qpos, key=qpos.get)
    m_full = 0
    while m_full < len(at_pos) and at_pos[m_full] in rep:
        m_full += 1
    by_pos = {qpos[i["task"]]: i for i in items}
    print(f"\n### Queue order (first attempts start smallest task first; {len(qpos)} tasks)\n")
    print(f"Complete queue prefix: the first {m_full} tasks in queue order have all finished.\n")
    print("| queue positions | finished / tasks | replica | bench, same tasks | diff [95% CI] | complete? |")
    print("|---|---|---|---|---|---|")
    out["queue"] = {"complete_prefix": m_full, "bins": [], "prefixes": []}
    rows = [(f"prefix 0-{m_full - 1}", 0, m_full)] if m_full else []
    rows += [(f"prefix 0-{min(m, len(qpos)) - 1}", 0, min(m, len(qpos))) for m in prefix_m if m != m_full]
    rows += [(f"{lo}-{min(lo + 40, len(qpos)) - 1}", lo, min(lo + 40, len(qpos))) for lo in range(0, len(qpos), 40)]
    for name, lo, hi in rows:
        sel = [by_pos[p] for p in range(lo, hi) if p in by_pos]
        s = summarise(sel, reps)
        fin = sum(1 for p in range(lo, hi) if at_pos[p] in rep)
        rec = dict(s or {}, name=name, lo=lo, hi=hi, finished=fin)
        out["queue"]["prefixes" if name.startswith("prefix") else "bins"].append(rec)
        if not s:
            print(f"| {name} | {fin} / {hi - lo} | - | - | - | {'yes' if fin == hi - lo else 'no'} |")
            continue
        ci = "" if s["ci"][0] is None else f" [{s['ci'][0]:+.1f}, {s['ci'][1]:+.1f}]"
        print(f"| {name} | {fin} / {hi - lo} | {s['replica']:.1f} | {s['bench']:.1f} | {s['diff']:+.1f}{ci} | "
              f"{'yes' if fin == hi - lo else 'no'} |")

    # time profile: by hour of finish, and cumulative cuts
    print("\n### Time profile (ARC task score; hour = t_end_h of the replica's first attempt)\n")
    print("| finished in hour | n | replica pass@1 | bench, same tasks | diff | cumulative n | cum. replica | cum. bench | "
          "cum. diff [95% CI] | cum. diff vs length-censored bench |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    prof = []
    for h in range(int(math.ceil(t_max + 1e-9))):
        sel = [i for i in items if h <= (i["t_end_h"] or 0) < h + 1]
        cum = [i for i in items if (i["t_end_h"] or 0) < h + 1]
        s, c = summarise(sel, reps), summarise(cum, reps)
        rep_cut = {t: r for t, r in rep.items() if (r["t_end_h"] or 0) < h + 1}
        cc = summarise(paired_items(rep_cut, bench, min(h + 1, at_h), censored=True), reps)
        prof.append({"hour": h, "bin": s, "cum": c, "cum_censored": cc})
        if not s:
            continue
        ci = "" if c["ci"][0] is None else f" [{c['ci'][0]:+.1f}, {c['ci'][1]:+.1f}]"
        print(f"| {h}-{h + 1} | {s['n']} | {s['replica']:.1f} | {s['bench']:.1f} | {s['diff']:+.1f} | {c['n']} | "
              f"{c['replica']:.1f} | {c['bench']:.1f} | {c['diff']:+.1f}{ci} | "
              f"{'-' if not cc else format(cc['diff'], '+.1f')} |")
    out["profile"] = prof

    # stops / length on the paired tasks
    br = [x for i in items for x in i["bench"]]
    rr = [i["rep"] for i in items]
    print("\n### Attempt shape on the paired tasks\n")
    print("| | attempts | stops | mean turns | mean output tokens | train_pass | tool errors / attempt | "
          "auto-repairs applied / attempt | cut streams |")
    print("|---|---|---|---|---|---|---|---|---|")
    out["shape"] = {}
    for name, rows in (("replica first attempts", rr), ("benchmark attempts (same tasks)", br)):
        sh = {"n": len(rows), "stops": dict(collections.Counter(r["stop"] for r in rows)),
              "turns": statistics.mean(r["turns"] for r in rows), "tokens": statistics.mean(r["tokens"] for r in rows),
              "train_pass": sum(r["train_pass"] is True for r in rows) / len(rows),
              "tool_errors": statistics.mean(r["tool_errors"] for r in rows),
              "auto_repairs": statistics.mean(r["auto_repairs"] for r in rows), "cut": sum(bool(r["cuts"]) for r in rows)}
        out["shape"][name] = sh
        print(f"| {name} | {len(rows)} | {stop_counts(rows)} | {sh['turns']:.1f} | {sh['tokens']:,.0f} | "
              f"{sh['train_pass']:.0%} | {sh['tool_errors']:.2f} | {sh['auto_repairs']:.2f} | {sh['cut']} |")
    ratio = [i["rep"]["tokens"] / statistics.mean(x["tokens"] for x in i["bench"])
             for i in items if statistics.mean(x["tokens"] for x in i["bench"]) > 0]
    if ratio:
        gm = math.exp(statistics.mean(math.log(max(x, 1e-6)) for x in ratio))
        print(f"\nOutput tokens, replica / benchmark mean on the same task: geometric mean {gm:.2f} (n {len(ratio)}).")

    # losses: the replica's attempt is wrong where the benchmark is right more often
    losses = sorted([i for i in items if i["r"] < i["b"] - 1e-9], key=lambda i: i["r"] - i["b"])
    print(f"\n### Losses: replica below the benchmark on the task ({len(losses)} of {len(items)})\n")
    if losses:
        lr = [i["rep"] for i in losses]
        print(f"- Replica stops: {stop_counts(lr)}; mean turns {statistics.mean(r['turns'] for r in lr):.1f}, "
              f"mean output tokens {statistics.mean(r['tokens'] for r in lr):,.0f}; train_pass "
              f"{sum(r['train_pass'] is True for r in lr)}/{len(lr)}; deadline / loop cuts "
              f"{sum(bool(r['cuts']) for r in lr)}; tool errors / attempt {statistics.mean(r['tool_errors'] for r in lr):.2f}.")
        lb = [x for i in losses for x in i["bench"]]
        print(f"- Benchmark attempts on those tasks: stops {stop_counts(lb)}; mean output tokens "
              f"{statistics.mean(r['tokens'] for r in lb):,.0f}; tool errors / attempt "
              f"{statistics.mean(r['tool_errors'] for r in lb):.2f}; right {sum(score(x['right'])[0] for x in lb):.1f} of "
              f"{len(lb)} attempts.")
        wins = [i for i in items if i["r"] > i["b"] + 1e-9]
        print(f"- For scale, wins ({len(wins)}): replica right where the benchmark mean is below 1; summed loss "
              f"{sum(i['b'] - i['r'] for i in losses):.2f} vs summed win {sum(i['r'] - i['b'] for i in wins):.2f} "
              f"tasks.")
        print("\n| workload task | orig | kind | t_end_h | replica: score / stop / turns / tokens / train_pass | "
              "bench attempts: score / stop / tokens |")
        print("|---|---|---|---|---|---|")
        for i in losses[:n_losses]:
            r = i["rep"]
            bs = "; ".join(f"{score(x['right'])[0]:.2g} {x['stop']} {x['tokens'] / 1e3:.1f}k ({x['src']})"
                           for x in i["bench"])
            print(f"| {i['task']} | {i['orig']} | {i['kind']} | {r['t_end_h']:.2f} | {i['r']:.2g} / {r['stop']} / "
                  f"{r['turns']} / {r['tokens'] / 1e3:.1f}k{'+' if r['tok_lb'] else ''} / {r['train_pass']} | {bs} |")
    out["losses"] = [{"task": i["task"], "orig": i["orig"], "r": i["r"], "b": i["b"], "stop": i["rep"]["stop"],
                      "turns": i["rep"]["turns"], "tokens": i["rep"]["tokens"]} for i in losses]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("replicas", nargs="+", help="[LABEL=]run dir or attempts.jsonl")
    ap.add_argument("--bench", action="append", required=True, help="[LABEL=]benchmark attempts / results file")
    ap.add_argument("--idmap", required=True)
    ap.add_argument("--challenges", help="workload challenges (default: next to the idmap)")
    ap.add_argument("--solutions", help="workload solutions (default: next to the idmap)")
    ap.add_argument("--cut", type=float, help="only replica records with t_end_h <= CUT")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--losses", type=int, default=40, help="rows in the loss table")
    ap.add_argument("--at-n", default="", help="extra N values for the progress profile, comma-separated")
    ap.add_argument("--prefix-m", default="", help="extra queue-prefix sizes M (first M tasks in queue order)")
    ap.add_argument("--json", help="write the numbers here")
    a = ap.parse_args()
    idp = Path(a.idmap)
    idmap = json.load(open(idp))
    ch = json.load(open(a.challenges or idp.with_name(idp.name.replace("_idmap.json", "_challenges.json"))))
    sol = json.load(open(a.solutions or idp.with_name(idp.name.replace("_idmap.json", "_solutions.json"))))
    ref_ch, ref_sol = json.load(open(REF_CH)), json.load(open(REF_SOL))
    bad = check_workload(ch, idmap, ref_ch)
    print(f"workload {len(ch)} tasks / {sum(len(v['test']) for v in ch.values())} pairs; transform check: "
          f"{'OK' if not bad else f'{len(bad)} copies do not invert: {bad[:5]}'}")
    bench = collections.defaultdict(list)
    for arg in a.bench:
        lab, path = labelled(arg)
        b, excl = load_bench(lab, path, ref_ch, ref_sol)
        flat = [x for v in b.values() for x in v]
        p1 = statistics.mean(statistics.mean(score(x["right"])[0] for x in v) for v in b.values())
        print(f"benchmark {lab}: `{path}`: {len(b)} tasks, {len(flat)} attempts, pass@1 {100 * p1:.1f} on its tasks"
              + (f"; left out (prompt differs from the Kaggle eval json): {', '.join(excl)}" if excl else ""))
        for t, v in b.items():
            bench[t].extend(v)
    print(f"pooled benchmark: {len(bench)} tasks, {sum(len(v) for v in bench.values())} attempts")
    res = []
    for arg in a.replicas:
        lab, path = labelled(arg)
        res.append(report(lab, path, ch, sol, idmap, bench, a.cut, a.boot, a.losses,
                          [int(x) for x in a.at_n.split(",") if x], [int(x) for x in a.prefix_m.split(",") if x]))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
