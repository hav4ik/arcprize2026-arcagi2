"""Build the ARC-AGI-2 experiment report (HTML + Markdown) from the result files.

Every score, token count and recorded cost is recomputed from `evals/results/*.jsonl` (latest record per
task/pair/attempt, as the resume logic writes them) against the GitHub ground truth
(`data/arc-agi-2-github/data/evaluation`). Kaggle numbers come from `kaggle/runs/`. Numbers that only exist in a
README (rental cost, ARC Prize's published scores, manual trace reviews) are footnoted as such.

Outputs (all under reports/):
- report_data.json                         every computed number + the report text (single source)
- arc_agi2_experiments_report.html          self-contained page (no external requests)
- arc_agi2_experiments_report.artifact.html same page without the doctype/html/head/body wrapper
- arc_agi2_experiments_report.md            GitHub-flavoured Markdown rendering of the same data

    .venv/bin/python reports/build_report.py

Read-only on every input file. Safe to run while evals are appending to their result files.
"""
import html
import math
import json
import random
import re
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evals/results"
OUT = ROOT / "reports"
REPORT_DATE = "2026-10-12"

TASKS = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "data/arc-agi-2-github/data/evaluation").glob("*.json"))}
FULL = [(t, p) for t in sorted(TASKS) for p in range(len(TASKS[t]["test"]))]
PILOT_TASKS = sorted(TASKS)[:15]
PILOT = [(t, p) for t, p in FULL if t in PILOT_TASKS]


def read_pairs(path):
    out = []
    for tok in Path(path).read_text().replace("\n", ",").split(","):
        if tok.strip():
            t, p = tok.strip().split(":")
            out.append((t, int(p)))
    return out


DEV = read_pairs(ROOT / "experiments/E009_tool_token_efficiency/dev_pairs.txt")
NONDEV = read_pairs(ROOT / "experiments/E012_qwen_tools/nondev_pairs.txt")
SETS = {"full": FULL, "pilot": PILOT, "dev": DEV}
FLASH_PRICE, PRO_PRICE = 0.60, 1.98  # $/M output tokens, off-peak list price
BOOT = 10000

# ----------------------------------------------------------------------------------------------- loading


def read_jsonl(name):
    recs, bad = [], 0
    path = RES / name
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:  # a line being written right now
                bad += 1
    return recs, bad


AUDIT = {"files": {}, "flag_mismatch": 0, "records_checked": 0}


_CACHE = {}


def load(name):
    """Latest record per (task, pair, attempt). NVIDIA-subset files use (task, 0, sample). Cached per file."""
    if name in _CACHE:
        return _CACHE[name]
    recs, bad = read_jsonl(name)
    lat = {}
    for r in recs:
        if "pair" in r:
            k = (r["task_id"], int(r["pair"]), int(r.get("attempt", 0)))
        else:
            k = (r["task_id"], 0, int(r.get("sample", 0)))
        lat[k] = r
    nv = bool(recs) and "pair" not in recs[0]
    for (t, p, a), r in lat.items():
        r["_err"] = "error" in r
        if r["_err"]:
            r["_ok"] = False
            continue
        if nv:
            r["_ok"] = bool(r.get("correct"))
        else:
            ok = r.get("pred") is not None and r.get("pred") == TASKS[t]["test"][p]["output"]
            AUDIT["records_checked"] += 1
            if ok != bool(r.get("correct")):
                AUDIT["flag_mismatch"] += 1
            r["_ok"] = ok
        r["_tok"] = tokens(r)
    AUDIT["files"][name] = {"lines": len(recs), "latest": len(lat), "bad_lines": bad,
                            "errors": sum(r["_err"] for r in lat.values()),
                            "mtime": datetime.fromtimestamp((RES / name).stat().st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    _CACHE[name] = (lat, recs)
    return lat, recs


def tokens(r):
    u = r.get("usage") or {}
    if "completion" in u:
        return u["completion"]
    if "completion_tokens" in r:
        return r["completion_tokens"]
    return u.get("completion_tokens", 0)


def running_tags():
    """Tags of eval processes alive right now (read from /proc; nothing is signalled)."""
    tags = set()
    for p in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = p.read_bytes().split(b"\0")
        except OSError:
            continue
        args = [a.decode(errors="replace") for a in args]
        if any(a.endswith(("run_qwen_tools.py", "run_deepseek_tools.py", "run_qwen_generate.py", "run_openrouter_arcprize.py")) for a in args):
            if "--tag" in args:
                tags.add(args[args.index("--tag") + 1])
    return tags


RUNNING = running_tags()

# ----------------------------------------------------------------------------------------------- scoring


def tw(pair_vals):
    """ARC Prize benchmarking (arc-agi-benchmarking scoring.py) task-weighted mean of {(t, p): value}: per task the fraction
    of its test pairs solved, then the mean over tasks."""
    tasks = defaultdict(list)
    for (t, p), v in pair_vals.items():
        tasks[t].append(v)
    return st.mean(st.mean(v) for v in tasks.values()) if tasks else None


def pw(pair_vals):
    """Kaggle competition metric, pair-weighted: the average over all task test outputs of {(t, p): value}."""
    return st.mean(pair_vals.values()) if pair_vals else None


METRICS = (("k_", pw), ("", tw))  # key prefix "k_" = Kaggle (pair-weighted, primary); no prefix = ARC Prize task-weighted


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def bootstrap(diff, seed=0):
    rng = random.Random(seed)
    vals = list(diff.values())
    means = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(BOOT))
    return means[int(0.025 * BOOT)], means[int(0.975 * BOOT)]


def bootstrap_cluster(task_sums, seed=0):
    """Pair-weighted mean difference, resampling tasks as clusters: task_sums = {task: (sum of pair diffs, n pairs)}."""
    rng = random.Random(seed)
    vals = list(task_sums.values())
    means = []
    for _ in range(BOOT):
        smp = rng.choices(vals, k=len(vals))
        means.append(sum(a for a, _ in smp) / sum(n for _, n in smp))
    means.sort()
    return means[int(0.025 * BOOT)], means[int(0.975 * BOOT)]


def merged(files):
    lat, allrecs = {}, []
    for f in files:
        l, recs = load(f)
        allrecs += recs
        for k, r in l.items():
            lat.setdefault(k, r)  # first file wins (E012: --skip-done-in)
    return lat, allrecs


def evaluate(spec):
    lat, allrecs = merged(spec["files"])
    if spec.get("attempts") is not None:
        lat = {k: r for k, r in lat.items() if k[2] in spec["attempts"]}
    setname = spec["set"]
    if setname in SETS:
        universe = SETS[setname]
        uset = set(universe)
        lat = {k: r for k, r in lat.items() if (k[0], k[1]) in uset}
    else:
        universe = sorted({(t, p) for t, p, _ in lat})
    atts = sorted({a for _, _, a in lat})
    per_att = {}
    for a in atts:
        present = [k for k in lat if k[2] == a]
        per_att[a] = {"present": len(present), "errors": sum(lat[k]["_err"] for k in present)}
    n_univ = len(universe)
    complete = bool(atts) and all(v["present"] == n_univ and v["errors"] <= max(1, int(0.01 * n_univ)) for v in per_att.values())
    exp_attempts = spec.get("expected_attempts")
    if exp_attempts and len(atts) < exp_attempts:
        complete = False
    ok = lambda k: bool(lat.get(k, {}).get("_ok"))
    res = {"id": spec["id"], "n_universe": n_univ, "attempt_indices": atts, "per_attempt": per_att, "complete": complete,
           "attempts_done": sum(not r["_err"] for r in lat.values()), "attempts_present": len(lat), "attempts_error": sum(r["_err"] for r in lat.values())}
    if complete:
        for pre, agg in METRICS:
            each = [agg({(t, p): ok((t, p, a)) for t, p in universe}) for a in atts]
            res[pre + "pass1_each"] = each
            res[pre + "pass1"] = st.mean(each)
            res[pre + "pass1_sd"] = st.stdev(each) if len(each) >= 2 else None
            if len(atts) >= 2:
                combos = list(combinations(atts, 2))
                res[pre + "pass2"] = st.mean(agg({(t, p): ok((t, p, a)) or ok((t, p, b)) for t, p in universe}) for a, b in combos)
                res[pre + "pass2_a01"] = agg({(t, p): ok((t, p, atts[0])) or ok((t, p, atts[1])) for t, p in universe})
                res["pass2_combos"] = len(combos)
        res["covered_pairs"] = n_univ
        res["covered_tasks"] = len({t for t, _ in universe})
    else:
        by_pair = defaultdict(list)
        for (t, p, a), r in lat.items():
            if not r["_err"]:
                by_pair[(t, p)].append(r["_ok"])
        res["covered_pairs"] = len(by_pair)
        res["covered_tasks"] = len({t for t, _ in by_pair})
        both = {k for k in by_pair if len(by_pair[k]) >= 2}
        for pre, agg in METRICS:
            res[pre + "pass1"] = agg({k: st.mean(v) for k, v in by_pair.items()})
            if len(atts) >= 2 and both:
                res[pre + "pass2"] = agg({(t, p): ok((t, p, atts[0])) or ok((t, p, atts[1])) for t, p in both
                                          if (t, p, atts[0]) in lat and (t, p, atts[1]) in lat})
                res["pass2_pairs"] = len(both)
    good = [r for r in lat.values() if not r["_err"]]
    toks = [r["_tok"] for r in good]
    ctoks = [r["_tok"] for r in good if r["_ok"]]
    res.update(tok_mean=st.mean(toks) if toks else None, tok_median=st.median(toks) if toks else None,
               tok_p90=pct(toks, 0.9), ctok_median=st.median(ctoks) if ctoks else None, n_correct=len(ctoks),
               unparsed=sum(r.get("pred") is None for r in good), n_good=len(good))
    if any("answer_source" in r for r in good):
        res["sources"] = dict(Counter(r.get("answer_source") or "none" for r in good).most_common())
        res["sources_correct"] = dict(Counter(r.get("answer_source") or "none" for r in good if r["_ok"]))
    hv = Counter((r.get("harness_version") or (r.get("config") or {}).get("harness_version") or "-") for r in good)
    res["harness_versions"] = dict(hv)
    n_samples = len(atts) if complete else None
    res["samples"] = n_samples
    # cost
    c = spec.get("cost", ("none",))
    if c[0] == "est":
        price = c[1]
        res["cost_total"] = sum(toks) * price / 1e6
        res["cost_kind"] = "est"
    elif c[0] == "rec":
        field = c[1]
        res["cost_total"] = sum(r.get(field) or 0 for r in good)
        res["cost_file_total"] = sum(r.get(field) or 0 for r in allrecs)
        res["cost_kind"] = "rec"
        if field == "est_cost_usd" and any("est_cost_peak_usd" in r for r in good):
            # --auto-peak-price runs: est_cost_usd already bills peak turns at 2x; est_cost_peak_usd is that (doubled) peak part
            by_att = defaultdict(lambda: [0.0, 0.0])
            for (_, _, a), r in lat.items():
                if not r["_err"]:
                    by_att[a][0] += r.get(field) or 0
                    by_att[a][1] += r.get("est_cost_peak_usd") or 0
            res["cost_by_attempt"] = {a: {"rec": v[0], "peak_part": v[1], "offpeak_equiv": v[0] - v[1] / 2} for a, v in sorted(by_att.items())}
            res["cost_offpeak_equiv"] = sum(v["offpeak_equiv"] for v in res["cost_by_attempt"].values())
    elif c[0] == "text":
        res["cost_kind"] = "text"
        res["cost_text"] = c[1]
    else:
        res["cost_kind"] = "none"
    if "cost_total" in res:
        res["cost_per_sample"] = res["cost_total"] / n_samples if n_samples else None
        if "cost_offpeak_equiv" in res and n_samples:
            res["cost_per_sample_offpeak"] = res["cost_offpeak_equiv"] / n_samples
    res["_lat"] = lat
    res["_universe"] = universe
    return res


def pair_rates(lat, keys, attempts=None, cap=None):
    out = {}
    for t, p in keys:
        vals = []
        for (tt, pp, a), r in lat.items():
            if tt == t and pp == p and not r["_err"] and (attempts is None or a in attempts):
                vals.append(r["_ok"] and (cap is None or r["_tok"] <= cap))
        if vals:
            out[(t, p)] = st.mean(vals)
    return out


def index_lat(lat):
    idx = defaultdict(dict)
    for (t, p, a), r in lat.items():
        idx[(t, p)][a] = r
    return idx


def paired(run, ref, match_attempts=False, cap=None):
    """Paired pass@1 difference run - ref over the pairs the run covers.

    Primary (diff/lo/hi/run/ref_score): Kaggle pair-weighted, 95% CI from a cluster bootstrap that resamples tasks (all pairs of a
    task move together). Secondary (tw_*): ARC Prize task-weighted, bootstrap over tasks."""
    ri, fi = index_lat(run["_lat"]), index_lat(ref["_lat"])
    run_r, ref_r = {}, {}
    for k, atts in ri.items():
        done = {a: r for a, r in atts.items() if not r["_err"] or run["complete"]}
        if not done or k not in fi:
            continue
        ref_atts = {a: r for a, r in fi[k].items() if not r["_err"]}
        if match_attempts:
            ref_atts = {a: r for a, r in ref_atts.items() if a in done}
        if not ref_atts:
            continue
        run_r[k] = st.mean(r["_ok"] for r in done.values())
        ref_r[k] = st.mean(r["_ok"] and (cap is None or r["_tok"] <= cap) for r in ref_atts.values())
    if not run_r:
        return None
    tasks = defaultdict(list)
    for k in run_r:
        tasks[k[0]].append((run_r[k], ref_r[k]))
    tdiff = {t: st.mean(a for a, _ in v) - st.mean(b for _, b in v) for t, v in tasks.items()}
    tlo, thi = bootstrap(tdiff)
    sums = {t: (sum(a - b for a, b in v), len(v)) for t, v in tasks.items()}
    lo, hi = bootstrap_cluster(sums)
    return {"ref": ref["id"], "cap": cap, "pairs": len(run_r), "tasks": len(tasks), "run": pw(run_r), "ref_score": pw(ref_r),
            "diff": pw(run_r) - pw(ref_r), "lo": lo, "hi": hi,
            "tw_run": tw(run_r), "tw_ref_score": tw(ref_r), "tw_diff": st.mean(tdiff.values()), "tw_lo": tlo, "tw_hi": thi}


def ranked_answers(recs, k=2):
    """Same rule as evals/train_check.py ranked_answers (the harness's Kaggle answer order): attempts ranked by
    train_pass (True first), then attempt order; the top-k distinct non-empty predictions."""
    order = sorted((r for r in recs if r.get("pred") is not None), key=lambda r: (r.get("train_pass") is not True, r["_att"]))
    out = []
    for r in order:
        if r["pred"] not in [o["pred"] for o in out]:
            out.append(r)
        if len(out) == k:
            break
    return out


def policy3(spec):
    """pass@2 with the conditional 3rd attempt (E017 --extra-attempt-if-no-train-pass): per pair, the harness's top-2 ranked
    answers over attempts 0, 1 and (when it fired) 2, against the plain a0+a1 pass@2 on the same file."""
    lat, _ = merged(spec["files"])
    idx = index_lat(lat)
    pol, base = {}, {}
    for t, p in FULL:
        recs = [dict(r, _att=a) for a, r in idx.get((t, p), {}).items() if not r["_err"]]
        pol[(t, p)] = any(r["_ok"] for r in ranked_answers(recs))
        base[(t, p)] = any(r["_ok"] for r in recs if r["_att"] in (0, 1))
    extra = [r for (t, p, a), r in lat.items() if a == 2 and not r["_err"]]
    tasks = defaultdict(list)
    for k in pol:
        tasks[k[0]].append(pol[k] - base[k])
    lo, hi = bootstrap_cluster({t: (sum(v), len(v)) for t, v in tasks.items()})
    base_tok = sum(r["_tok"] for (t, p, a), r in lat.items() if a in (0, 1) and not r["_err"])
    ex_tok = sum(r["_tok"] for r in extra)
    return {"fired": len(extra), "fired_pairs": len({(r["task_id"], r["pair"]) for r in extra}),
            "extra_correct": sum(r["_ok"] for r in extra), "extra_train_pass": sum(r.get("train_pass") is True for r in extra),
            "extra_train_pass_correct": sum(r.get("train_pass") is True and r["_ok"] for r in extra),
            "extra_no_answer": sum(r.get("pred") is None for r in extra),
            "extra_tok_mean": st.mean(r["_tok"] for r in extra) if extra else None, "extra_tok_total": ex_tok,
            "extra_cost": sum(r.get("est_cost_usd") or 0 for r in extra),
            "extra_cost_peak_part": sum(r.get("est_cost_peak_usd") or 0 for r in extra),
            "tok_per_pair_base": base_tok / len(FULL), "tok_per_pair_policy": (base_tok + ex_tok) / len(FULL),
            "rescued": sum(pol[k] and not base[k] for k in pol), "lost": sum(base[k] and not pol[k] for k in pol),
            "k_pass2_base": pw(base), "k_pass2_policy": pw(pol), "pass2_base": tw(base), "pass2_policy": tw(pol),
            "diff": pw(pol) - pw(base), "lo": lo, "hi": hi}


def train_check_stats(lat):
    """Live train-check verifier quality over the given attempts: P(correct | train_pass), P(correct | train_fail).
    Attempts without an answer (train_pass None) are left out of both conditionals, as in the E017 README."""
    good = [r for r in lat.values() if not r["_err"] and "train_pass" in r]
    ps = [r for r in good if r.get("train_pass") is True]
    fs = [r for r in good if r.get("train_pass") is False]
    return {"n": len(good), "pass": len(ps), "pass_correct": sum(r["_ok"] for r in ps), "fail": len(fs),
            "fail_correct": sum(r["_ok"] for r in fs), "no_answer": sum(r.get("pred") is None for r in good),
            "p_pass": sum(r["_ok"] for r in ps) / len(ps) if ps else None, "p_fail": sum(r["_ok"] for r in fs) / len(fs) if fs else None,
            "recall": sum(r["_ok"] for r in ps) / max(1, sum(r["_ok"] for r in good))}


E016_DIR = ROOT / "experiments/E016_cutoff_continuation"
E016_ARMS = [("toolnote", "E016 toolnote (thinking off)", "drop the truncated reasoning-only turn, append a \"time is up, call submit_answer now\" "
              "note to the last tool result; thinking off, 2,048-token cap, one extra turn if it runs code instead"),
             ("usermsg", "E016 usermsg (thinking off)", "keep the trace as sent and append the same text as a user message; thinking off, 2,048-token cap"),
             ("toolnote_think", "E016 toolnote (thinking on)", "toolnote messages with thinking on (low effort), 8,192-token cap")]


def e016_rows(s2060):
    """Table 2 rows for E016's continuation arms, recomputed from results/continuations.jsonl; scored with the official
    exact-match flag stored per row, and substituted into E010 s20x60 to show the effect on its full-set scores."""
    path = E016_DIR / "results/continuations.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    best = {}
    for r in rows:
        if "error" not in r:
            best[(r["task_id"], int(r["pair"]), int(r["attempt"]), r["variant"])] = r
    base = {k[:3]: r for k, r in best.items() if k[3] == "baseline"}
    out, summ = [], {}
    lat = s2060["_lat"]
    for var, rid, desc in E016_ARMS:
        rs = [r for k, r in best.items() if k[3] == var]
        n = len(rs)
        ans = sum(r.get("pred") is not None for r in rs)
        corr = [r for r in rs if r.get("correct")]
        src = Counter(r.get("source") or "none" for r in rs)
        toks = [r["usage"]["completion"] for r in rs]
        cost = sum(r.get("cost_usd") or 0 for r in rs)
        inp = sum(r["usage"]["prompt"] for r in rs)
        hit = sum(r["usage"].get("cache_hit", 0) for r in rs)
        # verify the official flag on the stored prediction
        mism = sum(bool(r.get("correct")) != (r.get("pred") is not None and r["pred"] == TASKS[r["task_id"]]["test"][int(r["pair"])]["output"])
                   for r in rs)
        # substitute rescued answers into s20x60 (attempts 0 and 1)
        sub = {k: r["_ok"] for k, r in lat.items() if not r["_err"]}
        for r in rs:
            sub[(r["task_id"], int(r["pair"]), int(r["attempt"]))] = bool(r.get("correct")) or sub.get((r["task_id"], int(r["pair"]), int(r["attempt"])), False)
        k1 = st.mean(pw({(t, p): sub.get((t, p, a), False) for t, p in FULL}) for a in (0, 1))
        k2 = pw({(t, p): sub.get((t, p, 0), False) or sub.get((t, p, 1), False) for t, p in FULL})
        per_pair = defaultdict(list)
        for r in rs:
            per_pair[(r["task_id"], int(r["pair"]))].append(bool(r.get("correct")))
        tw1 = tw({k: st.mean(v) for k, v in per_pair.items()})
        summ[var] = {"n": n, "answered": ans, "correct": len(corr), "cost": cost, "out_tok": sum(toks), "in_tok": inp, "cache_hit": hit,
                     "sources": dict(src), "s2060_k_pass1": k1, "s2060_k_pass2": k2, "flag_mismatch": mism,
                     "rescued": [f"{r['task_id']}.{r['pair']}#{r['attempt']}" for r in corr]}
        status = (f"done. {desc}. Answered {ans}/{n} (" + ", ".join(f"{k} {v}" for k, v in src.most_common()) + f"), correct {len(corr)}"
                  + (f" ({', '.join(summ[var]['rescued'])})" if corr else "") + f"; rec. ${cost:.3f} for {n} calls (off-peak); input "
                  f"{inp / 1000:.0f}k tokens ({hit / 1000:.0f}k cache hit), output {sum(toks) / 1000:.1f}k")
        if var == "toolnote":
            nb = len(base)
            status += (f". Zero-cost probe on the replayed sessions: `answer` set in {sum(bool(b.get('answer_present')) for b in base.values())}/{nb}, "
                       f"a function fitting every train pair in {sum(bool(b.get('functions_fitting_all_train')) for b in base.values())}, "
                       f"correct {sum(bool(b.get('any_function_correct')) for b in base.values())}. With thinking off DeepSeek drops earlier "
                       "reasoning_content from the prompt, so the turn sees only code and tool outputs [^e016]")
        out.append({
            "id": rid, "fam": DS, "model": FLASH, "harness": "tools-grid (continuation)", "set": f"s20x60 cut-off attempts ({n})",
            "coverage": f"1 × {n}/{n} attempts", "k_pass1": f"{f1(len(corr) / n)} ({len(corr)}/{n})", "k_pass2": "—",
            "pass1": f1(tw1), "pass2": "—", "tok": f"{fk(st.mean(toks))} / {fk(st.median(toks))}",
            "paired": (f"E010 s20x60 with these answers substituted: Kaggle pass@1 {f1(s2060['k_pass1'])} → {f1(k1)}, "
                       f"pass@2 {f1(s2060['k_pass2'])} → {f1(k2)}"),
            "status": status,
            "raw": {"k_pass1": len(corr) / n, "k_pass2": None, "pass1": tw1, "pass2": None, "tok_mean": st.mean(toks)}})
    summ["total_cost_all_rows"] = sum(r.get("cost_usd") or 0 for r in rows)
    return out, summ


# ----------------------------------------------------------------------------------------------- run specs
DS = "DeepSeek"
QW = "Qwen"
NM = "Nemotron"
FLASH = "DeepSeek V4.1-Flash (DeepSeek API)"
PRO = "DeepSeek V4-Pro (DeepSeek API)"
QWEN_NV = "Qwen3.8-27B NVFP4, self-hosted SGLang (vast.ai 4×RTX PRO 6000)"
QWEN36 = "Qwen3.6-35B-A3B NVFP4, self-hosted SGLang (vast.ai 4×RTX PRO 6000 Server)"
TOOLS_V3 = "120k total / 64k per turn / 24k forced final, harness persistent-v3"
SUB = "submit_answer tool, session-answer fallback (no forced final), lenient parse, 64k per turn"
GEN = "no tools (raw gen API)"
E017_H = "low effort, sub80s40 budgets (80k hard / 40k soft, notes on every output, `submit_answer`), harness persistent-v5"


def tool_file(prompt, effort, tag, mode="grid"):
    return f"tools_persistent_{mode}_{prompt}_deepseek-flash_{effort}_{tag}.jsonl" if prompt else f"tools_persistent_{mode}_deepseek-flash_{effort}_{tag}.jsonl"


REC_EST = ("rec", "est_cost_usd")
REC_OR = ("rec", "cost_usd")

T1_SPECS = [
    dict(id="E001d", fam=DS, model=FLASH, harness="no tools", set="full", files=["arcprize_deepseek-flash_max.jsonl"],
         desc="ARC Prize prompt verbatim, max effort, max_tokens 200k, one call per attempt", cost=("est", FLASH_PRICE),
         note="17 connection-error attempts were retried on 2026-10-03, so all 501 records are complete; ARC Prize parser."),
    dict(id="E004-GL", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2", "low", "full_a0")],
         desc=f"lean2 prompt, low effort, {TOOLS_V3}", cost=REC_EST, ref="E001d",
         note="Candidate from E004; the baseline for E009–E011."),
    dict(id="E004 grid-high", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2", "high", "full_a0")],
         desc=f"lean2 prompt, high effort, {TOOLS_V3}", cost=REC_EST, ref="E004-GL"),
    dict(id="E004 grid-max", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2", "max", "full_a0")],
         desc=f"lean2 prompt, max effort, {TOOLS_V3}", cost=REC_EST, ref="E004-GL"),
    dict(id="E004 cap100", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2", "low", "cap100")],
         desc="lean2 prompt, low effort, 100k total / 64k per turn / 24k forced final", cost=REC_EST, ref="E004-GL"),
    dict(id="E004 code-high", fam=DS, model=FLASH, harness="tools-code", set="full", files=[tool_file("lean2", "high", "full_a0", "code")],
         desc=f"submit_solution checked on train pairs, lean2, high effort, {TOOLS_V3}", cost=REC_EST, ref="E004-GL"),
    dict(id="E004 code-max", fam=DS, model=FLASH, harness="tools-code", set="full", files=[tool_file("lean2", "max", "full_a0", "code")],
         desc=f"submit_solution checked on train pairs, lean2, max effort, {TOOLS_V3}", cost=REC_EST, ref="E004-GL"),
    dict(id="E004 plain-low", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file(None, "low", "full_a0")],
         desc=f"plain ARC Prize prompt (no lean2 suffix), low effort, {TOOLS_V3}", cost=REC_EST, ref="E004-GL",
         note="Prompt ablation."),
    dict(id="E004 plain-max", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file(None, "max", "full_a0")],
         desc=f"plain ARC Prize prompt, max effort, {TOOLS_V3}", cost=REC_EST, ref="E004-GL", note="Prompt ablation."),
    dict(id="E009 sub80s40", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2sub", "low", "e009_full_sub80s40")],
         desc=f"lean2sub prompt, low effort, 80k hard cap, budget note on every tool output, 40k soft target; {SUB}",
         cost=REC_EST, ref="E004-GL"),
    dict(id="E009/E011 sub96n", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2sub", "low", "e009_full_sub96n")],
         desc=f"lean2sub prompt, low effort, 96k hard cap, budget notes from 50% used; {SUB}", cost=REC_EST, ref="E004-GL",
         note="Sample 0 from E009, sample 1 from E011 (same file). E011's pick for best accuracy."),
    dict(id="E009 sub96s40", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2sub", "low", "e009_full_sub96s40")],
         desc=f"lean2sub prompt, low effort, 96k hard cap, notes on every output, 40k soft target; {SUB}", cost=REC_EST, ref="E004-GL"),
    dict(id="E009 sub64a", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2sub", "low", "e009_full_sub64a")],
         attempts=[0], desc=f"lean2sub prompt, low effort, 64k hard cap, notes on every output, warn at 75%; {SUB}", cost=REC_EST,
         ref="E004-GL", note="Sample 0 only; the stopped 2nd sample (61 pairs) is in Table 2."),
    dict(id="E010 s20x60", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2sub", "low", "e010_full_s20x60")],
         desc=f"lean2sub prompt, low effort, 60k hard cap, 20k soft target with strong wording; {SUB}", cost=REC_EST, ref="E004-GL",
         note="E011's pick for ≲35k tokens/attempt."),
    dict(id="E010 s25x80", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2sub", "low", "e010_full_s25x80")],
         desc=f"lean2sub prompt, low effort, 80k hard cap, 25k soft target with strong wording; {SUB}", cost=REC_EST, ref="E004-GL"),
    dict(id="E011 sub80s40v", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2sub", "low", "e011_full_sub80s40v")],
         desc="sub80s40 + --verify-submit (train check before accepting submit_answer), lean2sub prompt", cost=REC_EST, ref="E004-GL"),
    dict(id="E011 subv80s40", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2subv", "low", "e011_full_subv80s40")],
         desc="sub80s40 + --verify-submit + lean2subv prompt (submit exactly solve(test_input))", cost=REC_EST, ref="E004-GL"),
    dict(id="E011 sub80s40h", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2sub", "high", "e011_full_sub80s40h")],
         desc="sub80s40 at high effort", cost=REC_EST, ref="E004-GL"),
    dict(id="E017 V1 summary", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2sub", "low", "e017_full_v1sum")],
         desc=f"sub80s40 + `--task-summary` (P1: host-computed perception pack in the first message: sizes, colours, object tables, "
              f"in→out object matching, \"outside train range\" line for the test input); `--train-check`; {E017_H}",
         cost=REC_EST, ref="E004-GL", ref_extra=["E009 sub80s40"], e017=True),
    dict(id="E017 V2 lean2roles", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2roles", "low", "e017_full_v2roles")],
         desc=f"sub80s40 with the lean2roles prompt (P2: ROLES note before solve(), no \"noise\" labels without code evidence, "
              f"test-vs-train difference audit after train passes); `--train-check`; {E017_H}",
         cost=REC_EST, ref="E004-GL", ref_extra=["E009 sub80s40"], e017=True),
    dict(id="E017 V3 all", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2roles", "low", "e017_full_v3all")],
         attempts=[0, 1], expected_attempts=2,
         desc=f"P1+P2+P3: lean2roles + `--task-summary` + `--helpers` (objects, crop, d8, panels, `check(solve)` in the session); "
              f"`--train-check`; conditional 3rd attempt when neither attempt passes the train check; {E017_H}",
         cost=REC_EST, ref="E004-GL", ref_extra=["E009 sub80s40"], e017=True, policy3=True,
         note="pass@1 / pass@2 use attempts 0–1 only; the conditional 3rd attempts (attempt index 2) are reported separately."),
    dict(id="E008 Qwen xhigh-240k", fam=QW, model=QWEN_NV, harness=GEN, set="full", files=["qwen_nvfp4_gen_xhigh_240k_t1.jsonl"],
         desc="ARC Prize prompt, tier xhigh, temp 1.0 / top_p 0.95 / top_k 20 (official), 240k budget in 32k chunks, mid-thought EOS continued, 4k forced final",
         cost=("text", "≈$27 rental [^rent-xhigh]"),
         note="First 52 records are the pilot run (copied in)."),
    dict(id="E008 Qwen medium-48k", fam=QW, model=QWEN_NV, harness=GEN, set="full", files=["qwen_nvfp4_gen_medium_48k.jsonl"],
         desc="ARC Prize prompt, tier medium, temp 0.6 / top_p 0.95 / top_k 20, 48k budget in 16k chunks, EOS continuation, 4k forced final",
         cost=("text", "≈$10 rental [^rent-medium]"), ref="E008 Qwen xhigh-240k", ref_match=True),
    dict(id="E008 Qwen3.6-35B", fam=QW, model=QWEN36, harness=GEN, set="full", files=["nvidia-qwen3-6-35b-a3b-nvfp4_gen_none_240k_t1pp15.jsonl"],
         desc="ARC Prize prompt, no tier (thinking on), temp 1.0 / top_p 0.95 / top_k 20, presence penalty 1.5, 240k budget in 32k chunks",
         cost=("text", "≈$19 rental [^rent-36]"), ref="E008 Qwen xhigh-240k", ref_match=True),
    dict(id="E012 Qwen tools full", fam=QW, model=QWEN_NV, harness="tools-grid", set="full", expected_attempts=2,
         files=["qwen_tools_medium_48k_dev_s24.jsonl", "qwen_tools_medium_48k_full_v3.jsonl"],
         desc="E009 sub80s40 port: lean2sub prompt + submit_answer, tier medium, 48k hard cap, 24k soft target, notes on every output, temp 1.0 / top_p 0.95 / top_k 20, forced 256-token submit at the cap",
         cost=("text", "box time [^rent-e012]"), ref="E008 Qwen xhigh-240k", ref_match=True, ref_cap=48000,
         note="Merged: dev-pair attempts and 81 attempt-0 records from qwen_tools_medium_48k_dev_s24 (harness qwen-tools-2), the rest from qwen_tools_medium_48k_full_v3 (qwen-tools-3, adds a salvage answer source).",
         running_tags=["full_v3"]),
    dict(id="E008 Nemotron Lightning", fam=NM, model="Nemotron 3.5 Lightning bf16, OpenRouter (CoreWeave)", harness="no tools", set="full",
         files=["openrouter_nvidia-nemotron-3.5-lightning_high.jsonl"],
         desc="ARC Prize prompt, high effort, max_tokens 200k, one call", cost=REC_OR,
         note="1 sample; pass@2 skipped as not worth the cost. Every attempt ends with an admitted guess [^nemo]."),
]

# Table 2: smaller sets, probes, partial runs.
def ds_dev(tag, prompt, desc, exp="E009"):
    return dict(id=f"{exp} dev {tag}", fam=DS, model=FLASH, harness="tools-grid", set="dev", files=[tool_file(prompt, "low", f"{exp.lower()}_dev_{tag}")],
                desc=desc, cost=REC_EST, ref="E004-GL", setlabel="dev-56", expected=56)


T2_SPECS = [
    # E012 full set (only lands here while incomplete; moves to Table 1 when done)
    # pilot-26
    dict(id="E008 Qwen OpenRouter pilot", fam=QW, model="Qwen3.8-27B bf16, OpenRouter (DeepInfra)", harness="no tools", set="pilot",
         files=["openrouter_qwen-qwen3.8-27b_high_pilot.jsonl"], desc="ARC Prize prompt, high effort, 200k", cost=REC_OR,
         ref="E008 Qwen xhigh-240k", setlabel="pilot-26", status="done",
         note="14/26 attempts ended with empty content (mid-thought stop)."),
    dict(id="E008 Nemotron Super pilot", fam=NM, model="Nemotron 3 Super 120B-A12B fp8, OpenRouter (DekaLLM)", harness="no tools", set="pilot",
         files=["openrouter_nvidia-nemotron-3-super-120b-a12b_high.jsonl"], desc="ARC Prize prompt, high effort, temp 1.0 / top_p 0.95, 200k",
         cost=REC_OR, setlabel="pilot-26", status="provider errors",
         note="Inconclusive: provider-side stream failures on 15–30 min streams; the completed attempts are all wrong."),
    dict(id="E008 Qwen xhigh pilot", fam=QW, model=QWEN_NV, harness=GEN, set="pilot", files=["qwen_nvfp4_gen_xhigh_240k_t1_pilot.jsonl"],
         desc="same settings as E008 Qwen xhigh-240k", cost=("none",), setlabel="pilot-26", status="merged",
         note="These 52 records were copied into the full run."),
    dict(id="E012 pilot medium-48k s24", fam=QW, model=QWEN_NV, harness="tools-grid", set="pilot", files=["qwen_tools_medium_48k_pilot_s24.jsonl"],
         desc="tier medium, 48k hard / 24k soft, qwen-tools-2 (early)", cost=("none",), ref="E008 Qwen xhigh-240k", ref_match=True, ref_cap=48000,
         setlabel="pilot-26", status="stopped", expected=52, tags=["pilot_s24"]),
    dict(id="E012 pilot xhigh-48k s24", fam=QW, model=QWEN_NV, harness="tools-grid", set="pilot", files=["qwen_tools_xhigh_48k_pilot_s24.jsonl"],
         desc="tier xhigh, 48k hard / 24k soft", cost=("none",), ref="E008 Qwen xhigh-240k", ref_match=True, ref_cap=48000,
         setlabel="pilot-26", status="stopped", expected=52),
    dict(id="E012 pilot xhigh-48k s24v", fam=QW, model=QWEN_NV, harness="tools-grid", set="pilot", files=["qwen_tools_xhigh_48k_pilot_s24v.jsonl"],
         desc="tier xhigh, 48k hard / 24k soft, --verify-submit", cost=("none",), ref="E008 Qwen xhigh-240k", ref_match=True, ref_cap=48000,
         setlabel="pilot-26", status="stopped", expected=52),
    dict(id="E012 pilot xhigh-80k s40", fam=QW, model=QWEN_NV, harness="tools-grid", set="pilot", files=["qwen_tools_xhigh_80k_pilot_s40.jsonl"],
         desc="tier xhigh, 80k hard / 40k soft", cost=("none",), ref="E008 Qwen xhigh-240k", ref_match=True, ref_cap=80000,
         setlabel="pilot-26", status="stopped", expected=52),
    # dev-56: E012
    dict(id="E012 dev medium-48k s24", fam=QW, model=QWEN_NV, harness="tools-grid", set="dev", files=["qwen_tools_medium_48k_dev_s24.jsonl"],
         desc="tier medium, 48k hard / 24k soft, notes on every output (the Kaggle v2 config)", cost=("none",), ref="E008 Qwen xhigh-240k",
         ref_match=True, ref_cap=48000, setlabel="dev-56", expected=112, tags=["dev_s24"]),
    dict(id="E012 dev medium-48k s24strong", fam=QW, model=QWEN_NV, harness="tools-grid", set="dev", files=["qwen_tools_medium_48k_dev_s24strong.jsonl"],
         desc="as dev medium-48k s24 + strong soft-target wording", cost=("none",), ref="E008 Qwen xhigh-240k", ref_match=True, ref_cap=48000,
         setlabel="dev-56", expected=112, tags=["dev_s24strong"]),
    dict(id="E012 dev medium-28k s14", fam=QW, model=QWEN_NV, harness="tools-grid", set="dev", files=["qwen_tools_medium_28k_dev_s14.jsonl"],
         desc="tier medium, 28k hard / 14k soft", cost=("none",), ref="E008 Qwen xhigh-240k", ref_match=True, ref_cap=28000,
         setlabel="dev-56", expected=112, tags=["dev_s14"]),
    dict(id="E012 dev xhigh-48k s24", fam=QW, model=QWEN_NV, harness="tools-grid", set="dev", files=["qwen_tools_xhigh_48k_dev_s24.jsonl"],
         desc="tier xhigh, 48k hard / 24k soft", cost=("none",), ref="E008 Qwen xhigh-240k", ref_match=True, ref_cap=48000,
         setlabel="dev-56", expected=112, tags=["xhigh_dev_s24"]),
    # dev-56: E009
    ds_dev("sub96", "lean2sub", "96k hard cap, no notes"),
    ds_dev("sub96n", "lean2sub", "96k hard cap, notes from 50% used"),
    ds_dev("brief96n", "lean2brief", "brevity prompt, 96k, notes from 50%"),
    ds_dev("brief80n", "lean2brief", "brevity prompt, 80k, notes from 50%"),
    ds_dev("brief80a", "lean2brief", "brevity prompt, 80k, notes on every output"),
    ds_dev("brief72a", "lean2brief", "brevity prompt, 72k, notes on every output"),
    ds_dev("brief64a", "lean2brief", "brevity prompt, 64k, notes on every output"),
    ds_dev("sub80a", "lean2sub", "80k, notes on every output"),
    ds_dev("sub72a", "lean2sub", "72k, notes on every output"),
    ds_dev("sub64a", "lean2sub", "64k, notes on every output"),
    ds_dev("sub80s40", "lean2sub", "80k, notes on every output, 40k soft target"),
    ds_dev("sub72s32", "lean2sub", "72k, 32k soft target"),
    ds_dev("sub64s32", "lean2sub", "64k, 32k soft target"),
    # dev-56: E010
    ds_dev("nt0s20x60", "lean2sub", "thinking off on every turn, 60k / 20k soft", "E010"),
    ds_dev("nt1s40x80", "lean2sub", "thinking on turn 1 only, 80k / 40k soft", "E010"),
    ds_dev("nt3s20x60", "lean2sub", "thinking on turns 1–3, 60k / 20k soft", "E010"),
    ds_dev("s10x40", "lean2sub", "40k hard / 10k soft, strong wording", "E010"),
    ds_dev("s15x45", "lean2sub", "45k hard / 15k soft, strong wording", "E010"),
    ds_dev("s20x40", "lean2sub", "40k hard / 20k soft, strong wording", "E010"),
    ds_dev("s20x60", "lean2sub", "60k hard / 20k soft, strong wording", "E010"),
    ds_dev("s25x80", "lean2sub", "80k hard / 25k soft, strong wording", "E010"),
    # E004a pilots on E004's own 20-pair subset
    dict(id="E004a grid plain max v1", fam=DS, model=FLASH, harness="tools-grid", set="e004", files=["tools_persistent_grid_deepseek-flash_max_pilot.jsonl"],
         desc="plain prompt, max, 200k total / 8k final, harness v1", cost=REC_EST, ref="E004-GL", setlabel="E004a-20"),
    dict(id="E004a code plain max v1", fam=DS, model=FLASH, harness="tools-code", set="e004", files=["tools_persistent_code_deepseek-flash_max_pilot.jsonl"],
         desc="plain prompt, max, harness v1 (buggy train check)", cost=REC_EST, ref="E004-GL", setlabel="E004a-20"),
    dict(id="E004a code plain max v2", fam=DS, model=FLASH, harness="tools-code", set="e004", files=["tools_persistent_code_deepseek-flash_max_pilot2plain.jsonl"],
         desc="plain prompt, max, harness v2", cost=REC_EST, ref="E004-GL", setlabel="E004a-20"),
    dict(id="E004a grid lean max", fam=DS, model=FLASH, harness="tools-grid", set="e004", files=["tools_persistent_grid_deepseek-flash_max_pilot2.jsonl"],
         desc="lean prompt, max, harness v2", cost=REC_EST, ref="E004-GL", setlabel="E004a-20"),
    dict(id="E004a code lean max", fam=DS, model=FLASH, harness="tools-code", set="e004", files=["tools_persistent_code_deepseek-flash_max_pilot2.jsonl"],
         desc="lean prompt, max, harness v2", cost=REC_EST, ref="E004-GL", setlabel="E004a-20"),
    dict(id="E004a grid lean high", fam=DS, model=FLASH, harness="tools-grid", set="e004", files=["tools_persistent_grid_deepseek-flash_high_pilot2.jsonl"],
         desc="lean prompt, high, harness v2", cost=REC_EST, ref="E004-GL", setlabel="E004a-20"),
    dict(id="E004a code lean high", fam=DS, model=FLASH, harness="tools-code", set="e004", files=["tools_persistent_code_deepseek-flash_high_pilot2.jsonl"],
         desc="lean prompt, high, harness v2", cost=REC_EST, ref="E004-GL", setlabel="E004a-20"),
    dict(id="E004 stateless v0", fam=DS, model=FLASH, harness="tools-grid", set="covered", files=["tools_deepseek-flash_max_pilot.jsonl"],
         desc="stateless sandbox (no filesystem isolation), max effort", cost=REC_EST, ref="E004-GL", setlabel="2 easy tasks", status="superseded"),
    dict(id="E004 code smoke", fam=DS, model=FLASH, harness="tools-code", set="covered", files=["tools_persistent_code_deepseek-flash_max_smoke.jsonl"],
         desc="harness v1 smoke test", cost=REC_EST, ref="E004-GL", setlabel="smoke", status="probe"),
    # E003 / E005: no tools, DeepSeek
    dict(id="E003 Flash max", fam=DS, model=FLASH, harness="no tools", set="covered", files=["arcprize_deepseek-flash_max_traces.jsonl"],
         desc="ARC Prize prompt, max effort; trace study", cost=("est", FLASH_PRICE), ref="E001d", setlabel="E003-4 (4 tasks)", status="done"),
    dict(id="E003 Pro max", fam=DS, model=PRO, harness="no tools", set="covered", files=["arcprize_deepseek-v4-pro_max_traces.jsonl"],
         desc="ARC Prize prompt, max effort; trace study", cost=("est", PRO_PRICE), ref="E001d", setlabel="E003-4 (4 tasks)", status="done"),
    dict(id="E005 Flash high", fam=DS, model=FLASH, harness="no tools", set="covered", files=["arcprize_deepseek-flash_high_a1.jsonl"],
         desc="ARC Prize prompt, high effort, 1 attempt", cost=("est", FLASH_PRICE), ref="E001d", setlabel="full-167 (partial)", status="paused",
         expected=167, note="Paused after a few calls; short pairs finish first, so not representative."),
    dict(id="E005 Flash low", fam=DS, model=FLASH, harness="no tools", set="covered", files=["arcprize_deepseek-flash_low_a1.jsonl"],
         desc="ARC Prize prompt, low effort, 1 attempt", cost=("est", FLASH_PRICE), ref="E001d", setlabel="full-167 (partial)", status="paused",
         expected=167, note="Paused; short pairs only."),
    # partial full-set runs
    dict(id="E009 sub64a 2nd sample", fam=DS, model=FLASH, harness="tools-grid", set="covered", files=[tool_file("lean2sub", "low", "e009_full_sub64a")],
         attempts=[1], desc="as E009 sub64a", cost=REC_EST, ref="E004-GL", setlabel="full-167 (partial)", status="stopped", expected=167,
         note="Stopped early because it was clearly losing to E004-GL."),
    dict(id="E010 s20x40", fam=DS, model=FLASH, harness="tools-grid", set="full", files=[tool_file("lean2sub", "low", "e010_full_s20x40")],
         desc="40k hard / 20k soft, strong wording", cost=REC_EST, ref="E004-GL", setlabel="full-167", status="done (8 API errors)",
         note="8 attempts lost to API connection errors and not retried, so it fails the completeness rule."),
    # provider / harness probes
    dict(id="E008 Qwen Reka diag", fam=QW, model="Qwen3.8-27B fp8, OpenRouter (Reka)", harness="no tools", set="covered",
         files=["openrouter_qwen-qwen3.8-27b_high_reka_diag.jsonl"], desc="high effort; provider fixes thinking at ~2k tokens", cost=REC_OR,
         setlabel="probe", status="probe"),
    dict(id="E008 Qwen Reka budget", fam=QW, model="Qwen3.8-27B fp8, OpenRouter (Reka)", harness="no tools", set="covered",
         files=["openrouter_qwen-qwen3.8-27b_high_reka_budget.jsonl"], desc="reasoning.max_tokens 150k (ignored by provider)", cost=REC_OR,
         setlabel="probe", status="probe"),
    dict(id="E008 Qwen Alibaba diag", fam=QW, model="Qwen3.8-27B, OpenRouter (Alibaba)", harness="no tools", set="covered",
         files=["openrouter_qwen-qwen3.8-27b_high_alibaba_diag.jsonl"], desc="reasoning.max_tokens 100k (provider fixes ~16k)", cost=REC_OR,
         setlabel="probe", status="probe"),
    dict(id="E008 Nemotron Lightning smoke", fam=NM, model="Nemotron 3.5 Lightning bf16, OpenRouter (CoreWeave)", harness="no tools", set="covered",
         files=["openrouter_nvidia-nemotron-3.5-lightning_high_smoke.jsonl"], desc="router smoke test", cost=REC_OR, setlabel="smoke", status="probe"),
    dict(id="E008 Qwen medium smoke", fam=QW, model=QWEN_NV, harness=GEN, set="covered", files=["qwen_nvfp4_gen_medium_48k_smoke.jsonl"],
         desc="generation-API harness smoke test", cost=("none",), setlabel="smoke", status="probe"),
    dict(id="E012 xhigh-80k smoke", fam=QW, model=QWEN_NV, harness="tools-grid", set="covered", files=["qwen_tools_xhigh_80k_smoke.jsonl"],
         desc="tool-harness smoke test, xhigh, 80k / 40k soft", cost=("none",), setlabel="smoke", status="probe"),
]

NVIDIA_SPECS = [
    dict(id="E001a", fam=DS, model=FLASH, file="arc2_deepseek-flash_max.jsonl", desc="NVIDIA transductive prompt, max effort, 200k", price=FLASH_PRICE),
    dict(id="E001b", fam=DS, model=FLASH, file="arc2_deepseek-flash_high.jsonl", desc="NVIDIA transductive prompt, high effort, 200k", price=FLASH_PRICE),
    dict(id="E001c", fam=DS, model=PRO, file="arc2_deepseek-v4-pro_max.jsonl", desc="NVIDIA transductive prompt, max effort, 200k", price=PRO_PRICE),
]

E004_SUBSET = None  # filled from the E004a pilot files

# ARC Prize's official DeepSeek V4.1 Flash results (arcprize.org/results/deepseek-v4-1-flash), quoted; see footnote "arcprize"
ARCPRIZE_V41 = {"max": 72.9, "high": 67.5, "low": 60.6, "none": 1.3, "arc1_max": 94.5, "public_max_tasks": 91, "agree_lo": 82, "agree_hi": 87}
QWEN_SEMI = 42.4


def nvidia_eval(spec):
    lat, _ = load(spec["file"])
    samples = sorted({a for _, _, a in lat})
    probs = sorted({t for t, _, _ in lat})
    ok = lambda t, s: bool(lat.get((t, 0, s), {}).get("_ok"))
    each = [st.mean(ok(t, s) for t in probs) for s in samples]
    toks = [r["_tok"] for r in lat.values()]
    res = {"id": spec["id"], "pass1": st.mean(each), "pass1_each": each, "samples": len(samples), "problems": len(probs),
           "tok_mean": st.mean(toks), "tok_median": st.median(toks), "cost_total": sum(toks) * spec["price"] / 1e6,
           "unparsed": sum(r.get("pred") is None for r in lat.values()), "n_good": len(lat)}
    if len(samples) >= 2:
        res["pass2"] = st.mean(ok(t, 0) or ok(t, 1) for t in probs)
    # one test input per problem: Kaggle (pair-weighted) and task-weighted scores coincide
    res.update({"k_" + k: res[k] for k in ("pass1", "pass1_each", "pass2") if k in res})
    return res


# ----------------------------------------------------------------------------------------------- formatting helpers


def f1(x):
    return "—" if x is None else f"{x * 100:.1f}"


def fk(x):
    return "—" if x is None else f"{x / 1000:.1f}k"


def fdiff(d):
    if not d:
        return "—"
    return f"{d['diff'] * 100:+.1f} [{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}]".replace("-", "−")


def sg(v, nd=1):
    """Signed number with a real minus sign."""
    return f"{v:+.{nd}f}".replace("-", "−")


def money(x):
    return "—" if x is None else (f"${x:.2f}" if x < 100 else f"${x:.0f}")


# ----------------------------------------------------------------------------------------------- build


def build():
    global E004_SUBSET
    pilot_lat, _ = load("tools_persistent_grid_deepseek-flash_max_pilot.jsonl")
    E004_SUBSET = sorted({(t, p) for t, p, _ in pilot_lat})
    SETS["e004"] = E004_SUBSET

    results = {}
    t1_rows, t2_rows = [], []
    all_specs = T1_SPECS + T2_SPECS
    for spec in all_specs:
        results[spec["id"]] = evaluate(spec)
    specs = {s["id"]: s for s in all_specs}

    # E004-GL long tail: pairs whose correct rollouts needed >40k tokens on average
    gl = results["E004-GL"]
    gi = index_lat(gl["_lat"])
    longtail = [k for k, atts in gi.items() if any(r["_ok"] for r in atts.values())
                and st.mean(r["_tok"] for r in atts.values() if r["_ok"]) > 40000]

    def longtail_acc(res):
        idx = index_lat(res["_lat"])
        vals = [st.mean(r["_ok"] for r in idx[k].values()) for k in longtail if k in idx and idx[k]]
        return st.mean(vals) if vals else None

    def paired_for(spec, res):
        out = []
        ref = spec.get("ref")
        if ref and ref in results and ref != spec["id"]:
            out.append(paired(res, results[ref], match_attempts=spec.get("ref_match", False)))
            if spec.get("ref_cap"):
                d = paired(res, results[ref], match_attempts=spec.get("ref_match", False), cap=spec["ref_cap"])
                out.append(d)
        for r2 in spec.get("ref_extra", []):
            out.append(paired(res, results[r2]))
        return [d for d in out if d]

    for spec in all_specs:
        res = results[spec["id"]]
        res["paired"] = paired_for(spec, res)
        if spec.get("e017"):
            res["train_check"] = train_check_stats(res["_lat"])
        if spec.get("policy3"):
            res["policy3"] = policy3(spec)
        if spec["fam"] == DS and spec["harness"].startswith("tools") and spec["set"] == "full":
            res["longtail_acc"] = longtail_acc(res)
        tags = spec.get("running_tags") or spec.get("tags") or []
        res["running"] = any(t in RUNNING for t in tags)

    nv = {s["id"]: nvidia_eval(s) for s in NVIDIA_SPECS}

    # ------------------------------------------------------------------ Table 1
    fam_order = {DS: 0, QW: 1, NM: 2}
    for spec in T1_SPECS:
        res = results[spec["id"]]
        if not (res["complete"] and spec["set"] == "full"):
            continue
        t1_rows.append(t1_row(spec, res))
    t1_rows.sort(key=lambda r: (fam_order[r["fam"]], -(r["raw"]["k_pass2"] if r["raw"]["k_pass2"] is not None else -1),
                                -(r["raw"]["k_pass1"] or 0)))
    mark_best(t1_rows, {"k_pass1": max, "k_pass2": max, "tok_mean": min, "tok_median": min, "tok_p90": min, "ctok_median": min},
              band={"k_pass1": NOISE_BAND, "k_pass2": NOISE_BAND})

    # ------------------------------------------------------------------ Table 2
    for spec in T1_SPECS:  # incomplete Table-1 candidates go to Table 2
        res = results[spec["id"]]
        if not res["complete"]:
            t2_rows.append(t2_row(spec, res, inprogress=True))
    for spec in T2_SPECS:
        t2_rows.append(t2_row(spec, results[spec["id"]]))
    e16_rows, e16 = e016_rows(results["E010 s20x60"])
    t2_rows += e16_rows
    for spec in NVIDIA_SPECS:
        r = nv[spec["id"]]
        t2_rows.append({
            "id": spec["id"], "fam": spec["fam"], "model": spec["model"], "harness": "no tools",
            "set": f"NVIDIA val subset ‡ ({r['problems']} problems)",
            "coverage": f"{r['samples']} × {r['problems']}/{r['problems']}",
            "k_pass1": f1(r["pass1"]) + (f" ({' / '.join(f1(x) for x in r['pass1_each'])})" if r["samples"] > 1 else ""),
            "k_pass2": f1(r.get("pass2")), "tok": f"{fk(r['tok_mean'])} / {fk(r['tok_median'])}", "paired": "— (different task set)",
            "pass1": f1(r["pass1"]), "pass2": f1(r.get("pass2")),
            "status": f"done. {spec['desc']}. One test input per problem, so Kaggle and task-weighted scores coincide. Score uses the stored flag "
                      f"against NVIDIA's expected_output; est. cost {money(r['cost_total'])}; {r['unparsed']} unparsed.",
            "raw": {"k_pass1": r["pass1"], "k_pass2": r.get("pass2"), "pass1": r["pass1"], "pass2": r.get("pass2"), "tok_mean": r["tok_mean"]}})
    kag = kaggle_data()
    for k in kag["local_tests"]:
        t2_rows.append(k)

    # ------------------------------------------------------------------ derived findings
    F = findings(results, specs, nv, longtail, kag, e16)
    chart = [{"id": r["id"], "fam": r["fam"], "tok_mean": r["raw"]["tok_mean"], "k_pass1": r["raw"]["k_pass1"], "k_pass2": r["raw"]["k_pass2"],
              "pass1": r["raw"]["pass1"], "pass2": r["raw"]["pass2"], "samples": r["raw"]["samples"]} for r in t1_rows]
    metric_gap = [{"id": r["id"], "samples": r["raw"]["samples"],
                   "k_pass1": r["raw"]["k_pass1"], "tw_pass1": r["raw"]["pass1"], "k_pass2": r["raw"]["k_pass2"], "tw_pass2": r["raw"]["pass2"],
                   "d_pass1": r["raw"]["k_pass1"] - r["raw"]["pass1"],
                   "d_pass2": (r["raw"]["k_pass2"] - r["raw"]["pass2"]) if r["raw"]["k_pass2"] is not None else None} for r in t1_rows]
    multi = sum(len(TASKS[t]["test"]) > 1 for t in TASKS)
    gaps = [g_[k_] for g_ in metric_gap for k_ in ("d_pass1", "d_pass2") if g_[k_] is not None]
    gap_txt = (f"at most {max(abs(x) for x in gaps) * 100:.1f} pp, and Kaggle is the lower of the two in {sum(x < 0 for x in gaps)} of "
               f"{len(gaps)} pass@1/pass@2 values" if gaps else "small")
    nt = sorted({len(TASKS[t]["test"]) for t in TASKS if len(TASKS[t]["test"]) > 1})
    mrange = f"{nt[0]}–{nt[-1]}" if len(nt) > 1 else str(nt[0])
    scoring = [x.replace("{multi}", str(multi)).replace("{gap}", gap_txt).replace("{mrange}", mrange) for x in SCORING]
    data = {
        "meta": {"title": "ARC-AGI-2 Experiment Report", "date": REPORT_DATE,
                 "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                 "command": ".venv/bin/python reports/build_report.py",
                 "metric_keys": "k_* = Kaggle competition score (pair-weighted: solved test outputs / all test outputs), primary; "
                                "unprefixed pass1/pass2 = ARC Prize benchmarking task-weighted score (arc-agi-benchmarking scoring.py). "
                                "paired[].diff/lo/hi/run/ref_score are Kaggle (cluster bootstrap over tasks); paired[].tw_* are task-weighted.",
                 "running_tags": sorted(RUNNING), "audit": AUDIT, "longtail_pairs": len(longtail),
                 "sets": {"full": len(FULL), "pilot": len(PILOT), "dev": len(DEV), "e004": len(E004_SUBSET)}},
        "table1": t1_rows, "table2": t2_rows, "chart": chart, "metric_gap": metric_gap, "findings": F["findings"], "kaggle": kag, "caveats": F["caveats"], "arcprize_cmp": F["arcprize_cmp"], "e016": e16,
        "footnotes": FOOTNOTES, "crosscheck": crosscheck(results, nv), "scope": SCOPE, "scoring": scoring,
        "runs": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}, "nvidia_runs": nv,
    }
    data["pareto"] = pareto(chart, results)
    v2, pl1 = kag.get("v2-v2", {}), kag.get("plain-v1", {})
    data["kaggle_text"] = dict(KAGGLE_TEXT, local=KAGGLE_TEXT["local"].format(
        v2=v2.get("k_score_txt", "?"), pl=pl1.get("k_score_txt", "?"), v2t=v2.get("score", "?"), plt=pl1.get("score", "?")))
    return data


def pareto_front(pts):
    front, best2 = [], -1.0
    for p_ in sorted((p_ for p_ in pts if p_["k_pass2"] is not None), key=lambda p_: p_["tok_mean"]):
        if p_["k_pass2"] > best2:
            front.append(p_)
            best2 = p_["k_pass2"]
    return front


def pareto(chart, results):
    """Kaggle pass@2 Pareto front over the multi-sample Table 1 runs, and how E017 changed it."""
    front = pareto_front(chart)
    before = pareto_front([p for p in chart if not p["id"].startswith("E017")])
    ids, bids = [p["id"] for p in front], [p["id"] for p in before]
    txt = ("Kaggle pass@2 Pareto front over all multi-sample runs (fewer tokens per attempt, higher pass@2): "
           + " → ".join(f"{p['id']} ({fk(p['tok_mean'])}, {f1(p['k_pass2'])}%)" for p in front) + ".")
    added, dropped = [i for i in ids if i not in bids], [i for i in bids if i not in ids]
    if added:
        txt += (f" {' and '.join(added)} joined the front" + (f", displacing {' and '.join(dropped)}" if dropped else "") + ".")
    pp = results.get("E017 V3 all", {}).get("policy3")
    if pp:
        txt += (f" Not plotted: with its conditional 3rd attempt E017 V3 reaches {f1(pp['k_pass2_policy'])}% Kaggle pass@2 at "
                f"{fk(pp['tok_per_pair_policy'])} tokens per pair ({fk(pp['tok_per_pair_base'])} without), a per-pair rather than per-attempt cost.")
    return {"front": ids, "front_before_e017": bids, "text": txt}


def coverage_text(res):
    pa = res["per_attempt"]
    if res["complete"]:
        errs = sum(v["errors"] for v in pa.values())
        s = f"{len(pa)} × {res['n_universe']}/{res['n_universe']}"
        if errs:
            s += f" ({errs} API-error attempt{'s' if errs > 1 else ''} scored wrong)"
        return s
    return None


def score_cells(res, pre):
    """pass@1 / pass@2 display strings for one metric (pre = "k_" Kaggle, "" task-weighted)."""
    ns = res["samples"]
    p1, p1_sub = f1(res[pre + "pass1"]), None
    if ns >= 2:
        p1 = f"{res[pre + 'pass1'] * 100:.1f} ± {res[pre + 'pass1_sd'] * 100:.1f}"
        p1_sub = " / ".join(f1(x) for x in res[pre + "pass1_each"])
    p2 = f1(res.get(pre + "pass2"))
    p2_sub = f"a0+a1 {f1(res.get(pre + 'pass2_a01'))}; {res['pass2_combos']} attempt pairs" if ns > 2 else None
    return p1, p1_sub, p2, p2_sub


def t1_row(spec, res):
    ns = res["samples"]
    p1, p1_sub, p2, p2_sub = score_cells(res, "k_")
    tp1, tp1_sub, tp2, tp2_sub = score_cells(res, "")
    cost_sub, cost_raw = None, res.get("cost_per_sample")
    cba = res.get("cost_by_attempt") or {}
    if cba and any(v["peak_part"] > 0 for v in cba.values()):
        def when(v):
            return "peak 2×" if v["peak_part"] >= v["rec"] - 1e-9 else ("off-peak" if v["peak_part"] == 0 else "partly peak")
        cost = " / ".join(f"{money(v['rec'])} ({when(v)})" for v in cba.values())
        cost_raw = res["cost_per_sample_offpeak"]
        cost_sub = ("rec. per sample" if len(cba) > 1 else "rec.") + f"; {money(cost_raw)}/sample off-peak equiv."
        if res.get("policy3"):
            cost_sub += f"; +{money(res['policy3']['extra_cost'])} for 3rd attempts"
    elif res["cost_kind"] in ("est", "rec"):
        cost = f"{money(res['cost_per_sample'])}/sample ({'est.' if res['cost_kind'] == 'est' else 'rec.'})"
    elif res["cost_kind"] == "text":
        cost = res["cost_text"]
    else:
        cost = "—"
    notes = []
    if spec.get("note"):
        notes.append(spec["note"])
    for d in res.get("paired", []):
        notes.append(f"Δ Kaggle pass@1 vs {d['ref']}{' capped ' + str(d['cap'] // 1000) + 'k' if d['cap'] else ''}: {fdiff(d)} pp")
    if res.get("longtail_acc") is not None:
        notes.append(f"long-tail pairs solved: {f1(res['longtail_acc'])}%")
    if res.get("sources"):
        srcs = res["sources"]
        tot = sum(srcs.values())
        top = ", ".join(f"{k} {v}" for k, v in list(srcs.items())[:3])
        notes.append(f"answer sources: {top} (of {tot})")
    unp = res["unparsed"]
    if unp and not res.get("sources"):
        notes.append(f"{unp}/{res['n_good']} attempts without a parsable grid")
    tc = res.get("train_check")
    if tc:
        notes.append(f"live train check: {tc['pass']}/{tc['n']} attempts pass train; P(correct | train pass) {tc['p_pass']:.0%} "
                     f"({tc['pass_correct']}/{tc['pass']}), P(correct | train fail) {tc['p_fail']:.0%} ({tc['fail_correct']}/{tc['fail']}); "
                     f"{tc['no_answer']} without an answer")
    pp = res.get("policy3")
    if pp:
        p2_sub = f"with cond. 3rd attempt: {f1(pp['k_pass2_policy'])}"
        tp2_sub = f"with 3rd attempt: {f1(pp['pass2_policy'])}"
        notes.append(f"Conditional 3rd attempt (attempt index 2, only when neither attempt 0 nor 1 passed the train check): fired on "
                     f"{pp['fired_pairs']}/{len(FULL)} pairs; {pp['extra_correct']} correct, {pp['extra_train_pass']} passed train "
                     f"({pp['extra_train_pass_correct']} of those correct), {pp['extra_no_answer']} without an answer; mean "
                     f"{fk(pp['extra_tok_mean'])} tokens, +{fk(pp['extra_tok_total'] / len(FULL))} tokens/pair "
                     f"({fk(pp['tok_per_pair_base'])} → {fk(pp['tok_per_pair_policy'])}, "
                     f"{sg((pp['tok_per_pair_policy'] / pp['tok_per_pair_base'] - 1) * 100, 0)}%), rec. {money(pp['extra_cost'])}{' off-peak' if not pp['extra_cost_peak_part'] else ' (partly peak)'}. "
                     f"Answers ranked as the harness does (train pass first, then attempt order, top-2 distinct): Kaggle pass@2 "
                     f"{f1(pp['k_pass2_base'])} → {f1(pp['k_pass2_policy'])} ({sg(pp['diff'] * 100)} pp [{sg(pp['lo'] * 100)}, "
                     f"{sg(pp['hi'] * 100)}]; {pp['rescued']} pairs rescued, {pp['lost']} lost); task-weighted {f1(pp['pass2_base'])} → "
                     f"{f1(pp['pass2_policy'])}")
    return {"id": spec["id"], "fam": spec["fam"], "model": spec["model"], "harness": spec["harness"], "desc": spec["desc"],
            "samples": coverage_text(res), "k_pass1": p1, "k_pass1_sub": p1_sub, "k_pass2": p2, "k_pass2_sub": p2_sub,
            "pass1": tp1, "pass1_sub": tp1_sub, "pass2": tp2, "pass2_sub": tp2_sub,
            "tok_mean": fk(res["tok_mean"]), "tok_median": fk(res["tok_median"]), "tok_p90": fk(res["tok_p90"]),
            "ctok_median": fk(res["ctok_median"]), "cost": cost, "cost_sub": cost_sub, "notes": notes,
            "raw": {"k_pass1": res["k_pass1"], "k_pass2": res.get("k_pass2"), "pass1": res["pass1"], "pass2": res.get("pass2"),
                    "tok_mean": res["tok_mean"], "tok_median": res["tok_median"],
                    "tok_p90": res["tok_p90"], "ctok_median": res["ctok_median"], "samples": ns,
                    "cost_per_sample": cost_raw, "k_pass2_policy": pp["k_pass2_policy"] if pp else None}, "best": {}}


# Scores within this many points (fraction) of a column's best are treated as tied with it (sample noise is ~2-3 pp).
NOISE_BAND = 0.03


def mark_best(rows, rules, band=None):
    band = band or {}
    for col, fn in rules.items():
        vals = [r["raw"][col] for r in rows if r["raw"].get(col) is not None]
        if not vals:
            continue
        b = fn(vals)
        for r in rows:
            v = r["raw"].get(col)
            r["best"][col] = v is not None and (v >= b - band[col] - 1e-9 if col in band else v == b)


def t2_row(spec, res, inprogress=False):
    exp = spec.get("expected")
    pa = res["per_attempt"]
    done = res["attempts_done"]
    if inprogress:
        total = len(FULL) * (spec.get("expected_attempts") or 1)
        cov = f"{done}/{total} attempts; {res['covered_pairs']}/{len(FULL)} pairs"
        setlabel = "full-167 (in progress)"
        status = ("running" if res.get("running") else "incomplete") + " — moves to Table 1 when all 334 attempts are done"
    else:
        setlabel = spec.get("setlabel", spec["set"])
        n_u = res["n_universe"] if spec["set"] in SETS else res["covered_pairs"]
        if spec["set"] in SETS or spec["set"] == "e004":
            setlabel = f"{setlabel} ({len(SETS[spec['set']])} pairs)" if spec["set"] in SETS else setlabel
        if exp:
            cov = f"{done}/{exp} attempts"
        elif res["complete"]:
            cov = f"{len(pa)} × {res['covered_pairs']}/{n_u}"
        else:
            cov = f"{done} attempts, {res['covered_pairs']} pairs / {res['covered_tasks']} tasks"
        if res["attempts_error"]:
            cov += f"; {res['attempts_error']} errors"
        status = spec.get("status") or ("done" if (not exp or res["attempts_present"] >= exp) else "stopped")
        if res.get("running"):
            status = "running"
    pairedtxt = "; ".join(f"vs {d['ref']}{' capped ' + str(d['cap'] // 1000) + 'k (sim.)' if d['cap'] else ''}: {fdiff(d)} "
                          f"({f1(d['run'])} vs {f1(d['ref_score'])}, {d['pairs']} pairs)" for d in res.get("paired", [])) or "—"
    extra = []
    if spec.get("note"):
        extra.append(spec["note"])
    if spec.get("desc"):
        extra.insert(0, spec["desc"])
    if res.get("cost_kind") in ("est", "rec") and res.get("cost_total") is not None:
        extra.append(f"{'est.' if res['cost_kind'] == 'est' else 'rec.'} {money(res['cost_total'])}")
    if res.get("sources"):
        extra.append("sources: " + ", ".join(f"{k} {v}" for k, v in res["sources"].items()))
    if res.get("harness_versions") and len(res["harness_versions"]) > 1:
        extra.append("harness versions: " + ", ".join(f"{k} {v}" for k, v in res["harness_versions"].items()))
    p2_sub = f"{res['pass2_pairs']} pairs with 2 attempts" if res.get("pass2_pairs") else None

    def p1txt(pre):
        each = res.get(pre + "pass1_each")
        return f1(res[pre + "pass1"]) + (f" ({' / '.join(f1(x) for x in each)})" if each and len(each) > 1 else "")
    return {"id": spec["id"], "fam": spec["fam"], "model": spec["model"], "harness": spec["harness"], "set": setlabel, "coverage": cov,
            "k_pass1": p1txt("k_"), "k_pass2": f1(res.get("k_pass2")), "k_pass2_sub": p2_sub,
            "pass1": p1txt(""), "pass2": f1(res.get("pass2")),
            "tok": f"{fk(res['tok_mean'])} / {fk(res['tok_median'])}", "paired": pairedtxt,
            "status": f"{status}. " + "; ".join(extra),
            "raw": {"k_pass1": res["k_pass1"], "k_pass2": res.get("k_pass2"), "pass1": res["pass1"], "pass2": res.get("pass2"),
                    "tok_mean": res["tok_mean"]}}


# ----------------------------------------------------------------------------------------------- Kaggle


def kaggle_data():
    kr = ROOT / "kaggle/runs"
    out = {"bench": {}, "local_tests": [], "submissions": []}
    for name in ("bench-mtp3-v3", "bench-mtp0-v3"):
        p = kr / name / "bench_results.json"
        if p.exists():
            b = json.loads(p.read_text())
            out["bench"][name] = {"mtp": b["mtp"], "kv_tokens": b["kv_tokens"], "concurrency": b["concurrency"],
                                  "decode_tok_s": b["decode_long"]["steady_decode_tok_s"], "ctx_mean": b["decode_long"]["ctx_mean"],
                                  "prefill_s": b["decode_long"]["prefill_phase_s"], "e2e": b["e2e"]}
    for name, label in (("v2-v2", "Kaggle v2 tools (commit run)"), ("plain-v1", "Kaggle plain v1 (commit run)")):
        p = kr / name / "summary_local.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        pp = s["per_pair"]
        out[name] = {k: s.get(k) for k in ("server", "agent", "concurrency", "kv_tokens", "phases_s", "attempts", "tokens", "gen_minutes",
                                             "answered", "errors", "tool_calls", "tool_errors", "parse_failures", "score", "kv_usage_max_pct")}
        tasks = sorted({x["task"] for x in pp})
        n_ok = sum(bool(x["correct"]) for x in pp)
        k_score = n_ok / len(pp)
        by_task = defaultdict(list)
        for x in pp:
            by_task[x["task"]].append(bool(x["correct"]))
        tw_score = st.mean(st.mean(v) for v in by_task.values())
        out[name]["k_score"] = k_score
        out[name]["k_score_txt"] = f"{n_ok}/{len(pp)}"
        out[name]["tw_score"] = tw_score
        out["local_tests"].append({
            "id": label, "fam": QW, "model": "Qwen3.8-27B FP8, vLLM 0.23 on Kaggle 4×L4", "harness": "tools-grid" if s["agent"] == "tools" else "no tools",
            "set": f"Kaggle local test ({len(tasks)} tasks / {len(pp)} pairs)", "coverage": f"1 × {len(pp)}/{len(pp)}",
            "k_pass1": f"{f1(k_score)} ({n_ok}/{len(pp)} outputs)", "k_pass2": "—", "pass1": f"{f1(tw_score)} ({s['score']})", "pass2": "—",
            "tok": f"{fk(st.mean(x['tokens'] for x in pp))} / {fk(st.median(x['tokens'] for x in pp))}",
            "paired": "—",
            "status": f"done (deadline-forced ~6 min after server ready). {s['tool_calls']} tool calls, {s['tool_errors']} tool errors, "
                      f"{s['parse_failures']} parse failures, peak KV {s['kv_usage_max_pct']}%; source kaggle/runs/{name}/summary_local.json",
            "raw": {"k_pass1": k_score, "k_pass2": None, "pass1": tw_score, "pass2": None, "tok_mean": st.mean(x["tokens"] for x in pp)}})
    p = kr / "submission_watch.jsonl"
    if p.exists():
        last = {}
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                last[r["id"]] = r
        out["submissions"] = [{k: r.get(k) for k in ("id", "description", "submitted_at", "status", "observed_at", "elapsed_h", "public_score")}
                              for r in last.values()]
    # per-attempt budget on Kaggle: rate x usable time / pairs
    usable_s = 11 * 3600 - (out.get("v2-v2", {}).get("phases_s") or {}).get("server_ready", 562)
    rates = {f"decode MTP=3 (measured, 16 seqs @40k ctx)": out["bench"].get("bench-mtp3-v3", {}).get("decode_tok_s"),
             "decode MTP=0 (measured)": out["bench"].get("bench-mtp0-v3", {}).get("decode_tok_s"),
             "E2E scheduler avg MTP=3 (README) [^kag-e2e]": 295.0}
    out["budget"] = [{"rate_name": k, "tok_s": v, "total_M": v * usable_s / 1e6, "per_pair_1att": v * usable_s / 259,
                      "per_att_2att": v * usable_s / 518} for k, v in rates.items() if v]
    out["usable_s"] = usable_s
    return out


# ----------------------------------------------------------------------------------------------- text

SCOPE = ("This report collects every model and harness experiment we have run on ARC-AGI-2 so far. Models: DeepSeek V4.1-Flash and V4-Pro "
         "(DeepSeek API); Qwen3.8-27B and Qwen3.6-35B-A3B (NVFP4 checkpoints on self-hosted SGLang, plus OpenRouter probes); Nemotron 3.5 "
         "Lightning and Nemotron 3 Super 120B (OpenRouter). Harnesses: a no-tool single call with the ARC Prize prompt; a generation API "
         "harness for Qwen that continues the raw completion through mid-thought EOS and forces a final answer at the budget; Python tool "
         "use in grid mode (answer as a grid, later handed over with `submit_answer`) and code mode (`submit_solution`, checked on the "
         "train pairs); token-budget variants of the tool harness (hard caps, budget notes, soft targets, thinking-off turns); E017's "
         "additions to it (a host-computed task summary, a roles prompt, session helpers, a live train check and a conditional 3rd attempt); "
         "and E016's \"time is up\" continuation of cut-off attempts. "
         "Every number below is recomputed from `evals/results/*.jsonl` and `kaggle/runs/` by `reports/build_report.py`.")

SCORING = [
    "**Primary metric: Kaggle score (pair-weighted).** The ARC Prize 2026 ARC-AGI-2 competition page on Kaggle allows 2 attempts per test "
    "input; a test output scores 1 if either attempt matches exactly, else 0, and \"The final score is the average across all task test "
    "outputs\". So the Kaggle score = solved test outputs / all test outputs (167 on the public eval set), and a task with {mrange} test inputs "
    "counts {mrange} times. All headline numbers, best-value shading, sorting, charts and paired Δs use this metric because it is what the "
    "competition scores.",
    "**Secondary metric: ARC Prize task-weighted.** The ARC Prize benchmarking harness (`arc-agi-benchmarking`, `scoring.py`) scores each "
    "task as the fraction of its test pairs solved and averages over the 120 tasks. ARC Prize's published leaderboard numbers use this "
    "convention, so it is shown in smaller muted columns and quoted in parentheses where we compare with ARC Prize's numbers. The two differ "
    "only through the {multi} tasks with more than one test input; for Table 1 runs the gap is {gap} (Appendix).",
    "**Full set** = the official ARC-AGI-2 public evaluation set from github.com/arcprize/ARC-AGI-2 (120 tasks / 167 test pairs). "
    "This is the public evaluation set, not the hidden Kaggle test set (240 tasks / 259 pairs) and not ARC Prize's semi-private set.",
    "**pass@1** = mean over attempts (samples) of the single-attempt score, ± sd over samples. **pass@2** = a test output counts if either of "
    "2 attempts is correct (the Kaggle rule); with more than 2 samples it is the mean over all attempt pairs (attempts 0+1 shown as a0+a1). "
    "**Paired Δ** = run minus reference on the same test outputs (Kaggle pass@1, pp), 95% CI from a cluster bootstrap that resamples tasks "
    "(all test outputs of a task move together; 10,000 resamples).",
    "**Completeness rule (Table 1).** A run is complete when every counted sample has a record for all 167 pairs. Attempts lost to an API "
    "error are scored wrong; a sample with more than 1 such attempt counts as incomplete and the run goes to Table 2.",
    "**Tokens/attempt** = output (completion) tokens summed over all turns of an attempt, over attempts that returned. "
    "**Cost**: rec. = recorded per-call cost (`est_cost_usd` at off-peak list price incl. input, or OpenRouter's `cost_usd`); "
    "est. = output tokens × list price ($0.60/M Flash, $1.98/M Pro); rental = vast.ai box time from the README (footnoted).",
    "**Other sets.** pilot-26 = the first 15 tasks in sorted id order (26 pairs); dev-56 = E009's stratified dev set "
    "(`experiments/E009_tool_token_efficiency/dev_pairs.txt`); E004a-20 = E004's hard-biased 20-pair subset. Partial runs are scored on "
    "their covered pairs only (Kaggle: mean over covered test outputs; task-weighted: a task's score is over its covered pairs and only "
    "covered tasks are averaged).",
]

FOOTNOTES = {
    "rent-xhigh": "Rental from experiments/E008_openrouter_models/README.md: about $5.42/h × 5.0 h (pilot 0.8 h + full 4.2 h) ≈ $27. API cost $0.",
    "rent-medium": "Not stated in the README. Derived from the README's wall time (13.07M tokens in 115.8 min) × the box's $5.42/h ≈ $10.",
    "rent-36": "Rental from the E008 README: ≈3.25 h × $5.93/h ≈ $19, including ~0.5 h of server setup.",
    "rent-e012": "Self-hosted on vast.ai instance 55443640 ($5.89/h, E012 README); box time per run is not itemised. API cost $0.",
    "nemo": "Manual reading of 40 traces (experiments/E008_openrouter_models/nemotron_traces/notes_{A,B,C,D}.md), not recomputed.",
    "kag-e2e": "295 tok/s average over 16 min in the end-to-end scheduler (MTP=3, contexts 2–20k) is from experiments/E013_kaggle_submission/README.md; the bench JSON stores only the decode benchmark.",
    "arcprize": ("ARC Prize's official DeepSeek V4.1 Flash results, https://arcprize.org/results/deepseek-v4-1-flash: verified semi-private ARC-AGI-2 "
                 "Max 72.9%, High 67.5%, Low 60.6%, None 1.3% (ARC-AGI-1 Max 94.5%); their per-task public-eval table at Max marks 91/120 tasks "
                 "solved. The per-task agreement with E001d (82–87% depending on the attempt pair) was computed against that table, which is not "
                 "stored in the repo, so it is quoted, not recomputed here. Qwen3.8-27B's 42.4% semi-private is quoted from the E008 README. The "
                 "older V4 Flash 0731 numbers in the E001 README (63.9% public / 61.4% semi-private) are for a different model version."),
    "e017fa": ("From experiments/E017_reasoning_efficiency/failure_analysis.md (phase 1B: offline replay of 835 sub80s40/sub96n attempts and 48 "
               "traces annotated by a single reader each; token shares are reader estimates) and the E017 README; not recomputed here."),
    "e017ps": ("From experiments/E017_reasoning_efficiency/policy_sim.md (phase 1A: 160 sub80s40 + s20x60 attempts on a stratified sample of 32 "
               "pairs, train-check labels by replay, population-weighted) and failure_analysis.md section 5; not recomputed here."),
    "e016": ("From experiments/E016_cutoff_continuation/README.md: for 142ca369.1#0 the thinking-off continuation had a 12.5k-token prompt "
             "against 45.7k for the original last turn; not recomputed here."),
    "audit": "From experiments/E008_openrouter_models/qwen_format_audit.md (450 Qwen attempts read and re-parsed); not recomputed here.",
    "contam": "From experiments/E004_tool_use/README.md (safety trace review): Flash listed real ARC-AGI-2 eval task ids while trying to recall answers.",
}


def findings(R, specs, nv, longtail, kag, e16=None):
    g = lambda k: R[k]
    gl, e1 = g("E004-GL"), g("E001d")
    d_gl = next(d for d in gl["paired"] if d["ref"] == "E001d")
    pl, pm = g("E004 plain-low"), g("E004 plain-max")
    s80, s96, s64, s2060, s2040 = g("E009 sub80s40"), g("E009/E011 sub96n"), g("E009 sub64a"), g("E010 s20x60"), g("E010 s20x40")
    def d_of(k):
        return next((d for d in g(k)["paired"] if d["ref"] == "E004-GL" and not d["cap"]), None)
    # no-answer share in E004-GL
    gl_good = [r for r in gl["_lat"].values() if not r["_err"]]
    na = [r for r in gl_good if r.get("pred") is None]
    na_share_tok = sum(r["_tok"] for r in na) / sum(r["_tok"] for r in gl_good)
    sub_src = s80.get("sources", {})
    sub_share = sub_src.get("submit", 0) / max(1, sum(sub_src.values()))
    # dev vs full
    pairs_df = [("sub80s40", "E009 dev sub80s40", "E009 sub80s40"), ("sub96n", "E009 dev sub96n", "E009/E011 sub96n"),
                ("sub64a", "E009 dev sub64a", "E009 sub64a"), ("s20x60", "E010 dev s20x60", "E010 s20x60"),
                ("s25x80", "E010 dev s25x80", "E010 s25x80"), ("s20x40", "E010 dev s20x40", "E010 s20x40")]
    dvf = []
    for name, dk, fk_ in pairs_df:
        dd = next((d for d in g(dk)["paired"] if d["ref"] == "E004-GL"), None)
        fd = d_of(fk_)
        if dd and fd:
            dvf.append((name, dd["diff"], fd["diff"]))
    gains = [(n, a, b) for n, a, b in dvf if a > 0]
    mean_dev = st.mean(a for _, a, _ in gains) if gains else None
    mean_full = st.mean(b for _, _, b in gains) if gains else None
    # Qwen
    qx, qm, q36 = g("E008 Qwen xhigh-240k"), g("E008 Qwen medium-48k"), g("E008 Qwen3.6-35B")
    def eos(res):
        rs = [r for r in res["_lat"].values() if not r["_err"]]
        return (sum((r.get("continuations") or 0) > 0 for r in rs) / len(rs), sum(bool(r.get("forced_final")) for r in rs) / len(rs))
    qm_eos, qm_forced = eos(qm)
    qx_eos, qx_forced = eos(qx)
    e12 = g("E012 Qwen tools full")
    e12_d = {(d["cap"] or 0): d for d in e12["paired"]}
    # Qwen no-tool at a cap close to the tool run's mean tokens
    cap_eq = int(round((e12["tok_mean"] or 0) / 4000.0) * 4000) or 32000
    d_eq = paired(e12, qx, match_attempts=True, cap=cap_eq)
    e12dev = g("E012 dev medium-48k s24")
    nemo = g("E008 Nemotron Lightning")
    # cap table for Qwen no-tool (Kaggle)
    cap_rows = []
    for cap in (28000, 36000, 48000, 64000, 96000, 128000, None):
        c = {k: r["_ok"] and (cap is None or r["_tok"] <= cap) for k, r in qx["_lat"].items()}
        row = {"cap": cap, "finished": st.mean((r["_tok"] <= cap) if cap else bool(r.get("closed_think")) for r in qx["_lat"].values())}
        for pre, agg in METRICS:
            row[pre + "pass1"] = st.mean(agg({(t, p): c[(t, p, a)] for t, p in FULL}) for a in (0, 1))
            row[pre + "pass2"] = agg({(t, p): c[(t, p, 0)] or c[(t, p, 1)] for t, p in FULL})
        cap_rows.append(row)
    kag["qwen_cap_table"] = cap_rows
    unp = {k: (R[k]["unparsed"], R[k]["n_good"]) for k in ("E008 Qwen xhigh-240k", "E008 Qwen medium-48k", "E008 Qwen3.6-35B")}

    nt_d = [d["diff"] * 100 for k in ("E010 dev nt0s20x60", "E010 dev nt1s40x80", "E010 dev nt3s20x60")
            for d in g(k)["paired"] if d["ref"] == "E004-GL" and not d["cap"]]
    ds_sd = [R[x["id"]]["k_pass1_sd"] * 100 for x in specs.values() if x["fam"] == DS and x["set"] == "full"
             and R[x["id"]]["complete"] and R[x["id"]].get("k_pass1_sd") is not None]
    pilot_pairs, pilot_tasks = len(PILOT), len(PILOT_TASKS)

    def ci(d):
        return f"{d['diff'] * 100:+.1f} pp [{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}]".replace("-", "−")

    F = []
    F.append(("Tools + lean2 prompt beat no tools for Flash; tools alone do not.",
              f"E004-GL (grid mode, lean2 prompt, low effort) scores pass@1 {f1(gl['k_pass1'])}% over {gl['samples']} samples against "
              f"{f1(e1['k_pass1'])}% for no-tool E001d ({e1['samples']} samples), paired {ci(d_gl)}, at {fk(gl['tok_mean'])} vs "
              f"{fk(e1['tok_mean'])} tokens/attempt. With the plain ARC Prize prompt the same tool harness drops to {f1(pl['k_pass1'])}% (low) "
              f"and {f1(pm['k_pass1'])}% (max, {fk(pm['tok_mean'])} tokens): the model reasons by hand, calls the tool rarely and runs out of budget. "
              f"pass@2 is {f1(gl['k_pass2'])}% vs {f1(e1['k_pass2'])}% (mean over attempt pairs). All scores in these findings are "
              "Kaggle (pair-weighted) scores."))
    F.append(("The `submit_answer` hand-off is what makes the harness both cheaper and more accurate.",
              f"In E004-GL, {len(na)}/{len(gl_good)} attempts end without a parsable grid and burn {na_share_tok:.0%} of all output tokens. "
              f"With `submit_answer(expr)` the grid is read from the Python session instead of being retyped; in sub80s40 "
              f"{sub_share:.0%} of answers arrive that way. sub80s40 reaches {f1(s80['k_pass1'])}% pass@1 / {f1(s80['k_pass2'])}% pass@2 at "
              f"{fk(s80['tok_mean'])} tokens (paired vs E004-GL {ci(d_of('E009 sub80s40'))}); sub96n reaches {f1(s96['k_pass1'])}% / "
              f"{f1(s96['k_pass2'])}% at {fk(s96['tok_mean'])}."))
    F.append(("Budget notes and soft targets beat hard caps; hard caps cut the long tail.",
              f"A 64k hard cap (sub64a) scores {f1(s64['k_pass1'])}% ({ci(d_of('E009 sub64a'))}); on the {len(longtail)} long-tail pairs "
              f"(E004-GL's correct rollouts average >40k tokens) it solves {f1(s64.get('longtail_acc'))}% against "
              f"{f1(gl['longtail_acc'])}% for E004-GL and {f1(s96.get('longtail_acc'))}% for sub96n. A 40k cap "
              f"(s20x40, 1 incomplete sample) falls to {f1(s2040['k_pass1'])}%. A soft target with a looser cap keeps the tail reachable: "
              f"s20x60 matches E004-GL on pass@2 ({f1(s2060['k_pass2'])}% vs {f1(gl['k_pass2'])}%) at {fk(s2060['tok_mean'])} tokens "
              f"({sg((s2060['tok_mean'] / gl['tok_mean'] - 1) * 100, 0)}%). Thinking-off turns (E010 nt*) save tokens but lose "
              f"{-max(nt_d):.0f}–{-min(nt_d):.0f} pp on dev."))
    v1, v2, v3 = g("E017 V1 summary"), g("E017 V2 lean2roles"), g("E017 V3 all")
    if all(x["complete"] for x in (v1, v2, v3)):
        def d_ref(res, ref):
            return next(d for d in res["paired"] if d["ref"] == ref and not d["cap"])
        def tokd(res):
            return sg((res["tok_mean"] / s80["tok_mean"] - 1) * 100, 0) + "%"
        pp, tc3, tc1, tc2 = v3["policy3"], v3["train_check"], v1["train_check"], v2["train_check"]
        F.append(("E017: a computed perception pack, a roles prompt and session helpers (V3) reach the top tier at fewer tokens; no accuracy change is significant.", [
            f"V3 (all three, harness persistent-v5) scores Kaggle pass@1 {f1(v3['k_pass1'])}% ({' / '.join(f1(x) for x in v3['k_pass1_each'])}) over "
            f"{v3['samples']} samples against {f1(s80['k_pass1'])}% for sub80s40 (paired {ci(d_ref(v3, 'E009 sub80s40'))}) at "
            f"{fk(v3['tok_mean'])} tokens/attempt ({tokd(v3)}); pass@2 {f1(v3['k_pass2'])}% vs {f1(s80['k_pass2'])}%. Single samples: "
            f"V1 (summary only) {f1(v1['k_pass1'])}% at {fk(v1['tok_mean'])} ({tokd(v1)}, shorter correct rollouts: median "
            f"{fk(v1['ctok_median'])} vs {fk(s80['ctok_median'])}); V2 (lean2roles only) {f1(v2['k_pass1'])}% at {fk(v2['tok_mean'])} ({tokd(v2)}). "
            f"One sample swings by several pp (sub80s40's own samples span {f1(min(s80['k_pass1_each']))}–{f1(max(s80['k_pass1_each']))}%), "
            "so these are 'no harm, slightly fewer tokens' results.",
            f"Conditional 3rd attempt (only when neither of 2 attempts passes the train check): fired on {pp['fired_pairs']} of {len(FULL)} pairs "
            f"and lifts V3's Kaggle pass@2 from {f1(pp['k_pass2_base'])}% to {f1(pp['k_pass2_policy'])}% "
            f"({sg(pp['diff'] * 100)} pp [{sg(pp['lo'] * 100)}, {sg(pp['hi'] * 100)}], {pp['rescued']} pairs rescued) for "
            f"{sg((pp['tok_per_pair_policy'] / pp['tok_per_pair_base'] - 1) * 100, 0)}% tokens per pair and {money(pp['extra_cost'])}; "
            f"task-weighted {f1(pp['pass2_base'])}% → {f1(pp['pass2_policy'])}%. Phase 1 had simulated +2.0 pp [+1.0, +3.2] for +23% tokens [^e017ps].",
            f"The live train check (re-running the answer cell on the training inputs; no model tokens) is a good ranker but a poor gate: "
            f"P(correct | train pass) = {tc3['p_pass']:.0%} ({tc3['pass_correct']}/{tc3['pass']}) on V3's attempts 0–1, {tc1['p_pass']:.0%} for V1, "
            f"{tc2['p_pass']:.0%} for V2 (phase-1 offline estimate 86% [^e017ps]); P(correct | train fail) = {tc3['p_fail']:.0%} on V3. The "
            f"{1 - tc3['p_pass']:.0%} of train-passing answers that are wrong mostly miss something the training pairs do not show.",
        ]))
        F.append(("E017 phase 1 (offline, $0): Flash spends its tokens on prose hypothesis search, and the test is almost always novel.", [
            "Token shares from 48 annotated sub80s40/sub96n traces (population-weighted reader estimates): hypothesis search in prose 51%, "
            "perception (reading or transcribing grids, hand counting) 20%, code writing 11%, everything after the first train pass 4%; readers "
            "judged about 42% of tokens wasted. Attempts whose longest single turn reaches 30k tokens are 26% correct, against 81% under 10k [^e017fa].",
            "In 155 of 167 pairs the test input lies outside the training inputs' range on at least one cheap feature (size, colours, component "
            "count, border contact, object sizes). The largest failure mode is a rule that fits every training pair and is wrong on the test "
            "(12 of 28 failing traces) [^e017fa].",
            "Restarting attempts that have no train pass by 20–50k tokens loses 7–12 pp pass@1 in simulation (slow successes and dead ends look "
            "alike until late), and gating the 2nd attempt on the train check loses 3.8 pp pass@2; only extra attempts where the check fails pay "
            "off [^e017ps].",
        ]))
    if e16:
        tn, um, tt = e16["toolnote"], e16["usermsg"], e16["toolnote_think"]
        F.append(("A \"time is up, submit now\" turn does not rescue cut-off attempts (E016).",
                  f"Of the {tn['n']} E010 s20x60 attempts that hit the 60k cap without an answer, a one-turn thinking-off continuation returns an "
                  f"answer for {tn['answered']} (tool note) or {um['answered']} (user message) and gets {tn['correct']} and {um['correct']} right; with "
                  f"thinking on (8k) it answers {tt['answered']} and gets {tt['correct']} right, which moves s20x60's Kaggle pass@1 from "
                  f"{f1(s2060['k_pass1'])}% to {f1(tt['s2060_k_pass1'])}% and leaves pass@2 at {f1(tt['s2060_k_pass2'])}%. The replayed sessions hold "
                  f"nothing to submit, so the lever for the long tail is budget, not hand-off. Total recorded spend "
                  f"${e16['total_cost_all_rows']:.2f}."))
    held = [(n, a, b) for n, a, b in dvf if a > 0 and b > 0]
    if gains:
        ratio = st.mean(a / b for _, a, b in held) if held else None
        F.append((f"The 56-pair dev set overstated gains" + (f" (about ×{ratio:.0f} for the configs that held up)." if ratio else "."),
                  f"Paired vs E004-GL, dev → full set: " + ", ".join(f"{n} {sg(a * 100)} → {sg(b * 100)}" for n, a, b in dvf) +
                  f" pp. The {len(gains)} configs that beat E004-GL on dev averaged {sg(mean_dev * 100)} pp there and "
                  f"{sg(mean_full * 100)} pp on the full set; sub64a flipped sign. Dev pairs under-represent the long tail of slow but "
                  "solvable pairs, and picking the best of many single-sample dev runs inflates the winner. Only full-set samples are used for decisions."))
    F.append(("Qwen3.8-27B needs its official settings; medium tier / temp 0.6 / 48k breaks it.",
              f"Official settings (tier xhigh, temp 1.0, 240k budget) give pass@1 {f1(qx['k_pass1'])}% / pass@2 {f1(qx['k_pass2'])}% at "
              f"{fk(qx['tok_mean'])} tokens/attempt. Medium tier, temp 0.6 and a 48k budget give {f1(qm['k_pass1'])}% / {f1(qm['k_pass2'])}%: "
              f"{qm_eos:.0%} of attempts emit EOS mid-thought (continued by the harness) and {qm_forced:.0%} are forced to answer. "
              f"Under the official settings mid-thought EOS falls to {qx_eos:.0%} and forced answers to {qx_forced:.1%}. "
              f"ARC Prize reports 42.4% semi-private for this model [^arcprize] (task-weighted; our public task-weighted pass@2 is "
              f"{f1(qx['pass2'])}%)."))
    if e12_d.get(48000) and d_eq:
        F.append(("Tools roughly double Qwen's accuracy at equal tokens.",
                  f"E012 (sub80s40 port, tier medium, 48k hard / 24k soft; {e12['attempts_done']} of 334 attempts done) scores "
                  f"pass@1 {f1(e12['k_pass1'])}% at {fk(e12['tok_mean'])} tokens/attempt. The no-tool xhigh run on the same attempts scores "
                  f"{f1(e12_d[48000]['ref_score'])}% when capped at 48k and {f1(d_eq['ref_score'])}% when capped at {cap_eq // 1000}k "
                  f"(simulated: correct and finished within the cap), and {f1(e12_d[0]['ref_score'])}% uncapped at 240k "
                  f"({fk(qx['tok_mean'])} tokens). Paired vs the {cap_eq // 1000}k cap: {ci(d_eq)}. "
                  f"On dev-56 the same config scores {f1(e12dev['k_pass1'])}% at {fk(e12dev['tok_mean'])}."))
    F.append(("Qwen3.6-35B-A3B and Nemotron are far behind.",
              f"Qwen3.6-35B-A3B: pass@1 {f1(q36['k_pass1'])}%, pass@2 {f1(q36['k_pass2'])}% at {fk(q36['tok_mean'])} tokens, a clean run with "
              f"every attempt closing its thinking on its own. Nemotron 3.5 Lightning: {f1(nemo['k_pass1'])}% ({nemo['n_correct']}/{nemo['n_good']}) at "
              f"{fk(nemo['tok_mean'])} tokens. Nemotron 3 Super 120B is inconclusive (provider stream failures on most pilot attempts)."))
    F.append(("Nemotron failure pattern (40 traces read).",
              "No trace finds the correct rule; every answer is an admitted guess or give-up (copy the test input, constant fill). "
              "It invents time pressure (\"given the time…\") although the prompt sets no limit and quits at 50–95k of 200k tokens; it misreads grid "
              "sizes in about 30/40 traces and never re-checks; its written grids drop cells or whole rows (the source of the jagged answers); "
              "and it tries dozens of one-example ideas without checking any rule on all training pairs [^nemo]."))
    F.append(("Format audit: the parser loses nothing.",
              "Re-parsing 450 Qwen attempts finds 0 incorrect attempts that contain the correct grid anywhere, in the answer or in the thinking; "
              "`\\boxed` is never used [^audit]. The one real output loss is dropping a row while typing a large grid: about 1–2% of attempts "
              "(1 for Qwen3.6, 4 exact + 2 near for Qwen3.8). Recomputed here: attempts without any parsable grid are " +
              ", ".join(f"{k.replace('E008 ', '')} {a}/{b}" for k, (a, b) in unp.items()) +
              f"; the recomputed correct flag matches the stored flag on all {AUDIT['records_checked']} public-eval records "
              f"({AUDIT['flag_mismatch']} mismatches)."))
    # ARC Prize comparison for DeepSeek V4.1 Flash (E001d uses ARC Prize's prompt at max effort, like their Max column)
    e1lat, e1atts = e1["_lat"], e1["attempt_indices"]
    solved = {}
    for a, b in combinations(e1atts, 2):
        solved[(a, b)] = sum(all(e1lat.get((t, p, a), {}).get("_ok") or e1lat.get((t, p, b), {}).get("_ok") for p in range(len(TASKS[t]["test"])))
                             for t in TASKS)
    a01 = solved[(e1atts[0], e1atts[1])]
    nT = len(TASKS)
    ap = ARCPRIZE_V41
    arc_cmp = {
        "title": "ARC Prize vs our E001d: DeepSeek V4.1 Flash, no tools, ARC Prize prompt",
        "rows": [
            ["ARC Prize, verified [^arcprize]", "ARC-AGI-2 semi-private", "Max / High / Low / None effort",
             " / ".join(f"{v:.1f}" for v in (ap["max"], ap["high"], ap["low"], ap["none"])) + "% (task-weighted)"],
            ["ARC Prize per-task table [^arcprize]", "ARC-AGI-2 public eval", "Max effort",
             f"{ap['public_max_tasks']}/{nT} tasks ✓ ({ap['public_max_tasks'] / nT * 100:.1f}%)"],
            ["Ours, E001d", "ARC-AGI-2 public eval", "max effort, attempts 0+1", f"{a01}/{nT} tasks fully solved ({a01 / nT * 100:.1f}%)"],
            ["Ours, E001d", "ARC-AGI-2 public eval", f"max effort, mean over {len(solved)} attempt pairs",
             f"task-weighted pass@2 {f1(e1['pass2'])}%; Kaggle pass@2 {f1(e1['k_pass2'])}%; tasks fully solved "
             + " / ".join(str(v) for v in solved.values()) + f" of {nT} (" + " / ".join(f"a{a}+a{b}" for a, b in solved) + ")"],
            ["Per-task agreement [^arcprize]", "public eval, 120 tasks", "E001d attempt pair vs ARC Prize Max ✓",
             f"{ap['agree_lo']}–{ap['agree_hi']}% of tasks, depending on the attempt pair"],
        ],
        "note": (f"Public − semi-private for V4.1 Flash at Max: {ap['public_max_tasks'] / nT * 100:.1f} − {ap['max']:.1f} = "
                 f"{ap['public_max_tasks'] / nT * 100 - ap['max']:.1f} pp. ARC-AGI-1 semi-private at Max: {ap['arc1_max']:.1f}%."),
    }
    caveats = [
        f"**Public vs semi-private.** All full-set numbers are on the public evaluation set. ARC Prize's verified semi-private ARC-AGI-2 scores "
        f"for DeepSeek V4.1 Flash are {ap['max']:.1f}% (Max), {ap['high']:.1f}% (High), {ap['low']:.1f}% (Low) and {ap['none']:.1f}% (None) "
        f"[^arcprize], task-weighted; their public-eval table at Max marks {ap['public_max_tasks']}/{nT} tasks solved. Our no-tool E001d (same "
        f"prompt, max effort) fully solves {a01}/{nT} tasks ({a01 / nT * 100:.1f}%) with attempts 0+1 (other attempt pairs: "
        + " and ".join(str(v) for k, v in solved.items() if k != (e1atts[0], e1atts[1])) +
        f") and scores {f1(e1['pass2'])}% task-weighted pass@2 averaged over attempt pairs ({f1(e1['k_pass2'])}% Kaggle); per-task agreement "
        f"with ARC Prize's Max column is {ap['agree_lo']}–{ap['agree_hi']}% depending on the attempt pair. So our Flash numbers sit close to "
        f"ARC Prize's public ones and should be read against V4.1's {ap['max']:.1f}% semi-private, not against the older V4 Flash 0731 numbers. "
        f"Qwen3.8-27B: {QWEN_SEMI:.1f}% semi-private "
        f"[^arcprize]; our public pass@2 for Qwen3.8 xhigh is {f1(qx['k_pass2'])}% Kaggle ({f1(qx['pass2'])}% task-weighted). Table below.",
        f"**Contamination.** The public set has been out since 2025, and in E004 traces Flash listed real ARC-AGI-2 eval task ids while trying "
        f"to recall answers [^contam], so some memorisation is possible. But V4.1 Flash's public score at Max is only "
        f"~{ap['public_max_tasks'] / nT * 100 - ap['max']:.0f} pp above its semi-private score, so any inflation of absolute Flash numbers "
        "looks small; comparisons within one model are not affected.",
        f"**Sample noise is a few pp.** E004-GL's {gl['samples']} samples span {f1(min(gl['k_pass1_each']))}–{f1(max(gl['k_pass1_each']))} "
        f"(sd {gl['k_pass1_sd'] * 100:.1f}); E001d's span {f1(min(e1['k_pass1_each']))}–{f1(max(e1['k_pass1_each']))} (sd {e1['k_pass1_sd'] * 100:.1f}); "
        f"across multi-sample DeepSeek runs the per-sample sd is {min(ds_sd):.1f}–{max(ds_sd):.1f} pp. Most paired CIs vs E004-GL "
        "include zero; only the losses (sub64a, s20x40, plain prompt) are clearly significant.",
        f"**Single samples.** Rows with 1 sample (grid-max, cap100, code mode, plain prompt, sub96s40, sub64a, E017 V1/V2, Nemotron) have no pass@2 and "
        f"carry the full sample noise. pilot-26 and dev-56 numbers are noisier still (one test output moves the pilot by "
        f"{100 / pilot_pairs:.1f} pp on the Kaggle metric; one task by {100 / pilot_tasks:.1f} pp task-weighted).",
        "**Budgets differ by harness.** No-tool DeepSeek runs use one 200k call; E004 tool runs a 120k total budget; E009–E011 80–96k; "
        "Qwen gen runs 48k or 240k; E012 48k. Compare tokens/attempt, not caps.",
    ]
    return {"findings": [{"title": t, "body": b} for t, b in F], "caveats": caveats, "arcprize_cmp": arc_cmp}


def longtail_rate(res, longtail):
    idx = index_lat(res["_lat"])
    return st.mean(st.mean(r["_ok"] for r in idx[k].values()) for k in longtail)


# ----------------------------------------------------------------------------------------------- cross-check

CLAIMS = [
    # (run id, metric, claimed value (% or tokens), source)
    ("E001d", "pass1", 59.9, "E004 README / results_summary.md"), ("E001d", "pass2", 74.6, "E004 README"),
    ("E001d", "pass2_a01", 79.3, "E004 README / TSV"), ("E001d", "tok_mean", 55.7e3, "E004 README"),
    ("E001d", "pass1", 63.1, "experiments.tsv E001d headline (attempt 0, pre-retry)"),
    ("E001d", "pass2_a01", 78.1, "experiments.tsv E001d headline (pre-retry)"),
    ("E004-GL", "pass1", 66.0, "E004 README"), ("E004-GL", "pass2", 76.4, "E004 README"), ("E004-GL", "tok_mean", 49.5e3, "E004 README"),
    ("E004 grid-high", "pass1", 68.3, "E004 README"), ("E004 grid-high", "pass2", 78.2, "E004 README"), ("E004 grid-high", "tok_mean", 55.2e3, "E004 README"),
    ("E004 grid-max", "pass1", 65.3, "E004 README"), ("E004 grid-max", "tok_mean", 65.0e3, "E004 README"),
    ("E004 cap100", "pass1", 61.9, "E004 README"), ("E004 cap100", "tok_mean", 45.5e3, "results_summary.md"),
    ("E004 code-high", "pass1", 61.9, "E004 README"), ("E004 code-high", "tok_mean", 57.8e3, "E004 README"),
    ("E004 code-max", "pass1", 60.7, "E004 README"), ("E004 code-max", "tok_mean", 65.2e3, "E004 README"),
    ("E004 plain-low", "pass1", 51.3, "E004 README"), ("E004 plain-low", "tok_mean", 61.0e3, "E004 README"),
    ("E004 plain-max", "pass1", 47.6, "E004 README"), ("E004 plain-max", "tok_mean", 86.2e3, "E004 README"),
    ("E009 sub80s40", "pass1", 67.8, "E009/E011 README"), ("E009 sub80s40", "pass2", 77.5, "E011 README"),
    ("E009 sub80s40", "pass2", 77.4, "E009 README"), ("E009 sub80s40", "tok_mean", 40.7e3, "E011 README"),
    ("E009/E011 sub96n", "pass1", 68.3, "E011 README"), ("E009/E011 sub96n", "pass2", 78.5, "E011 README"), ("E009/E011 sub96n", "tok_mean", 45.7e3, "E011 README"),
    ("E009 sub96s40", "pass1", 67.8, "E009 README"), ("E009 sub96s40", "tok_mean", 42.1e3, "E009 README"),
    ("E009 sub64a", "pass1", 58.2, "E009 README"), ("E009 sub64a", "tok_mean", 38.7e3, "E009 README"),
    ("E010 s20x60", "pass1", 62.4, "E010 README"), ("E010 s20x60", "pass2", 76.5, "E010 README"), ("E010 s20x60", "tok_mean", 32.9e3, "E010 README"),
    ("E010 s25x80", "pass1", 64.7, "E010 README"), ("E010 s25x80", "pass2", 75.3, "E010 README"), ("E010 s25x80", "tok_mean", 35.9e3, "E010 README"),
    ("E010 s20x40", "pass1", 47.8, "E010 README"), ("E010 s20x40", "tok_mean", 29.3e3, "E010 README"),
    ("E011 sub80s40v", "pass1", 66.5, "E011 README"), ("E011 sub80s40v", "pass2", 77.2, "E011 README"), ("E011 sub80s40v", "tok_mean", 41.9e3, "E011 README"),
    ("E011 subv80s40", "pass1", 65.7, "E011 README"), ("E011 subv80s40", "pass2", 76.5, "E011 README"), ("E011 subv80s40", "tok_mean", 41.4e3, "E011 README"),
    ("E011 sub80s40h", "pass1", 66.0, "E011 README"), ("E011 sub80s40h", "pass2", 77.5, "E011 README"), ("E011 sub80s40h", "tok_mean", 47.1e3, "E011 README"),
    ("E008 Qwen xhigh-240k", "pass1", 36.4, "E008 README"), ("E008 Qwen xhigh-240k", "pass2", 48.2, "E008 README"), ("E008 Qwen xhigh-240k", "tok_mean", 73.6e3, "E008 README"),
    ("E008 Qwen medium-48k", "pass1", 9.7, "E008 README"), ("E008 Qwen medium-48k", "pass2", 13.3, "E008 README"), ("E008 Qwen medium-48k", "tok_mean", 39.1e3, "E008 README"),
    ("E008 Qwen3.6-35B", "pass1", 6.2, "E008 README"), ("E008 Qwen3.6-35B", "pass2", 7.5, "E008 README"), ("E008 Qwen3.6-35B", "tok_mean", 53.0e3, "E008 README"),
    ("E008 Qwen3.6-35B", "pass1", 6.1, "results_summary.md (interim snapshot, 203/334)"),
    ("E008 Nemotron Lightning", "pass1", 0.0, "E008 README"), ("E008 Nemotron Lightning", "tok_mean", 68.0e3, "E008 README"),
    ("E008 Qwen OpenRouter pilot", "pass1", 20.0, "E008 README"), ("E008 Qwen OpenRouter pilot", "tok_mean", 35.2e3, "E008 README"),
    ("E012 Qwen tools full", "pass1", 26.4, "experiments.tsv E012 (interim snapshot)"),
    ("E012 Qwen tools full", "tok_mean", 32.8e3, "experiments.tsv E012 (interim snapshot)"),
    ("E009 dev sub96", "pass1", 73.1, "E009 README"), ("E009 dev sub96n", "pass1", 74.1, "E009 README"),
    ("E009 dev brief96n", "pass1", 66.0, "E009 README"), ("E009 dev brief80n", "pass1", 65.0, "E009 README"),
    ("E009 dev brief80a", "pass1", 72.8, "E009 README"), ("E009 dev brief72a", "pass1", 65.0, "E009 README"),
    ("E009 dev brief64a", "pass1", 58.8, "E009 README"), ("E009 dev sub80a", "pass1", 68.7, "E009 README"),
    ("E009 dev sub72a", "pass1", 66.3, "E009 README"), ("E009 dev sub64a", "pass1", 73.1, "E009 README"),
    ("E009 dev sub80s40", "pass1", 71.1, "E009 README"), ("E009 dev sub72s32", "pass1", 75.2, "E009 README"),
    ("E009 dev sub64s32", "pass1", 65.6, "E009 README"), ("E009 dev sub64a", "tok_mean", 37.2e3, "E009 README"),
    ("E009 dev sub80s40", "tok_mean", 38.8e3, "E009 README"),
    ("E010 dev nt0s20x60", "pass1", 27.3, "E010 README"), ("E010 dev nt1s40x80", "pass1", 36.1, "E010 README"),
    ("E010 dev nt3s20x60", "pass1", 40.5, "E010 README"), ("E010 dev s10x40", "pass1", 50.7, "E010 README"),
    ("E010 dev s15x45", "pass1", 48.0, "E010 README"), ("E010 dev s20x40", "pass1", 55.0, "E010 README"),
    ("E010 dev s20x60", "pass1", 60.9, "E010 README"), ("E010 dev s25x80", "pass1", 67.0, "E010 README"),
    ("E017 V1 summary", "pass1", 66.0, "E017 README"), ("E017 V1 summary", "k_pass1", 67.1, "E017 README (pair-weighted)"),
    ("E017 V1 summary", "tok_mean", 37.1e3, "E017 README"),
    ("E017 V2 lean2roles", "pass1", 67.8, "E017 README"), ("E017 V2 lean2roles", "k_pass1", 67.1, "E017 README (pair-weighted)"),
    ("E017 V2 lean2roles", "tok_mean", 42.9e3, "E017 README"),
    ("E017 V3 all", "pass1", 68.3, "E017 README"), ("E017 V3 all", "pass2", 76.0, "E017 README"),
    ("E017 V3 all", "k_pass1", 69.2, "E017 README (pair-weighted)"), ("E017 V3 all", "k_pass2", 77.2, "E017 README (pair-weighted)"),
    ("E017 V3 all", "tok_mean", 39.5e3, "E017 README"),
    ("E001a", "pass1", 52.9, "E001 README"), ("E001a", "pass2", 60.8, "E001 README"), ("E001b", "pass1", 40.8, "E001 README"),
    ("E001b", "pass2", 56.7, "E001 README"), ("E001c", "pass1", 40.0, "E001 README"),
]
STALE = {"experiments.tsv E001d headline (attempt 0, pre-retry)": "predates the 2026-10-03 retry of 17 errored attempts; compare attempt 0 = ",
         "experiments.tsv E001d headline (pre-retry)": "predates the 2026-10-03 retry",
         "results_summary.md (interim snapshot, 203/334)": "interim snapshot; the run has since finished",
         "experiments.tsv E012 (interim snapshot)": "interim snapshot of a running job"}


def crosscheck(R, nv):
    rows = []
    for rid, metric, claimed, src in CLAIMS:
        res = R.get(rid) or nv.get(rid)
        if not res:
            continue
        val = res.get(metric)
        if val is None:
            continue
        if metric.startswith("tok"):
            diff = val - claimed
            flag = abs(diff) > max(1000, 0.02 * claimed)
            comp, cl, df = fk(val), fk(claimed), f"{diff / 1000:+.1f}k"
        else:
            diff = val * 100 - claimed
            flag = abs(diff) > 0.5
            comp, cl, df = f1(val), f"{claimed:.1f}", f"{diff:+.1f}"
        note = STALE.get(src, "")
        if src.startswith("experiments.tsv E001d headline (attempt 0"):
            note += f1(R["E001d"]["pass1_each"][0])
        rows.append({"run": rid, "metric": metric, "claimed": cl, "computed": comp, "diff": df, "source": src, "flag": flag, "note": note})
    return rows


# ----------------------------------------------------------------------------------------------- SVG chart

FAM_CLASS = {DS: "ds", QW: "qw", NM: "nm"}
FAM_SHAPE = {DS: "circle", QW: "square", NM: "triangle"}


def marker(shape, x, y, r, cls, filled=True):
    fill = f'class="mk {cls}{"" if filled else " ring"}"'
    if shape == "circle":
        return f'<circle {fill} cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'
    if shape == "square":
        return f'<rect {fill} x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" rx="1.5"/>'
    h = r * 1.15
    return f'<path {fill} d="M{x:.1f},{y - h:.1f} L{x + h:.1f},{y + h * 0.8:.1f} L{x - h:.1f},{y + h * 0.8:.1f} Z"/>'


def short_label(rid):
    return (rid.replace("E009/E011 ", "").replace("E009 ", "").replace("E010 ", "").replace("E011 ", "").replace("E004 ", "")
            .replace("E008 ", "").replace("Qwen ", "Qwen ").replace("Nemotron Lightning", "Nemotron L."))


def svg_chart(points, title, x_dom, y_dom, x_ticks, y_ticks, width=760, height=440, label_filter=None, box=None, fig_id="fig"):
    ml, mr, mt, mb = 56, 18, 18, 46
    pw, ph = width - ml - mr, height - mt - mb
    X = lambda v: ml + (v - x_dom[0]) / (x_dom[1] - x_dom[0]) * pw
    Y = lambda v: mt + ph - (v - y_dom[0]) / (y_dom[1] - y_dom[0]) * ph
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{fig_id}-t" preserveAspectRatio="xMidYMid meet">',
             f'<title id="{fig_id}-t">{html.escape(title)}</title>']
    for v in y_ticks:
        parts.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        parts.append(f'<text class="tick" x="{ml - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:g}%</text>')
    for v in x_ticks:
        parts.append(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{mt}" y2="{mt + ph}"/>')
        parts.append(f'<text class="tick" x="{X(v):.1f}" y="{mt + ph + 18}" text-anchor="middle">{v / 1000:g}k</text>')
    parts.append(f'<line class="axis" x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}"/>')
    parts.append(f'<line class="axis" x1="{ml}" x2="{ml}" y1="{mt}" y2="{mt + ph}"/>')
    parts.append(f'<text class="axlabel" x="{ml + pw / 2:.1f}" y="{height - 6}" text-anchor="middle">mean output tokens per attempt (all turns)</text>')
    parts.append(f'<text class="axlabel" transform="translate(14,{mt + ph / 2:.1f}) rotate(-90)" text-anchor="middle">Kaggle score (pair-weighted)</text>')
    if box:
        (bx0, by0), (bx1, by1) = box[0], box[1]
        parts.append(f'<rect class="zoombox" x="{X(bx0):.1f}" y="{Y(by1):.1f}" width="{X(bx1) - X(bx0):.1f}" height="{Y(by0) - Y(by1):.1f}"/>')
        parts.append(f'<text class="tick" x="{X(bx0) + 4:.1f}" y="{Y(by1) - 5:.1f}">{html.escape(box[2])}</text>')
    obstacles = []
    marks = []
    for pnt in points:
        x = X(pnt["tok_mean"])
        sh, cls = FAM_SHAPE[pnt["fam"]], FAM_CLASS[pnt["fam"]]
        y1 = Y(pnt["k_pass1"] * 100)
        tip = (f"{pnt['id']}: {fk(pnt['tok_mean'])} tokens/attempt, Kaggle pass@1 {f1(pnt['k_pass1'])}%"
               + (f", Kaggle pass@2 {f1(pnt['k_pass2'])}%" if pnt["k_pass2"] is not None else "") + f", {pnt['samples']} sample(s)"
               + (f"; task-weighted {f1(pnt['pass1'])}" + (f" / {f1(pnt['pass2'])}" if pnt.get("pass2") is not None else "") + "%"
                  if pnt.get("pass1") is not None else ""))
        g = [f'<g class="pt"><title>{html.escape(tip)}</title>']
        if pnt["k_pass2"] is not None:
            y2 = Y(pnt["k_pass2"] * 100)
            g.append(f'<line class="stem {cls}" x1="{x:.1f}" x2="{x:.1f}" y1="{y1:.1f}" y2="{y2:.1f}"/>')
            g.append(marker(sh, x, y2, 4.5, cls, filled=False))
            obstacles.append((x - 6, y2 - 6, x + 6, y2 + 6))
        g.append(marker(sh, x, y1, 4.5, cls))
        g.append(f'<circle class="hit" cx="{x:.1f}" cy="{y1:.1f}" r="11"/>')
        g.append("</g>")
        marks.append("".join(g))
        obstacles.append((x - 6, y1 - 6, x + 6, y1 + 6))
    parts += marks
    # greedy label placement (deterministic), leader line when the label sits away from its point
    placed = []
    fs, cw = 10.5, 6.35
    order = sorted(points, key=lambda p: -p["k_pass1"])
    for pnt in order:
        if label_filter and not label_filter(pnt):
            continue
        lab = short_label(pnt["id"])
        w, h = len(lab) * cw + 2, fs + 2
        x, y = X(pnt["tok_mean"]), Y(pnt["k_pass1"] * 100)
        best = None
        for r in (9, 16, 26, 38, 52, 68, 86):
            for ang in (0, 180, -30, 30, -150, 150, -90, 90, -60, 60, -120, 120):
                dx, dy = r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang))
                cx, cy = x + dx, y + dy
                cos = math.cos(math.radians(ang))
                x0 = cx if cos > 0.3 else (cx - w if cos < -0.3 else cx - w / 2)
                y0 = cy - h / 2
                bb = (x0, y0, x0 + w, y0 + h)
                if bb[0] < ml + 2 or bb[2] > ml + pw - 2 or bb[1] < mt + 1 or bb[3] > mt + ph - 2:
                    continue
                if any(not (bb[2] < o[0] or bb[0] > o[2] or bb[3] < o[1] or bb[1] > o[3]) for o in obstacles + placed):
                    continue
                best = (bb, r, cx, cy)
                break
            if best:
                break
        if not best:
            continue
        bb, r, cx, cy = best
        placed.append((bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1))
        if r > 12:
            lx = min(max(x, bb[0]), bb[2])
            ly = min(max(y, bb[1]), bb[3])
            parts.append(f'<line class="leader" x1="{x:.1f}" y1="{y:.1f}" x2="{lx:.1f}" y2="{ly:.1f}"/>')
        parts.append(f'<text class="plabel" x="{bb[0] + 1:.1f}" y="{bb[3] - 3:.1f}">{html.escape(lab)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------------------------------------- HTML rendering


def gap_line(gaps):
    """One line: Kaggle pass@2 minus task-weighted pass@2 per Table 1 run (pass@1 for single-sample runs), in pp."""
    items = []
    for g_ in gaps:
        if g_["d_pass2"] is not None:
            items.append(f"{g_['id']} {sg(g_['d_pass2'] * 100)}")
        else:
            items.append(f"{g_['id']} n/a (1 sample; pass@1 {sg(g_['d_pass1'] * 100)})")
    two = [g_["d_pass2"] * 100 for g_ in gaps if g_["d_pass2"] is not None]
    return ("Kaggle pass@2 − task-weighted pass@2 (pp): " + "; ".join(items) + ". "
            + (f"Range {sg(min(two))} to {sg(max(two))} pp, mean {sg(st.mean(two))} pp." if two else ""))

FN_RE = re.compile(r"\[\^([a-z0-9-]+)\]")
_FN_SEEN = Counter()  # footnote reference occurrences per render, so every back-reference id is unique


def inline(s, fn_order):
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)

    def fn(m):
        k = m.group(1)
        if k not in fn_order:
            fn_order.append(k)
        n = fn_order.index(k) + 1
        _FN_SEEN[k] += 1
        c = _FN_SEEN[k]
        return f'<sup class="fn"><a href="#fn-{k}" id="fnref-{k}-{n}{"" if c == 1 else "-" + str(c)}">{n}</a></sup>'
    return FN_RE.sub(fn, s)


CSS = r"""
/* Layout: one centred column (max 1280px) with a 20px gutter; prose capped at 68ch, tables full width in scroll boxes. */
:root{
  --bg:#f6f7f8; --surface:#ffffff; --fg:#14181d; --fg2:#3f4852; --muted:#66707b; --rule:#d9dee3; --rule2:#eceff2;
  --zebra:#f3f5f7; --accent:#24557f; --accent-soft:#e3edf6; --best:#0f4c2e; --best-bg:#e2f2e8; --code-bg:#eef1f4;
  --ds:#2a78d6; --qw:#eb6834; --nm:#1baf7a;
  --f-head:"Iowan Old Style","Palatino Linotype",Palatino,Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif;
  --f-body:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue","Noto Sans",Arial,sans-serif;
  --f-mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#111417; --surface:#171b1f; --fg:#e8ebee; --fg2:#c3cad1; --muted:#929ba5; --rule:#2c3238; --rule2:#22272c;
  --zebra:#1b2025; --accent:#8db8e0; --accent-soft:#1d2b38; --best:#9fe0b8; --best-bg:#16301f; --code-bg:#1f252b;
  --ds:#3987e5; --qw:#d95926; --nm:#199e70; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#111417; --surface:#171b1f; --fg:#e8ebee; --fg2:#c3cad1; --muted:#929ba5; --rule:#2c3238; --rule2:#22272c;
  --zebra:#1b2025; --accent:#8db8e0; --accent-soft:#1d2b38; --best:#9fe0b8; --best-bg:#16301f; --code-bg:#1f252b;
  --ds:#3987e5; --qw:#d95926; --nm:#199e70; color-scheme:dark}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 var(--f-body);overflow-wrap:break-word}
.footnotes,.prose,.findings{overflow-wrap:anywhere}
.wrap{max-width:1280px;margin:0 auto;padding-inline:20px;padding-block:28px 64px}
h1,h2,h3{font-family:var(--f-head);font-weight:600;line-height:1.2;text-wrap:balance;color:var(--fg)}
h1{font-size:2.05rem;margin:0 0 .3rem;letter-spacing:-.01em}
h2{font-size:1.45rem;margin:2.6rem 0 .6rem;padding-top:.9rem;border-top:1px solid var(--rule)}
h3{font-size:1.1rem;margin:1.4rem 0 .4rem}
p,li{max-width:68ch}
.prose{max-width:68ch}
.meta{color:var(--muted);font-size:.9rem;display:flex;flex-wrap:wrap;gap:.25rem 1.2rem;margin-bottom:1rem}
.meta span{white-space:nowrap}
code,.mono{font-family:var(--f-mono);font-size:.86em}
code{background:var(--code-bg);padding:.05em .3em;border-radius:3px;overflow-wrap:anywhere}
a{color:var(--accent)}
.convention{background:var(--surface);border:1px solid var(--rule);border-left:3px solid var(--accent);padding:.9rem 1.1rem;margin:1.2rem 0;max-width:86ch}
.convention h3{margin-top:0;font-size:1rem;letter-spacing:.04em;text-transform:uppercase;font-family:var(--f-body);color:var(--fg2)}
.convention ul{margin:.3rem 0 0;padding-left:1.1rem;display:grid;gap:.35rem}
.convention li{max-width:none}
.tbl-note{color:var(--muted);font-size:.86rem;max-width:90ch;margin:.3rem 0 .6rem}
.tbl-wrap{overflow:auto;max-height:82vh;border:1px solid var(--rule);background:var(--surface);border-radius:4px}
table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%;font-size:.84rem;font-variant-numeric:tabular-nums}
th,td{padding:.45rem .6rem;border-bottom:1px solid var(--rule2);vertical-align:top;text-align:left}
thead th{position:sticky;top:0;background:var(--surface);z-index:2;border-bottom:1px solid var(--rule);font-weight:600;color:var(--fg2);white-space:nowrap;box-shadow:0 1px 0 var(--rule)}
tbody tr:nth-child(even) td{background:var(--zebra)}
td.num,th.num{text-align:right;white-space:nowrap}
td.num{font-family:var(--f-mono);font-size:.8rem}
td.num.cost{white-space:normal;min-width:13ch;max-width:24ch}
td.id{font-family:var(--f-mono);font-size:.78rem;white-space:nowrap;font-weight:600}
td.desc{width:34ch;min-width:28ch;max-width:38ch;color:var(--fg2);white-space:normal}
td.notes{width:40ch;min-width:30ch;max-width:44ch;color:var(--fg2);font-size:.8rem;white-space:normal}
td.model{width:20ch;min-width:16ch;max-width:22ch;white-space:normal}
td.cov{min-width:11ch;max-width:18ch;font-family:var(--f-mono);font-size:.76rem;white-space:normal}
td.id,thead th:first-child{position:sticky;left:0;z-index:1;background:var(--surface);box-shadow:1px 0 0 var(--rule2)}
thead th:first-child{z-index:3}
td.id{max-width:24ch;white-space:normal}
td .sub{display:block;color:var(--muted);font-size:.72rem;margin-top:.1rem}
td.best{color:var(--best);font-weight:700;background:var(--best-bg)!important}
th.prim{color:var(--fg)} td.prim{font-size:.84rem;font-weight:600;color:var(--fg)}
th.sec{color:var(--muted);font-weight:500;font-size:.74rem} td.sec{color:var(--muted);font-size:.72rem}
th.sec.first,td.sec.first{border-left:1px solid var(--rule)}
td.sec .sub{font-size:.68rem}
tr.fam-row td{background:var(--accent-soft)!important;font-weight:600;color:var(--fg2);font-size:.78rem;letter-spacing:.05em;text-transform:uppercase}
.badge{display:inline-block;width:.62rem;height:.62rem;border-radius:50%;margin-right:.35rem;vertical-align:baseline}
.badge.ds{background:var(--ds)} .badge.qw{background:var(--qw);border-radius:2px} .badge.nm{background:var(--nm);border-radius:0;clip-path:polygon(50% 0,100% 100%,0 100%)}
th button{all:unset;cursor:pointer;display:inline-flex;gap:.25rem;align-items:center}
th button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
th button .arr{color:var(--muted);font-size:.7em}
th[aria-sort="ascending"] .arr::after{content:"▲"} th[aria-sort="descending"] .arr::after{content:"▼"}
th:not([aria-sort]) .arr::after{content:"↕"}
.findings{display:grid;gap:.9rem;padding:0;list-style:none}
.findings li{max-width:80ch}
.findings ul{margin:.3rem 0 0;padding-left:1.1rem;display:grid;gap:.3rem}
.findings strong.ft{display:block;font-family:var(--f-head);font-size:1.02rem;margin-bottom:.15rem}
figure{margin:1.2rem 0;max-width:900px}
figcaption{color:var(--muted);font-size:.86rem;margin-top:.4rem;max-width:80ch}
.chart-scroll{overflow-x:auto}
.chart{width:100%;min-width:620px;height:auto;display:block;background:var(--surface);border:1px solid var(--rule);border-radius:4px}
.chart .grid{stroke:var(--rule2);stroke-width:1}
.chart .axis{stroke:var(--muted);stroke-width:1}
.chart .tick{fill:var(--muted);font:11px var(--f-body);font-variant-numeric:tabular-nums}
.chart .axlabel{fill:var(--fg2);font:12px var(--f-body)}
.chart .plabel{fill:var(--fg);font:10.5px var(--f-mono)}
.chart .leader{stroke:var(--muted);stroke-width:.8}
.chart .zoombox{fill:none;stroke:var(--muted);stroke-dasharray:4 3}
.chart .mk{stroke:var(--surface);stroke-width:1.5}
.chart .mk.ds{fill:var(--ds)} .chart .mk.qw{fill:var(--qw)} .chart .mk.nm{fill:var(--nm)}
.chart .mk.ring{fill:var(--surface);stroke-width:2}
.chart .mk.ring.ds{stroke:var(--ds)} .chart .mk.ring.qw{stroke:var(--qw)} .chart .mk.ring.nm{stroke:var(--nm)}
.chart .stem{stroke-width:1.5;opacity:.55} .chart .stem.ds{stroke:var(--ds)} .chart .stem.qw{stroke:var(--qw)} .chart .stem.nm{stroke:var(--nm)}
.chart .hit{fill:transparent}
.chart .pt:hover .mk{stroke:var(--fg)}
.legend{display:flex;flex-wrap:wrap;gap:.4rem 1.2rem;font-size:.85rem;color:var(--fg2);margin:.5rem 0}
.legend span{display:inline-flex;align-items:center;gap:.35rem}
.legend svg{width:14px;height:14px}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.6rem 1.2rem;max-width:900px;margin:.8rem 0}
.kv div{border-top:1px solid var(--rule);padding-top:.4rem;min-width:0}
.kv b{display:block;font-family:var(--f-mono);font-size:1.05rem;font-variant-numeric:tabular-nums}
.kv small{color:var(--muted)}
.small-tbl{max-width:900px}
#arcprize{width:100%} #arcprize td{min-width:12ch}
.footnotes{font-size:.86rem;color:var(--fg2)}
.footnotes li{max-width:90ch;margin-bottom:.3rem}
.flag{color:var(--qw);font-weight:600}
@media (max-width:640px){body{font-size:14.5px} h1{font-size:1.6rem} .wrap{padding-inline:16px} .meta span{white-space:normal;min-width:0}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = r"""
(function(){
  document.querySelectorAll('table.sortable').forEach(function(tbl){
    var heads = tbl.querySelectorAll('thead th');
    heads.forEach(function(th, i){
      var btn = th.querySelector('button'); if(!btn) return;
      btn.addEventListener('click', function(){
        var asc = th.getAttribute('aria-sort') !== 'ascending';
        heads.forEach(function(h){ h.removeAttribute('aria-sort'); });
        th.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
        var body = tbl.tBodies[0];
        var rows = Array.prototype.slice.call(body.rows).filter(function(r){ return !r.classList.contains('fam-row'); });
        body.querySelectorAll('tr.fam-row').forEach(function(r){ r.remove(); });
        var num = th.classList.contains('num');
        rows.sort(function(a, b){
          var x = a.cells[i].getAttribute('data-v'), y = b.cells[i].getAttribute('data-v');
          if(num){ x = x === '' || x === null ? NaN : +x; y = y === '' || y === null ? NaN : +y;
            if(isNaN(x) && isNaN(y)) return 0; if(isNaN(x)) return 1; if(isNaN(y)) return -1;
            return asc ? x - y : y - x; }
          x = (x || a.cells[i].textContent).toLowerCase(); y = (y || b.cells[i].textContent).toLowerCase();
          return asc ? x.localeCompare(y) : y.localeCompare(x);
        });
        rows.forEach(function(r){ body.appendChild(r); });
      });
    });
  });
})();
"""


def th(label, num=False, sortable=True, extra=""):
    c = " ".join(x for x in ("num" if num else "", extra) if x)
    cls = f' class="{c}"' if c else ""
    if sortable:
        return f'<th{cls} scope="col"><button type="button">{label}<span class="arr" aria-hidden="true"></span></button></th>'
    return f"<th{cls} scope=\"col\">{label}</th>"


def td(text, cls="", v=None, sub=None, best=False, fn_order=None):
    c = (cls + (" best" if best else "")).strip()
    attr = f' class="{c}"' if c else ""
    if v is not None:
        attr += f' data-v="{v}"'
    body = inline(text, fn_order) if fn_order is not None else html.escape(text)
    if sub:
        body += f'<span class="sub">{html.escape(sub)}</span>'
    return f"<td{attr}>{body}</td>"


def rawv(x):
    return "" if x is None else f"{x:.6g}"


def render_html(data, wrapper=True):
    fo = []  # footnote order
    _FN_SEEN.clear()
    m = data["meta"]
    t1, t2 = data["table1"], data["table2"]
    fam_name = {DS: "DeepSeek", QW: "Qwen", NM: "Nemotron"}
    head = f"<title>{html.escape(m['title'])}</title>\n<style>{CSS}</style>"
    out = ['<div class="wrap">']
    out.append(f"<header><h1>{html.escape(m['title'])}</h1>")
    out.append(f'<div class="meta"><span>{m["date"]}</span><span>Data snapshot {m["generated_utc"]}</span>'
               f'<span>Table 1: {len(t1)} runs</span><span>Table 2: {len(t2)} runs</span>'
               f'<span>Regenerate: <code>{m["command"]}</code></span></div>')
    out.append(f'<p class="prose">{inline(data["scope"], fo)}</p></header>')
    out.append('<section class="convention" aria-labelledby="conv-h"><h3 id="conv-h">Scoring convention</h3><ul>')
    out += [f"<li>{inline(s, fo)}</li>" for s in data["scoring"]]
    out.append("</ul></section>")

    # Table 1
    out.append('<h2 id="t1">Table 1 — complete runs on the full public evaluation set</h2>')
    out.append(f'<p class="tbl-note">Runs that cover all {m["sets"]["full"]} test outputs for every counted sample. Scores in %. '
               '<strong>Kaggle pass@1 / pass@2</strong> (pair-weighted, the competition metric) are primary; the muted ARC Prize columns are the '
               'task-weighted benchmark convention. Grouped by model family, then sorted by Kaggle pass@2 (runs without pass@2 last, by Kaggle '
               'pass@1). Click a header to sort. Shaded cells mark the top tier of a column: Kaggle scores within 3 pp of the column’s best (treated as tied, since one sample swings by ~2–3 pp) and the fewest tokens. Δ values in '
               'Notes are paired Kaggle pass@1 differences in pp with a 95% cluster-bootstrap CI (tasks resampled).</p>')
    out.append('<div class="tbl-wrap"><table class="sortable" id="table1"><thead><tr>')
    out += [th("Experiment ID"), th("LLM model"), th("Harness"), th("Description"), th("Samples"), th("Kaggle pass@1", True, extra="prim"),
            th("Kaggle pass@2", True, extra="prim"), th("ARC Prize pass@1 (task-wt.)", True, extra="sec first"),
            th("ARC Prize pass@2 (task-wt.)", True, extra="sec"),
            th("Tok mean", True), th("Tok median", True), th("Tok p90", True), th("Correct tok median", True), th("Cost", True), th("Notes", sortable=False)]
    out.append("</tr></thead><tbody>")
    cur = None
    for r in t1:
        if r["fam"] != cur:
            cur = r["fam"]
            out.append(f'<tr class="fam-row"><td colspan="15">{fam_name[cur]}</td></tr>')
        raw, b = r["raw"], r["best"]
        out.append("<tr>")
        out.append(f'<td class="id" data-v="{html.escape(r["id"])}"><span class="badge {FAM_CLASS[r["fam"]]}" aria-hidden="true"></span>{html.escape(r["id"])}</td>')
        out.append(td(r["model"], "model"))
        out.append(td(r["harness"]))
        out.append(td(r["desc"], "desc", fn_order=fo))
        out.append(td(r["samples"], "cov", v=raw["samples"]))
        out.append(td(r["k_pass1"], "num prim", v=rawv(raw["k_pass1"]), sub=r["k_pass1_sub"], best=b.get("k_pass1")))
        out.append(td(r["k_pass2"], "num prim", v=rawv(raw["k_pass2"]), sub=r["k_pass2_sub"], best=b.get("k_pass2")))
        out.append(td(r["pass1"], "num sec first", v=rawv(raw["pass1"]), sub=r["pass1_sub"]))
        out.append(td(r["pass2"], "num sec", v=rawv(raw["pass2"]), sub=r["pass2_sub"]))
        for col in ("tok_mean", "tok_median", "tok_p90", "ctok_median"):
            out.append(td(r[col], "num", v=rawv(raw[col]), best=b.get(col)))
        out.append(td(r["cost"], "num", v=rawv(raw.get("cost_per_sample")), sub=r.get("cost_sub"), fn_order=fo).replace('<td class="num"', '<td class="num cost"', 1))
        out.append(f'<td class="notes">{"<br>".join(inline(n, fo) for n in r["notes"])}</td>')
        out.append("</tr>")
    out.append("</tbody></table></div>")

    # Table 2
    out.append('<h2 id="t2">Table 2 — smaller sets, probes and partial runs</h2>')
    out.append('<p class="tbl-note">Pilot-26, dev-56, E004a-20, provider probes, stopped or running full-set runs, the NVIDIA-subset runs '
               '(‡ different task set: 120 problems with one test input each, NVIDIA prompt) and the Kaggle commit-run local tests. '
               'Kaggle pass@1 / pass@2 = mean over the covered test outputs (primary); the muted columns are ARC Prize task-weighted over covered tasks. '
               'Paired diff = run minus reference on the test outputs the run covers (Kaggle pass@1, pp, 95% cluster-bootstrap CI over tasks). References: E004-GL (4-sample mean) for '
               'DeepSeek tool runs, E001d for DeepSeek no-tool runs, no-tool Qwen xhigh-240k on the same attempt indices for Qwen runs, '
               'and the same Qwen run capped at the tool run\'s hard cap (simulated: correct and finished within the cap).</p>')
    out.append('<div class="tbl-wrap"><table class="sortable" id="table2"><thead><tr>')
    out += [th("Experiment ID"), th("Model"), th("Harness"), th("Set"), th("Samples / coverage"), th("Kaggle pass@1", True, extra="prim"),
            th("Kaggle pass@2", True, extra="prim"), th("ARC Prize pass@1 (task-wt.)", True, extra="sec first"),
            th("ARC Prize pass@2 (task-wt.)", True, extra="sec"), th("Tok mean / median", True), th("Paired diff vs reference", sortable=False), th("Status / notes", sortable=False)]
    out.append("</tr></thead><tbody>")
    for r in t2:
        raw = r["raw"]
        out.append("<tr>")
        out.append(f'<td class="id" data-v="{html.escape(r["id"])}"><span class="badge {FAM_CLASS[r["fam"]]}" aria-hidden="true"></span>{html.escape(r["id"])}</td>')
        out.append(td(r["model"], "model"))
        out.append(td(r["harness"]))
        out.append(td(r["set"]))
        out.append(td(r["coverage"], "cov"))
        out.append(td(r["k_pass1"], "num prim", v=rawv(raw["k_pass1"])))
        out.append(td(r["k_pass2"], "num prim", v=rawv(raw["k_pass2"]), sub=r.get("k_pass2_sub")))
        out.append(td(r["pass1"], "num sec first", v=rawv(raw.get("pass1"))))
        out.append(td(r["pass2"], "num sec", v=rawv(raw.get("pass2"))))
        out.append(td(r["tok"], "num", v=rawv(raw["tok_mean"])))
        out.append(td(r["paired"], "notes"))
        out.append(td(r["status"], "notes", fn_order=fo))
        out.append("</tr>")
    out.append("</tbody></table></div>")

    # Findings
    out.append('<h2 id="findings">Key findings</h2><p class="tbl-note">Scores are Kaggle (pair-weighted) unless marked task-weighted; '
               'task-weighted values appear only where a finding is compared with ARC Prize\'s published numbers.</p><ul class="findings">')
    for f in data["findings"]:
        body = f["body"]
        body = ("<ul>" + "".join(f"<li>{inline(b, fo)}</li>" for b in body) + "</ul>") if isinstance(body, list) else inline(body, fo)
        out.append(f'<li><strong class="ft">{inline(f["title"], fo)}</strong>{body}</li>')
    out.append("</ul>")

    # Chart
    pts = data["chart"]
    out.append('<h2 id="chart">Tokens vs accuracy</h2>')
    out.append('<div class="legend" aria-label="Legend">'
               '<span><svg viewBox="0 0 14 14"><circle cx="7" cy="7" r="5" style="fill:var(--ds)"/></svg>DeepSeek V4.1-Flash</span>'
               '<span><svg viewBox="0 0 14 14"><rect x="2" y="2" width="10" height="10" rx="1.5" style="fill:var(--qw)"/></svg>Qwen</span>'
               '<span><svg viewBox="0 0 14 14"><path d="M7,1.5 L12.5,12 L1.5,12 Z" style="fill:var(--nm)"/></svg>Nemotron</span>'
               '<span><svg viewBox="0 0 14 14"><circle cx="7" cy="7" r="4.5" style="fill:var(--surface);stroke:var(--fg2)" stroke-width="2"/></svg>Kaggle pass@2 (runs with ≥2 samples)</span>'
               '<span><svg viewBox="0 0 14 14"><circle cx="7" cy="7" r="5" style="fill:var(--fg2)"/></svg>Kaggle pass@1</span></div>')
    ds_pts = [p for p in pts if p["fam"] == DS]
    xmax = max(p["tok_mean"] for p in pts)
    xhi = 10000 * (int(xmax / 10000) + 1)
    svg1 = svg_chart(pts, "All Table 1 runs: Kaggle pass@1 (filled) and Kaggle pass@2 (ring) vs mean output tokens per attempt",
                     (30000, xhi), (0, 85), list(range(30000, xhi + 1, 10000)), [0, 20, 40, 60, 80],
                     label_filter=lambda p: p["fam"] != DS or p["id"] in ("E001d", "E004-GL", "E004 plain-max"), fig_id="fig1")
    out.append(f'<figure><div class="chart-scroll">{svg1}</div><figcaption>Fig. 1. Every Table 1 run, y = Kaggle score (pair-weighted, '
               'solved test outputs / 167). Filled marker = Kaggle pass@1 (mean over samples); ring = Kaggle pass@2 (mean over attempt pairs), joined by a stem. Hover a marker for its values. x and y are linear and start at 30k tokens / 0%. '
               'The unlabelled DeepSeek cluster is enlarged in Fig. 2.</figcaption></figure>')
    dmin = min(p["k_pass1"] for p in ds_pts) * 100
    ylo = 5 * int((dmin - 2) / 5)
    svg2 = svg_chart(ds_pts, "DeepSeek V4.1-Flash runs only, enlarged: Kaggle pass@1 and pass@2 vs tokens", (30000, xhi), (ylo, 82), list(range(30000, xhi + 1, 10000)),
                     list(range(ylo, 81, 5)), height=520, fig_id="fig2")
    out.append(f'<figure><div class="chart-scroll">{svg2}</div><figcaption>Fig. 2. DeepSeek runs only, y = Kaggle score from {ylo}%. '
               f'{html.escape(data["pareto"]["text"])}</figcaption></figure>')

    # Kaggle
    k = data["kaggle"]
    out.append('<h2 id="kaggle">Kaggle feasibility</h2>')
    out.append(kaggle_html(k, fo, data["kaggle_text"]))

    # Caveats
    out.append('<h2 id="caveats">Caveats</h2><ul>')
    out += [f"<li>{inline(c, fo)}</li>" for c in data["caveats"]]
    out.append("</ul>")
    ac = data["arcprize_cmp"]
    out.append(f'<h3 id="arcprize-cmp">{html.escape(ac["title"])}</h3>')
    out.append('<div class="tbl-wrap small-tbl"><table id="arcprize"><thead><tr><th scope="col">Source</th><th scope="col">Set</th>'
               '<th scope="col">Setting</th><th scope="col">Score</th></tr></thead><tbody>')
    for row in ac["rows"]:
        out.append("<tr>" + "".join("<td>" + inline(c, fo) + "</td>" for c in row) + "</tr>")
    out.append(f'</tbody></table></div><p class="tbl-note">{inline(ac["note"], fo)}</p>')

    # Appendix
    cc = data["crosscheck"]
    flagged = [c for c in cc if c["flag"]]
    out.append('<h2 id="appendix">Appendix — cross-check against READMEs and the registry</h2>')
    out.append(f'<p class="prose">{len(cc)} README / registry numbers were compared with the recomputed values. {len(flagged)} differ by more than '
               '0.5 pt (scores) or more than 1k and 2% (tokens). Everything else agrees within those limits.</p>')
    out.append('<div class="tbl-wrap small-tbl"><table id="xcheck"><thead><tr><th scope="col">Run</th><th scope="col">Metric</th>'
               '<th class="num" scope="col">README</th><th class="num" scope="col">Recomputed</th><th class="num" scope="col">Diff</th>'
               '<th scope="col">Source</th><th scope="col">Explanation</th></tr></thead><tbody>')
    for c in flagged:
        out.append(f'<tr><td class="id">{html.escape(c["run"])}</td><td>{c["metric"]}</td><td class="num">{c["claimed"]}</td>'
                   f'<td class="num">{c["computed"]}</td><td class="num flag">{c["diff"]}</td><td>{html.escape(c["source"])}</td>'
                   f'<td>{html.escape(c["note"] or "unexplained: check the README")}</td></tr>')
    out.append("</tbody></table></div>")
    out.append(f'<p class="prose">The cross-check compares task-weighted values (k_* rows: Kaggle pair-weighted, where the source quotes both), because the READMEs and the registry quote ARC Prize\'s '
               'task-weighted convention.</p>')
    out.append(f'<h3 id="metric-gap">Kaggle minus task-weighted, per Table 1 run</h3><p class="prose">{html.escape(gap_line(data["metric_gap"]))}</p>')
    a = m["audit"]
    out.append(f'<p class="tbl-note">Input audit: {len(a["files"])} result files read; recomputed correctness matches the stored flag on '
               f'{a["records_checked"] - a["flag_mismatch"]}/{a["records_checked"]} public-eval records. Eval processes running at build time: '
               f'{", ".join(m["running_tags"]) or "none"}.</p>')

    # Footnotes
    out.append('<h2 id="notes">Notes</h2><ol class="footnotes">')
    for key in fo:
        out.append(f'<li id="fn-{key}">{html.escape(data["footnotes"][key])}</li>')
    out.append("</ol></div>")
    out.append(f"<script>{JS}</script>")
    body = "\n".join(out)
    if not wrapper:
        return head + "\n" + body + "\n"
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f"{head}\n</head>\n<body>\n{body}\n</body>\n</html>\n")


def kaggle_html(k, fo, kt):
    b3 = k["bench"].get("bench-mtp3-v3", {})
    b0 = k["bench"].get("bench-mtp0-v3", {})
    v2 = k.get("v2-v2", {})
    s = []
    s.append('<p class="prose">Target: the Kaggle rerun on 4×L4 (24 GB, SM89), 12 h, no internet, 240 tasks / 259 test inputs. '
             'Measured on Kaggle with Qwen3.8-27B-FP8 (outputs in <code>kaggle/runs/</code>):</p><div class="kv">')
    s.append(f'<div><b>{b3.get("decode_tok_s", 0):.0f} tok/s</b><small>decode, MTP=3, {b3.get("concurrency")} seqs at {b3.get("ctx_mean", 0) / 1000:.0f}k context (vLLM 0.19)</small></div>')
    s.append(f'<div><b>{b0.get("decode_tok_s", 0):.0f} tok/s</b><small>same without MTP</small></div>')
    s.append(f'<div><b>{b3.get("kv_tokens", 0) / 1000:.0f}k → {v2.get("kv_tokens", 0) / 1e6:.2f}M</b><small>KV tokens, vLLM 0.19 → 0.23 (prefix caching on)</small></div>')
    s.append(f'<div><b>{v2.get("concurrency")} seqs</b><small>concurrency in the v2 commit run (KV / 40k)</small></div>')
    s.append(f'<div><b>{(v2.get("phases_s") or {}).get("server_ready", 0) / 60:.1f} min</b><small>notebook start → server ready (v2 commit run)</small></div>')
    s.append("</div>")
    s.append('<h3>Budget per attempt</h3><p class="prose">Generated tokens = rate × usable time (force-answer deadline at 11 h minus server start-up), '
             'divided by 259 pairs (one attempt each) or 518 attempts (two each).</p>')
    s.append('<div class="tbl-wrap small-tbl"><table><thead><tr><th scope="col">Throughput assumption</th><th class="num" scope="col">tok/s</th>'
             '<th class="num" scope="col">Total</th><th class="num" scope="col">Per pair (1 attempt)</th><th class="num" scope="col">Per attempt (2 attempts)</th></tr></thead><tbody>')
    for r in k["budget"]:
        s.append(f'<tr><td>{inline(r["rate_name"], fo)}</td><td class="num">{r["tok_s"]:.0f}</td><td class="num">{r["total_M"]:.1f}M</td>'
                 f'<td class="num">{fk(r["per_pair_1att"])}</td><td class="num">{fk(r["per_att_2att"])}</td></tr>')
    s.append("</tbody></table></div>")
    s.append('<h3>What that budget buys (no tools, Qwen3.8 xhigh, simulated caps)</h3><p class="prose">From the full xhigh-240k run: an attempt '
             'counts only if it was correct and finished within the cap. Kaggle score (pair-weighted) primary; ARC Prize task-weighted muted.</p>')
    s.append('<div class="tbl-wrap small-tbl"><table><thead><tr><th class="num" scope="col">Cap</th><th class="num" scope="col">Finished within cap</th>'
             '<th class="num prim" scope="col">Kaggle pass@1</th><th class="num prim" scope="col">Kaggle pass@2</th>'
             '<th class="num sec first" scope="col">Task-wt. pass@1</th><th class="num sec" scope="col">Task-wt. pass@2</th></tr></thead><tbody>')
    for r in k["qwen_cap_table"]:
        s.append(f'<tr><td class="num">{str(r["cap"] // 1000) + "k" if r["cap"] else "240k budget (uncapped)"}</td><td class="num">{r["finished"]:.0%}</td>'
                 f'<td class="num prim">{f1(r["k_pass1"])}</td><td class="num prim">{f1(r["k_pass2"])}</td>'
                 f'<td class="num sec first">{f1(r["pass1"])}</td><td class="num sec">{f1(r["pass2"])}</td></tr>')
    s.append("</tbody></table></div>")
    s.append(f'<ul><li>{inline(kt["tools"], fo)}</li>')
    subs = k.get("submissions") or []
    for sub in subs:
        s.append(f'<li><strong>Submitted config:</strong> {html.escape(sub["description"])} — submission {sub["id"]}, submitted '
                 f'{sub["submitted_at"][:16].replace("T", " ")} UTC, status {sub["status"]} at {sub["observed_at"][:16].replace("T", " ")} UTC'
                 f'{", public score " + str(sub["public_score"]) if sub.get("public_score") is not None else ""}. '
                 f'{inline(kt["submitted"], fo)}</li>')
    s.append(f'<li>{inline(kt["prefix"], fo)}</li><li>{inline(kt["local"], fo)}</li></ul>')
    return "".join(s)


KAGGLE_TEXT = {
    "tools": "**Why tools for Kaggle.** At ~30–37k tokens per pair, no-tool Qwen sits on the steep part of its cap curve (table above), "
             "while the E012 tool agent gets twice the no-tool accuracy at the same spend: about two-thirds of its attempts finish under the 48k cap, and every correct answer is handed over by `submit_answer`.",
    "submitted": "Config (kaggle/README.md, v2): E012 tool agent (`agent_qwen.solve`, qwen-tools-3), tier medium, 24k soft target, hard cap "
                 "min(48k, adaptive per-pair share), budget notes on every tool output, temp 1.0 / top_p 0.95 / top_k 20, LocalExecutor, vLLM 0.23 "
                 "from our own wheel source, force answers at 11h00m, hard stop 11h30m, emergency writer 11h42m, notebook guard 11h45m (all from the first cell).",
    "prefix": "**Prefix-caching rule** (AGENTS.md): tool use re-sends the whole growing transcript each turn, so prefix caching (vLLM "
              "`--enable-prefix-caching` / SGLang radix cache) is mandatory; never benchmark, tune or submit a tool config with it off, size "
              "concurrency by cache health (hit rate, evictions, preemptions) and not only KV capacity, and log those metrics during real runs. "
              "On vLLM 0.19 prefix caching put the hybrid model in Mamba `align` mode and cut KV to ~304k tokens; vLLM 0.23 reports 1.40M.",
    "local": "The commit runs only validate the pipeline: the local test is deadline-forced about 6 minutes after the server is up, so its "
             "scores of {v2} (tools) and {pl} (plain) solved test outputs ({v2t} and {plt} task-weighted) say nothing about accuracy (Table 2).",
}

# ----------------------------------------------------------------------------------------------- Markdown rendering


def md_cell(s):
    return (s or "").replace("|", "\\|").replace("\n", " ")


def render_md(data):
    m = data["meta"]
    fo = []
    extra_fn = []  # (key, text) generated footnotes for long notes

    def md(s):
        for k in FN_RE.findall(s):
            if k not in fo:
                fo.append(k)
        return s

    def best(text, flag):
        return f"**{text}**" if flag and text not in ("—", "") else text

    L = [f"# {m['title']}", "", f"{m['date']} · data snapshot {m['generated_utc']} · regenerate with `{m['command']}`", "",
         md(data["scope"]), "", "## Scoring convention", ""]
    L += [f"- {md(s)}" for s in data["scoring"]]
    L += ["", f"## Table 1 — complete runs on the full public evaluation set ({len(data['table1'])} runs)", "",
          "Runs covering all 167 test outputs for every counted sample. Scores in %. **Kaggle pass@1 / pass@2** (pair-weighted, the competition "
          "metric) are primary; *ARC Prize task-weighted* columns (italic) are secondary. Grouped by model family and sorted by Kaggle pass@2 "
          "(runs without pass@2 last). **Bold** = top tier of the column: Kaggle scores within 3 pp of the best (treated as tied; one sample swings by ~2–3 pp) and the fewest tokens. Δ in notes = paired Kaggle "
          "pass@1 difference in pp, 95% cluster-bootstrap CI (tasks resampled).", "",
          "| Experiment ID | LLM model | Harness | Description | Samples | Kaggle pass@1 | Kaggle pass@2 | *ARC Prize pass@1 (task-wt.)* | *ARC Prize pass@2 (task-wt.)* | Tok mean | Tok median | Tok p90 | Correct tok median | Cost | Notes |",
          "|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
    for i, r in enumerate(data["table1"]):
        b = r["best"]
        p1 = best(r["k_pass1"], b.get("k_pass1")) + (f"<br><sub>{r['k_pass1_sub']}</sub>" if r["k_pass1_sub"] else "")
        p2 = best(r["k_pass2"], b.get("k_pass2")) + (f"<br><sub>{r['k_pass2_sub']}</sub>" if r["k_pass2_sub"] else "")
        it = lambda x: f"*{x}*" if x not in ("—", "") else x
        tp1 = it(r["pass1"]) + (f"<br><sub>{r['pass1_sub']}</sub>" if r["pass1_sub"] else "")
        tp2 = it(r["pass2"]) + (f"<br><sub>{r['pass2_sub']}</sub>" if r["pass2_sub"] else "")
        key = f"t1-{i + 1}"
        if r["notes"]:
            extra_fn.append((key, " · ".join(r["notes"])))
        desc = r["desc"]
        if len(desc) > 90:
            dkey = f"t1d-{i + 1}"
            extra_fn.append((dkey, desc))
            desc = desc[:80].rsplit(" ", 1)[0] + f" … [^{dkey}]"
        L.append("| " + " | ".join(md_cell(x) for x in [
            f"`{r['id']}`", r["model"], r["harness"], md(desc), r["samples"], p1, p2, tp1, tp2, best(r["tok_mean"], b.get("tok_mean")),
            best(r["tok_median"], b.get("tok_median")), best(r["tok_p90"], b.get("tok_p90")), best(r["ctok_median"], b.get("ctok_median")),
            md(r["cost"]) + (f"<br><sub>{r['cost_sub']}</sub>" if r.get("cost_sub") else ""), f"[^{key}]" if r["notes"] else ""]) + " |")
    L += ["", f"## Table 2 — smaller sets, probes and partial runs ({len(data['table2'])} runs)", "",
          "‡ = NVIDIA validation subset (different task set). Kaggle pass@1 / pass@2 = mean over covered test outputs (primary); *task-weighted* "
          "columns are secondary. Paired diff = run minus reference on the test outputs the run covers (Kaggle pass@1, pp, 95% cluster-bootstrap CI over tasks); "
          "references: E004-GL for DeepSeek tools, E001d for DeepSeek no-tool, Qwen xhigh-240k (same attempt indices; also capped at the run's hard cap, simulated) for Qwen.", "",
          "| Experiment ID | Model | Harness | Set | Samples / coverage | Kaggle pass@1 | Kaggle pass@2 | *ARC Prize pass@1 (task-wt.)* | *ARC Prize pass@2 (task-wt.)* | Tok mean / median | Paired diff vs reference | Status / notes |",
          "|---|---|---|---|---|--:|--:|--:|--:|--:|---|---|"]
    for i, r in enumerate(data["table2"]):
        key = f"t2-{i + 1}"
        st_ = r["status"]
        if len(st_) > 70:
            extra_fn.append((key, st_))
            st_ = st_.split(".")[0] + f" [^{key}]"
        pr = r["paired"]
        if len(pr) > 90:
            pk = f"t2p-{i + 1}"
            extra_fn.append((pk, pr))
            pr = pr.split(";")[0] + f" [^{pk}]"
        p2 = r["k_pass2"] + (f"<br><sub>{r['k_pass2_sub']}</sub>" if r.get("k_pass2_sub") else "")
        itx = lambda x: f"*{x}*" if x not in ("—", "") else x
        L.append("| " + " | ".join(md_cell(x) for x in [f"`{r['id']}`", r["model"], r["harness"], r["set"], r["coverage"], r["k_pass1"],
                                                          p2, itx(r["pass1"]), itx(r["pass2"]), r["tok"], md(pr), md(st_)]) + " |")
    L += ["", "## Key findings", "", "Scores are Kaggle (pair-weighted) unless marked task-weighted; task-weighted values appear only where a "
          "finding is compared with ARC Prize's published numbers.", ""]
    for f in data["findings"]:
        if isinstance(f["body"], list):
            L.append(f"- **{md(f['title']).replace('**', '')}**")
            L += [f"  - {md(b)}" for b in f["body"]]
        else:
            L.append(f"- **{md(f['title']).replace('**', '')}** {md(f['body'])}")
    L += ["", "## Tokens vs accuracy", "",
          "Points plotted in the HTML report (Fig. 1–2): every Table 1 run; x = mean output tokens per attempt, y = Kaggle score "
          "(pair-weighted): Kaggle pass@1, and Kaggle pass@2 where ≥2 samples.", "",
          "| Run | Family | Tokens / attempt | Kaggle pass@1 | Kaggle pass@2 | Samples |", "|---|---|--:|--:|--:|--:|"]
    for p in sorted(data["chart"], key=lambda p: p["tok_mean"]):
        L.append(f"| `{p['id']}` | {p['fam']} | {fk(p['tok_mean'])} | {f1(p['k_pass1'])} | {f1(p['k_pass2'])} | {p['samples']} |")
    L += ["", data["pareto"]["text"]]
    k = data["kaggle"]
    b3, b0, v2 = k["bench"].get("bench-mtp3-v3", {}), k["bench"].get("bench-mtp0-v3", {}), k.get("v2-v2", {})
    L += ["", "## Kaggle feasibility", "",
          "Target: Kaggle rerun on 4×L4 (24 GB, SM89), 12 h, no internet, 240 tasks / 259 test inputs. Measured on Kaggle with Qwen3.8-27B-FP8 (`kaggle/runs/`):", "",
          f"- Decode {b3.get('decode_tok_s', 0):.0f} tok/s with MTP=3 ({b0.get('decode_tok_s', 0):.0f} without), {b3.get('concurrency')} sequences at "
          f"{b3.get('ctx_mean', 0) / 1000:.0f}k context, vLLM 0.19.",
          f"- KV cache {b3.get('kv_tokens', 0) / 1000:.0f}k tokens on vLLM 0.19 → {v2.get('kv_tokens', 0) / 1e6:.2f}M on vLLM 0.23 (prefix caching on); "
          f"v2 commit run concurrency {v2.get('concurrency')}, server ready {(v2.get('phases_s') or {}).get('server_ready', 0) / 60:.1f} min after notebook start.", "",
          "| Throughput assumption | tok/s | Total | Per pair (1 attempt) | Per attempt (2 attempts) |", "|---|--:|--:|--:|--:|"]
    for r in k["budget"]:
        L.append(f"| {md(r['rate_name'])} | {r['tok_s']:.0f} | {r['total_M']:.1f}M | {fk(r['per_pair_1att'])} | {fk(r['per_att_2att'])} |")
    L += ["", "No-tool Qwen3.8 xhigh at simulated caps (correct and finished within the cap); Kaggle score primary, task-weighted secondary:", "",
          "| Cap | Finished within cap | Kaggle pass@1 | Kaggle pass@2 | *Task-wt. pass@1* | *Task-wt. pass@2* |", "|--:|--:|--:|--:|--:|--:|"]
    for r in k["qwen_cap_table"]:
        L.append(f"| {str(r['cap'] // 1000) + 'k' if r['cap'] else '240k budget (uncapped)'} | {r['finished']:.0%} | {f1(r['k_pass1'])} | "
                 f"{f1(r['k_pass2'])} | *{f1(r['pass1'])}* | *{f1(r['pass2'])}* |")
    kt = data["kaggle_text"]
    L += ["", f"- {md(kt['tools'])}"]
    for sub in k.get("submissions") or []:
        L.append(f"- **Submitted config:** {sub['description']} — submission {sub['id']}, submitted {sub['submitted_at'][:16].replace('T', ' ')} UTC, "
                 f"status {sub['status']} at {sub['observed_at'][:16].replace('T', ' ')} UTC. {md(kt['submitted'])}")
    L += [f"- {md(kt['prefix'])}", f"- {md(kt['local'])}", "", "## Caveats", ""]
    L += [f"- {md(c)}" for c in data["caveats"]]
    ac = data["arcprize_cmp"]
    L += ["", f"### {ac['title']}", "", "| Source | Set | Setting | Score |", "|---|---|---|---|"]
    L += ["| " + " | ".join(md_cell(md(c)) for c in row) + " |" for row in ac["rows"]]
    L += ["", md(ac["note"])]
    cc = [c for c in data["crosscheck"] if c["flag"]]
    L += ["", "## Appendix — cross-check against READMEs and the registry", "",
          f"{len(data['crosscheck'])} README / registry numbers were compared with the recomputed values; {len(cc)} differ by more than 0.5 pt "
          "(scores) or more than 1k and 2% (tokens). The cross-check compares task-weighted values (k_* rows: Kaggle pair-weighted, where the source quotes both), because the READMEs and the registry "
          "quote ARC Prize's task-weighted convention.", "",
          "| Run | Metric | README | Recomputed | Diff | Source | Explanation |", "|---|---|--:|--:|--:|---|---|"]
    for c in cc:
        L.append(f"| `{c['run']}` | {c['metric']} | {c['claimed']} | {c['computed']} | {c['diff']} | {md_cell(c['source'])} | {md_cell(c['note'] or 'unexplained: check the README')} |")
    a = m["audit"]
    L += ["", "### Kaggle minus task-weighted, per Table 1 run", "", gap_line(data["metric_gap"])]
    L += ["", f"Input audit: {len(a['files'])} result files read; recomputed correctness matches the stored flag on "
              f"{a['records_checked'] - a['flag_mismatch']}/{a['records_checked']} public-eval records. Eval processes running at build time: "
              f"{', '.join(m['running_tags']) or 'none'}.", ""]
    for key in fo:
        if key in data["footnotes"]:
            L.append(f"[^{key}]: {data['footnotes'][key]}")
    for key, text in extra_fn:
        # footnotes referenced from table cells; inner [^x] refs are flattened to keep GitHub happy
        L.append(f"[^{key}]: {FN_RE.sub(lambda mm: '(' + data['footnotes'].get(mm.group(1), mm.group(1)) + ')', text)}")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------------------------- main


def main():
    OUT.mkdir(exist_ok=True)
    data = build()
    (OUT / "report_data.json").write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str))
    data = json.loads((OUT / "report_data.json").read_text())  # render only from the dumped data
    (OUT / "arc_agi2_experiments_report.html").write_text(render_html(data, wrapper=True))
    (OUT / "arc_agi2_experiments_report.artifact.html").write_text(render_html(data, wrapper=False))
    (OUT / "arc_agi2_experiments_report.md").write_text(render_md(data))
    print(f"Table 1: {len(data['table1'])} rows; Table 2: {len(data['table2'])} rows; "
          f"cross-check flags: {sum(c['flag'] for c in data['crosscheck'])}; running: {data['meta']['running_tags']}")
    for p in ("arc_agi2_experiments_report.html", "arc_agi2_experiments_report.artifact.html", "arc_agi2_experiments_report.md", "report_data.json"):
        print(f"  reports/{p}: {(OUT / p).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
