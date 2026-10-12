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
    """ARC Prize task-weighted mean of {(t, p): value}."""
    tasks = defaultdict(list)
    for (t, p), v in pair_vals.items():
        tasks[t].append(v)
    return st.mean(st.mean(v) for v in tasks.values()) if tasks else None


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def bootstrap(diff, seed=0):
    rng = random.Random(seed)
    vals = list(diff.values())
    means = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(BOOT))
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
        each = [tw({(t, p): ok((t, p, a)) for t, p in universe}) for a in atts]
        res["pass1_each"] = each
        res["pass1"] = st.mean(each)
        res["pass1_sd"] = st.stdev(each) if len(each) >= 2 else None
        if len(atts) >= 2:
            combos = list(combinations(atts, 2))
            res["pass2"] = st.mean(tw({(t, p): ok((t, p, a)) or ok((t, p, b)) for t, p in universe}) for a, b in combos)
            res["pass2_a01"] = tw({(t, p): ok((t, p, atts[0])) or ok((t, p, atts[1])) for t, p in universe})
            res["pass2_combos"] = len(combos)
        res["covered_pairs"] = n_univ
        res["covered_tasks"] = len({t for t, _ in universe})
    else:
        by_pair = defaultdict(list)
        for (t, p, a), r in lat.items():
            if not r["_err"]:
                by_pair[(t, p)].append(r["_ok"])
        res["pass1"] = tw({k: st.mean(v) for k, v in by_pair.items()})
        res["covered_pairs"] = len(by_pair)
        res["covered_tasks"] = len({t for t, _ in by_pair})
        both = {k for k in by_pair if len(by_pair[k]) >= 2}
        if len(atts) >= 2 and both:
            res["pass2"] = tw({(t, p): ok((t, p, atts[0])) or ok((t, p, atts[1])) for t, p in both
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
    elif c[0] == "text":
        res["cost_kind"] = "text"
        res["cost_text"] = c[1]
    else:
        res["cost_kind"] = "none"
    if "cost_total" in res:
        res["cost_per_sample"] = res["cost_total"] / n_samples if n_samples else None
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
    """Paired pass@1 difference run - ref over the pairs the run covers (task-weighted, bootstrap over tasks)."""
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
    diff = {t: st.mean(a for a, _ in v) - st.mean(b for _, b in v) for t, v in tasks.items()}
    lo, hi = bootstrap(diff)
    return {"ref": ref["id"], "cap": cap, "pairs": len(run_r), "tasks": len(tasks), "run": tw(run_r), "ref_score": tw(ref_r),
            "diff": st.mean(diff.values()), "lo": lo, "hi": hi}


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
    dict(id="E008 Qwen xhigh-240k", fam=QW, model=QWEN_NV, harness="gen API", set="full", files=["qwen_nvfp4_gen_xhigh_240k_t1.jsonl"],
         desc="ARC Prize prompt, tier xhigh, temp 1.0 / top_p 0.95 / top_k 20 (official), 240k budget in 32k chunks, mid-thought EOS continued, 4k forced final",
         cost=("text", "≈$27 rental [^rent-xhigh]"),
         note="First 52 records are the pilot run (copied in)."),
    dict(id="E008 Qwen medium-48k", fam=QW, model=QWEN_NV, harness="gen API", set="full", files=["qwen_nvfp4_gen_medium_48k.jsonl"],
         desc="ARC Prize prompt, tier medium, temp 0.6 / top_p 0.95 / top_k 20, 48k budget in 16k chunks, EOS continuation, 4k forced final",
         cost=("text", "≈$10 rental [^rent-medium]"), ref="E008 Qwen xhigh-240k", ref_match=True),
    dict(id="E008 Qwen3.6-35B", fam=QW, model=QWEN36, harness="gen API", set="full", files=["nvidia-qwen3-6-35b-a3b-nvfp4_gen_none_240k_t1pp15.jsonl"],
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
    dict(id="E008 Qwen xhigh pilot", fam=QW, model=QWEN_NV, harness="gen API", set="pilot", files=["qwen_nvfp4_gen_xhigh_240k_t1_pilot.jsonl"],
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
    dict(id="E008 Qwen medium smoke", fam=QW, model=QWEN_NV, harness="gen API", set="covered", files=["qwen_nvfp4_gen_medium_48k_smoke.jsonl"],
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
        return [d for d in out if d]

    for spec in all_specs:
        res = results[spec["id"]]
        res["paired"] = paired_for(spec, res)
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
    t1_rows.sort(key=lambda r: (fam_order[r["fam"]], -(r["raw"]["pass2"] if r["raw"]["pass2"] is not None else -1),
                                -(r["raw"]["pass1"] or 0)))
    mark_best(t1_rows, {"pass1": max, "pass2": max, "tok_mean": min, "tok_median": min, "tok_p90": min, "ctok_median": min})

    # ------------------------------------------------------------------ Table 2
    for spec in T1_SPECS:  # incomplete Table-1 candidates go to Table 2
        res = results[spec["id"]]
        if not res["complete"]:
            t2_rows.append(t2_row(spec, res, inprogress=True))
    for spec in T2_SPECS:
        t2_rows.append(t2_row(spec, results[spec["id"]]))
    for spec in NVIDIA_SPECS:
        r = nv[spec["id"]]
        t2_rows.append({
            "id": spec["id"], "fam": spec["fam"], "model": spec["model"], "harness": "no tools",
            "set": f"NVIDIA val subset ‡ ({r['problems']} problems)",
            "coverage": f"{r['samples']} × {r['problems']}/{r['problems']}",
            "pass1": f1(r["pass1"]) + (f" ({' / '.join(f1(x) for x in r['pass1_each'])})" if r["samples"] > 1 else ""),
            "pass2": f1(r.get("pass2")), "tok": f"{fk(r['tok_mean'])} / {fk(r['tok_median'])}", "paired": "— (different task set)",
            "status": f"done. {spec['desc']}. Score uses the stored flag against NVIDIA's expected_output; est. cost {money(r['cost_total'])}; {r['unparsed']} unparsed.",
            "raw": {"pass1": r["pass1"], "pass2": r.get("pass2"), "tok_mean": r["tok_mean"]}})
    kag = kaggle_data()
    for k in kag["local_tests"]:
        t2_rows.append(k)

    # ------------------------------------------------------------------ derived findings
    F = findings(results, specs, nv, longtail, kag)
    chart = [{"id": r["id"], "fam": r["fam"], "tok_mean": r["raw"]["tok_mean"], "pass1": r["raw"]["pass1"], "pass2": r["raw"]["pass2"],
              "samples": r["raw"]["samples"]} for r in t1_rows]
    data = {
        "meta": {"title": "ARC-AGI-2 Experiment Report", "date": REPORT_DATE,
                 "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                 "command": ".venv/bin/python reports/build_report.py",
                 "running_tags": sorted(RUNNING), "audit": AUDIT, "longtail_pairs": len(longtail),
                 "sets": {"full": len(FULL), "pilot": len(PILOT), "dev": len(DEV), "e004": len(E004_SUBSET)}},
        "table1": t1_rows, "table2": t2_rows, "chart": chart, "findings": F["findings"], "kaggle": kag, "caveats": F["caveats"],
        "footnotes": FOOTNOTES, "crosscheck": crosscheck(results, nv), "scope": SCOPE, "scoring": SCORING,
        "runs": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}, "nvidia_runs": nv,
    }
    return data


def coverage_text(res):
    pa = res["per_attempt"]
    if res["complete"]:
        errs = sum(v["errors"] for v in pa.values())
        s = f"{len(pa)} × {res['n_universe']}/{res['n_universe']}"
        if errs:
            s += f" ({errs} API-error attempt{'s' if errs > 1 else ''} scored wrong)"
        return s
    return None


def t1_row(spec, res):
    ns = res["samples"]
    p1 = f1(res["pass1"])
    p1_sub = None
    if ns >= 2:
        p1 = f"{res['pass1'] * 100:.1f} ± {res['pass1_sd'] * 100:.1f}"
        p1_sub = " / ".join(f1(x) for x in res["pass1_each"])
    p2 = f1(res.get("pass2"))
    p2_sub = f"a0+a1 {f1(res.get('pass2_a01'))}; {res['pass2_combos']} pairs" if ns > 2 else None
    if res["cost_kind"] in ("est", "rec"):
        cost = f"{money(res['cost_per_sample'])}/sample ({'est.' if res['cost_kind'] == 'est' else 'rec.'})"
    elif res["cost_kind"] == "text":
        cost = res["cost_text"]
    else:
        cost = "—"
    notes = []
    if spec.get("note"):
        notes.append(spec["note"])
    for d in res.get("paired", []):
        notes.append(f"Δ pass@1 vs {d['ref']}{' capped ' + str(d['cap'] // 1000) + 'k' if d['cap'] else ''}: {fdiff(d)} pp")
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
    return {"id": spec["id"], "fam": spec["fam"], "model": spec["model"], "harness": spec["harness"], "desc": spec["desc"],
            "samples": coverage_text(res), "pass1": p1, "pass1_sub": p1_sub, "pass2": p2, "pass2_sub": p2_sub,
            "tok_mean": fk(res["tok_mean"]), "tok_median": fk(res["tok_median"]), "tok_p90": fk(res["tok_p90"]),
            "ctok_median": fk(res["ctok_median"]), "cost": cost, "notes": notes,
            "raw": {"pass1": res["pass1"], "pass2": res.get("pass2"), "tok_mean": res["tok_mean"], "tok_median": res["tok_median"],
                    "tok_p90": res["tok_p90"], "ctok_median": res["ctok_median"], "samples": ns,
                    "cost_per_sample": res.get("cost_per_sample")}, "best": {}}


def mark_best(rows, rules):
    for col, fn in rules.items():
        vals = [r["raw"][col] for r in rows if r["raw"].get(col) is not None]
        if not vals:
            continue
        b = fn(vals)
        for r in rows:
            r["best"][col] = r["raw"].get(col) == b


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
    p2 = f1(res.get("pass2"))
    p2_sub = f"{res['pass2_pairs']} pairs with 2 attempts" if res.get("pass2_pairs") else None
    return {"id": spec["id"], "fam": spec["fam"], "model": spec["model"], "harness": spec["harness"], "set": setlabel, "coverage": cov,
            "pass1": f1(res["pass1"]) + (f" ({' / '.join(f1(x) for x in res['pass1_each'])})" if res.get("pass1_each") and len(res["pass1_each"]) > 1 else ""),
            "pass2": p2, "pass2_sub": p2_sub, "tok": f"{fk(res['tok_mean'])} / {fk(res['tok_median'])}", "paired": pairedtxt,
            "status": f"{status}. " + "; ".join(extra),
            "raw": {"pass1": res["pass1"], "pass2": res.get("pass2"), "tok_mean": res["tok_mean"]}}


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
        out["local_tests"].append({
            "id": label, "fam": QW, "model": "Qwen3.8-27B FP8, vLLM 0.23 on Kaggle 4×L4", "harness": "tools-grid" if s["agent"] == "tools" else "no tools",
            "set": f"Kaggle local test ({len(tasks)} tasks / {len(pp)} pairs)", "coverage": f"1 × {len(pp)}/{len(pp)}",
            "pass1": f"{s['score']}", "pass2": "—", "tok": f"{fk(st.mean(x['tokens'] for x in pp))} / {fk(st.median(x['tokens'] for x in pp))}",
            "paired": "—",
            "status": f"done (deadline-forced ~6 min after server ready). {s['tool_calls']} tool calls, {s['tool_errors']} tool errors, "
                      f"{s['parse_failures']} parse failures, peak KV {s['kv_usage_max_pct']}%; source kaggle/runs/{name}/summary_local.json",
            "raw": {"pass1": None, "pass2": None, "tok_mean": st.mean(x["tokens"] for x in pp)}})
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
         "train pairs); and token-budget variants of the tool harness (hard caps, budget notes, soft targets, thinking-off turns). "
         "Every number below is recomputed from `evals/results/*.jsonl` and `kaggle/runs/` by `reports/build_report.py`.")

SCORING = [
    "**ARC Prize scoring, task-weighted.** Each task scores the fraction of its test pairs solved; the score is the mean over the 120 tasks.",
    "**Full set** = the official ARC-AGI-2 public evaluation set from github.com/arcprize/ARC-AGI-2 (120 tasks / 167 test pairs). "
    "This is the public evaluation set, not the hidden Kaggle test set (240 tasks / 259 pairs) and not ARC Prize's semi-private set.",
    "**pass@1** = mean over attempts (samples) of the single-attempt score. **pass@2** = a pair counts if either of 2 attempts is correct; "
    "with more than 2 attempts it is the mean over all attempt pairs (attempts 0+1 shown as a0+a1).",
    "**Completeness rule (Table 1).** A run is complete when every counted sample has a record for all 167 pairs. Attempts lost to an API "
    "error are scored wrong; a sample with more than 1 such attempt counts as incomplete and the run goes to Table 2.",
    "**Tokens/attempt** = output (completion) tokens summed over all turns of an attempt, over attempts that returned. "
    "**Cost**: rec. = recorded per-call cost (`est_cost_usd` at off-peak list price incl. input, or OpenRouter's `cost_usd`); "
    "est. = output tokens × list price ($0.60/M Flash, $1.98/M Pro); rental = vast.ai box time from the README (footnoted).",
    "**Other sets.** pilot-26 = the first 15 tasks in sorted id order (26 pairs); dev-56 = E009's stratified dev set "
    "(`experiments/E009_tool_token_efficiency/dev_pairs.txt`); E004a-20 = E004's hard-biased 20-pair subset. Partial runs are scored on "
    "their covered pairs only (a task's score is over its covered pairs; only covered tasks are averaged).",
]

FOOTNOTES = {
    "rent-xhigh": "Rental from experiments/E008_openrouter_models/README.md: about $5.42/h × 5.0 h (pilot 0.8 h + full 4.2 h) ≈ $27. API cost $0.",
    "rent-medium": "Not stated in the README. Derived from the README's wall time (13.07M tokens in 115.8 min) × the box's $5.42/h ≈ $10.",
    "rent-36": "Rental from the E008 README: ≈3.25 h × $5.93/h ≈ $19, including ~0.5 h of server setup.",
    "rent-e012": "Self-hosted on vast.ai instance 55443640 ($5.89/h, E012 README); box time per run is not itemised. API cost $0.",
    "nemo": "Manual reading of 40 traces (experiments/E008_openrouter_models/nemotron_traces/notes_{A,B,C,D}.md), not recomputed.",
    "kag-e2e": "295 tok/s average over 16 min in the end-to-end scheduler (MTP=3, contexts 2–20k) is from experiments/E013_kaggle_submission/README.md; the bench JSON stores only the decode benchmark.",
    "arcprize": "ARC Prize's published scores (V4 Flash 0731: 63.9% public / 61.4% semi-private; Qwen3.8-27B: 42.4% semi-private) are quoted from the E001 and E008 READMEs.",
    "audit": "From experiments/E008_openrouter_models/qwen_format_audit.md (450 Qwen attempts read and re-parsed); not recomputed here.",
    "contam": "From experiments/E004_tool_use/README.md (safety trace review): Flash listed real ARC-AGI-2 eval task ids while trying to recall answers.",
}


def findings(R, specs, nv, longtail, kag):
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
        p1 = st.mean(tw({(t, p): c[(t, p, a)] for t, p in FULL}) for a in (0, 1))
        p2 = tw({(t, p): c[(t, p, 0)] or c[(t, p, 1)] for t, p in FULL})
        fin = st.mean((r["_tok"] <= cap) if cap else bool(r.get("closed_think")) for r in qx["_lat"].values())
        cap_rows.append({"cap": cap, "finished": fin, "pass1": p1, "pass2": p2})
    kag["qwen_cap_table"] = cap_rows
    unp = {k: (R[k]["unparsed"], R[k]["n_good"]) for k in ("E008 Qwen xhigh-240k", "E008 Qwen medium-48k", "E008 Qwen3.6-35B")}

    def ci(d):
        return f"{d['diff'] * 100:+.1f} pp [{d['lo'] * 100:+.1f}, {d['hi'] * 100:+.1f}]".replace("-", "−")

    F = []
    F.append(("Tools + lean2 prompt beat no tools for Flash; tools alone do not.",
              f"E004-GL (grid mode, lean2 prompt, low effort) scores pass@1 {f1(gl['pass1'])}% over {gl['samples']} samples against "
              f"{f1(e1['pass1'])}% for no-tool E001d ({e1['samples']} samples), paired {ci(d_gl)}, at {fk(gl['tok_mean'])} vs "
              f"{fk(e1['tok_mean'])} tokens/attempt. With the plain ARC Prize prompt the same tool harness drops to {f1(pl['pass1'])}% (low) "
              f"and {f1(pm['pass1'])}% (max, {fk(pm['tok_mean'])} tokens): the model reasons by hand, calls the tool rarely and runs out of budget. "
              f"pass@2 is {f1(gl['pass2'])}% vs {f1(e1['pass2'])}% (mean over attempt pairs)."))
    F.append(("The `submit_answer` hand-off is what makes the harness both cheaper and more accurate.",
              f"In E004-GL, {len(na)}/{len(gl_good)} attempts end without a parsable grid and burn {na_share_tok:.0%} of all output tokens. "
              f"With `submit_answer(expr)` the grid is read from the Python session instead of being retyped; in sub80s40 "
              f"{sub_share:.0%} of answers arrive that way. sub80s40 reaches {f1(s80['pass1'])}% pass@1 / {f1(s80['pass2'])}% pass@2 at "
              f"{fk(s80['tok_mean'])} tokens (paired vs E004-GL {ci(d_of('E009 sub80s40'))}); sub96n reaches {f1(s96['pass1'])}% / "
              f"{f1(s96['pass2'])}% at {fk(s96['tok_mean'])}."))
    F.append(("Budget notes and soft targets beat hard caps; hard caps cut the long tail.",
              f"A 64k hard cap (sub64a) scores {f1(s64['pass1'])}% ({ci(d_of('E009 sub64a'))}); on the {len(longtail)} long-tail pairs "
              f"(E004-GL's correct rollouts average >40k tokens) it solves {f1(s64.get('longtail_acc'))}% against "
              f"{f1(gl['longtail_acc'])}% for E004-GL and {f1(s96.get('longtail_acc'))}% for sub96n. A 40k cap "
              f"(s20x40, 1 incomplete sample) falls to {f1(s2040['pass1'])}%. A soft target with a looser cap keeps the tail reachable: "
              f"s20x60 matches E004-GL on pass@2 ({f1(s2060['pass2'])}% vs {f1(gl['pass2'])}%) at {fk(s2060['tok_mean'])} tokens "
              f"({sg((s2060['tok_mean'] / gl['tok_mean'] - 1) * 100, 0)}%). Thinking-off turns (E010 nt*) save tokens but lose 28–45 pp on dev."))
    held = [(n, a, b) for n, a, b in dvf if a > 0 and b > 0]
    if gains:
        ratio = st.mean(a / b for _, a, b in held) if held else None
        F.append((f"The 56-pair dev set overstated gains" + (f" (about ×{ratio:.0f} for the configs that held up)." if ratio else "."),
                  f"Paired vs E004-GL, dev → full set: " + ", ".join(f"{n} {sg(a * 100)} → {sg(b * 100)}" for n, a, b in dvf) +
                  f" pp. The {len(gains)} configs that beat E004-GL on dev averaged {sg(mean_dev * 100)} pp there and "
                  f"{sg(mean_full * 100)} pp on the full set; sub64a flipped sign. Dev pairs under-represent the long tail of slow but "
                  "solvable pairs, and picking the best of many single-sample dev runs inflates the winner. Only full-set samples are used for decisions."))
    F.append(("Qwen3.8-27B needs its official settings; medium tier / temp 0.6 / 48k breaks it.",
              f"Official settings (tier xhigh, temp 1.0, 240k budget) give pass@1 {f1(qx['pass1'])}% / pass@2 {f1(qx['pass2'])}% at "
              f"{fk(qx['tok_mean'])} tokens/attempt. Medium tier, temp 0.6 and a 48k budget give {f1(qm['pass1'])}% / {f1(qm['pass2'])}%: "
              f"{qm_eos:.0%} of attempts emit EOS mid-thought (continued by the harness) and {qm_forced:.0%} are forced to answer. "
              f"Under the official settings mid-thought EOS falls to {qx_eos:.0%} and forced answers to {qx_forced:.1%}. "
              f"ARC Prize reports 42.4% semi-private for this model [^arcprize]."))
    if e12_d.get(48000) and d_eq:
        F.append(("Tools roughly double Qwen's accuracy at equal tokens.",
                  f"E012 (sub80s40 port, tier medium, 48k hard / 24k soft; {e12['attempts_done']} of 334 attempts done) scores "
                  f"pass@1 {f1(e12['pass1'])}% at {fk(e12['tok_mean'])} tokens/attempt. The no-tool xhigh run on the same attempts scores "
                  f"{f1(e12_d[48000]['ref_score'])}% when capped at 48k and {f1(d_eq['ref_score'])}% when capped at {cap_eq // 1000}k "
                  f"(simulated: correct and finished within the cap), and {f1(e12_d[0]['ref_score'])}% uncapped at 240k "
                  f"({fk(qx['tok_mean'])} tokens). Paired vs the {cap_eq // 1000}k cap: {ci(d_eq)}. "
                  f"On dev-56 the same config scores {f1(e12dev['pass1'])}% at {fk(e12dev['tok_mean'])}."))
    F.append(("Qwen3.6-35B-A3B and Nemotron are far behind.",
              f"Qwen3.6-35B-A3B: pass@1 {f1(q36['pass1'])}%, pass@2 {f1(q36['pass2'])}% at {fk(q36['tok_mean'])} tokens, a clean run with "
              f"every attempt closing its thinking on its own. Nemotron 3.5 Lightning: {f1(nemo['pass1'])}% ({nemo['n_correct']}/{nemo['n_good']}) at "
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
    caveats = [
        f"**Public vs semi-private.** All full-set numbers are on the public evaluation set. ARC Prize reports V4 Flash 0731 at 63.9% public / "
        f"61.4% semi-private and Qwen3.8-27B at 42.4% semi-private [^arcprize]; our public pass@2 for Qwen3.8 xhigh is {f1(qx['pass2'])}%.",
        "**Contamination signs for Flash.** The public set predates V4.1; Flash's no-tool pass@2 here is well above ARC Prize's public number for "
        "V4 Flash 0731, and in E004 traces Flash listed real ARC-AGI-2 eval task ids while trying to recall answers [^contam]. Absolute Flash scores "
        "may be inflated; comparisons within one model are not affected.",
        f"**Sample noise is ±3–5 pp.** E004-GL's {gl['samples']} samples span {f1(min(gl['pass1_each']))}–{f1(max(gl['pass1_each']))} "
        f"(sd {gl['pass1_sd'] * 100:.1f}); E001d's span {f1(min(e1['pass1_each']))}–{f1(max(e1['pass1_each']))}. Most paired CIs vs E004-GL "
        "include zero; only the losses (sub64a, s20x40, plain prompt) are clearly significant.",
        "**Single samples.** Rows with 1 sample (grid-max, cap100, code mode, plain prompt, sub96s40, sub64a, Nemotron) have no pass@2 and "
        "carry the full ±3–5 pp noise. pilot-26 and dev-56 numbers are noisier still (one task moves the pilot by 6.7 pp).",
        "**Budgets differ by harness.** No-tool DeepSeek runs use one 200k call; E004 tool runs a 120k total budget; E009–E011 80–96k; "
        "Qwen gen runs 48k or 240k; E012 48k. Compare tokens/attempt, not caps.",
    ]
    return {"findings": [{"title": t, "body": b} for t, b in F], "caveats": caveats}


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
    parts.append(f'<text class="axlabel" transform="translate(14,{mt + ph / 2:.1f}) rotate(-90)" text-anchor="middle">score (ARC Prize, task-weighted)</text>')
    if box:
        (bx0, by0), (bx1, by1) = box[0], box[1]
        parts.append(f'<rect class="zoombox" x="{X(bx0):.1f}" y="{Y(by1):.1f}" width="{X(bx1) - X(bx0):.1f}" height="{Y(by0) - Y(by1):.1f}"/>')
        parts.append(f'<text class="tick" x="{X(bx0) + 4:.1f}" y="{Y(by1) - 5:.1f}">{html.escape(box[2])}</text>')
    obstacles = []
    marks = []
    for pnt in points:
        x = X(pnt["tok_mean"])
        sh, cls = FAM_SHAPE[pnt["fam"]], FAM_CLASS[pnt["fam"]]
        y1 = Y(pnt["pass1"] * 100)
        tip = (f"{pnt['id']}: {fk(pnt['tok_mean'])} tokens/attempt, pass@1 {f1(pnt['pass1'])}%"
               + (f", pass@2 {f1(pnt['pass2'])}%" if pnt["pass2"] is not None else "") + f", {pnt['samples']} sample(s)")
        g = [f'<g class="pt"><title>{html.escape(tip)}</title>']
        if pnt["pass2"] is not None:
            y2 = Y(pnt["pass2"] * 100)
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
    order = sorted(points, key=lambda p: -p["pass1"])
    for pnt in order:
        if label_filter and not label_filter(pnt):
            continue
        lab = short_label(pnt["id"])
        w, h = len(lab) * cw + 2, fs + 2
        x, y = X(pnt["tok_mean"]), Y(pnt["pass1"] * 100)
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

FN_RE = re.compile(r"\[\^([a-z0-9-]+)\]")


def inline(s, fn_order):
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)

    def fn(m):
        k = m.group(1)
        if k not in fn_order:
            fn_order.append(k)
        n = fn_order.index(k) + 1
        return f'<sup class="fn"><a href="#fn-{k}" id="fnref-{k}-{n}">{n}</a></sup>'
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
.footnotes{font-size:.86rem;color:var(--fg2)}
.footnotes li{max-width:90ch;margin-bottom:.3rem}
.flag{color:var(--qw);font-weight:600}
@media (max-width:640px){body{font-size:14.5px} h1{font-size:1.6rem} .wrap{padding-inline:16px}}
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


def th(label, num=False, sortable=True):
    cls = ' class="num"' if num else ""
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
    out.append(f'<p class="tbl-note">Runs that cover all {m["sets"]["full"]} pairs for every counted sample. Grouped by model family, then '
               'sorted by pass@2 (runs without pass@2 last, by pass@1). Click a header to sort. Shaded cells mark the best value in a column '
               '(highest score, fewest tokens). Scores in %. Δ values in Notes are paired pass@1 differences in pp with a 95% bootstrap CI over tasks.</p>')
    out.append('<div class="tbl-wrap"><table class="sortable" id="table1"><thead><tr>')
    out += [th("Experiment ID"), th("LLM model"), th("Harness"), th("Description"), th("Samples"), th("pass@1", True), th("pass@2", True),
            th("Tok mean", True), th("Tok median", True), th("Tok p90", True), th("Correct tok median", True), th("Cost", True), th("Notes", sortable=False)]
    out.append("</tr></thead><tbody>")
    cur = None
    for r in t1:
        if r["fam"] != cur:
            cur = r["fam"]
            out.append(f'<tr class="fam-row"><td colspan="13">{fam_name[cur]}</td></tr>')
        raw, b = r["raw"], r["best"]
        out.append("<tr>")
        out.append(f'<td class="id" data-v="{html.escape(r["id"])}"><span class="badge {FAM_CLASS[r["fam"]]}" aria-hidden="true"></span>{html.escape(r["id"])}</td>')
        out.append(td(r["model"], "model"))
        out.append(td(r["harness"]))
        out.append(td(r["desc"], "desc", fn_order=fo))
        out.append(td(r["samples"], "cov", v=raw["samples"]))
        out.append(td(r["pass1"], "num", v=rawv(raw["pass1"]), sub=r["pass1_sub"], best=b.get("pass1")))
        out.append(td(r["pass2"], "num", v=rawv(raw["pass2"]), sub=r["pass2_sub"], best=b.get("pass2")))
        for col in ("tok_mean", "tok_median", "tok_p90", "ctok_median"):
            out.append(td(r[col], "num", v=rawv(raw[col]), best=b.get(col)))
        out.append(td(r["cost"], "num", v=rawv(raw.get("cost_per_sample")), fn_order=fo))
        out.append(f'<td class="notes">{"<br>".join(inline(n, fo) for n in r["notes"])}</td>')
        out.append("</tr>")
    out.append("</tbody></table></div>")

    # Table 2
    out.append('<h2 id="t2">Table 2 — smaller sets, probes and partial runs</h2>')
    out.append('<p class="tbl-note">Pilot-26, dev-56, E004a-20, provider probes, stopped or running full-set runs, the NVIDIA-subset runs '
               '(‡ different task set: 120 problems with one test input each, NVIDIA prompt) and the Kaggle commit-run local tests. '
               'Paired diff = run minus reference on the pairs the run covers (pass@1, pp, 95% CI). References: E004-GL (4-sample mean) for '
               'DeepSeek tool runs, E001d for DeepSeek no-tool runs, no-tool Qwen xhigh-240k on the same attempt indices for Qwen runs, '
               'and the same Qwen run capped at the tool run\'s hard cap (simulated: correct and finished within the cap).</p>')
    out.append('<div class="tbl-wrap"><table class="sortable" id="table2"><thead><tr>')
    out += [th("Experiment ID"), th("Model"), th("Harness"), th("Set"), th("Samples / coverage"), th("pass@1", True), th("pass@2", True),
            th("Tok mean / median", True), th("Paired diff vs reference", sortable=False), th("Status / notes", sortable=False)]
    out.append("</tr></thead><tbody>")
    for r in t2:
        raw = r["raw"]
        out.append("<tr>")
        out.append(f'<td class="id" data-v="{html.escape(r["id"])}"><span class="badge {FAM_CLASS[r["fam"]]}" aria-hidden="true"></span>{html.escape(r["id"])}</td>')
        out.append(td(r["model"], "model"))
        out.append(td(r["harness"]))
        out.append(td(r["set"]))
        out.append(td(r["coverage"], "cov"))
        out.append(td(r["pass1"], "num", v=rawv(raw["pass1"])))
        out.append(td(r["pass2"], "num", v=rawv(raw["pass2"]), sub=r.get("pass2_sub")))
        out.append(td(r["tok"], "num", v=rawv(raw["tok_mean"])))
        out.append(td(r["paired"], "notes"))
        out.append(td(r["status"], "notes", fn_order=fo))
        out.append("</tr>")
    out.append("</tbody></table></div>")

    # Findings
    out.append('<h2 id="findings">Key findings</h2><ul class="findings">')
    for f in data["findings"]:
        out.append(f'<li><strong class="ft">{inline(f["title"], fo)}</strong>{inline(f["body"], fo)}</li>')
    out.append("</ul>")

    # Chart
    pts = data["chart"]
    out.append('<h2 id="chart">Tokens vs accuracy</h2>')
    out.append('<div class="legend" aria-label="Legend">'
               '<span><svg viewBox="0 0 14 14"><circle cx="7" cy="7" r="5" style="fill:var(--ds)"/></svg>DeepSeek V4.1-Flash</span>'
               '<span><svg viewBox="0 0 14 14"><rect x="2" y="2" width="10" height="10" rx="1.5" style="fill:var(--qw)"/></svg>Qwen</span>'
               '<span><svg viewBox="0 0 14 14"><path d="M7,1.5 L12.5,12 L1.5,12 Z" style="fill:var(--nm)"/></svg>Nemotron</span>'
               '<span><svg viewBox="0 0 14 14"><circle cx="7" cy="7" r="4.5" style="fill:var(--surface);stroke:var(--fg2)" stroke-width="2"/></svg>pass@2 (runs with ≥2 samples)</span>'
               '<span><svg viewBox="0 0 14 14"><circle cx="7" cy="7" r="5" style="fill:var(--fg2)"/></svg>pass@1</span></div>')
    ds_pts = [p for p in pts if p["fam"] == DS]
    xmax = max(p["tok_mean"] for p in pts)
    xhi = 10000 * (int(xmax / 10000) + 1)
    svg1 = svg_chart(pts, "All Table 1 runs: pass@1 (filled) and pass@2 (ring) vs mean output tokens per attempt",
                     (30000, xhi), (0, 85), list(range(30000, xhi + 1, 10000)), [0, 20, 40, 60, 80],
                     label_filter=lambda p: p["fam"] != DS or p["id"] in ("E001d", "E004-GL", "E004 plain-max"), fig_id="fig1")
    out.append(f'<figure><div class="chart-scroll">{svg1}</div><figcaption>Fig. 1. Every Table 1 run. Filled marker = pass@1 (mean over samples); ring = pass@2 '
               '(mean over attempt pairs), joined by a stem. Hover a marker for its values. x and y are linear and start at 30k tokens / 0%. '
               'The unlabelled DeepSeek cluster is enlarged in Fig. 2.</figcaption></figure>')
    dmin = min(p["pass1"] for p in ds_pts) * 100
    ylo = 5 * int((dmin - 2) / 5)
    svg2 = svg_chart(ds_pts, "DeepSeek V4.1-Flash runs only, enlarged", (30000, xhi), (ylo, 82), list(range(30000, xhi + 1, 10000)),
                     list(range(ylo, 81, 5)), height=520, fig_id="fig2")
    front, best2 = [], -1.0
    for p_ in sorted((p_ for p_ in pts if p_["pass2"] is not None), key=lambda p_: p_["tok_mean"]):
        if p_["pass2"] > best2:
            front.append(p_)
            best2 = p_["pass2"]
    ftxt = " → ".join(f"{html.escape(p_['id'])} ({fk(p_['tok_mean'])}, {f1(p_['pass2'])}%)" for p_ in front)
    out.append(f'<figure><div class="chart-scroll">{svg2}</div><figcaption>Fig. 2. DeepSeek runs only, y axis from {ylo}%. pass@2 Pareto front over '
               f'all multi-sample runs (fewer tokens, higher pass@2): {ftxt}.</figcaption></figure>')

    # Kaggle
    k = data["kaggle"]
    out.append('<h2 id="kaggle">Kaggle feasibility</h2>')
    out.append(kaggle_html(k, fo))

    # Caveats
    out.append('<h2 id="caveats">Caveats</h2><ul>')
    out += [f"<li>{inline(c, fo)}</li>" for c in data["caveats"]]
    out.append("</ul>")

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


def kaggle_html(k, fo):
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
             'counts only if it was correct and finished within the cap.</p>')
    s.append('<div class="tbl-wrap small-tbl"><table><thead><tr><th class="num" scope="col">Cap</th><th class="num" scope="col">Finished within cap</th>'
             '<th class="num" scope="col">pass@1</th><th class="num" scope="col">pass@2</th></tr></thead><tbody>')
    for r in k["qwen_cap_table"]:
        s.append(f'<tr><td class="num">{str(r["cap"] // 1000) + "k" if r["cap"] else "240k budget (uncapped)"}</td><td class="num">{r["finished"]:.0%}</td><td class="num">{f1(r["pass1"])}</td><td class="num">{f1(r["pass2"])}</td></tr>')
    s.append("</tbody></table></div>")
    s.append(f'<ul><li>{inline(KAGGLE_TEXT["tools"], fo)}</li>')
    subs = k.get("submissions") or []
    for sub in subs:
        s.append(f'<li><strong>Submitted config:</strong> {html.escape(sub["description"])} — submission {sub["id"]}, submitted '
                 f'{sub["submitted_at"][:16].replace("T", " ")} UTC, status {sub["status"]} at {sub["observed_at"][:16].replace("T", " ")} UTC'
                 f'{", public score " + str(sub["public_score"]) if sub.get("public_score") is not None else ""}. '
                 f'{inline(KAGGLE_TEXT["submitted"], fo)}</li>')
    s.append(f'<li>{inline(KAGGLE_TEXT["prefix"], fo)}</li><li>{inline(KAGGLE_TEXT["local"], fo)}</li></ul>')
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
    "local": "The commit runs only validate the pipeline: the local test is deadline-forced about 6 minutes after the server is up, so its 2/4 "
             "(tools) and 1/4 (plain) task scores say nothing about accuracy (Table 2).",
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
          "Runs covering all 167 pairs for every counted sample, grouped by model family and sorted by pass@2 (runs without pass@2 last). "
          "**Bold** = best value in the column (highest score, fewest tokens). Scores in %; Δ in notes = paired pass@1 difference in pp, 95% bootstrap CI over tasks.", "",
          "| Experiment ID | LLM model | Harness | Description | Samples | pass@1 | pass@2 | Tok mean | Tok median | Tok p90 | Correct tok median | Cost | Notes |",
          "|---|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|---|"]
    for i, r in enumerate(data["table1"]):
        b = r["best"]
        p1 = best(r["pass1"], b.get("pass1")) + (f"<br><sub>{r['pass1_sub']}</sub>" if r["pass1_sub"] else "")
        p2 = best(r["pass2"], b.get("pass2")) + (f"<br><sub>{r['pass2_sub']}</sub>" if r["pass2_sub"] else "")
        key = f"t1-{i + 1}"
        if r["notes"]:
            extra_fn.append((key, " · ".join(r["notes"])))
        desc = r["desc"]
        if len(desc) > 90:
            dkey = f"t1d-{i + 1}"
            extra_fn.append((dkey, desc))
            desc = desc[:80].rsplit(" ", 1)[0] + f" … [^{dkey}]"
        L.append("| " + " | ".join(md_cell(x) for x in [
            f"`{r['id']}`", r["model"], r["harness"], md(desc), r["samples"], p1, p2, best(r["tok_mean"], b.get("tok_mean")),
            best(r["tok_median"], b.get("tok_median")), best(r["tok_p90"], b.get("tok_p90")), best(r["ctok_median"], b.get("ctok_median")),
            md(r["cost"]), f"[^{key}]" if r["notes"] else ""]) + " |")
    L += ["", f"## Table 2 — smaller sets, probes and partial runs ({len(data['table2'])} runs)", "",
          "‡ = NVIDIA validation subset (different task set). Paired diff = run minus reference on the pairs the run covers (pass@1, pp, 95% CI); "
          "references: E004-GL for DeepSeek tools, E001d for DeepSeek no-tool, Qwen xhigh-240k (same attempt indices; also capped at the run's hard cap, simulated) for Qwen.", "",
          "| Experiment ID | Model | Harness | Set | Samples / coverage | pass@1 | pass@2 | Tok mean / median | Paired diff vs reference | Status / notes |",
          "|---|---|---|---|---|--:|--:|--:|---|---|"]
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
        p2 = r["pass2"] + (f"<br><sub>{r['pass2_sub']}</sub>" if r.get("pass2_sub") else "")
        L.append("| " + " | ".join(md_cell(x) for x in [f"`{r['id']}`", r["model"], r["harness"], r["set"], r["coverage"], r["pass1"],
                                                          p2, r["tok"], md(pr), md(st_)]) + " |")
    L += ["", "## Key findings", ""]
    for f in data["findings"]:
        L.append(f"- **{md(f['title']).replace('**', '')}** {md(f['body'])}")
    L += ["", "## Tokens vs accuracy", "",
          "Points plotted in the HTML report (Fig. 1–2): every Table 1 run; x = mean output tokens per attempt, y = pass@1 (and pass@2 where ≥2 samples).", "",
          "| Run | Family | Tokens / attempt | pass@1 | pass@2 | Samples |", "|---|---|--:|--:|--:|--:|"]
    for p in sorted(data["chart"], key=lambda p: p["tok_mean"]):
        L.append(f"| `{p['id']}` | {p['fam']} | {fk(p['tok_mean'])} | {f1(p['pass1'])} | {f1(p['pass2'])} | {p['samples']} |")
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
    L += ["", "No-tool Qwen3.8 xhigh at simulated caps (correct and finished within the cap):", "",
          "| Cap | Finished within cap | pass@1 | pass@2 |", "|--:|--:|--:|--:|"]
    for r in k["qwen_cap_table"]:
        L.append(f"| {str(r['cap'] // 1000) + 'k' if r['cap'] else '240k budget (uncapped)'} | {r['finished']:.0%} | {f1(r['pass1'])} | {f1(r['pass2'])} |")
    L += ["", f"- {md(KAGGLE_TEXT['tools'])}"]
    for sub in k.get("submissions") or []:
        L.append(f"- **Submitted config:** {sub['description']} — submission {sub['id']}, submitted {sub['submitted_at'][:16].replace('T', ' ')} UTC, "
                 f"status {sub['status']} at {sub['observed_at'][:16].replace('T', ' ')} UTC. {md(KAGGLE_TEXT['submitted'])}")
    L += [f"- {md(KAGGLE_TEXT['prefix'])}", f"- {md(KAGGLE_TEXT['local'])}", "", "## Caveats", ""]
    L += [f"- {md(c)}" for c in data["caveats"]]
    cc = [c for c in data["crosscheck"] if c["flag"]]
    L += ["", "## Appendix — cross-check against READMEs and the registry", "",
          f"{len(data['crosscheck'])} README / registry numbers were compared with the recomputed values; {len(cc)} differ by more than 0.5 pt "
          "(scores) or more than 1k and 2% (tokens).", "",
          "| Run | Metric | README | Recomputed | Diff | Source | Explanation |", "|---|---|--:|--:|--:|---|---|"]
    for c in cc:
        L.append(f"| `{c['run']}` | {c['metric']} | {c['claimed']} | {c['computed']} | {c['diff']} | {md_cell(c['source'])} | {md_cell(c['note'] or 'unexplained: check the README')} |")
    a = m["audit"]
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
