#!/usr/bin/env python3
"""E043: self-contained HTML report of the tool-use pruning work, focused on the tool E128 2-bit and E96 2/3-bit models.

Every accuracy, length, turn and tool-error number is recomputed from the harness result files and the saved
reasoning traces in evals/results/; dev screens come from results/tool_nll_tool_dev.jsonl and
results/tool_gen_probe_tool_dev.jsonl; the no-tool table from results/lowbit_summary.json. 4xL4 serving numbers are
E042's measurements (dsv4-infer, dummy weights in the real schemas), entered below in L4 with their sources.

  make_tool_report.py reports/tool_pruning_report.html
"""
import collections
import glob
import gzip
import html
import json
import math
import re
import statistics as st
import sys
from datetime import datetime, timezone
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVR = HERE / "../../../evals/results"
RES = HERE / "../results"
OUT = sys.argv[1]
W = 720

FULL, E128, E96 = "e030_hdrmin", "e043_e128t_w10_u2_gptq_eng64_sub80", "e043_e96t_w10_u2u3_gptq_eng64_sub80"
MT64, FREF = "e043_e96t_w10_u2u3_gptq_eng64_sub80_mt64", "e043_full_ref_sub80"
MODELS = [  # (tag, label, slot, experts x format)
    (FULL, "Full model (DeepSeek API)", 1, "384 x MXFP4 (4.25 bit)"),
    (E128, "Tool E128 2-bit", 2, "128 x uint2 GPTQ (2.19 bit)"),
    (E96, "Tool E96 2/3-bit", 3, "96 x uint2 gate/up + uint3 down (2.53 bit)"),
]
NLL_TAG = {FULL: "full", E128: "e128t_w10_u2_gptq_eng64_sub80", E96: "e96t_w10_u2u3_gptq_eng64_sub80"}

# E042 (dsv4-infer) 4xL4 measurements: dummy weights in these schemas, vLLM fork, util 0.93, prefix caching, 64 GiB tool
# Engram in host RAM. E128 T4 replays its own tool turns; E96 T2 replays the full model's (its own not measured yet).
L4 = {
    FULL: None,
    E128: dict(weights=16.87, kv_gib=2.10, kv_tok=1469523, streams=40, tps=318.6, hits=96.7, kv_peak=0.76,
               shapes="own tool turns (T4)", pre="0 in 60 min"),
    E96: dict(weights=15.18, kv_gib=None, kv_tok=2720000, streams=80, tps=453.5, hits=96.5, kv_peak=0.64,
              shapes="full model's tool turns (T2)", pre="0 in 70 min"),
}
HF = {FULL: "-", E128: "chankhavu/DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit (131.1 GB)",
      E96: "not exported (GPTQ tensors lost with the box's RAM disk; ~5 min to rebuild)"}


def esc(s):
    return html.escape(str(s), quote=True)


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def kfmt(x):
    return f"{x / 1000:.1f}K"


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0


def pfmt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.2f}"


def sign_p(a, b):
    n = a + b
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n) if n else 1.0


# ---------------------------------------------------------------- data
def load(tag, ok_only=False):
    (f,) = glob.glob(str(EVR / f"tools_*_{tag}.jsonl"))
    d = {}
    for line in open(f):
        r = json.loads(line)
        d[(r["task_id"], r["pair"], r["attempt"])] = r
    return {k: r for k, r in d.items() if not ok_only or r.get("stop_reason") != "error"}


def tool_class(o):
    if o.startswith("[tool error:"):
        return "format"
    if o.startswith("[answer not accepted") or "not accepted" in o[:80]:
        return "rejected"
    if "Traceback" in o or "Error" in o[:400]:
        return "python"
    if "timed out" in o.lower() or "timeout" in o.lower()[:200]:
        return "timeout"
    return "ok"


NAME_RE = re.compile(r"NameError: name '(\w+)' is not defined")
MOD_RE = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")
EXC_RE = re.compile(r"\n(\w+(?:Error|Exception)): ")


def traces(rows):
    """Per-turn usage and tool-output classes from the saved transcripts."""
    out = dict(prompts=[], comps=[], cls=collections.Counter(), exc=collections.Counter(),
               names=collections.Counter(), mods=collections.Counter())
    for r in rows.values():
        p = r.get("trace_path")
        if not p or not Path(p).exists():
            continue
        tr = json.load(gzip.open(p))
        for e in tr.get("events", []):
            u = e.get("usage")
            if u:
                out["prompts"].append(u.get("prompt_tokens", 0))
                out["comps"].append(u.get("completion_tokens", 0))
        for m in tr["messages"]:
            if m.get("role") != "tool":
                continue
            o = m.get("content") or ""
            c = tool_class(o)
            out["cls"][c] += 1
            if c == "python":
                ex = EXC_RE.findall(o)
                out["exc"][ex[-1] if ex else "other"] += 1
                out["names"].update(NAME_RE.findall(o)[-1:])
                out["mods"].update(MOD_RE.findall(o)[-1:])
    return out


def task_weighted(rows, attempts=2):
    by = collections.defaultdict(dict)
    for (t, p, a), r in rows.items():
        by[t].setdefault(p, {})[a] = bool(r.get("correct"))
    p1, p2 = [], []
    for t, pairs in by.items():
        if not all(len(v) == attempts for v in pairs.values()):
            continue
        p1.append(st.mean(st.mean(v.values()) for v in pairs.values()))
        p2.append(st.mean(any(v.values()) for v in pairs.values()))
    return len(p1), st.mean(p1), st.mean(p2)


def summarize(tag, ref=None):
    rows = load(tag)
    n_tasks, p1, p2 = task_weighted(rows)
    pairs = collections.defaultdict(list)
    for (t, p, a), r in rows.items():
        pairs[(t, p)].append(bool(r.get("correct")))
    toks = [r["usage"]["completion"] for r in rows.values()]
    turns = [r.get("turns", 0) for r in rows.values()]
    calls = sum(r.get("tool_calls", 0) for r in rows.values())
    stops = collections.Counter(r.get("stop_reason") for r in rows.values())
    s = dict(tag=tag, rows=rows, n=len(rows), tasks=n_tasks, p1=p1, p2=p2,
             pairs2=sum(any(v) for v in pairs.values()), npairs=len(pairs),
             correct=sum(bool(r.get("correct")) for r in rows.values()),
             tok_mean=st.mean(toks), tok_med=st.median(toks), tok_p90=q(toks, 0.9), tok_max=max(toks), toks=toks,
             turn_mean=st.mean(turns), turn_med=st.median(turns), turn_p90=q(turns, 0.9), turns=turns,
             calls=calls, calls_per=calls / len(rows),
             err=sum(r.get("tool_errors", 0) for r in rows.values()) / max(calls, 1),
             rescues=sum(r.get("rescues", 0) or 0 for r in rows.values()),
             budget=stops.get("output_budget", 0), cap=stops.get("max_turns", 0), stops=stops,
             correct_turns=st.mean(r["turns"] for r in rows.values() if r.get("correct")),
             wrong_turns=st.mean(r["turns"] for r in rows.values() if not r.get("correct")))
    tr = traces(rows)
    s.update(turn_prompt_p50=q(tr["prompts"], 0.5), turn_prompt_p90=q(tr["prompts"], 0.9),
             turn_comp_mean=st.mean(tr["comps"]) if tr["comps"] else 0, tr=tr)
    if ref is not None:
        rp = collections.defaultdict(list)
        for (t, p, a), r in ref["rows"].items():
            rp[(t, p)].append(bool(r.get("correct")))
        both = sum(any(pairs[k]) and any(rp[k]) for k in pairs)
        only_ref = sum(any(rp[k]) and not any(pairs[k]) for k in pairs)
        only_me = sum(any(pairs[k]) and not any(rp[k]) for k in pairs)
        neither = len(pairs) - both - only_ref - only_me
        ka = [k for k in rows if k in ref["rows"]]
        a_me = sum(bool(rows[k].get("correct")) and not ref["rows"][k].get("correct") for k in ka)
        a_ref = sum(bool(ref["rows"][k].get("correct")) and not rows[k].get("correct") for k in ka)
        # where the losses are: pairs bucketed by how many of the reference's 2 attempts solved them
        buckets = {}
        for b in (0, 1, 2):
            ks = [k for k in pairs if sum(rp[k]) == b]
            buckets[b] = (len(ks), sum(any(pairs[k]) for k in ks), sum(sum(pairs[k]) for k in ks) / max(2 * len(ks), 1))
        s.update(both=both, only_ref=only_ref, only_me=only_me, neither=neither, a_me=a_me, a_ref=a_ref,
                 a_p=sign_p(a_me, a_ref), buckets=buckets)
    return s


def nll_screen():
    agg = collections.defaultdict(lambda: [0.0, 0, 0.0, 0, 0])
    for line in open(RES / "tool_nll_tool_dev.jsonl"):
        r = json.loads(line)
        x = agg[r["tag"]]
        x[0] += r["nll_gen"]; x[1] += r["n_gen"]; x[2] += r["nll_blk"]; x[3] += r["n_blk"]; x[4] += r["top1_blk"]
    out = {t: dict(nll=x[0] / x[1], blk=x[2] / x[3], blk_top1=x[4] / x[3]) for t, x in agg.items()}
    probe = collections.defaultdict(lambda: [0, 0])
    for line in open(RES / "tool_gen_probe_tool_dev.jsonl"):
        r = json.loads(line)
        probe[r["config"]][0] += bool(r["valid"]); probe[r["config"]][1] += 1
    for t, (v, n) in probe.items():
        out.setdefault(t, {})["probe"] = (v, n)
    return out


def partial(tag, refs):
    rows = load(tag, ok_only=True)
    res = dict(n=len(rows), correct=sum(bool(r.get("correct")) for r in rows.values()), vs=[])
    for name, ref in refs:
        ks = [k for k in rows if k in ref]
        me = sum(bool(rows[k].get("correct")) for k in ks)
        other = sum(bool(ref[k].get("correct")) for k in ks)
        a = sum(bool(rows[k].get("correct")) and not ref[k].get("correct") for k in ks)
        b = sum(bool(ref[k].get("correct")) and not rows[k].get("correct") for k in ks)
        res["vs"].append((name, len(ks), me, other, a, b, sign_p(a, b),
                          st.mean(rows[k]["usage"]["completion"] for k in ks), st.mean(ref[k]["usage"]["completion"] for k in ks),
                          st.mean(rows[k]["turns"] for k in ks), st.mean(ref[k]["turns"] for k in ks)))
    calls = sum(r.get("tool_calls", 0) for r in rows.values())
    res["err"] = sum(r.get("tool_errors", 0) for r in rows.values()) / max(calls, 1)
    res["max_turns"] = max(r["turns"] for r in rows.values())
    return res


def simple_run(tag, attempt0=True):
    rows = load(tag)
    if attempt0:
        rows = {k: r for k, r in rows.items() if k[2] == 0}
    calls = sum(r.get("tool_calls", 0) for r in rows.values())
    return dict(n=len(rows), correct=sum(bool(r.get("correct")) for r in rows.values()),
                err=sum(r.get("tool_errors", 0) for r in rows.values()) / max(calls, 1), calls=calls,
                tok_mean=st.mean(r["usage"]["completion"] for r in rows.values()),
                rescues=sum(r.get("rescues", 0) or 0 for r in rows.values()))


S = {}
S[FULL] = summarize(FULL)
for t in (E128, E96):
    S[t] = summarize(t, ref=S[FULL])
NLL = nll_screen()
FULL_ROWS = S[FULL]["rows"]
P_REF = partial(FREF, [("API full model", FULL_ROWS), ("tool E128 2-bit", S[E128]["rows"]), ("tool E96 2/3-bit, 40 turns", S[E96]["rows"])])
P_MT64 = partial(MT64, [("API full model", FULL_ROWS), ("tool E128 2-bit", S[E128]["rows"]), ("tool E96 2/3-bit, 40 turns", S[E96]["rows"]),
                        ("self-hosted full (partial)", load(FREF, ok_only=True))])
DEVS = [(lab, simple_run(t)) for lab, t in (("temperature 1.0", "e043_devs_e128t_t1"), ("+ min_p 0.05", "e043_devs_e128t_minp05"),
                                            ("top_p 0.95", "e043_devs_e128t_topp95"))]
R1 = [(lab, simple_run(t)) for lab, t in (("Full model (self-hosted)", "e043_full_a1"), ("Full + Engram 64 GiB", "e043_eng64_a1"),
                                          ("E72 4-bit, tool-block x30 + rescue", "e043_e72_w30_eng64_rescue2"),
                                          ("E64 4-bit, tool-block x30 + rescue", "e043_e64_w30_eng64_rescue2"),
                                          ("E48 4-bit, tool-block x30 + rescue", "e043_e48_w30_eng64_rescue2"))]
NT = json.load(open(RES / "lowbit_summary.json"))
for t in (E128, E96):
    L4[t]["att_h"] = L4[t]["tps"] * 3600 / S[t]["tok_mean"]
    L4[t]["h334"] = 334 / L4[t]["att_h"]


# ---------------------------------------------------------------- charts
def hbar(cid, rows, vmax, fmt, label_w=260, unit="", nticks=6, tickfmt=None):
    """rows: [(label, value, tip, css_class)] horizontal bars, value label at the tip."""
    row_h, padr = 34, 70
    h = 16 + row_h * len(rows) + 30
    iw = W - label_w - padr
    s = [f'<svg id="{cid}" class="chart" viewBox="0 0 {W} {h}" role="img" aria-label="{esc(unit)}">']
    for k in range(nticks + 1):
        x = label_w + iw * k / nticks
        s.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="8" y2="{h - 26}"/>')
        s.append(f'<text class="tick" x="{x:.1f}" y="{h - 10}" text-anchor="middle">{esc((tickfmt or fmt)(vmax * k / nticks))}</text>')
    for i, (lab, v, tip, cls) in enumerate(rows):
        y = 14 + i * row_h
        w = max(2, iw * v / vmax)
        s.append(f'<text class="lab" x="{label_w - 10}" y="{y + 15}" text-anchor="end">{esc(lab)}</text>')
        s.append(f'<g class="hit" tabindex="0" data-tip="{esc(tip)}"><rect x="{label_w}" y="{y - 4}" width="{iw}" height="30" '
                 f'fill="transparent"/><path class="{cls}" d="M{label_w},{y} H{label_w + w - 4} Q{label_w + w},{y} '
                 f'{label_w + w},{y + 4} V{y + 18} Q{label_w + w},{y + 22} {label_w + w - 4},{y + 22} H{label_w} Z"/></g>')
        s.append(f'<text class="val" x="{label_w + w + 8}" y="{y + 15}">{esc(fmt(v))}</text>')
    s.append("</svg>")
    return "\n".join(s)


def cdf_lines(cid, series, xmax, xstep, xfmt, xlabel, xticks, ref_x=None, ref_label="", h=300):
    """series: [(name, slot, values)] -> step CDF lines (share of attempts <= x)."""
    padl, padr, padt, padb = 56, 24, 14, 44
    iw, ih = W - padl - padr, h - padt - padb
    sx = lambda v: padl + iw * v / xmax
    sy = lambda v: padt + ih * (1 - v)
    s = [f'<svg id="{cid}" class="chart" viewBox="0 0 {W} {h}" role="img">']
    for k in range(6):
        v = k / 5
        s.append(f'<line class="grid" x1="{padl}" x2="{padl + iw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>')
        s.append(f'<text class="tick" x="{padl - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{100 * v:.0f}%</text>')
    for v in xticks:
        s.append(f'<text class="tick" x="{sx(v):.1f}" y="{padt + ih + 16}" text-anchor="middle">{esc(xfmt(v))}</text>')
    s.append(f'<line class="axis" x1="{padl}" x2="{padl + iw}" y1="{padt + ih}" y2="{padt + ih}"/>')
    if ref_x is not None:
        s.append(f'<line class="ref" x1="{sx(ref_x):.1f}" x2="{sx(ref_x):.1f}" y1="{padt}" y2="{padt + ih}"/>')
        s.append(f'<text class="tick" x="{sx(ref_x) - 6:.1f}" y="{padt + ih - 8}" text-anchor="end">{esc(ref_label)}</text>')
    xs = [xstep * i for i in range(int(xmax / xstep) + 1)]
    for name, slot, vals in series:
        n = len(vals)
        pts = [(x, sum(v <= x for v in vals) / n) for x in xs]
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
        s.append(f'<path class="line s{slot}" d="{d}"/>')
        for x, y in pts[1::max(1, len(pts) // 16)]:
            s.append(f'<g class="hit" tabindex="0" data-tip="{esc(f"{100 * y:.0f}% of attempts|{name}: at most {xfmt(x)}")}">'
                     f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="10" fill="transparent"/>'
                     f'<circle class="dot s{slot}" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5"/></g>')
    s.append(f'<text class="axlabel" x="{padl + iw / 2:.1f}" y="{h - 6}" text-anchor="middle">{esc(xlabel)}</text>')
    s.append("</svg>")
    return "\n".join(s)


def legend(items):
    return '<div class="legend">' + "".join(f'<span class="lg"><span class="key line s{slot}"></span>{esc(n)}</span>'
                                            for n, slot in items) + "</div>"


# ---------------------------------------------------------------- tables
def main_table():
    cols = [S[t] for t, *_ in MODELS]
    hdr = "".join(f'<th class="{"hl" if t != FULL else ""}">{esc(lab)}</th>' for t, lab, *_ in MODELS)
    groups = []

    def row(label, f, note=""):
        cells = "".join(f'<td class="{"hl" if t != FULL else ""}">{f(t, S[t])}</td>' for t, *_ in MODELS)
        return f'<tr><td>{esc(label)}{f"<span class=note> {esc(note)}</span>" if note else ""}</td>{cells}</tr>'

    def grp(title):
        return f'<tr class="grp"><td colspan="{len(MODELS) + 1}">{esc(title)}</td></tr>'

    d = lambda t, s, k: (f'{"+" if s[k] >= 0 else ""}{s[k]:.1f}' if t != FULL else "-")
    groups += [grp("Model"),
               row("Routed experts x format", lambda t, s: esc(next(m[3] for m in MODELS if m[0] == t))),
               row("Engram", lambda t, s: "full table" if t == FULL else "64 GiB tool table (33.9% of rows)"),
               grp("Accuracy (ARC-AGI-2 eval, 120 tasks / 167 pairs x 2 attempts)"),
               row("pass@1, task-weighted", lambda t, s: f"<b>{pct(s['p1'])}</b>"),
               row("pass@2, task-weighted", lambda t, s: f"<b>{pct(s['p2'])}</b>"),
               row("pass@2 vs full, pp", lambda t, s: "-" if t == FULL else f"{100 * (s['p2'] - S[FULL]['p2']):+.1f}"),
               row("Pairs solved (pass@2)", lambda t, s: f"{s['pairs2']} / {s['npairs']}"),
               row("Correct attempts", lambda t, s: f"{s['correct']} / {s['n']}"),
               row("Pairs: both / full only / this only", lambda t, s: "-" if t == FULL else f"{s['both']} / {s['only_ref']} / {s['only_me']}"),
               row("Attempts: this-only vs full-only wins (sign test p)", lambda t, s: "-" if t == FULL else f"{s['a_me']} vs {s['a_ref']} ({pfmt(s['a_p'])})"),
               grp("Output length per attempt (tokens, reasoning + content)"),
               row("Mean", lambda t, s: kfmt(s["tok_mean"])),
               row("Median", lambda t, s: kfmt(s["tok_med"])),
               row("p90", lambda t, s: kfmt(s["tok_p90"])),
               row("Max", lambda t, s: kfmt(s["tok_max"])),
               row("Mean vs full", lambda t, s: "-" if t == FULL else f"{100 * (s['tok_mean'] / S[FULL]['tok_mean'] - 1):+.0f}%"),
               row("Stopped by the 80K output budget", lambda t, s: f"{s['budget']}"),
               grp("Turns and tool use"),
               row("Turns per attempt: mean / median / p90", lambda t, s: f"{s['turn_mean']:.1f} / {s['turn_med']:.0f} / {s['turn_p90']}"),
               row("Turns: correct vs wrong attempts", lambda t, s: f"{s['correct_turns']:.1f} vs {s['wrong_turns']:.1f}"),
               row("Stopped at the 40-turn cap", lambda t, s: f"{s['cap']}"),
               row("Turn prompt p50 / p90 (tokens)", lambda t, s: f"{kfmt(s['turn_prompt_p50'])} / {kfmt(s['turn_prompt_p90'])}"),
               row("Output tokens per turn, mean", lambda t, s: f"{s['turn_comp_mean']:,.0f}"),
               row("Tool calls per attempt", lambda t, s: f"{s['calls_per']:.1f}"),
               row("Tool calls that raised an error", lambda t, s: f"<b>{pct(s['err'])}</b>"),
               row("Rescued tool calls (--rescue-code)", lambda t, s: f"{s['rescues']}" if t != FULL else "n/a"),
               grp("Quality proxies (64 family-disjoint dev tool traces, teacher-forced)"),
               row("Dev NLL (generated tokens)", lambda t, s: f"{NLL[NLL_TAG[t]]['nll']:.4f}"),
               row("Dev NLL vs full", lambda t, s: "-" if t == FULL else f"+{100 * (NLL[NLL_TAG[t]]['nll'] / NLL['full']['nll'] - 1):.1f}%"),
               row("Tool-call-block NLL / top-1", lambda t, s: f"{NLL[NLL_TAG[t]]['blk']:.3f} / {pct(NLL[NLL_TAG[t]]['blk_top1'])}"),
               row("Valid first-turn tool calls (25 dev prompts)",
                   lambda t, s: "25 / 25 (+Engram 64 GiB)" if t == FULL else "{} / {}".format(*NLL[NLL_TAG[t]]["probe"])),
               grp("4xL4 deployment (E042: dummy weights in this schema, util 0.93, prefix caching, TP4)"),
               row("Weights per GPU", lambda t, s: "does not fit (510 GB)" if t == FULL else f"{L4[t]['weights']:.2f} GiB"),
               row("KV cache per GPU", lambda t, s: "-" if t == FULL else
                   (f"{L4[t]['kv_gib']:.2f} GiB = {L4[t]['kv_tok'] / 1e6:.2f}M tokens" if L4[t]["kv_gib"] else f"{L4[t]['kv_tok'] / 1e6:.2f}M tokens")),
               row("Replayed traffic", lambda t, s: "-" if t == FULL else esc(L4[t]["shapes"])),
               row("Concurrent attempts (streams)", lambda t, s: "-" if t == FULL else f"{L4[t]['streams']}"),
               row("Generation throughput", lambda t, s: "-" if t == FULL else f"<b>{L4[t]['tps']:.1f} tok/s</b>"),
               row("Prefix-cache hit rate", lambda t, s: "-" if t == FULL else f"{L4[t]['hits']:.1f}%"),
               row("KV peak / preemptions", lambda t, s: "-" if t == FULL else f"{L4[t]['kv_peak']:.2f} / {L4[t]['pre']}"),
               row("Attempts per hour (at this model's mean length)", lambda t, s: "-" if t == FULL else
                   f"<b>{L4[t]['att_h']:.1f}</b>{'' if t == E128 else ' (est.)'}"),
               row("Hours for a 334-attempt eval", lambda t, s: "-" if t == FULL else f"{L4[t]['h334']:.1f} h"),
               grp("Artifact"),
               row("Hugging Face (private, chankhavu)", lambda t, s: esc(HF[t]))]
    return f'<div class="scroll"><table class="main"><thead><tr><th>Metric</th>{hdr}</tr></thead><tbody>{"".join(groups)}</tbody></table></div>'


def bucket_table():
    hd = "<tr><th>Pairs, by how often the full model solved them</th><th>pairs</th>" + "".join(
        f"<th>{esc(lab)}: pass@2</th><th>{esc(lab)}: attempt accuracy</th>" for t, lab, *_ in MODELS[1:]) + "</tr>"
    names = {2: "full solved 2/2 (easy)", 1: "full solved 1/2 (borderline)", 0: "full solved 0/2 (hard)"}
    body = ""
    for b in (2, 1, 0):
        n = S[E128]["buckets"][b][0]
        cells = "".join(f"<td>{S[t]['buckets'][b][1]} / {n}</td><td>{pct(S[t]['buckets'][b][2], 0)}</td>" for t, *_ in MODELS[1:])
        body += f"<tr><td>{names[b]}</td><td>{n}</td>{cells}</tr>"
    return f'<div class="scroll"><table>{hd}{body}</table></div>'


def typo_table():
    out = []
    for t, lab, *_ in MODELS:
        tr = S[t]["tr"]
        py = tr["cls"]["python"]
        exc = ", ".join(f"{k} {v}" for k, v in tr["exc"].most_common(5))
        names = ", ".join(f"<code>{esc(k)}</code> {v}" for k, v in tr["names"].most_common(8))
        mods = ", ".join(f"<code>{esc(k)}</code> {v}" for k, v in tr["mods"].most_common(6))
        out.append(f"<tr><td>{esc(lab)}</td><td>{py}</td><td>{tr['cls']['format']}</td><td class=l>{esc(exc)}</td>"
                   f"<td class=l>{names or '-'}</td><td class=l>{mods or '-'}</td></tr>")
    return ('<div class="scroll"><table><tr><th>Model</th><th>Python errors</th><th>DSML format errors</th><th class=l>Top exception types</th>'
            '<th class=l>Top undefined names</th><th class=l>Top missing modules</th></tr>' + "".join(out) + "</table></div>")


def dev_screen_table():
    rows = [("full", "Full model"), ("full_eng64", "Full + Engram 64 GiB"),
            ("e128t_w0", "E128 native MXFP4, tool-block weight 0"), ("e128t_w10", "E128 native MXFP4, tool-block weight 10 (chosen)"),
            ("e128t_w30", "E128 native MXFP4, tool-block weight 30"), ("e128t_w10_u2_gptq_eng64_sub80", "<b>Tool E128 2-bit GPTQ (final)</b>"),
            ("e96t_w10", "E96 native MXFP4, tool-block weight 10 (chosen)"), ("e96t_w30", "E96 native MXFP4, tool-block weight 30"),
            ("e96t_w10_u2u3_gptq_eng64_sub80", "<b>Tool E96 2/3-bit GPTQ (final)</b>")]
    base = NLL["full"]["nll"]
    body = "".join(
        f"<tr><td>{lab}</td><td>{NLL[t]['nll']:.4f}</td><td>{'+' if NLL[t]['nll'] >= base else ''}{100 * (NLL[t]['nll'] / base - 1):.1f}%</td>"
        f"<td>{NLL[t]['blk']:.3f}</td><td>{pct(NLL[t]['blk_top1'])}</td><td>{'{} / {}'.format(*NLL[t]['probe']) if 'probe' in NLL[t] else '-'}</td></tr>"
        for t, lab in rows if t in NLL)
    return ('<div class="scroll"><table><tr><th>Configuration (64 GiB tool Engram unless noted)</th><th>dev NLL</th><th>vs full</th>'
            '<th>tool-block NLL</th><th>tool-block top-1</th><th>valid first-turn calls</th></tr>' + body + "</table></div>")


def engram_table():
    rows = [("e128t_u2_eng64_recheck", "64 GiB tool table (shipped)"), ("e128t_u2_eng112", "112 GiB"), ("e128t_u2_fulleng", "full table")]
    return "<table><tr><th>Engram on tool E128 2-bit</th><th>dev NLL</th><th>tool-block NLL</th></tr>" + "".join(
        f"<tr><td>{lab}</td><td>{NLL[t]['nll']:.5f}</td><td>{NLL[t]['blk']:.4f}</td></tr>" for t, lab in rows) + "</table>"


def partial_table(p, title):
    body = "".join(f"<tr><td>{esc(n)}</td><td>{k}</td><td>{me}</td><td>{o}</td><td>{a} vs {b}</td><td>{pfmt(pv)}</td>"
                   f"<td>{kfmt(tm)} vs {kfmt(to)}</td><td>{um:.1f} vs {uo:.1f}</td></tr>"
                   for n, k, me, o, a, b, pv, tm, to, um, uo in p["vs"])
    return (f"<p class=cap>{esc(title)}: {p['n']} of 334 attempts finished, {p['correct']} correct; tool-call error rate "
            f"{pct(p['err'])}; longest attempt {p['max_turns']} turns.</p>"
            '<div class="scroll"><table><tr><th>compared with</th><th>same attempts</th><th>this run correct</th><th>other run correct</th>'
            '<th>this-only vs other-only</th><th>sign test p</th><th>output tokens</th><th>turns</th></tr>' + body + "</table></div>")


def devs_table():
    body = "".join(f"<tr><td>{esc(lab)}</td><td>{r['correct']} / {r['n']}</td><td>{r['calls']}</td><td>{pct(r['err'])}</td></tr>" for lab, r in DEVS)
    return ("<table><tr><th>Sampler (tool E128 2-bit, 1 attempt, 32 dev-family training tasks)</th><th>solved / attempts done</th>"
            "<th>tool calls</th><th>calls with an error</th></tr>" + body + "</table>")


def r1_table():
    body = "".join(f"<tr><td>{esc(lab)}</td><td>{r['correct']} / {r['n']}</td><td>{pct(r['correct'] / r['n'])}</td><td>{pct(r['err'])}</td>"
                   f"<td>{kfmt(r['tok_mean'])}</td></tr>" for lab, r in R1)
    return ("<table><tr><th>First round (s20x60 + header, 1 attempt per pair, self-hosted)</th><th>correct pairs</th><th>pass@1</th>"
            "<th>tool-call errors</th><th>mean output</th></tr>" + body + "</table>")


def notool_table():
    keys = ["full_a1", "e96_u2u3_gptq_eng56", "e128_u2_gptq_eng56", "e72_w10_eng48", "e64_w10_eng48"]
    body = ""
    for k in keys:
        v = NT[k]
        p2 = pct(v["pass2_tw"]) if v.get("pass2_tw") is not None else "-"
        body += (f"<tr><td>{esc(v['label'])}</td><td>{esc(v['format'])}</td><td>{esc(v['engram'])}</td><td>{pct(v['pass1'])}</td><td>{p2}</td>"
                 f"<td>{kfmt(v['tokens_mean'])} / {kfmt(v['tokens_median'])} / {kfmt(v['tokens_p90'])}</td>"
                 f"<td>{'+' if v['dev_nll_vs_full'] > 0 else ''}{100 * v['dev_nll_vs_full']:.1f}%</td><td class=l>{esc(v.get('l4') or '-')}</td></tr>")
    return ('<div class="scroll"><table><tr><th>No-tool model</th><th>experts</th><th>Engram</th><th>pass@1 (attempt 1)</th>'
            '<th>pass@2, task-weighted</th><th>output mean / median / p90</th><th>no-tool dev NLL vs full</th><th class=l>4xL4 (E042)</th></tr>'
            + body + "</table></div>")


# ---------------------------------------------------------------- page
CSS = open(HERE / "make_engram_report.py").read().split("<style>")[1].split("</style>")[0].replace("{{", "{").replace("}}", "}")
EXTRA = """
main { max-width:1040px; }
.lab { fill:var(--ink2); font-size:12px; } .val { fill:var(--ink); font-size:12px; font-variant-numeric:tabular-nums; }
.s1f { fill:var(--s1); } .s2f { fill:var(--s2); } .s3f { fill:var(--s3); } .s0f { fill:var(--axis); }
.hit:hover .s1f, .hit:hover .s2f, .hit:hover .s3f, .hit:hover .s0f { opacity:0.75; }
.ref { stroke:var(--axis); stroke-width:2; stroke-dasharray:4 3; }
table.main td:first-child { min-width:230px; } table.main th { min-width:150px; }
table.main td.hl, table.main th.hl { background:color-mix(in srgb, var(--s2) 7%, transparent); }
table.main th.hl:last-child, table.main td.hl:last-child { background:color-mix(in srgb, var(--s3) 7%, transparent); }
tr.grp td { text-align:left; font-weight:600; color:var(--ink); padding-top:14px; border-bottom:1px solid var(--axis); }
td .note, .note { color:var(--muted); font-size:12px; } td.l, th.l { text-align:left; }
.cap { font-size:13px; color:var(--ink2); margin:12px 0 4px; }
.verdict { border-left:3px solid var(--s2); } .warn { border-left:3px solid var(--axis); }
.tile .val small { display:block; font-size:13px; font-weight:400; color:var(--ink2); margin-top:2px; }
h3 { font-size:15px; margin:20px 0 4px; }
"""


def tile(lab, val, note):
    return f'<div class="card tile"><div class="lab2">{lab}</div><div class="val2">{val}</div><div class="note">{note}</div></div>'


f, a, b = S[FULL], S[E128], S[E96]
err_rows = [("Full model (API)", f["err"], f"{pct(f['err'])}|full model, {f['calls']} tool calls", "s1f"),
            ("Tool E128 2-bit", a["err"], f"{pct(a['err'])}|tool E128 2-bit, {a['calls']} tool calls", "s2f"),
            ("Tool E96 2/3-bit", b["err"], f"{pct(b['err'])}|tool E96 2/3-bit, {b['calls']} tool calls", "s3f")]
for lab, r in R1[2:4]:
    err_rows.append((lab.split(",")[0] + " (round 1)", r["err"], f"{pct(r['err'])}|{lab}; first-round harness (s20x60, 1 attempt)", "s0f"))
items = [(lab, slot) for _, lab, slot, _ in MODELS]
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Tool-use pruning report</title>
<style>{CSS}{EXTRA}</style></head><body><main>
<h1>Pruning DeepSeek-V4.1-Flash for tool-integrated ARC on 4xL4</h1>
<p class="sub">Experiment E043, 2026-10-23 to 2026-10-24. Generated {now} by <code>scripts/make_tool_report.py</code> from the
harness results and reasoning traces. Focus: the tool-use E128 2-bit and E96 2/3-bit models. Earlier reports:
<code>pruning_results.html</code> (first round), <code>engram_census.html</code> (Engram rows).</p>

<div class="tiles">
{tile("Tool E128 2-bit, pass@2", f"{pct(a['p2'])} <small>vs {pct(f['p2'])} full</small>", f"pass@1 {pct(a['p1'])} vs {pct(f['p1'])}; 120 tasks x 2 attempts")}
{tile("Tool E96 2/3-bit, pass@2", f"{pct(b['p2'])} <small>vs {pct(f['p2'])} full</small>", f"pass@1 {pct(b['p1'])}; turn-limited (40-turn cap)")}
{tile("E128 on 4xL4", f"{L4[E128]['att_h']:.1f} <small>attempts/h</small>", f"{L4[E128]['tps']} tok/s at {L4[E128]['streams']} streams, {L4[E128]['hits']}% prefix hits")}
{tile("E128 VRAM per L4", f"{L4[E128]['weights']} <small>GiB weights</small>", f"KV {L4[E128]['kv_gib']:.2f} GiB = {L4[E128]['kv_tok'] / 1e6:.2f}M tokens; Engram in host RAM")}
</div>

<div class="card verdict"><b>Verdict.</b>
<ul>
<li><b>Tool E128 2-bit is the deliverable.</b> {pct(a['p2'])} pass@2 against {pct(f['p2'])} for the full model through the API,
inside the 2-3 pp noise band of a single 2-attempt run, with a third of the routed experts at 2 bits. Same attempts:
{a['a_me']} won only by E128, {a['a_ref']} only by the full model (sign test {pfmt(a['a_p'])}).</li>
<li>Its cost is length: {100 * (a['tok_mean'] / f['tok_mean'] - 1):+.0f}% output tokens and {a['turn_mean'] - f['turn_mean']:+.1f} turns per attempt, mostly from
Python typos ({pct(a['err'])} of tool calls error, against {pct(f['err'])}). On 4xL4 it runs {L4[E128]['att_h']:.1f} attempts/h
({L4[E128]['h334']:.1f} h per 334-attempt eval).</li>
<li><b>Tool E96 2/3-bit is not there yet:</b> {100 * (b['p2'] - f['p2']):+.1f} pp pass@2. {pct(b['err'])} of its tool calls error, and
{b['cap']} attempts hit the 40-turn cap. It would be ~1.6x faster on 4xL4, so its 64-turn rerun (interrupted, see below) decides
whether it is worth it.</li>
<li>The errors are character-level identifier typos from <b>expert pruning, not quantization</b>; truncated sampling does not
remove them. More experts, fewer typos.</li>
</ul></div>

<div class="card warn"><b>Status (2026-10-24).</b> The Vast account's credit ran out at 14:46 UTC and the 8x RTX PRO 6000 box was
stopped mid-run. Two runs are partial: the self-hosted full-model reference and the E96 64-turn rerun (section 6). A
real-weight 4xL4 ARC run of the HF E128 checkpoint is running on dsv4-infer's Runpod pod.</div>

<h2>1. Main metrics (tool harness)</h2>
<p>Harness <code>persistent-v6</code>, the sub80s40 + minimal-effort-header configuration (80K total output, 40K soft
budget, 64K per turn, 40 turns, temperature 1.0); the pruned models add <code>--rescue-code</code>. ARC-AGI-2 public
eval: 120 tasks, 167 test pairs, 2 independent attempts per pair = 334 attempts per model, all finished. The 120 eval
tasks were never used to choose pruning settings. pass@1 is the mean attempt accuracy, pass@2 counts a pair solved if
either attempt is right; both are averaged per task, then over tasks. "(est.)": E96's throughput was measured on the full
model's turn shapes, then converted at its own mean length.</p>
<div class="card">{main_table()}</div>

<h2>2. Where the accuracy goes</h2>
<p>Pairs grouped by how often the full model solved them in its own 2 attempts. E128's gap is almost all in the
borderline pairs, where the full model itself is a coin flip; it keeps {S[E128]['buckets'][2][1]} of the
{S[E128]['buckets'][2][0]} easy pairs. E96 also drops {S[E96]['buckets'][2][0] - S[E96]['buckets'][2][1]} easy pairs (attempt accuracy
{pct(S[E96]['buckets'][2][2], 0)} vs {pct(S[E128]['buckets'][2][2], 0)} for E128): pairs whose idea is easy but whose execution fails,
the signature of typos and the turn cap. Both pruned models solve about a third of the pairs the full model missed twice.
Bucketing by the reference's own results favours the reference (regression to the mean): a second full-model run would
also lose some borderline pairs and win some hard ones.</p>
<div class="card">{bucket_table()}</div>

<h2>3. Length and turns</h2>
<h3>Output tokens per attempt (share of attempts at or below x)</h3>
<div class="card">{legend(items)}{cdf_lines("cdf_tok", [(lab, slot, S[t]["toks"]) for t, lab, slot, _ in MODELS], 90000, 2500,
    lambda v: f"{v / 1000:.0f}K", "output tokens per attempt", range(0, 90001, 10000), ref_x=80000, ref_label="80K budget")}</div>
<p>E128's curve sits right of the full model's along its whole length: it reasons longer, and
{a['budget']} attempts run into the 80K budget (full: {f['budget']}). E96 sits between the two: it often stops at the turn cap before it runs out of tokens.</p>
<h3>Turns per attempt</h3>
<div class="card">{legend(items)}{cdf_lines("cdf_turns", [(lab, slot, S[t]["turns"]) for t, lab, slot, _ in MODELS], 40, 1,
    lambda v: f"{v:.0f}", "turns per attempt", range(0, 41, 5), ref_x=40, ref_label="40-turn cap")}</div>
<p>E96 has {b['cap']} attempts at the cap, which ended with only {kfmt(st.mean(r['usage']['completion'] for r in b['rows'].values() if r.get('stop_reason') == 'max_turns'))}
of their 80K tokens used on average: it runs out of turns, not tokens.</p>

<h2>4. The typos</h2>
<p>Share of tool calls that raised an error. The first-round rows (grey) are 4-bit native experts in the older
s20x60 harness with 1 attempt, so they are comparable only in trend.</p>
<div class="card">{hbar("err", err_rows, 0.6, lambda v: f"{100 * v:.1f}%", unit="tool-call error rate", nticks=6, tickfmt=lambda v: f"{100 * v:.0f}%")}</div>
<p>Almost all errors are Python exceptions in the model's own code, not tool-call format failures. The undefined names and
missing modules are one-character slips (<code>coms</code> for <code>comps</code>, <code>scpy</code> for
<code>scipy</code>): the pruned models lose some exact identifier copying. The model reads the traceback and fixes the
name, which costs a turn.</p>
<div class="card">{typo_table()}</div>
<h3>Truncated sampling does not fix them</h3>
<p>Tool E128 2-bit on 32 dev-family ARC training tasks (never eval tasks). The typos survive min_p 0.05, so their
probability is at least 5% of the top token's: they come from the model, not the sampler's tail. These tasks are easy
(about 4.5 turns), so the test cannot rank samplers on accuracy.</p>
<div class="card">{devs_table()}</div>

<h2>5. 4xL4 serving (E042, dsv4-infer)</h2>
<p>Kaggle-like 4xL4 (22.0 GiB usable per GPU, PCIe, no P2P), vLLM DeepSeek-V4.1 SM89 fork with Humming 2/3-bit expert
kernels, TP4, util 0.93, prefix caching on. Weights are dummy tensors in the exact schemas; throughput is a multi-turn
replay of real tool traffic, so prefix-cache reuse is realistic. The Engram table lives in pinned host RAM, not VRAM.
The KV cache is tiny per token (compressed attention, ~1.5 KB per token per GPU).</p>
<div class="card scroll"><table>
<tr><th>Per GPU</th><th>Tool E128 2-bit</th><th>Tool E96 2/3-bit</th><th>E72 4-bit (reference)</th></tr>
<tr><td>Usable / vLLM budget (util 0.93)</td><td>22.04 / 20.50 GiB</td><td>22.04 / 20.50 GiB</td><td>22.04 / 20.50 GiB</td></tr>
<tr><td>Weights</td><td><b>16.87 GiB</b></td><td>15.18 GiB</td><td>16.56 GiB</td></tr>
<tr><td>KV cache</td><td><b>2.10 GiB = 1.47M tokens</b></td><td>~3.8 GiB = 2.72M tokens</td><td>2.28 GiB = 1.59M tokens</td></tr>
<tr><td>Activations + CUDA graphs (rest)</td><td>~1.5 GiB</td><td>~1.5 GiB</td><td>~1.7 GiB</td></tr>
</table></div>
<div class="card scroll"><table>
<tr><th>Tool-traffic replay</th><th>streams</th><th>gen tok/s</th><th>prefix hits</th><th>KV peak</th><th>preemptions</th></tr>
<tr><td><b>E128 2-bit, its own tool turns (T4)</b></td><td>40</td><td><b>318.6</b></td><td>96.7%</td><td>0.76</td><td>0 in 60 min</td></tr>
<tr><td>E128 2-bit, full-model tool turns (T2)</td><td>48</td><td>340.5</td><td>96.1%</td><td>0.77</td><td>0 in 58 min</td></tr>
<tr><td><b>E96 2/3-bit, full-model tool turns (T2)</b></td><td>80</td><td><b>453.5</b></td><td>96.5%</td><td>0.64</td><td>0 in 70 min</td></tr>
<tr><td>E96 2/3-bit, full-model tool turns (T3)</td><td>96</td><td>291.3</td><td>85.8%</td><td>0.80</td><td>0 (cache thrash)</td></tr>
<tr><td>E72 4-bit, tool mode, util 0.88</td><td>32</td><td>205</td><td>87.8%</td><td></td><td></td></tr>
</table></div>
<p>A KV peak of ~0.8 is the practical edge: past it the prefix cache thrashes (uncached prefill triples at E96 x 96)
before anything is preempted. E128 therefore runs at 40 streams and E96 at 80 or fewer. Throughput is close to the
no-tool rate (E128 no-tool: 33.8 attempts/h) because 96-97% of each turn's prompt is a cached prefix.</p>

<h2>6. Interrupted runs (partial)</h2>
<p>The box stopped at 14:46 UTC. The finished attempts are the shorter ones, so only same-attempt comparisons are
meaningful. Both runs will resume with the same tags (the harness re-runs only the attempts recorded as errors).</p>
<div class="card">{partial_table(P_REF, "Self-hosted full model, same harness and sampler as the pruned models")}
<p class=note>If this holds over the full run, the self-hosted full model (same vLLM stack, temperature 1.0) is the
fairer baseline, and tool E128 2-bit matches it. The API's thinking-mode sampler is not known to be the same.</p></div>
<div class="card">{partial_table(P_MT64, "Tool E96 2/3-bit with a 64-turn cap")}</div>

<h2>7. How the models were built</h2>
<ol>
<li><b>Expert statistics</b> from 10M tokens of training-task tool rollouts (E031/E036/E038, DeepSeek-V4.1-Flash-1017,
s20x60 + header), replayed through the full model on the 8x RTX PRO 6000 box with an observer on every MoE layer:
selections, router weights and per-expert output amplitudes.</li>
<li><b>Keep sets</b> by contribution on generated tokens, with tool-call-block tokens weighted x10. Without the weight
the DSML tool-call format breaks first (E72: 0/10 valid first calls); x10 recovers most tool-block quality at less
general-NLL cost than x30. The router (gate weight, bias, bias_vl) is sliced to the kept rows; top-6 routing stays.</li>
<li><b>Engram:</b> a 64 GiB top-frequency table (35.4% / 32.3% of the rows on layers 1 / 14) from a 252M-token census of
the training rollouts. A weaker 2-bit model does not lean on Engram more:</li>
</ol>
<div class="card">{engram_table()}</div>
<ol start="4">
<li><b>Low-bit experts:</b> per-expert GPTQ with Humming's packing. Gate/up uint2 group 128; down uint2 (E128) or uint3
(E96) group 64; asymmetric, bf16 scales, damping 1.0 (experts see only ~3.4K calibration rows). Calibration: 1.4M
tokens of tool rollouts replayed under each model's own mask, 10% of rows. Relative output error over the 40 layers:
gate/up 2.5% (RTN 10.9%), down 5.5% (RTN 12.7%) for E128. 2-bit RTN is not usable.</li>
<li><b>Export</b> as a GPTQ-only checkpoint: backbone without routed experts, compact Engram files, <code>gptq/layer_LL.safetensors</code>.
Parity on 64 dev traces: 0.79640 (export) vs 0.79652 (runtime masks). The HF copy was re-verified against the original
checkpoint by byte ranges: router rows bit-exact at the kept ids, sampled tensors and Engram rows byte-equal, and every
sampled dequantized expert matches its own original expert (cosine 0.89-0.93) and not its neighbour (~0).</li>
</ol>
<h3>Dev screen (teacher-forced, 64 family-disjoint dev tool traces; probe: 25 first-turn prompts)</h3>
<div class="card">{dev_screen_table()}</div>

<h2>8. Earlier rounds (different harnesses; not comparable to section 1)</h2>
<h3>First round: pruning to fit 4xL4 at 4 bits, tool harness</h3>
<div class="card">{r1_table()}</div>
<p>At 72 experts and below, the tool harness fails: the format collapses first and, once fixed, the typos and semantic
errors remain. That is what moved the work to more experts at 2-3 bits.</p>
<h3>No-tool round and low-bit no-tool models (official no-tool prompt, 1 turn, 2 attempts)</h3>
<div class="card">{notool_table()}</div>
<p>Without tools, E96 2/3-bit matches the full model's pass@2 pairs (125/167) and is the fastest on 4xL4. These numbers
come from no-tool statistics and the no-tool harness; they are never pooled with the tool results above.</p>

<h2>9. Artifacts and caveats</h2>
<ul>
<li>Checkpoints (private, chankhavu): <code>DeepSeek-V4.1-Flash-1017-arc-tool-E128-2bit</code> (131.1 GB),
<code>-arc-notool-E96-2bit3bit</code> (116.5 GB), <code>-arc-notool-E128-2bit</code> (123.1 GB), <code>-arc-notool-E72</code> (118.5 GB).
Datasets: <code>dsv41-flash1017-expert-importance</code>, <code>dsv41-flash1017-e043-pruning-artifacts</code>,
<code>dsv41-flash1017-e043-gptq-artifacts</code>.</li>
<li>Serving: the vLLM DeepSeek-V4.1 SM89/SM120 fork with <code>apply_compact_engram.py</code> and
<code>apply_e043_humming.py</code>, <code>--language-model-only</code>, and the env vars in the model card.</li>
<li>Noise: one run of 2 attempts per model. Differences under 2-3 pp of pass@2 are ties.</li>
<li>The keep sets and calibration come from s20x60 rollouts; the eval uses sub80s40 (experts assumed stable across budgets).</li>
<li>4xL4 numbers use dummy weights in the real schemas; the real-weight check is running.</li>
<li>Full story, commands and every intermediate result: <code>experiments/E043_dsv41_real_pruning/README.md</code>.</li>
</ul>
</main><div id="tip" role="tooltip"></div>
<script>
const tip=document.getElementById('tip');
function show(e,g){{const[a,b]=g.dataset.tip.split('|');tip.replaceChildren();const s=document.createElement('b');s.textContent=a;
const t=document.createElement('span');t.textContent=b;tip.append(s,t);tip.style.display='block';const r=g.getBoundingClientRect();
const x=e&&e.clientX!==undefined?e.clientX:r.left+r.width/2;const y=e&&e.clientY!==undefined?e.clientY:r.top;
tip.style.left=Math.min(x+12,window.innerWidth-tip.offsetWidth-8)+'px';tip.style.top=Math.max(8,y-tip.offsetHeight-12)+'px';}}
document.querySelectorAll('.hit').forEach(g=>{{g.addEventListener('pointermove',e=>show(e,g));g.addEventListener('focus',()=>show(null,g));
g.addEventListener('pointerleave',()=>tip.style.display='none');g.addEventListener('blur',()=>tip.style.display='none');}});
</script></body></html>"""
page = page.replace('class="lab2"', 'class="lab"').replace('class="val2"', 'class="val"')
open(OUT, "w").write(page)
print(OUT, len(page))
