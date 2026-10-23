#!/usr/bin/env python3
"""E043: self-contained HTML report of the expert-pruning results (ARC evals, routing statistics, NLL screens,
probes, throughput). Inputs are the small JSON/JSONL files in results/ (eval_stats.json from eval_report_stats.py,
expert_stats.json from expert_report_stats.py, nll_*.jsonl, gen_probe*.jsonl, throughput_series.json).

  make_results_report.py <e043_results_dir> <out.html>
"""
import collections
import html
import json
import math
import sys
from pathlib import Path

RES, OUT = Path(sys.argv[1]), sys.argv[2]
EVS = json.load(open(RES / "eval_stats.json"))
EXS = json.load(open(RES / "expert_stats.json"))
W = 720


def esc(s):
    return html.escape(str(s), quote=True)


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ---------------------------------------------------------------- chart helpers
def hbar(cid, rows, vmax=1.0, fmt=pct, label_w=250):
    """rows: [(label, value, tip)] single series, horizontal bars with value at the tip."""
    row_h, padr = 34, 70
    h = 16 + row_h * len(rows) + 30
    iw = W - label_w - padr
    s = [f'<svg id="{cid}" class="chart" viewBox="0 0 {W} {h}" role="img">']
    for k in range(6):
        x = label_w + iw * k / 5
        s.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="8" y2="{h - 26}"/>')
        s.append(f'<text class="tick" x="{x:.1f}" y="{h - 10}" text-anchor="middle">{fmt(vmax * k / 5) if fmt is not pct else f"{100 * vmax * k / 5:.0f}%"}</text>')
    for i, (lab, v, tip) in enumerate(rows):
        y = 14 + i * row_h
        w = max(2, iw * v / vmax)
        s.append(f'<text class="lab" x="{label_w - 10}" y="{y + 15}" text-anchor="end">{esc(lab)}</text>')
        s.append(f'<g class="hit" tabindex="0" data-tip="{esc(tip)}"><rect x="{label_w}" y="{y}" width="{iw}" height="22" '
                 f'fill="transparent"/><path class="bar s1f" d="M{label_w},{y} H{label_w + w - 4} Q{label_w + w},{y} '
                 f'{label_w + w},{y + 4} V{y + 18} Q{label_w + w},{y + 22} {label_w + w - 4},{y + 22} H{label_w} Z"/></g>')
        s.append(f'<text class="val" x="{label_w + w + 8}" y="{y + 15}">{esc(fmt(v))}</text>')
    s.append("</svg>")
    return "\n".join(s)


def stacked(cid, rows, cats, label_w=200):
    """rows: [(label, [values per cat])]; cats: [(name, slot)]; 100% stacked horizontal bars with 2px gaps."""
    row_h, padr = 34, 20
    h = 14 + row_h * len(rows) + 10
    iw = W - label_w - padr
    s = [f'<svg id="{cid}" class="chart" viewBox="0 0 {W} {h}" role="img">']
    for i, (lab, vals) in enumerate(rows):
        y = 10 + i * row_h
        tot = sum(vals)
        x = label_w
        s.append(f'<text class="lab" x="{label_w - 10}" y="{y + 15}" text-anchor="end">{esc(lab)}</text>')
        for (cn, slot), v in zip(cats, vals):
            w = iw * v / tot
            if w <= 0:
                continue
            tip = f"{v} pairs|{cn} ({lab})"
            s.append(f'<g class="hit" tabindex="0" data-tip="{esc(tip)}"><rect class="s{slot}f" x="{x:.1f}" y="{y}" '
                     f'width="{max(0.5, w - 2):.1f}" height="22"/></g>')
            if w > 28:
                s.append(f'<text class="inbar" x="{x + w / 2 - 1:.1f}" y="{y + 15}" text-anchor="middle">{v}</text>')
            x += w
    s.append("</svg>")
    return "\n".join(s)


def lines(cid, series, xmin, xmax, ymin, ymax, xlabel, yfmt, xticks, xfmt=lambda v: f"{v:g}", logx=False, h=300,
          ref=None, markers=True, endlabels=True):
    """series: [(name, slot, [(x, y)])]."""
    padl, padr, padt, padb = 64, 170, 14, 44
    iw, ih = W - padl - padr, h - padt - padb
    tx = (lambda v: math.log(v)) if logx else (lambda v: v)
    sx = lambda v: padl + iw * (tx(v) - tx(xmin)) / (tx(xmax) - tx(xmin))
    sy = lambda v: padt + ih * (1 - (v - ymin) / (ymax - ymin))
    s = [f'<svg id="{cid}" class="chart" viewBox="0 0 {W} {h}" role="img">']
    for k in range(6):
        v = ymin + (ymax - ymin) * k / 5
        s.append(f'<line class="grid" x1="{padl}" x2="{padl + iw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>')
        s.append(f'<text class="tick" x="{padl - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{yfmt(v)}</text>')
    for v in xticks:
        s.append(f'<text class="tick" x="{sx(v):.1f}" y="{padt + ih + 16}" text-anchor="middle">{esc(xfmt(v))}</text>')
    s.append(f'<line class="axis" x1="{padl}" x2="{padl + iw}" y1="{padt + ih}" y2="{padt + ih}"/>')
    if ref:
        rv, rl = ref
        s.append(f'<line class="ref" x1="{padl}" x2="{padl + iw}" y1="{sy(rv):.1f}" y2="{sy(rv):.1f}"/>')
        s.append(f'<text class="endlabel" x="{padl + iw + 8}" y="{sy(rv) + 4:.1f}">{esc(rl)}</text>')
    ends = []
    for name, slot, pts in series:
        pts = [(x, y) for x, y in pts if y is not None]
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
        s.append(f'<path class="line s{slot}" d="{d}"/>')
        if markers:
            for x, y in pts:
                tip = f"{yfmt(y)}|{name}, {xfmt(x)}"
                s.append(f'<g class="hit" tabindex="0" data-tip="{esc(tip)}"><circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" '
                         f'r="11" fill="transparent"/><circle class="dot s{slot}" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4"/></g>')
        ends.append([sy(pts[-1][1]), name, sx(pts[-1][0])])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 14)
    for y, name, x in (ends if endlabels else []):
        s.append(f'<text class="endlabel" x="{padl + iw + 8}" y="{y + 4:.1f}">{esc(name)}</text>')
    s.append(f'<text class="axlabel" x="{padl + iw / 2:.1f}" y="{h - 6}" text-anchor="middle">{esc(xlabel)}</text>')
    s.append("</svg>")
    return "\n".join(s)


def columns(cid, vals, labels, ymax, yfmt, slot=1, h=240, xlabel="", hl=None):
    """Single-series column chart (e.g. per layer); hl: indices drawn in slot 2 for emphasis."""
    padl, padr, padt, padb = 56, 16, 12, 44
    iw, ih = W - padl - padr, h - padt - padb
    n = len(vals)
    band = iw / n
    bw = min(24, band * 0.7)
    s = [f'<svg id="{cid}" class="chart" viewBox="0 0 {W} {h}" role="img">']
    for k in range(5):
        v = ymax * k / 4
        y = padt + ih * (1 - v / ymax)
        s.append(f'<line class="grid" x1="{padl}" x2="{W - padr}" y1="{y:.1f}" y2="{y:.1f}"/>')
        s.append(f'<text class="tick" x="{padl - 8}" y="{y + 4:.1f}" text-anchor="end">{yfmt(v)}</text>')
    for i, v in enumerate(vals):
        x = padl + band * i + (band - bw) / 2
        hh = ih * min(v, ymax) / ymax
        y = padt + ih - hh
        r = min(3, hh / 2, bw / 2)
        cls = f"s{2 if hl and i in hl else slot}f"
        tip = f"{yfmt(v)}|{labels[i]}"
        s.append(f'<g class="hit" tabindex="0" data-tip="{esc(tip)}"><rect x="{padl + band * i:.1f}" y="{padt}" width="{band:.1f}" '
                 f'height="{ih}" fill="transparent"/><path class="{cls}" d="M{x:.1f},{padt + ih} V{y + r:.1f} Q{x:.1f},{y:.1f} '
                 f'{x + r:.1f},{y:.1f} H{x + bw - r:.1f} Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} V{padt + ih} Z"/></g>')
        if i % 5 == 0:
            s.append(f'<text class="tick" x="{padl + band * (i + 0.5):.1f}" y="{padt + ih + 16}" text-anchor="middle">{esc(labels[i])}</text>')
    s.append(f'<text class="axlabel" x="{padl + iw / 2:.1f}" y="{h - 6}" text-anchor="middle">{esc(xlabel)}</text>')
    s.append("</svg>")
    return "\n".join(s)


def scatter(cid, pts, xlabel, ylabel, xr, yr, h=320):
    """pts: [(label, x, y, slot)] with direct labels."""
    padl, padr, padt, padb = 64, 30, 14, 44
    iw, ih = W - padl - padr, h - padt - padb
    sx = lambda v: padl + iw * (v - xr[0]) / (xr[1] - xr[0])
    sy = lambda v: padt + ih * (1 - (v - yr[0]) / (yr[1] - yr[0]))
    s = [f'<svg id="{cid}" class="chart" viewBox="0 0 {W} {h}" role="img">']
    for k in range(6):
        v = yr[0] + (yr[1] - yr[0]) * k / 5
        s.append(f'<line class="grid" x1="{padl}" x2="{padl + iw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>')
        s.append(f'<text class="tick" x="{padl - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
        v = xr[0] + (xr[1] - xr[0]) * k / 5
        s.append(f'<text class="tick" x="{sx(v):.1f}" y="{padt + ih + 16}" text-anchor="middle">{v:.2f}</text>')
    for lab, x, y, slot in pts:
        tip = f"{x:.3f} / {y:.3f}|{lab}: NLL generated / NLL tool block"
        s.append(f'<g class="hit" tabindex="0" data-tip="{esc(tip)}"><circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="12" '
                 f'fill="transparent"/><circle class="dot s{slot}" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5"/></g>')
        s.append(f'<text class="lab" x="{sx(x) + 8:.1f}" y="{sy(y) - 6:.1f}">{esc(lab)}</text>')
    s.append(f'<text class="axlabel" x="{padl + iw / 2:.1f}" y="{h - 6}" text-anchor="middle">{esc(xlabel)}</text>')
    s.append(f'<text class="axlabel" x="14" y="{padt + ih / 2:.1f}" transform="rotate(-90 14 {padt + ih / 2:.1f})" '
             f'text-anchor="middle">{esc(ylabel)}</text>')
    s.append("</svg>")
    return "\n".join(s)


def legend(items, kind="line"):
    return '<div class="legend">' + "".join(
        f'<span class="lg"><span class="key {kind} s{slot}{"f" if kind == "box" else ""}"></span>{esc(n)}</span>'
        for n, slot in items) + "</div>"


def table(head, rows):
    return ('<div class="scroll"><table><tr>' + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table></div>")


def nll(fname):
    agg = collections.defaultdict(collections.Counter)
    for l in open(RES / fname):
        r = json.loads(l)
        for k in ("n_gen", "nll_gen", "top1_gen", "n_blk", "nll_blk", "top1_blk"):
            agg[r["tag"]][k] += r.get(k, 0) or 0
    return {t: {"nll": a["nll_gen"] / a["n_gen"], "top1": a["top1_gen"] / a["n_gen"],
                "nll_blk": a["nll_blk"] / a["n_blk"] if a["n_blk"] else None,
                "top1_blk": a["top1_blk"] / a["n_blk"] if a["n_blk"] else None} for t, a in agg.items()}


# ---------------------------------------------------------------- sections
MAIN = [("full", "Full: 384 experts, full Engram"), ("eng64", "384 experts + Engram 64 GiB"),
        ("E72", "E72 w30 + Engram 64 GiB"), ("E64", "E64 w30 + Engram 64 GiB"), ("E48", "E48 w30 + Engram 64 GiB")]
ev = {k: EVS[k] for k, _ in MAIN if k in EVS}

c_pass = hbar("pass", [(n, ev[k]["pass1"], f"{pct(ev[k]['pass1'])} pass@1|{ev[k]['correct']}/{ev[k]['attempts']} pairs, "
                        f"task-weighted {pct(ev[k]['task_weighted'])}") for k, n in MAIN if k in ev])
c_pair = stacked("pair", [(n, [ev[k]["vs_full"][c] for c in ("both", "full_only", "pruned_only", "neither")])
                          for k, n in MAIN if k in ev and "vs_full" in ev[k]],
                 [("solved by both", 1), ("full model only", 2), ("this model only", 3), ("neither", 4)])


def cdf(vals, xs):
    v = sorted(vals)
    return [(x, sum(1 for t in v if t <= x) / len(v)) for x in xs]


xs = list(range(0, 62001, 2000))
SHORT = {"full": "Full", "eng64": "Engram 64 GiB only", "E72": "E72 w30", "E64": "E64 w30", "E48": "E48 w30"}
c_tok = lines("tok", [(SHORT[k], i + 1, cdf(ev[k]["tokens"], xs)) for i, (k, n) in enumerate(MAIN) if k in ev],
              0, 62000, 0, 1, "output tokens per attempt (cumulative share of attempts)", lambda v: f"{100 * v:.0f}%",
              [0, 10000, 20000, 30000, 40000, 50000, 60000], xfmt=lambda v: f"{v / 1000:.0f}K", markers=False,
              endlabels=False)
ev_tab = table(["model", "pairs correct", "pass@1", "task-weighted", "mean turns", "mean output tokens",
                "tool-call error rate", "rescued calls / nudges", "train_pass (correct)", "stop reasons"],
               [[esc(n), f"{ev[k]['correct']}/{ev[k]['attempts']}", pct(ev[k]["pass1"]), pct(ev[k]["task_weighted"]),
                 f"{sum(ev[k]['turns']) / len(ev[k]['turns']):.1f}", f"{sum(ev[k]['tokens']) / len(ev[k]['tokens']):,.0f}",
                 pct(ev[k]["tool_error_rate"]), f"{ev[k]['rescued_calls']} / {ev[k]['nudges']}",
                 f"{ev[k]['train_pass']} ({ev[k]['train_pass_correct']})",
                 esc(", ".join(f"{a} {b}" for a, b in ev[k]["stops"].items()))] for k, n in MAIN if k in ev])
fail_tab = table(["early run (stopped)", "pairs correct", "stop reasons"],
                 [[esc(k), f"{v['correct']}/{v['attempts']}", esc(", ".join(f"{a} {b}" for a, b in v["stops"].items()))]
                  for k, v in EVS.items() if "no rescue" in k])

cov = EXS["coverage"]
Es = [8, 16, 32, 48, 64, 72, 96, 128, 192, 256, 384]
c_cov = lines("cov", [(name, slot, [(e, cov[key][str(e)]["mean"]) for e in Es])
                      for name, slot, key in [("frequency (max possible)", 1, "frequency (oracle for routing share)"),
                                              ("contribution total", 2, "contribution"),
                                              ("tool-block weight w30", 3, "w30"), ("plain REAP", 4, "plain REAP")]],
              8, 384, 0, 1, "routed experts kept per layer (log scale)", lambda v: f"{100 * v:.0f}%",
              [8, 16, 32, 64, 128, 256, 384], logx=True, h=320, endlabels=False)
eff = EXS["effective_experts"]
c_eff = columns("eff", eff, [f"layer {i}" for i in range(40)], 400, lambda v: f"{v:.0f}",
                xlabel="backbone layer (layers 20-39: decoder half); effective experts = exp(routing entropy)")
pls = EXS["per_layer_kept_share"]
c_layer = lines("layer", [(lab, slot, list(enumerate(pls[key]))) for lab, slot, key in
                          [("E72 w30", 1, "contrib_dec_sp0_w30_e72"), ("E64 w30", 2, "contrib_dec_sp0_w30_e64"),
                           ("E48 w30", 3, "contrib_dec_sp0_w30_e48")] if key in pls],
                0, 39, 0.3, 0.9, "backbone layer", lambda v: f"{100 * v:.0f}%", [0, 5, 10, 15, 20, 25, 30, 35, 39],
                h=280, markers=False)
ov = EXS["toolblock_vs_general_top72_overlap"]
c_ov = columns("ov", ov, [f"layer {i}" for i in range(40)], 1.0, lambda v: f"{100 * v:.0f}%",
               xlabel="backbone layer: share of the top-72 tool-block experts that are also top-72 overall")

ne, nt = nll("nll_experts.jsonl"), nll("nll_toolblock.jsonl")
pts = [(lab, nt[t]["nll"], nt[t]["nll_blk"], slot) for t, lab, slot in
       [("full", "full", 1), ("e72_cdec", "E72 contribution", 2), ("e72_sp16", "E72 protect 16", 2), ("e72_w10", "E72 w10", 2),
        ("e72_w30", "E72 w30", 2), ("e72_tb", "E72 tool-block only", 2), ("e64_w30", "E64 w30", 3), ("e64_tb", "E64 tool-block only", 3),
        ("e48_w30", "E48 w30", 4), ("e48_tb", "E48 tool-block only", 4)] if t in nt]
c_sc = scatter("sc", pts, "NLL of all generated tokens (lower = better reasoning text)",
               "NLL of tool-call blocks", (0.6, 1.35), (0.1, 0.32))
rule_rows = [("full", "full model"), ("eng64", "384 experts + Engram 64 GiB"), ("e64_contrib_dec", "E64 contribution, decoder on gen"),
             ("e64_contrib", "E64 contribution total"), ("e64_csqrt", "E64 sqrt(freq) x mean amp"), ("e64_reap", "E64 plain REAP"),
             ("e64_wsum", "E64 router-weight mass"), ("e64_freq", "E64 frequency"), ("e64_iter", "E64 iterative"),
             ("e72_contrib_dec", "E72 contribution, decoder on gen"), ("e72_iter", "E72 iterative"),
             ("e48_contrib_dec", "E48 contribution, decoder on gen"), ("e48_iter", "E48 iterative"),
             ("e64x_export_contrib_dec", "E64 physical export (parity check)")]
rule_tab = table(["model", "NLL gen", "top-1"], [[esc(n), f"{ne[t]['nll']:.3f}", pct(ne[t]["top1"])] for t, n in rule_rows if t in ne])
tb_tab = table(["model", "NLL gen", "top-1", "NLL tool block", "top-1 tool block"],
               [[esc(t), f"{v['nll']:.3f}", pct(v["top1"]), f"{v['nll_blk']:.3f}", pct(v["top1_blk"])] for t, v in nt.items()])
probe = collections.defaultdict(lambda: [0, 0])
for f in sorted(RES.glob("gen_probe*.jsonl")):
    if f.name == "gen_probe.jsonl":
        continue  # first probe run: prefix-cache contaminated
    for l in open(f):
        r = json.loads(l)
        probe[r["config"]][0] += bool(r["valid"])
        probe[r["config"]][1] += 1
probe_tab = table(["config", "valid first-turn tool calls"], [[esc(k), f"{v[0]}/{v[1]}"] for k, v in probe.items()])
cov_tab = table(["rule", *[f"E{e}" for e in (48, 64, 72, 96, 128, 192)]],
                [[esc(name), *[f"{pct(v[str(e)]['mean'])} (tb {pct(v[str(e)]['tb_mean'], 0)})" for e in (48, 64, 72, 96, 128, 192)]]
                 for name, v in cov.items()])
jac = EXS["jaccard"]
jac_tab = table(["", *jac["names"]], [[esc(a), *[f"{x:.2f}" for x in row]] for a, row in zip(jac["names"], jac["matrix"])])

thr = json.load(open(RES / "throughput_series.json"))


def window(run, t0, t1):
    rows = [r for r in thr.get(run, []) if t0 <= r[0] <= t1]
    if not rows:
        return None
    busy = [r for r in rows if r[3] >= 100]
    return (sum(r[2] for r in rows) / len(rows), sum(r[2] for r in busy) / max(1, len(busy)), max(r[2] for r in rows),
            sorted(r[3] for r in rows)[len(rows) // 2])


thr_rows = []
for lab, run, t0, t1 in [("E64 physical export", "tp4A_e64x", "15:16:00", "15:40:00"), ("E64 w30 (runtime masks)", "tp4A", "17:05:00", "18:10:00"),
                         ("384 experts + Engram 64 GiB", "tp4A", "18:45:00", "19:45:00"), ("E72 w30 (runtime masks)", "tp4B", "17:05:00", "17:50:00"),
                         ("E48 w30 (runtime masks, degenerate turns)", "tp4B", "18:20:00", "19:10:00")]:
    w = window(run, t0, t1)
    if w:
        thr_rows.append([esc(lab), f"{w[0]:,.0f}", f"{w[1]:,.0f}", f"{w[2]:,.0f}", str(w[3])])
thr_tab = table(["model (4 GPUs, TP4+EP4)", "mean gen tok/s", "mean at >=100 streams", "peak", "median running"], thr_rows)

CSS = open(Path(__file__).with_name("make_engram_report.py")).read().split("<style>")[1].split("</style>")[0]
CSS = CSS.replace("{{", "{").replace("}}", "}")
EXTRA = """
.lab { fill:var(--ink2); font-size:12px; } .val { fill:var(--ink); font-size:12px; font-variant-numeric:tabular-nums; }
.s1f { fill:var(--s1); } .s2f { fill:var(--s2); } .s3f { fill:var(--s3); } .s4f { fill:var(--s4); } .s5f { fill:var(--s5); }
.hit:hover .s1f, .hit:hover .s2f, .hit:hover .s3f, .hit:hover .s4f { opacity:0.75; }
.line.s4 { stroke:var(--s4); } .line.s5 { stroke:var(--s5); } .dot.s4 { fill:var(--s4); } .dot.s5 { fill:var(--s5); }
.key.s4, .key.s4f { background:var(--s4); } .key.s5 { background:var(--s5); } .key.s1f { background:var(--s1); }
.key.s2f { background:var(--s2); } .key.s3f { background:var(--s3); }
.key.box { width:12px; height:12px; border-radius:2px; }
.ref { stroke:var(--axis); stroke-width:2; } .inbar { fill:#fff; font-size:11px; font-weight:600; }
:root { --s4:#eda100; --s5:#e87ba4; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --s4:#c98500; --s5:#d55181; } }
:root[data-theme="dark"] { --s4:#c98500; --s5:#d55181; }
"""
pi = lambda k, f: f(ev[k]) if k in ev else "-"
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Expert pruning results</title>
<style>{CSS}{EXTRA}</style></head><body><main>
<h1>Expert pruning results</h1>
<p class="sub">DeepSeek-V4.1-Flash-1017, experiment E043 (2026-10-23). ARC-AGI-2 public eval (120 tasks / 167 test pairs), one
attempt per pair, the E031 tool harness (s20x60 + minimal effort header) on the self-hosted vLLM fork (8x RTX PRO 6000,
two TP4+EP4 servers). Pruned models keep top-6 routing; "w30" keep sets weight tool-call-block tokens x30 in the expert
score; pruned runs use the harness safety net <code>--rescue-code</code>. Engram details: <code>engram_census.html</code>.</p>
<div class="tiles">
 <div class="card tile"><div class="lab2">Full model</div><div class="val2">{pi('full', lambda e: pct(e['pass1']))}</div><div class="note">pass@1, 167 pairs</div></div>
 <div class="card tile"><div class="lab2">Engram 64 GiB only</div><div class="val2">{pi('eng64', lambda e: pct(e['pass1']))}</div><div class="note">all 384 experts (control)</div></div>
 <div class="card tile"><div class="lab2">E72 w30</div><div class="val2">{pi('E72', lambda e: pct(e['pass1']))}</div><div class="note">largest size that fits 4xL4</div></div>
 <div class="card tile"><div class="lab2">E64 / E48 w30</div><div class="val2">{pi('E64', lambda e: pct(e['pass1'], 0))} / {pi('E48', lambda e: pct(e['pass1'], 0))}</div><div class="note">E48: 120 of 167 pairs</div></div>
</div>

<h2>ARC pass@1</h2>
<div class="card">{c_pass}<details><summary>Table view</summary>{ev_tab}</details></div>
<h2>Same pairs, paired with the full model</h2>
<p>Single attempts at temperature 1 are noisy per pair: the Engram-only control and the full model each solve ~110-116
pairs but disagree on 40. The pruned models mostly lose pairs the full model solves; they rarely solve new ones.</p>
<div class="card">{legend([("solved by both", 1), ("full model only", 2), ("this model only", 3), ("neither", 4)], "box")}{c_pair}</div>
<h2>Output tokens per attempt</h2>
<p>Cumulative share of attempts by output tokens. E48 runs to the 60K budget in a single turn on almost every attempt;
E64/E72 spend more tokens than the full model and hit the budget more often.</p>
<div class="card">{legend([(SHORT[k], i + 1) for i, (k, n) in enumerate(MAIN) if k in ev])}{c_tok}</div>
<details><summary>Early pruned runs without the safety net (all stopped)</summary>{fail_tab}</details>

<h2>How much routing traffic survives</h2>
<p>Mean over layers of the share of the full model's routing selections that still land on a kept expert (10M
calibration tokens). Frequency is the upper bound for this measure; plain REAP keeps strong-but-rare experts and covers
far less traffic; the w30 rule gives up some general traffic to keep the experts that write tool calls.</p>
<div class="card">{legend([("frequency (max possible)", 1), ("contribution total", 2), ("tool-block weight w30", 3), ("plain REAP", 4)])}{c_cov}<details><summary>Table view (tb = share of tool-call-block selections kept)</summary>{cov_tab}</details></div>
<h2>Routing is spread over many experts</h2>
<p>Effective number of experts per layer, exp(entropy of routing frequency): 83-196 of 384. Keeping 48-72 experts
therefore cuts into experts that carry real traffic in every layer.</p>
<div class="card">{c_eff}</div>
<h2>Kept routing share per layer (w30 keep sets)</h2>
<div class="card">{legend([("E72 w30", 1), ("E64 w30", 2), ("E48 w30", 3)])}{c_layer}</div>
<h2>Tool-call experts are a partly separate set</h2>
<p>Per layer, the share of the 72 most important experts for tool-call tokens that are also in the 72 most important
overall: 42-64% in early layers. This is why general importance scores dropped the tool-call format.</p>
<div class="card">{c_ov}</div>

<h2>Reasoning text vs tool-call structure</h2>
<p>Teacher-forced NLL on 32 held-out eval traces: x = all generated tokens, y = tokens inside tool-call blocks. Moving
expert budget to tool-call structure lowers y and raises x; free generation needs y close to the full model.</p>
<div class="card">{legend([("full", 1), ("E72", 2), ("E64", 3), ("E48", 4)], "box")}{c_sc}<details><summary>Table view</summary>{tb_tab}</details></div>
<h2>E64 scoring rules and iterative pruning (one-shot statistics win)</h2><div class="card">{rule_tab}</div>
<h2>Free-generation probes</h2><p>Valid first-turn tool calls (simple prompt + 3 ARC prompts, greedy + T=1 samples),
cache-salted per configuration.</p><div class="card">{probe_tab}</div>
<h2>Keep-set overlap (Jaccard, mean over layers)</h2><div class="card">{jac_tab}</div>
<h2>Serving throughput on the RTX box</h2><p>Generation tokens/s per 4-GPU server during the evals. Agentic traffic
is CPU-bound on eager mixed prefill/decode steps; the pure-decode ceiling is ~2.5-2.7K tok/s at ~128 streams. Not L4
numbers.</p><div class="card">{thr_tab}</div>

<h2>Findings</h2><ul>
<li>Engram rows: a 64 GiB top-frequency table (34% of rows) is lossless on ARC (control {pi('eng64', lambda e: pct(e['pass1']))} vs full {pi('full', lambda e: pct(e['pass1']))}).</li>
<li>Expert pruning to E72/E64/E48 costs most of the accuracy even after fixing the tool-call format: routing is spread
over 83-196 effective experts per layer, and the kept sets cover 53-65% of the routing traffic.</li>
<li>E72 mostly keeps the puzzle's main idea and loses on execution (near-miss answers, 45% tool-call errors); E64 has
semantic damage (hallucinated module names); E48 never leaves its first turn.</li>
<li>Teacher-forced NLL understates the damage: the format collapse and the looping only show up in free generation.</li>
</ul></main><div id="tip" role="tooltip"></div>
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
