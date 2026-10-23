#!/usr/bin/env python3
"""E043: self-contained HTML report of the expert-pruning results (ARC evals, NLL screens, probes).

  make_results_report.py <evals_results_dir> <e043_results_dir> <out.html>
"""
import collections
import html
import json
import sys
from pathlib import Path

EV, RES, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
PFX = "tools_persistent_grid_lean2sub_dsv41_low_e043_"
RUNS = [("full_a1", "Full: 384 experts, full Engram"),
        ("eng64_a1", "384 experts + Engram 64 GiB"),
        ("e72_w30_eng64_rescue2", "E72 w30 + Engram 64 GiB"),
        ("e64_w30_eng64_rescue2", "E64 w30 + Engram 64 GiB"),
        ("e48_w30_eng64_rescue2", "E48 w30 + Engram 64 GiB")]


def esc(s):
    return html.escape(str(s), quote=True)


def load(tag):
    p = EV / f"{PFX}{tag}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def arc_row(tag, name):
    rows = load(tag)
    if not rows:
        return None
    t = collections.defaultdict(list)
    for r in rows:
        t[r["task_id"]].append(bool(r.get("correct")))
    stops = collections.Counter(r.get("stop_reason") for r in rows)
    return {"name": name, "n": len(rows), "correct": sum(bool(r.get("correct")) for r in rows),
            "taskw": sum(sum(v) / len(v) for v in t.values()) / len(t),
            "turns": sum(r.get("turns") or 0 for r in rows) / len(rows),
            "tok": sum((r.get("usage") or {}).get("completion", 0) for r in rows) / len(rows),
            "stops": ", ".join(f"{k} {v}" for k, v in stops.most_common()),
            "rescues": sum(r.get("rescues") or 0 for r in rows)}


def nll(fname):
    agg = collections.defaultdict(lambda: collections.Counter())
    for l in open(RES / fname):
        r = json.loads(l)
        a = agg[r["tag"]]
        for k in ("n_gen", "nll_gen", "top1_gen", "n_blk", "nll_blk", "top1_blk"):
            a[k] += r.get(k, 0) or 0
    return {t: {"nll": a["nll_gen"] / a["n_gen"], "top1": a["top1_gen"] / a["n_gen"],
                "nll_blk": a["nll_blk"] / a["n_blk"] if a["n_blk"] else None,
                "top1_blk": a["top1_blk"] / a["n_blk"] if a["n_blk"] else None} for t, a in agg.items()}


arc = [x for x in (arc_row(t, n) for t, n in RUNS) if x]
ne = nll("nll_experts.jsonl")
nt = nll("nll_toolblock.jsonl")

# --- chart 1: ARC pass@1 (horizontal bars, single series) ---
W, LBL, PADR, ROW = 720, 250, 70, 34
H = 20 + ROW * len(arc) + 30
svg = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="ARC pass@1 by model">']
iw = W - LBL - PADR
for k in range(0, 101, 20):
    x = LBL + iw * k / 100
    svg.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="10" y2="{H - 26}"/>')
    svg.append(f'<text class="tick" x="{x:.1f}" y="{H - 10}" text-anchor="middle">{k}%</text>')
for i, a in enumerate(arc):
    y = 16 + i * ROW
    p = a["correct"] / a["n"]
    w = max(2, iw * p)
    tip = esc(f"{100 * p:.1f}% pass@1|{a['correct']}/{a['n']} pairs, task-weighted {100 * a['taskw']:.1f}%")
    svg.append(f'<text class="lab" x="{LBL - 10}" y="{y + 15}" text-anchor="end">{esc(a["name"])}</text>')
    svg.append(f'<g class="hit" tabindex="0" data-tip="{tip}"><rect x="{LBL}" y="{y}" width="{iw}" height="22" '
               f'fill="transparent"/><path class="bar" d="M{LBL},{y} H{LBL + w - 4} Q{LBL + w},{y} {LBL + w},{y + 4} '
               f'V{y + 18} Q{LBL + w},{y + 22} {LBL + w - 4},{y + 22} H{LBL} Z"/></g>')
    svg.append(f'<text class="val" x="{LBL + w + 8}" y="{y + 15}">{100 * p:.1f}%</text>')
svg.append("</svg>")
chart_arc = "\n".join(svg)

# --- chart 2: NLL on generated tokens vs experts per layer, three rules ---
series = [("one-shot contribution", 1, {48: ne.get("e48_contrib_dec"), 64: ne.get("e64_contrib_dec"),
                                         72: ne.get("e72_contrib_dec")}),
          ("iterative", 2, {48: ne.get("e48_iter"), 64: ne.get("e64_iter"), 72: ne.get("e72_iter")}),
          ("tool-block w30", 3, {48: nt.get("e48_w30"), 64: nt.get("e64_w30"), 72: nt.get("e72_w30")})]
W2, H2, L2, R2, T2, B2 = 720, 300, 64, 150, 16, 44
xs = {48: 0, 64: 1, 72: 2}
ymin, ymax = 0.6, 1.2
sx = lambda e: L2 + (W2 - L2 - R2) * xs[e] / 2
sy = lambda v: T2 + (H2 - T2 - B2) * (1 - (v - ymin) / (ymax - ymin))
s2 = [f'<svg class="chart" viewBox="0 0 {W2} {H2}" role="img" aria-label="NLL vs experts">']
for k in range(7):
    v = ymin + (ymax - ymin) * k / 6
    s2.append(f'<line class="grid" x1="{L2}" x2="{W2 - R2}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>')
    s2.append(f'<text class="tick" x="{L2 - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:.1f}</text>')
full = ne.get("full", {}).get("nll", 0.664)
s2.append(f'<line class="ref" x1="{L2}" x2="{W2 - R2}" y1="{sy(full):.1f}" y2="{sy(full):.1f}"/>')
s2.append(f'<text class="endlabel" x="{W2 - R2 + 8}" y="{sy(full) + 4:.1f}">full model {full:.3f}</text>')
for e in xs:
    s2.append(f'<text class="tick" x="{sx(e):.1f}" y="{H2 - B2 + 18}" text-anchor="middle">E{e}</text>')
s2.append(f'<text class="axlabel" x="{(L2 + W2 - R2) / 2:.1f}" y="{H2 - 6}" text-anchor="middle">routed experts kept per layer '
          '(NLL of generated tokens, 32 eval traces)</text>')
ends = []
for name, slot, pts in series:
    pp = [(e, v["nll"]) for e, v in sorted(pts.items()) if v]
    d = " ".join(f"{'M' if i == 0 else 'L'}{sx(e):.1f},{sy(v):.1f}" for i, (e, v) in enumerate(pp))
    s2.append(f'<path class="line s{slot}" d="{d}"/>')
    for e, v in pp:
        tip = esc(f"{v:.3f} nats/token|{name}, E{e}")
        s2.append(f'<g class="hit" tabindex="0" data-tip="{tip}"><circle cx="{sx(e):.1f}" cy="{sy(v):.1f}" r="12" '
                  f'fill="transparent"/><circle class="dot s{slot}" cx="{sx(e):.1f}" cy="{sy(v):.1f}" r="4.5"/></g>')
    ends.append([sy(pp[-1][1]), name])
ends.sort()
for i in range(1, len(ends)):
    ends[i][0] = max(ends[i][0], ends[i - 1][0] + 14)
for y, name in ends:
    s2.append(f'<text class="endlabel" x="{sx(72) + 10:.1f}" y="{y + 4:.1f}">{esc(name)}</text>')
s2.append("</svg>")
chart_nll = "\n".join(s2)


def table(head, rows):
    return ('<div class="scroll"><table><tr>' + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table></div>")


arc_tab = table(["model", "pairs correct", "pass@1", "task-weighted", "turns", "output tokens", "stop reasons"],
                [[esc(a["name"]), f"{a['correct']}/{a['n']}", f"{100 * a['correct'] / a['n']:.1f}%",
                  f"{100 * a['taskw']:.1f}%", f"{a['turns']:.1f}", f"{a['tok']:,.0f}", esc(a["stops"])] for a in arc])
rule_rows = [("full", "full model"), ("eng64", "384 experts + Engram 64 GiB"), ("e64_contrib_dec", "E64 contribution, decoder on gen"),
             ("e64_contrib", "E64 contribution total"), ("e64_csqrt", "E64 sqrt(freq) x mean amp"), ("e64_reap", "E64 plain REAP"),
             ("e64_wsum", "E64 router-weight mass"), ("e64_freq", "E64 frequency"), ("e64_iter", "E64 iterative"),
             ("e72_contrib_dec", "E72 contribution, decoder on gen"), ("e72_iter", "E72 iterative"),
             ("e48_contrib_dec", "E48 contribution, decoder on gen"), ("e48_iter", "E48 iterative"),
             ("e64x_export_contrib_dec", "E64 physical export (parity)")]
rule_tab = table(["model", "NLL gen", "top-1"], [[esc(n), f"{ne[t]['nll']:.3f}", f"{100 * ne[t]['top1']:.1f}%"]
                                                 for t, n in rule_rows if t in ne])
tb_tab = table(["model", "NLL gen", "top-1", "NLL tool block", "top-1 tool block"],
               [[esc(t), f"{v['nll']:.3f}", f"{100 * v['top1']:.1f}%", f"{v['nll_blk']:.3f}", f"{100 * v['top1_blk']:.1f}%"]
                for t, v in nt.items()])
probe = collections.defaultdict(lambda: [0, 0])
for f in ("gen_probe2.jsonl", "gen_probe3.jsonl", "gen_probe4.jsonl", "gen_probe5.jsonl"):
    p = RES / f
    if p.exists():
        for l in open(p):
            r = json.loads(l)
            probe[r["config"]][0] += bool(r["valid"])
            probe[r["config"]][1] += 1
probe_tab = table(["config", "valid first-turn tool calls"], [[esc(k), f"{v[0]}/{v[1]}"] for k, v in probe.items()])

CSS = open(Path(__file__).with_name("make_engram_report.py")).read().split("<style>")[1].split("</style>")[0]
CSS = CSS.replace("{{", "{").replace("}}", "}")
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Expert pruning results</title>
<style>{CSS}
.lab {{ fill:var(--ink2); font-size:12px; }} .val {{ fill:var(--ink); font-size:12px; font-variant-numeric:tabular-nums; }}
.bar {{ fill:var(--s1); }} .hit:hover .bar {{ opacity:0.75; }} .ref {{ stroke:var(--axis); stroke-width:2; }}
</style></head><body><main>
<h1>Expert pruning results</h1>
<p class="sub">DeepSeek-V4.1-Flash-1017, experiment E043. ARC-AGI-2 public eval (120 tasks / 167 test pairs), one
attempt per pair, the E031 tool harness (s20x60 + minimal effort header) on the self-hosted fork (8x RTX PRO 6000).
Pruned models keep the original top-6 routing; "w30" keep sets weight tool-call-block tokens x30 in the
contribution score; pruned runs use the harness safety net <code>--rescue-code</code>.</p>
<h2>ARC pass@1</h2>
<div class="card">{chart_arc}
<details open><summary>Table view</summary>{arc_tab}</details></div>
<h2>Teacher-forced NLL vs experts kept</h2>
<p>NLL of the model's own generated tokens on 32 held-out eval-task traces (lower is better). Iterative re-observation
loses to one-shot statistics at every size; the tool-block weighting trades general NLL for intact tool-call
structure, which is what free generation needs.</p>
<div class="card"><div class="legend"><span class="lg"><span class="key line s1"></span>one-shot contribution</span>
<span class="lg"><span class="key line s2"></span>iterative</span><span class="lg"><span class="key line s3"></span>tool-block w30</span></div>
{chart_nll}</div>
<h2>E64 scoring rules (one-shot)</h2><div class="card">{rule_tab}</div>
<h2>Tool-block screen</h2><p>NLL / top-1 on targets inside tool-call blocks (from <code>&lt;/think&gt;</code> to EOS).</p>
<div class="card">{tb_tab}</div>
<h2>Free-generation probes</h2><p>Valid first-turn tool calls on a simple prompt and 3 ARC prompts (greedy + T=1
samples), cache-salted per configuration.</p><div class="card">{probe_tab}</div>
<h2>Findings</h2><ul>
<li>Engram rows: a 64 GiB top-frequency table (34% of rows) is within NLL noise of the full table and generates
normally; misses come from novel n-grams in the model's own reasoning (93% of unseen lookups).</li>
<li>All-token NLL and contribution-based keep sets hide a collapse of the DSML tool-call format at E72 and below
(0/10 valid first-turn calls); weighting tool-call-block tokens x30 restores 10/10 at E64/E72.</li>
<li>Trace reviews: E64 also has semantic damage (hallucinated module names, 57% of tool calls error vs 1.7% for the
full model); E72 mostly keeps the puzzle's main idea and loses on execution (near-miss answers).</li>
<li>E96 and above generate cleanly but need ~21 GiB/GPU of weights, above the 4xL4 budget; E72 is the ceiling.</li>
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
open(OUT, "w").write(page)
print(OUT, len(page))
