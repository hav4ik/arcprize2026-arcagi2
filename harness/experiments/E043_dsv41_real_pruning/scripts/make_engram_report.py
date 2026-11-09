#!/usr/bin/env python3
"""E043: self-contained HTML report of the Engram row census (inline SVG, no external assets).

  make_engram_report.py results/engram_stats.json reports/engram_census.html
"""
import html
import json
import math
import sys

S = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
H = S["hist"]
L0 = H["layers"][0]
L1 = H["layers"][1]

W, PADL, PADR, PADT, PADB = 720, 64, 24, 16, 44


def fmt_int(n):
    for d, s in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= d:
            return f"{n / d:.1f}{s}"
    return str(int(n))


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def bar_chart(cid, labels, values, ylabel, tip_fmt, height=260):
    """Single-series column chart (values as fractions), 4px rounded caps, hover per bar."""
    ih = height - PADT - PADB
    iw = W - PADL - PADR
    vmax = max(values)
    top = math.ceil(vmax * 100 / 5) * 5 / 100 or 0.05
    n = len(values)
    band = iw / n
    bw = min(24, band * 0.6)
    out = [f'<svg id="{cid}" class="chart" viewBox="0 0 {W} {height}" role="img" aria-label="{html.escape(ylabel)}">']
    for k in range(6):
        v = top * k / 5
        y = PADT + ih - ih * v / top
        out.append(f'<line class="grid" x1="{PADL}" x2="{W - PADR}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{PADL - 8}" y="{y + 4:.1f}" text-anchor="end">{100 * v:.0f}%</text>')
    out.append(f'<line class="axis" x1="{PADL}" x2="{W - PADR}" y1="{PADT + ih}" y2="{PADT + ih}"/>')
    for i, (lab, v) in enumerate(zip(labels, values)):
        x = PADL + band * i + (band - bw) / 2
        h = ih * v / top
        y = PADT + ih - h
        r = min(4, h / 2, bw / 2)
        # Rounded data end, square at the baseline.
        path = (f"M{x:.1f},{PADT + ih:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
                f"H{x + bw - r:.1f} Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} V{PADT + ih:.1f} Z")
        tip = html.escape(tip_fmt(i), quote=True)
        out.append(f'<g class="hit" tabindex="0" data-tip="{tip}"><rect x="{PADL + band * i:.1f}" y="{PADT}" '
                   f'width="{band:.1f}" height="{ih}" fill="transparent"/><path class="mark s1" d="{path}"/></g>')
        short = lab.split("-")[0].lstrip(">") + ("+" if lab.startswith(">") else "")
        out.append(f'<text class="tick" x="{PADL + band * (i + 0.5):.1f}" y="{PADT + ih + 16}" '
                   f'text-anchor="middle">{html.escape(short)}</text>')
    out.append(f'<text class="axlabel" x="{PADL + iw / 2:.1f}" y="{height - 6}" text-anchor="middle">'
               'lookups per row (bin start; hover for the range)</text>')
    out.append("</svg>")
    return "\n".join(out)


def line_chart(cid, series, xmax, ymin, ymax, xlabel, height=300):
    """Multi-series line chart. series: [(name, slot, [(x, y)], markers, tipfmt)]. y as fractions."""
    ih = height - PADT - PADB
    iw = W - PADL - PADR - 90
    sx = lambda x: PADL + iw * x / xmax
    sy = lambda y: PADT + ih - ih * (y - ymin) / (ymax - ymin)
    out = [f'<svg id="{cid}" class="chart" viewBox="0 0 {W} {height}" role="img">']
    for k in range(6):
        v = ymin + (ymax - ymin) * k / 5
        out.append(f'<line class="grid" x1="{PADL}" x2="{PADL + iw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}"/>')
        out.append(f'<text class="tick" x="{PADL - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{100 * v:.0f}%</text>')
    for gx in range(0, int(xmax) + 1, 16):
        out.append(f'<text class="tick" x="{sx(gx):.1f}" y="{PADT + ih + 16}" text-anchor="middle">{gx}</text>')
    out.append(f'<line class="axis" x1="{PADL}" x2="{PADL + iw}" y1="{PADT + ih}" y2="{PADT + ih}"/>')
    labels_y = []
    for name, slot, pts, markers, tipf in series:
        pts = [(x, y) for x, y in pts if y >= ymin]
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
        out.append(f'<path class="line s{slot}" d="{d}"/>')
        if markers:
            for i, (x, y) in enumerate(pts):
                tip = html.escape(tipf(i), quote=True)
                out.append(f'<g class="hit" tabindex="0" data-tip="{tip}"><circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" '
                           f'r="12" fill="transparent"/><circle class="dot s{slot}" cx="{sx(x):.1f}" '
                           f'cy="{sy(y):.1f}" r="4.5"/></g>')
        x, y = pts[-1]
        labels_y.append([sy(y), name.split(",")[-1].strip().replace(" (in-sample)", ""), sx(x)])
    # End labels with simple collision avoidance (leader lines when nudged).
    labels_y.sort()
    for i in range(1, len(labels_y)):
        if labels_y[i][0] - labels_y[i - 1][0] < 14:
            labels_y[i][0] = labels_y[i - 1][0] + 14
    for y, name, x in labels_y:
        out.append(f'<text class="endlabel" x="{x + 10:.1f}" y="{y + 4:.1f}">{html.escape(name)}</text>')
    out.append(f'<text class="axlabel" x="{PADL + iw / 2:.1f}" y="{height - 6}" text-anchor="middle">'
               f'{html.escape(xlabel)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def legend(items, kind="line"):
    return '<div class="legend">' + "".join(
        f'<span class="lg"><span class="key {kind} s{slot}"></span>{html.escape(name)}</span>' for name, slot in items) + "</div>"


labels = H["labels"]
tot_rows = sum(L0["rows"])
tot_lk = sum(L0["lookups"])
rows_frac = [r / tot_rows for r in L0["rows"]]
mass_frac = [m / tot_lk for m in L0["lookups"]]
c_hist_rows = bar_chart(
    "hrows", labels, rows_frac, "share of touched rows",
    lambda i: f"{pct(rows_frac[i])} of touched rows|{fmt_int(L0['rows'][i])} rows hit {labels[i]} times (layer 1)")
c_hist_mass = bar_chart(
    "hmass", labels, mass_frac, "share of all lookups",
    lambda i: f"{pct(mass_frac[i])} of lookups|rows hit {labels[i]} times carry {fmt_int(L0['lookups'][i])} lookups")

ev = S["eval"]
curve = S["calib_curve"]
xmax = 112
ser_cov = [
    ("calibration (in-sample)", 1, [(x, y) for x, y in curve if x <= xmax], False, None),
    ("eval traces, all lookups", 2, [(e["kept_gib"], e["eval_hit"]) for e in ev], True,
     lambda i: f"{pct(ev[i]['eval_hit'])} hit rate|{ev[i]['kept_gib']:.0f} GiB kept, held-out eval traces"),
    ("eval traces, generated tokens", 3, [(e["kept_gib"], e["eval_hit_gen"]) for e in ev], True,
     lambda i: f"{pct(ev[i]['eval_hit_gen'])} hit rate|{ev[i]['kept_gib']:.0f} GiB kept, generated-token inputs"),
]
c_cov = line_chart("cov", ser_cov, xmax, 0.6, 1.0, "retained Engram table (GiB, top-frequency rows over both layers)")
ser_ord = [(f"{o}-gram", o - 1, [(e["kept_gib"], e["eval_hit_by_order"][o - 2]) for e in ev], True,
            (lambda o: lambda i: f"{pct(ev[i]['eval_hit_by_order'][o - 2])} hit rate|{o}-gram lookups, "
                                 f"{ev[i]['kept_gib']:.0f} GiB kept")(o)) for o in (2, 3, 4)]
c_ord = line_chart("ord", ser_ord, xmax, 0.6, 1.0, "retained Engram table (GiB)")

bank_rows = []
for L in H["layers"]:
    for o in (2, 3, 4):
        b = [x for x in L["banks"] if x["order"] == o]
        size = sum(x["size"] for x in b)
        dist = sum(x["distinct"] for x in b)
        lk = sum(x["lookups"] for x in b)
        top = sum(x["top10pct_share"] for x in b) / len(b)
        bank_rows.append(f"<tr><td>{L['layer']}</td><td>{o}-gram</td><td>{fmt_int(size)}</td><td>{fmt_int(dist)}</td>"
                         f"<td>{pct(dist / size)}</td><td>{fmt_int(lk)}</td><td>{pct(top)}</td></tr>")
mask_rows = "".join(
    f"<tr><td>{e['mask'].replace('engram_', '')}</td><td>{e['kept_gib']:.1f}</td>"
    f"<td>{pct(e['kept_gib'] / S['full_table_gib'])}</td><td>{pct(e['calib_coverage'])}</td>"
    f"<td>{pct(e['eval_hit'])}</td><td>{pct(e['eval_hit_prompt'])}</td><td>{pct(e['eval_hit_gen'])}</td>"
    f"<td>{' / '.join(pct(x) for x in e['eval_hit_by_order'])}</td><td>{pct(e['eval_all48'])}</td></tr>" for e in ev)
hist_rows = "".join(f"<tr><td>{lab}</td><td>{fmt_int(r)}</td><td>{pct(r / tot_rows)}</td><td>{fmt_int(m)}</td>"
                    f"<td>{pct(m / tot_lk)}</td></tr>" for lab, r, m in zip(labels, L0["rows"], L0["lookups"]))
rr = L0["role_rows"]
ntok = S["tokens_prompt"] + S["tokens_gen"]
seen_gib = 2 * tot_rows * 264 / 2**30

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Engram row census</title>
<style>
:root {{ color-scheme: light; --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10); --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --surface:#1a1a19;
  --page:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --ring:rgba(255,255,255,0.10); --s1:#3987e5; --s2:#d95926; --s3:#199e70; }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --surface:#1a1a19; --page:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10); --s1:#3987e5; --s2:#d95926; --s3:#199e70; }}
body {{ margin:0; background:var(--page); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:960px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:32px 0 4px; }}
p, li {{ color:var(--ink2); }} .sub {{ color:var(--ink2); margin:0 0 20px; }}
.card {{ background:var(--surface); border:1px solid var(--ring); border-radius:10px; padding:16px; margin:12px 0; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }}
.tile .lab {{ color:var(--ink2); font-size:13px; }} .tile .val {{ font-size:28px; font-weight:600; }}
.tile .note {{ color:var(--muted); font-size:12px; }}
svg.chart {{ width:100%; height:auto; display:block; }}
.grid {{ stroke:var(--grid); stroke-width:1; }} .axis {{ stroke:var(--axis); stroke-width:1; }}
.tick {{ fill:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }}
.axlabel {{ fill:var(--ink2); font-size:12px; }} .endlabel {{ fill:var(--ink2); font-size:12px; }}
.mark.s1 {{ fill:var(--s1); }} .hit:hover .mark, .hit:focus .mark {{ opacity:0.75; }} .hit {{ outline:none; cursor:default; }}
.line {{ fill:none; stroke-width:2; stroke-linejoin:round; stroke-linecap:round; }}
.line.s1 {{ stroke:var(--s1); }} .line.s2 {{ stroke:var(--s2); }} .line.s3 {{ stroke:var(--s3); }}
.dot {{ stroke:var(--surface); stroke-width:2; }} .dot.s1 {{ fill:var(--s1); }} .dot.s2 {{ fill:var(--s2); }} .dot.s3 {{ fill:var(--s3); }}
.hit:hover .dot, .hit:focus .dot {{ r:6; }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:13px; color:var(--ink2); margin:4px 0 8px; }}
.lg {{ display:inline-flex; align-items:center; gap:6px; }}
.key.line {{ width:16px; height:2px; border-radius:1px; }} .key.s1 {{ background:var(--s1); }}
.key.s2 {{ background:var(--s2); }} .key.s3 {{ background:var(--s3); }}
table {{ border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }}
th, td {{ text-align:right; padding:4px 8px; border-bottom:1px solid var(--grid); }}
th:first-child, td:first-child {{ text-align:left; }} th {{ color:var(--ink2); font-weight:600; }}
details summary {{ cursor:pointer; color:var(--ink2); font-size:13px; margin-top:8px; }}
.scroll {{ overflow-x:auto; }}
#tip {{ position:fixed; pointer-events:none; background:var(--surface); border:1px solid var(--ring); border-radius:8px;
  padding:6px 10px; font-size:13px; box-shadow:0 2px 8px rgba(0,0,0,0.12); display:none; max-width:280px; }}
#tip b {{ display:block; color:var(--ink); }} #tip span {{ color:var(--ink2); }}
code {{ font-size:13px; }}
</style></head><body><main>
<h1>Engram row census</h1>
<p class="sub">DeepSeek-V4.1-Flash-1017 (revision 1bae0e10), experiment E043, 2026-10-23. Calibration: 10,001 tool-use
rollouts (E031/E036/E038, s20x60 + minimal header). Held out: 334 eval-task traces (E030 s20x60 hdr, 120 ARC-AGI-2
eval tasks). Addresses from the official <code>inference/engram.py</code>, recomputed on CPU.</p>

<div class="tiles">
 <div class="card tile"><div class="lab">Consumed input tokens</div><div class="val">{fmt_int(ntok)}</div>
  <div class="note">{fmt_int(S['tokens_prompt'])} prompt/tool, {fmt_int(S['tokens_gen'])} generated</div></div>
 <div class="card tile"><div class="lab">Lookups per Engram layer</div><div class="val">{fmt_int(tot_lk)}</div>
  <div class="note">24 per token: 2/3/4-gram × 8 heads</div></div>
 <div class="card tile"><div class="lab">Distinct rows touched</div><div class="val">{pct(tot_rows / L0['total'])}</div>
  <div class="note">{fmt_int(tot_rows)} of {fmt_int(L0['total'])} per layer</div></div>
 <div class="card tile"><div class="lab">All observed rows</div><div class="val">{seen_gib:.0f} GiB</div>
  <div class="note">of the {S['full_table_gib']:.0f} GiB table (264 B/row)</div></div>
</div>

<h2>How often each touched row is hit</h2>
<p>Share of touched rows by lookup count (layer 1; layer 14 is identical to the first decimal). Most touched rows
are hit a handful of times.</p>
<div class="card">{c_hist_rows}</div>
<h2>Where the lookups go</h2>
<p>Share of all lookups carried by rows in each count bin. A few very hot rows carry most of the traffic
(2-gram banks concentrate hardest, see the bank table below); rows hit at most 4 times carry {pct(sum(mass_frac[:3]))}.</p>
<div class="card">{c_hist_mass}
<details><summary>Table view</summary><div class="scroll"><table><tr><th>lookups per row</th><th>rows</th>
<th>share of rows</th><th>lookups</th><th>share of lookups</th></tr>{hist_rows}</table></div></details></div>

<h2>Coverage vs retained table size</h2>
<p>Keep the most frequent rows (one global ranking over both layers, original hash geometry, a dropped row reads as
a zero row before Wkv). In-sample coverage reaches 100% at {seen_gib:.0f} GiB, where every observed row is kept; the
held-out eval traces still miss {pct(1 - ev[-1]['eval_hit'])} of lookups there.</p>
<div class="card">{legend([("calibration (in-sample)", 1), ("eval traces, all lookups", 2), ("eval traces, generated tokens", 3)])}
{c_cov}</div>

<h2>Held-out hit rate by n-gram order</h2>
<p>2-gram lookups are almost always covered; the misses are 3- and 4-grams of new reasoning text.</p>
<div class="card">{legend([("2-gram", 1), ("3-gram", 2), ("4-gram", 3)])}{c_ord}
<details open><summary>Table view: candidate masks</summary><div class="scroll"><table><tr><th>mask</th><th>kept GiB</th>
<th>of table</th><th>calib coverage</th><th>eval hit</th><th>eval prompt</th><th>eval generated</th>
<th>eval 2/3/4-gram</th><th>tokens with all 48 hit</th></tr>{mask_rows}</table></div></details></div>

<h2>Bank occupancy by n-gram order</h2>
<p>Each (order, head) bank is a prime-sized range of about 16M rows. 4-gram banks are nearly saturated (long
reasoning makes most 4-grams new, so buckets collide), while 2-gram banks use a small, very concentrated set.</p>
<div class="card scroll"><table><tr><th>layer</th><th>order</th><th>rows (8 heads)</th><th>touched</th><th>touched share</th>
<th>lookups</th><th>top 10% rows' share</th></tr>{''.join(bank_rows)}</table></div>
<p>Touched rows by input role (layer 1): {fmt_int(rr['gen_only'])} only from generated tokens,
{fmt_int(rr['prompt_only'])} only from prompt/tool tokens, {fmt_int(rr['both'])} from both.</p>

<h2>Notes</h2>
<ul>
<li>Counts use consumed inputs only (each rollout minus its final token), once per position, 48 lookups per token over
both layers. Token sequences were rebuilt with the model's <code>encoding.py</code> and match the API's per-turn
prompt token counts on all 78,992 turns.</li>
<li>Hit rate is a selection diagnostic, not a quality result. Each candidate mask is checked next by teacher-forced
NLL on the eval traces and by full ARC eval runs on the self-hosted model.</li>
<li>Kaggle 4×L4 host RAM leaves about 122 GiB for the pinned table plus a 2.9 GiB original-to-compact map.</li>
</ul>
</main><div id="tip" role="tooltip"></div>
<script>
const tip = document.getElementById('tip');
function show(e, g) {{
  const [a, b] = g.dataset.tip.split('|');
  tip.replaceChildren();
  const s = document.createElement('b'); s.textContent = a; const t = document.createElement('span'); t.textContent = b;
  tip.append(s, t); tip.style.display = 'block';
  const r = g.getBoundingClientRect(); const x = e && e.clientX !== undefined ? e.clientX : r.left + r.width / 2;
  const y = e && e.clientY !== undefined ? e.clientY : r.top;
  tip.style.left = Math.min(x + 12, window.innerWidth - tip.offsetWidth - 8) + 'px';
  tip.style.top = Math.max(8, y - tip.offsetHeight - 12) + 'px';
}}
document.querySelectorAll('.hit').forEach(g => {{
  g.addEventListener('pointermove', e => show(e, g)); g.addEventListener('focus', () => show(null, g));
  g.addEventListener('pointerleave', () => tip.style.display = 'none'); g.addEventListener('blur', () => tip.style.display = 'none');
}});
</script></body></html>
"""
open(OUT, "w").write(page)
print(OUT, len(page))
