"""Render the selected OLD/NEW sub80s40 traces (selection.json) to plain text for close reading. No API calls.
Adapted from experiments/E026_trace_strategy/scripts/ds_render.py: same layout, plus approximate cumulative
token positions ("[~tok 12.3k]") inserted every ~1500 reasoning chars (interpolated by chars within a turn),
and a per-turn summary table at the top.
Usage: .venv/bin/python -I render_pairs.py OUTDIR"""
import gzip, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent

def marks(text, start_tok, turn_tok, every=1500):
    if not text: return text
    n = len(text); out = []; i = 0
    while i < n:
        j = min(n, i + every)
        k = text.rfind("\n", i, j) if j < n else n
        if k <= i: k = j
        out.append(f"[~tok {(start_tok + turn_tok * i / n) / 1000:.1f}k] " + text[i:k])
        i = k
    return "".join(out)

def render(rec, label):
    d = json.load(gzip.open(rec["trace_path"], "rt"))
    msgs = d["messages"]; usage = [e["usage"] for e in d["events"] if "usage" in e]
    out, table, cum, ti = [], [], 0, 0
    for m in msgs:
        if m["role"] == "user":
            out.append("### USER (prompt + task grids omitted; see the task JSON)")
        elif m["role"] == "assistant":
            t = usage[ti]["completion_tokens"] if ti < len(usage) else 0
            start = cum; cum += t; ti += 1
            rc = m.get("reasoning_content") or ""
            calls = [c["function"]["name"] for c in (m.get("tool_calls") or [])]
            table.append(f"turn {ti:2d}: {t:6d} tok (cum {cum/1000:5.1f}k) reasoning {len(rc):6d} ch, calls {calls}")
            out.append(f"### ASSISTANT turn {ti}  [{t} output tokens; tokens {start} -> {cum} of {rec['usage']['completion']}]")
            if rc: out.append("[reasoning]\n" + marks(rc, start, t))
            if m.get("content"): out.append("[content]\n" + m["content"])
            for c in m.get("tool_calls") or []:
                a = c["function"]["arguments"]
                try: a = json.loads(a); a = a.get("code") or a.get("expr") or json.dumps(a)
                except Exception: pass
                out.append(f"[tool call {c['function']['name']}]\n{a}")
        elif m["role"] == "tool":
            c = m["content"]
            out.append("### TOOL OUTPUT\n" + (c if len(c) < 5000 else c[:3500] + "\n...[truncated]...\n" + c[-1200:]))
    hdr = (f"{label} {rec['task_id']} pair {rec['pair']} attempt {rec['attempt']}: correct={rec['correct']} "
           f"completion_tokens={rec['usage']['completion']} turns={rec['turns']} stop={rec.get('stop_reason')} "
           f"answer_source={rec.get('answer_source')}\n" + "\n".join(table) + "\n\n")
    return hdr + "\n\n".join(out)

def find(trace_path):
    jl = Path(trace_path).parent.with_suffix(".jsonl")
    for l in open(jl):
        r = json.loads(l)
        if r.get("trace_path") == trace_path: return r

od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
for s in json.load(open(HERE / "selection.json")):
    for lab in ("old", "new"):
        r = find(s[f"{lab}_trace"])
        (od / f"{s['task']}_p{s['pair']}_{lab.upper()}.txt").write_text(render(r, lab.upper() + (" (V4.1-Flash, E009)" if lab == "old" else " (V4.1-Flash-1017, E029)")))
print("rendered", od)
