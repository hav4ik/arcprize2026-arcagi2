"""Render sampled DeepSeek traces (NT / S80 attempt 0 / V1) to plain text for reading. No API calls.
Usage: .venv/bin/python -I ds_render.py OUTDIR task:pair [task:pair ...]"""
import gzip, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "evals/results"
F = {"NT": RES / "arcprize_deepseek-flash_max.jsonl",
     "S80": RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40.jsonl",
     "V1": RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e017_full_v1sum.jsonl"}

def load(f):
    d = {}
    for l in open(f):
        r = json.loads(l)
        if "error" not in r:
            d[(r["task_id"], int(r["pair"]), int(r["attempt"]))] = r
    return d

def render_tool(r, show_prompt):
    d = json.load(gzip.open(r["trace_path"], "rt"))
    msgs = d["messages"]; usage = [e["usage"] for e in d["events"] if "usage" in e]
    out, cum, ti = [], 0, 0
    for m in msgs:
        if m["role"] == "user":
            c = m["content"]
            if show_prompt:
                out.append("### USER (first message; prompt + task" + (" + computed summary" if "computed summary" in c else "") + ")\n" + c)
            else:
                i = c.find("A computed summary")
                out.append("### USER (prompt+grids omitted; same task JSON as given to you)" + ("\n" + c[i:] if i >= 0 else ""))
        elif m["role"] == "assistant":
            t = usage[ti]["completion_tokens"] if ti < len(usage) else 0
            cum += t; ti += 1
            out.append(f"### ASSISTANT turn {ti}  [{t} output tokens; cumulative {cum} of {r['usage']['completion']}]")
            if m.get("reasoning_content"): out.append("[reasoning]\n" + m["reasoning_content"])
            if m.get("content"): out.append("[content]\n" + m["content"])
            for c in m.get("tool_calls") or []:
                a = c["function"]["arguments"]
                try: a = json.loads(a); a = a.get("code") or a.get("expr") or json.dumps(a)
                except Exception: pass
                out.append(f"[tool call {c['function']['name']}]\n{a}")
        elif m["role"] == "tool":
            c = m["content"]
            out.append("### TOOL OUTPUT\n" + (c if len(c) < 4000 else c[:3000] + "\n...[truncated]...\n" + c[-800:]))
    return "\n\n".join(out)

def main():
    od = Path(sys.argv[1]); od.mkdir(parents=True, exist_ok=True)
    data = {k: load(v) for k, v in F.items()}
    for tp in sys.argv[2:]:
        t, p = tp.split(":"); p = int(p)
        r = data["NT"][(t, p, 2)]
        txt = gzip.open(ROOT / r["reasoning_path"], "rt").read()
        hdr = f"NT (no tools, max effort) {t} pair {p}: correct={r['correct']} completion_tokens={r['usage']['completion_tokens']} reasoning_chars={len(txt)}\n"
        (od / f"{t}_p{p}_NT.txt").write_text(hdr + "\n[reasoning]\n" + txt + "\n\n[final content]\n" + str(r.get("content"))[:3000])
        for run in ("S80", "V1"):
            r = data[run][(t, p, 0)]
            hdr = f"{run} {t} pair {p} attempt 0: correct={r['correct']} train_pass={r.get('train_pass')} completion_tokens={r['usage']['completion']} turns={r['turns']}\n"
            (od / f"{t}_p{p}_{run}.txt").write_text(hdr + render_tool(r, show_prompt=False))
main()
