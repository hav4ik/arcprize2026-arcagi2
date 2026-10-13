"""Render sampled traces as readable text for annotation (no API). Writes rendered/<sid>.txt."""
import gzip, json, re, sys
from pathlib import Path
OUT = Path(__file__).resolve().parents[1]
DIG = re.compile(r"^[\s\[\],0-9.|:x=-]*$")

def collapse(text, keep=4):
    lines, out, run = text.splitlines(), [], []
    def flush():
        if len(run) > keep + 2:
            out.extend(run[:keep]); out.append(f"      [... {len(run) - keep - 1} more digit-only lines ...]"); out.append(run[-1])
        else:
            out.extend(run)
        run.clear()
    for l in lines:
        if len(l.strip()) >= 7 and DIG.match(l) and sum(c.isdigit() for c in l) >= 5:
            run.append(l)
        else:
            flush(); out.append(l)
    flush()
    return "\n".join(out)

def render(s, tool_cap=2500):
    d = json.load(gzip.open(s["trace_path"]))
    msgs, usage = d["messages"], [e for e in d["events"] if "usage" in e]
    total = s["tokens"]
    parts = [f"# {s['sid']} task {s['task_id']} pair {s['pair']} attempt {s['attempt']} run {s['run']} | outcome {s['outcome']} | total output tokens {total} | turns {s['turns']}\n"]
    cum, ai = 0, 0
    for m in msgs[1:]:
        if m["role"] == "assistant":
            t = usage[ai]["usage"].get("completion_tokens", 0) if ai < len(usage) else 0
            ai += 1
            start = cum; cum += t
            parts.append(f"\n===== TURN {ai} | {t} tokens | cumulative {start}->{cum} ({100*start/total:.0f}%->{100*cum/total:.0f}% of attempt) =====")
            if m.get("reasoning_content"):
                parts.append("[REASONING]\n" + collapse(m["reasoning_content"]))
            if m.get("content"):
                parts.append("[CONTENT]\n" + m["content"][:3000])
            for c in m.get("tool_calls") or []:
                try:
                    code = json.loads(c["function"]["arguments"]).get("code", "")
                except Exception:
                    code = c["function"]["arguments"]
                parts.append(f"[TOOL CALL {c['function']['name']}]\n{code}")
        elif m["role"] == "tool":
            c = m["content"]
            parts.append("[TOOL OUTPUT]\n" + (c if len(c) <= tool_cap else c[:tool_cap] + f"\n[... {len(c) - tool_cap} chars cut by renderer ...]" + c[-300:]))
        elif m["role"] == "user":
            parts.append("[USER]\n" + m["content"][:2000])
    return "\n".join(parts)

if __name__ == "__main__":
    (OUT / "rendered").mkdir(exist_ok=True)
    for l in open(OUT / "sample.jsonl"):
        s = json.loads(l)
        txt = render(s)
        (OUT / "rendered" / f"{s['sid']}.txt").write_text(txt)
        print(s["sid"], s["outcome"], s["tokens"], len(txt))
