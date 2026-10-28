#!/usr/bin/env python3
"""E060: reasoning-length statistics from harness traces (run_deepseek_tools.py records + their messages.json.gz).

Per assistant turn: output tokens (the server's usage.completion_tokens), reasoning length in characters and in
tokens (the model's tokenizer.json, so API and self-hosted runs compare), finish reason (length = hit the per-turn
cap). Per attempt: output tokens, turns, reasoning tokens, thinking share (reasoning tokens / output tokens), stop
reason (output_budget = hit the per-attempt cap, max_turns = hit the turn cap).

    python reasoning_stats.py --tokenizer tokenizer.json RUN.jsonl[+RUN_retry.jsonl] [...]

Trace paths come from each record's trace_path; --trace-root rewrites a different prefix (e.g. a box path) to a
local copy: --trace-root /root/e060/runs=experiments/E060_flashnext_selfhost/results
"""
import argparse
import gzip
import json
import statistics as st
from collections import Counter
from pathlib import Path


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def dist(xs, scale=1.0, unit=""):
    if not xs:
        return "-"
    return (f"mean {st.mean(xs) / scale:.1f}{unit} / median {st.median(xs) / scale:.1f}{unit} / "
            f"p90 {pct(xs, 0.9) / scale:.1f}{unit} / max {max(xs) / scale:.1f}{unit}")


def load(spec):
    out = {}
    for f in spec.split("+"):
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                k = (r["task_id"], int(r["pair"]), int(r["attempt"]))
                if "error" in r and k in out and "error" not in out[k]:
                    continue
                out[k] = r
    return {k: r for k, r in out.items() if "error" not in r and not r.get("multi_share")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--trace-root", default="", help="OLD=NEW prefix rewrite for trace paths")
    ap.add_argument("--json", help="write per-run summary JSON here")
    args = ap.parse_args()
    import tokenizers
    tok = tokenizers.Tokenizer.from_file(args.tokenizer)
    old, new = args.trace_root.split("=", 1) if args.trace_root else ("", "")
    summary = {}
    for spec in args.runs:
        recs = load(spec)
        t_out, t_rchars, t_rtok, t_fin, a_out, a_rtok, a_share, a_turns, a_stop = [], [], [], Counter(), [], [], [], [], Counter()
        missing = 0
        for r in recs.values():
            p = r.get("trace_path", "")
            if old and p.startswith(old):
                p = new + p[len(old):]
            try:
                trace = json.load(gzip.open(p))
            except Exception:
                missing += 1
                continue
            msgs = [m for m in trace["messages"] if m.get("role") == "assistant"]
            usages = [e for e in trace["events"] if "usage" in e]
            rt_sum = 0
            for m, e in zip(msgs, usages):
                rc = m.get("reasoning_content") or ""
                n_r = len(tok.encode(rc, add_special_tokens=False).ids) if rc else 0
                rt_sum += n_r
                u = e["usage"] or {}
                t_out.append(u.get("completion_tokens", 0))
                t_rchars.append(len(rc))
                t_rtok.append(n_r)
                t_fin[e.get("finish_reason") or "none"] += 1
            out_tok = r["usage"]["completion"]
            a_out.append(out_tok)
            a_rtok.append(rt_sum)
            a_share.append(rt_sum / out_tok if out_tok else 0)
            a_turns.append(r.get("turns", len(msgs)))
            a_stop[r.get("stop_reason") or "none"] += 1
        name = Path(spec.split("+")[0]).stem
        share_all = sum(a_rtok) / sum(a_out) if sum(a_out) else 0
        print(f"\n## {name}: {len(a_out)} attempts ({missing} traces missing), {len(t_out)} turns")
        print(f"- output tokens / attempt: {dist(a_out, 1e3, 'k')}")
        print(f"- turns / attempt: {dist(a_turns)}")
        print(f"- reasoning tokens / attempt: {dist(a_rtok, 1e3, 'k')}; thinking share of output {share_all:.1%} "
              f"(per-attempt mean {st.mean(a_share) if a_share else 0:.1%})")
        print(f"- output tokens / turn: {dist(t_out, 1e3, 'k')}")
        print(f"- reasoning tokens / turn: {dist(t_rtok, 1e3, 'k')}")
        print(f"- reasoning chars / turn: {dist(t_rchars, 1e3, 'k')}")
        print(f"- turn finish reasons: {dict(t_fin.most_common())} (length = per-turn cap hit)")
        print(f"- attempt stop reasons: {dict(a_stop.most_common())} (output_budget = attempt cap, max_turns = turn cap)")
        summary[name] = {"attempts": len(a_out), "turns": len(t_out), "out_per_attempt": a_out, "turns_per_attempt": a_turns,
                         "reasoning_tok_per_attempt": a_rtok, "reasoning_tok_per_turn": t_rtok,
                         "reasoning_chars_per_turn": t_rchars, "out_per_turn": t_out, "turn_finish": dict(t_fin),
                         "stops": dict(a_stop), "thinking_share": share_all}
    if args.json:
        Path(args.json).write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
