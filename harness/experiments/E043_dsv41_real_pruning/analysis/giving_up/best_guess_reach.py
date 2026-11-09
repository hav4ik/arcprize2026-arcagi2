#!/usr/bin/env python3
"""I7 offline estimate: which E128 attempts would `--best-guess-at F --best-guess-turns K` reach, and what they do now.

The nudge fires once, on the tool result of the first turn t (1-based) that has tool results, is not the accepted
submit, and has (output tokens through t >= F x 80k or t >= 40 - K), while another turn can still follow
(tokens < 80k and t < 40). Read-only: E128 records + traces, giving_up/results.json for the train-pass columns."""
import collections
import gzip
import json
import sys

ROOT = "/home/vu/Workspace/ARC-AGI-2"
RUN = f"{ROOT}/evals/results/tools_persistent_grid_lean2sub_dsv41_low_e043_e128t_w10_u2_gptq_eng64_sub80.jsonl"
ROWS = f"{ROOT}/experiments/E043_dsv41_real_pruning/analysis/giving_up/results.json"


def trigger(rec, f, k):
    cfg = rec["config"]
    budget, max_turns = cfg["max_total_output"], cfg["max_turns"]
    trace = json.load(gzip.open(rec["trace_path"]))
    msgs = trace["messages"]
    usage = [e["usage"] for e in trace["events"] if "usage" in e]
    asst = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    assert len(asst) == len(usage) == rec["turns"], (rec["task_id"], len(asst), len(usage), rec["turns"])
    cum = 0
    for t, (i, u) in enumerate(zip(asst, usage), 1):
        cum += u.get("completion_tokens") or 0
        has_tool = bool(msgs[i].get("tool_calls")) and i + 1 < len(msgs) and msgs[i + 1]["role"] == "tool"
        accepted = rec["stop_reason"] == "submitted" and t == len(asst)
        if not has_tool or accepted:
            continue
        if (f and cum >= f * budget) or (k and t >= max_turns - k):
            if cum < budget and t < max_turns:
                why = "tokens" if f and cum >= f * budget else "turns"
                return {"turn": t, "tokens": cum, "why": why, "total": rec["usage"]["completion"],
                        "turns": len(asst)}
            return None
    return None


def main(f=0.8, k=3):
    recs = [json.loads(line) for line in open(RUN)]
    rows = {tuple(r["key"]): r for r in json.load(open(ROWS))["rows"] if r["model"] == "E128"}
    total = sum(r["usage"]["completion"] for r in recs)
    hit = []
    for r in recs:
        tr = trigger(r, f, k)
        if tr:
            row = rows.get((r["task_id"], r["pair"], r["attempt"]), {})
            fp = row.get("first_pass_tok")
            tr.update(stop=r["stop_reason"], correct=bool(r.get("correct")), train_pass=r.get("train_pass"),
                      pass_before=fp is not None and fp <= tr["tokens"], any_pass=row.get("any_pass"),
                      key=(r["task_id"], r["pair"], r["attempt"]))
            hit.append(tr)
    after = sum(h["total"] - h["tokens"] for h in hit)
    print(f"E128: {len(recs)} attempts, {total / 1e6:.2f}M output tokens; F={f} K={k}")
    print(f"reached: {len(hit)} ({collections.Counter(h['why'] for h in hit)})")
    print("current stop / correct:", sorted(collections.Counter((h["stop"], h["correct"]) for h in hit).items()))
    print("all-train-pass output anywhere in the attempt:", collections.Counter(h["any_pass"] for h in hit),
          "| before the trigger:", sum(h["pass_before"] for h in hit))
    sub = [h for h in hit if h["stop"] == "submitted"]
    print(f"submitted later anyway: {len(sub)}, correct {sum(h['correct'] for h in sub)}; "
          f"turns after trigger median {sorted(h['turns'] - h['turn'] for h in sub)[len(sub) // 2] if sub else '-'}")
    nos = [h for h in hit if h["stop"] != "submitted"]
    print(f"ran to a cap: {len(nos)}, correct {sum(h['correct'] for h in nos)} "
          f"(never pass: {sum(not h['any_pass'] for h in nos)})")
    print(f"tokens after the trigger: {after / 1e3:.0f}k ({100 * after / total:.1f}% of all E128 output); "
          f"in cap-ended attempts {sum(h['total'] - h['tokens'] for h in nos) / 1e3:.0f}k, "
          f"in later submits {sum(h['total'] - h['tokens'] for h in sub) / 1e3:.0f}k")
    print(f"trigger point: tokens median {sorted(h['tokens'] for h in hit)[len(hit) // 2] / 1e3:.1f}k, "
          f"turn median {sorted(h['turn'] for h in hit)[len(hit) // 2]}")
    never = [r for r in recs if not rows.get((r["task_id"], r["pair"], r["attempt"]), {}).get("any_pass")
             and r["stop_reason"] != "submitted"]
    print(f"never-pass attempts that ran to a cap: {len(never)}; reached: "
          f"{sum(any(h['key'] == (r['task_id'], r['pair'], r['attempt']) for h in hit) for r in never)}")
    return hit


if __name__ == "__main__":
    main(*(float(a) for a in sys.argv[1:2]), *(int(a) for a in sys.argv[2:3]))
