#!/usr/bin/env python3
"""E043: pack no-tool Flash-1017 eval runs (run_deepseek_arcprize.py results) into the rollout parquet schema.

Rebuilds each request exactly as the harness sent it (official prompt + optional task summary + optional effort
header) and appends the assistant turn from the saved reasoning trace and final content.

  make_notool_eval_parquet.py --runs e044_notool_deepinfra_eval120 arcprize_deepseek-flash_low_e035_hdrsum --out x.parquet
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/vu/Workspace/ARC-AGI-2")
sys.path.insert(0, str(ROOT / "evals"))
from run_deepseek_arcprize import EFFORT_HEADERS, NOTOOL_SUMMARY_INTRO, SUMMARY_INTRO, TASKS, build_prompt, task_summary  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = {p.stem: json.load(open(p)) for p in sorted(TASKS.glob("*.json"))}
    out = []
    for run in a.runs:
        for line in open(ROOT / f"evals/results/{run}.jsonl"):
            r = json.loads(line)
            if "error" in r or not r.get("reasoning_path"):
                continue
            task = tasks[r["task_id"]]
            prompt = build_prompt(task["train"], task["test"][r["pair"]]["input"])
            if r.get("task_summary"):
                prompt += task_summary(task["train"], task["test"][r["pair"]]["input"]).replace(
                    SUMMARY_INTRO, NOTOOL_SUMMARY_INTRO, 1)
            msgs = [{"role": "user", "content": prompt}]
            if r.get("effort_header"):
                msgs.insert(0, {"role": "system", "content": EFFORT_HEADERS[r["effort_header"]]})
            rp = Path(r["reasoning_path"])
            if not rp.is_absolute():
                rp = ROOT / rp
            if not rp.exists():
                rp = ROOT / "evals/results" / run / rp.name
            opener = gzip.open if str(rp).endswith(".gz") else open
            reasoning = opener(rp, "rt").read()
            msgs.append({"role": "assistant", "reasoning_content": reasoning, "content": r.get("content") or ""})
            out.append({"task_id": r["task_id"], "pair": r["pair"], "attempt": r["attempt"], "batch": run,
                        "parent": r["task_id"], "correct": bool(r.get("correct")), "train_pass": None,
                        "messages": json.dumps(msgs), "events": json.dumps([]), "usage": json.dumps(r.get("usage") or {})})
    df = pd.DataFrame(out)
    df.to_parquet(a.out)
    print(len(df), df.groupby("batch").correct.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
