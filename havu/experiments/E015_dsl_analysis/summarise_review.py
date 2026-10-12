"""Tally the manual Q3 review (review/solved.jsonl, review/hard.jsonl).

    .venv/bin/python -I experiments/E015_dsl_analysis/summarise_review.py
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
for name, opkey in (("solved", "ops"), ("hard", "concepts")):
    rows = [json.loads(line) for line in open(f"{HERE}/review/{name}.jsonl")]
    v = Counter(r["dsl_verdict"] for r in rows)
    print(f"== {name}: n={len(rows)} verdicts={dict(v)}")
    print("  ops:", Counter(o for r in rows for o in r[opkey]).most_common())
    print("  missing:", Counter(m for r in rows for m in r["missing"]).most_common())
    for verdict in ("yes", "extra", "no"):
        print(f"  ops among {verdict}:", Counter(o for r in rows if r["dsl_verdict"] == verdict
                                                for o in r[opkey]).most_common(6))
    if name == "solved":
        progs = {json.loads(line)["task"]: json.loads(line) for line in open(f"{HERE}/programs.jsonl")}
        for verdict in ("yes", "extra", "no"):
            locs = sorted(progs[r["task"]]["loc"] for r in rows if r["dsl_verdict"] == verdict)
            est = [r["est_dsl_lines"] for r in rows if r["dsl_verdict"] == verdict and r["est_dsl_lines"]]
            print(f"  {verdict}: median DeepSeek LOC {locs[len(locs) // 2]}; median est DSL lines "
                  f"{sorted(est)[len(est) // 2] if est else None}")
    else:
        print("  dsl_would_help_discovery:", Counter(r["dsl_would_help_discovery"] for r in rows))
