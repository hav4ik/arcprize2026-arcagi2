"""Compare the AgentConfig the Kaggle notebook runs with (solver.build_agent_config under a settings file) with the
config E020 recorded for an eval arm (first record of its results file), field by field.

    .venv/bin/python experiments/E021_kaggle_v3/scripts/k3_config_diff.py experiments/E021_kaggle_v3/v4_settings.json \
        evals/results/qwen_tools_e020_K3_all.jsonl
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
settings = json.load(open(sys.argv[1]))
run_cfg = json.loads(open(sys.argv[2]).readline())["config"]
for k, v in settings.items():
    if k.startswith("ARC_"):
        os.environ[k] = str(v)
sys.path[:0] = [str(ROOT / "kaggle"), str(ROOT / "evals")]
import build_notebook  # noqa: E402
ns = {"__name__": "k3diff"}
exec(compile(build_notebook.solver_source(), "solver.py", "exec"), ns)  # module level only (no main, no kills)
from arc_harness import agent_qwen as AQ  # noqa: E402
cfg = AQ.config_dict(ns["build_agent_config"](AQ, int(ns["TOOLS_CAP"])), run_cfg.get("model"))
keys = sorted(set(cfg) | set(run_cfg))
diff = [(k, run_cfg.get(k, "<absent>"), cfg.get(k, "<absent>")) for k in keys if run_cfg.get(k, "<absent>") != cfg.get(k, "<absent>")]
print(f"{len(keys)} fields; {len(keys) - len(diff)} equal; differences (run vs notebook):")
for k, a, b in diff:
    print(f"  {k}: {a!r} -> {b!r}")
