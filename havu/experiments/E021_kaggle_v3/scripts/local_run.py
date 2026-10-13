"""E021 functional run of the v3 notebook code off Kaggle, on the rerun path with short deadlines.

Uses the modules extracted from the built notebook (kaggle/tests/check_notebook.py --variant v3 --extract-to DIR),
the notebook's own guard cell and final cell (paths remapped), and an already running OpenAI-compatible server.
The input dir holds an `arc-agi_test_challenges.json` made of public-eval tasks: its ids differ from the public
test file, so the solver takes the competition-rerun path (mode submit, deadlines from T0). Scored afterwards
against the eval solutions.

    .venv/bin/python experiments/E021_kaggle_v3/scripts/local_run.py --bundle /tmp/e021_nb_v3 --out DIR \
        --tasks 1818057f,2ba387bc --force 900 --hard 915 --emergency 1000 --guard 1040 --env '{"ARC_...": ...}'
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "kaggle"))
import build_notebook  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--server", default="http://localhost:30000/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--force", type=float, required=True)
    ap.add_argument("--hard", type=float, required=True)
    ap.add_argument("--emergency", type=float, required=True)
    ap.add_argument("--guard", type=float, required=True)
    ap.add_argument("--env", default="{}")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    work, inp = out / "working", out / "input"
    (inp / "arc-prize-2026-arc-agi-2").mkdir(parents=True)
    shutil.copytree(a.bundle, work / "arc_solver", ignore=shutil.ignore_patterns("__pycache__"))
    ch = json.load(open(ROOT / "data/arc-agi_evaluation_challenges.json"))
    sol = json.load(open(ROOT / "data/arc-agi_evaluation_solutions.json"))
    ids = a.tasks.split(",")
    tasks = {t: {"train": ch[t]["train"], "test": [{"input": p["input"]} for p in ch[t]["test"]]} for t in ids}
    (inp / "arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json").write_text(json.dumps(tasks))
    t0 = time.time()
    env = dict(os.environ, ARC_T0=str(t0), ARC_MODE="local", ARC_WORK=str(work), ARC_INPUT=str(inp),
               ARC_EXTERNAL_SERVER=a.server, ARC_MODEL_ID=a.model, ARC_FORCE_AT_S=str(a.force),
               ARC_HARD_STOP_S=str(a.hard), ARC_EMERGENCY_S=str(a.emergency), ARC_GUARD_S=str(a.guard))
    env.pop("KAGGLE_IS_COMPETITION_RERUN", None)
    env.pop("PYTHONPATH", None)
    env.update({k: str(v) for k, v in json.loads(a.env).items()})
    guard = build_notebook.GUARD_CELL.replace("/kaggle/working", str(work))
    final = (build_notebook.FINAL_CELL.replace("/kaggle/working", str(work)).replace("/kaggle/input", str(inp))
             .split("# Commit-run gate")[0])
    log = open(out / "run.log", "w")
    r = subprocess.run([sys.executable, "-u", "-c", guard], env=env, cwd=str(work), stdout=log,
                       stderr=subprocess.STDOUT)
    r2 = subprocess.run([sys.executable, "-c", final], env=env, cwd=str(work), capture_output=True, text=True)
    log.write("\nFINAL CELL: " + r2.stdout + r2.stderr)
    log.close()
    sub = json.load(open(work / "submission.json"))
    per, tot = [], 0.0
    for t in ids:
        n = len(ch[t]["test"])
        ok = 0
        for p in range(n):
            s = sub[t][p]
            c1, c2 = s["attempt_1"] == sol[t][p], s["attempt_2"] == sol[t][p]
            ok += c1 or c2
            per.append({"task": t, "pair": p, "attempt_1": c1, "attempt_2": c2,
                        "a1_is_input": s["attempt_1"] == ch[t]["test"][p]["input"],
                        "a2_is_input": s["attempt_2"] == ch[t]["test"][p]["input"]})
        tot += ok / n
    res = {"guard_rc": r.returncode, "final_rc": r2.returncode, "wall_s": round(time.time() - t0),
           "score": f"{tot:.2f}/{len(ids)} tasks", "pairs": per,
           "leftover_procs": subprocess.run(["pgrep", "-f", str(work)], capture_output=True, text=True).stdout.split()}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}))


if __name__ == "__main__":
    main()
