#!/usr/bin/env python3
"""Fixed short ARC diagnostics; selection uses challenge geometry, never outcomes.

prepare/score/watch run locally. run_on_server is called by smoke.py during the
three verification arms, reusing their loaded engine. Answer keys stay local.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/E074_active_params"
TAGS = ("uniform6_verify", "uniform5_verify", "mean5_profiled_verify")


def prepare():
    source = ROOT / "data/arc-agi_evaluation_challenges.json"
    tasks = json.loads(source.read_text())
    def area(task):
        grids = [p[k] for split in ("train", "test") for p in task[split]
                 for k in ("input", "output") if k in p]
        return max(len(g) * len(g[0]) for g in grids)
    sizes = sorted((area(t), name) for name, t in tasks.items())
    size_bucket = {name: min(2, 3 * i // len(sizes)) for i, (_, name) in enumerate(sizes)}
    priority = lambda name: hashlib.sha256(("E074-short-74:" + name).encode()).hexdigest()
    chosen, strata = [], {}
    for multi in (False, True):
        for size in range(3):
            pool = [name for name, task in tasks.items()
                    if (len(task["test"]) > 1) == multi and size_bucket[name] == size]
            selected = sorted(pool, key=priority)[:2]
            chosen.extend(selected)
            strata[f"{'multi' if multi else 'single'}_size{size}"] = selected
    chosen.extend(sorted(set(tasks) - set(chosen), key=priority)[:12 - len(chosen)])
    assert len(chosen) == len(set(chosen)) == 12
    manifest = {"version": "Kaggle ARC-AGI-2 evaluation", "task_ids": sorted(chosen),
                "challenge_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "selection": "Hash seed E074-short-74; two per max-grid-area tercile x single/multi-test stratum; no outcomes",
                "strata": strata, "attempts": 1, "max_total_output": 24000,
                "budget_soft": 12000, "multi_budget": .5, "max_turns": 20,
                "context_limit": 65536, "workers": 12,
                "role": "Early diagnostic only; not used to allocate layers or choose candidate"}
    path = EXP / "data/short_eval_manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Refusing to alter an existing diagnostic subset"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def command(root, exp, out, tag, manifest):
    return [str(root / "tvenv/bin/python"),
            str(root / "repo/experiments/E066_dsv41_image_pruning/scripts/b32_local_eval.py"),
            "--mode", "grid", "--effort", "low", "--prompt", "lean2sub", "--answer-tool",
            "--final-mode", "session", "--lenient-parse", "--max-tokens", "24000",
            "--max-total-output", str(manifest["max_total_output"]),
            "--budget-soft", str(manifest["budget_soft"]), "--budget-notes", "0.01", "--budget-warn", "0.75",
            "--effort-header", "minimal", "--turns-note", "5", "--train-check", "--rescue-code",
            "--model", "dsv41", "--fallback-solve", "--multi-test", "--auto-repair",
            "--multi-budget", str(manifest["multi_budget"]), "--context-limit", str(manifest["context_limit"]),
            "--max-turns", str(manifest["max_turns"]), "--grid-format", "text+image", "--image-cell", "0",
            "--base-url", "http://127.0.0.1:18074/v1",
            "--extra-body", json.dumps({"chat_template_kwargs": {"thinking": True, "reasoning_effort": 50},
                                        "temperature": 1.0, "top_p": 1.0, "cache_salt": "e074-short-" + tag}),
            "--sandbox-url", "local", "--attempts", "1", "--workers", str(manifest["workers"]),
            "--tasks", ",".join(manifest["task_ids"]),
            "--challenges", str(root / "repo/data/arc-agi_evaluation_challenges.json"),
            "--out", str(out / "short_arc.jsonl")]


def run_on_server(root, exp, out, tag, env, remaining):
    manifest = json.loads((exp / "data/short_eval_manifest.json").read_text())
    (out / "short_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cmd = command(root, exp, out, tag, manifest)
    (out / "short_command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    assert remaining() >= 600, "Insufficient arm time remaining to start the short diagnostic"
    started = time.monotonic()
    print("START_SHORT_ARC", tag, "12 tasks x1, 24k base budget", flush=True)
    with (out / "short_eval.log").open("w") as log:
        subprocess.run(cmd, cwd=root / "repo/evals", env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=remaining())
    rows = [json.loads(line) for line in (out / "short_arc.jsonl").read_text().splitlines()]
    assert {r["task_id"] for r in rows} == set(manifest["task_ids"]), "Short diagnostic did not finish all tasks"
    assert all(r["attempt"] == 0 for r in rows)
    (out / "short_complete.json").write_text(json.dumps({
        "tasks": 12, "attempts": 1, "elapsed_s": time.monotonic() - started,
        "note": "24k base output budget; diagnostic, not full-budget accuracy"}) + "\n")
    print("DONE_SHORT_ARC", tag, flush=True)


def score():
    path = ROOT / "experiments/E066_dsv41_image_pruning/scripts/score_b32_engram.py"
    spec = importlib.util.spec_from_file_location("arc_scoring", path)
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    manifest = json.loads((EXP / "data/short_eval_manifest.json").read_text())
    tasks = manifest["task_ids"]
    data, arms = {}, {}
    for tag in TAGS:
        folder = EXP / "results" / tag
        if not (folder / "short_complete.json").exists():
            continue
        assert json.loads((folder / "short_manifest.json").read_text()) == manifest
        rows = [json.loads(line) for line in (folder / "short_arc.jsonl").read_text().splitlines()]
        assert {r["task_id"] for r in rows} == set(tasks)
        latest = {(r["task_id"], r["pair"], r["attempt"]): r for r in rows}
        assert all(r["attempt"] == 0 for r in rows)
        assert all((t, p, 0) in latest or any(r["task_id"] == t and "error" in r for r in rows)
                   for t in tasks for p in range(len(scoring.SOL[t]))), "Missing test-input records"
        data[tag] = scoring.load_run(folder / "short_arc.jsonl")
        p1, _ = scoring.score(data[tag], tasks, (0,))
        elapsed = json.loads((folder / "short_complete.json").read_text())["elapsed_s"]
        token_count = sum(r.get("usage", {}).get("completion", 0) for r in latest.values())
        arms[tag] = {"pass_at_1": p1, "elapsed_seconds": elapsed, "output_tokens": token_count,
                     "output_tokens_per_task": token_count / len(tasks),
                     "end_to_end_output_tokens_per_second": token_count / elapsed,
                     "error_tasks": len({r["task_id"] for r in rows if "error" in r}),
                     "tasks_with_all_predictions": sum(all(latest.get((t, p, 0), {}).get("pred") is not None
                         for p in range(len(scoring.SOL[t]))) for t in tasks),
                     "stop_counts_by_task": dict(Counter(r.get("stop_reason", r.get("stop", "unknown"))
                         for (t, p, a), r in latest.items() if p == 0)),
                     "per_task_pass1": {t: scoring.score(data[tag], [t], (0,))[0] for t in tasks}}
    comparisons = {}
    if "uniform6_verify" in data:
        for tag in TAGS[1:]:
            if tag in data:
                delta, lo, hi, better, worse = scoring.paired(data[tag], data["uniform6_verify"], tasks, (0,))
                comparisons[tag] = {"pass1_difference_pp": delta, "paired_task_bootstrap95_pp": [lo, hi],
                                    "tasks_better": better, "tasks_worse": worse}
    report = {"version": manifest["version"], "task_count": len(tasks), "attempts": 1,
              "weighting": "Equal task weight; mean correctness over each task's test inputs",
              "max_total_output": 24000, "arms": arms, "comparisons_vs_uniform6": comparisons,
              "limitation": "12 tasks, one draw, shortened budget: diagnostic only; does not establish the 2pp target",
              "selection_policy": "Layer schedule and candidate selection remain based on training-family NLL",
              "updated_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    path = EXP / "results/short_arc_report.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(path)
    print(json.dumps(report, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("prepare", "score", "watch"))
    ap.add_argument("--hours", type=float, default=16)
    args = ap.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "score":
        score()
    else:
        deadline, previous = time.monotonic() + args.hours * 3600, None
        while time.monotonic() < deadline:
            signature = tuple(tag for tag in TAGS if (EXP / "results" / tag / "short_complete.json").exists())
            if signature and signature != previous:
                try:
                    score()
                    previous = signature
                except (FileNotFoundError, json.JSONDecodeError, AssertionError) as error:
                    print(f"Waiting for complete result sync: {type(error).__name__}: {error}", flush=True)
            if len(signature) == len(TAGS):
                if previous == signature:
                    return
            state = json.loads((EXP / "results/batch_status.json").read_text())
            if state.get("state") in ("stopped", "complete"):
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
