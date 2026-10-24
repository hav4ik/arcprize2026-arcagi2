"""Package E031 + E036 + E038 (+ E037 probe) rollouts for HF, in the E025 dataset format."""
import gzip, json, shutil, sys
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
ROOT = Path("/home/vu/Workspace/ARC-AGI-2"); R = ROOT / "evals/results"
OUT = Path(sys.argv[1]); E025 = ROOT / "experiments/E025_train_rollouts/data"
man = {json.loads(l)["task_id"]: json.loads(l) for l in open(E025 / "manifest.jsonl")}
P = "tools_persistent_grid_lean2sub_deepseek-flash_low_"
SPLITS = {
    "e031_arc2_train": [P + "e031_arc2_train.jsonl"],
    "e031_nvarc_b0": [P + "e031_nvarc_b0.jsonl"],
    "e036_nvarc_b1": [P + "e036_nvarc_b1.jsonl", P + "e036_nvarc_b1x.jsonl"],
    "e038_openrouter_deepinfra_b1r": ["e038_deepinfra_nvarc_b1r.jsonl"],
    "e037_openrouter_deepinfra": ["e037_openrouter_deepinfra_smoke.jsonl", "e037_openrouter_deepinfra_probe.jsonl"],
}
def latest(files):
    recs = {}
    for f in files:
        for r in map(json.loads, open(R / f)):
            recs[(r["task_id"], r["pair"], r["attempt"])] = r  # resumed attempts: last record wins
    return list(recs.values())
configs = {}
for name, files in SPLITS.items():
    rows = []
    for r in latest(files):
        m = man.get(r["task_id"], {})
        tr = {}
        if r.get("trace_path") and Path(r["trace_path"]).exists():
            tr = json.load(gzip.open(r["trace_path"], "rt"))
        u = r.get("usage") or {}
        rows.append(dict(task_id=r["task_id"], batch=m.get("batch", "arc2_train" if "arc2" in name else None),
            source=m.get("source", "kaggle_arc2_training" if "arc2" in name else None), parent=m.get("parent"),
            pair=r["pair"], attempt=r["attempt"], correct=r.get("correct"), train_pass=r.get("train_pass"),
            train_frac=r.get("train_frac"), answer_source=r.get("answer_source"), stop_reason=r.get("stop_reason"),
            error=r.get("error"), pred=json.dumps(r.get("pred")), turns=r.get("turns"), tool_calls=r.get("tool_calls"),
            prompt_tokens=u.get("prompt"), cache_hit_tokens=u.get("cache_hit"), completion_tokens=u.get("completion"),
            seconds=r.get("seconds"), est_cost_usd=r.get("est_cost_usd"), model=r.get("model"),
            provider="DeepInfra (OpenRouter, fp8)" if name.startswith(("e037", "e038")) else "DeepSeek API",
            messages=json.dumps(tr.get("messages")), events=json.dumps(tr.get("events"))))
        cfg = dict(r["config"]); cfg["challenges"] = str(Path(cfg["challenges"]).relative_to(ROOT))
        if cfg.get("solutions"): cfg["solutions"] = str(Path(cfg["solutions"]).relative_to(ROOT))
        configs[name] = cfg
    pq.write_table(pa.Table.from_pylist(rows), OUT / "rollouts" / f"{name}.parquet", compression="zstd")
    print(name, len(rows), sum(bool(x["correct"]) for x in rows), sum(x["messages"] == "null" for x in rows), "no-trace")
json.dump(configs, open(OUT / "run_config.json", "w"), indent=1)
for f in ["manifest.jsonl", "nvarc_order.txt", "nvarc_b0_challenges.json", "nvarc_b0_solutions.json",
          "nvarc_b1_challenges.json", "nvarc_b1_solutions.json"]:
    shutil.copy(E025 / f, OUT / "tasks" / f)
for src in ["E031_flash1017_hdr_rollouts/pairs_arc2_train.txt", "E031_flash1017_hdr_rollouts/pairs_nvarc_b0.txt",
            "E036_nvarc_balanced_rollouts/pairs_nvarc_b1.txt", "E036_nvarc_balanced_rollouts/pairs_nvarc_b1x.txt",
            "E037_openrouter_probe/pairs_probe.txt", "E038_deepinfra_b1_finish/pairs_nvarc_b1r.txt", "E037_openrouter_probe/pairs_smoke.txt"]:
    a, b = src.split("/"); shutil.copy(ROOT / "experiments" / src, OUT / "tasks" / f"{a.split('_')[0].lower()}_{b}")
shutil.copy(ROOT / "experiments/E025_train_rollouts/build_dataset.py", OUT / "build_dataset.py")
