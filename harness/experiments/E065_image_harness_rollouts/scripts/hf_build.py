"""E065: package the image set (e065_*) and text twin (e065t_*) as configs of the HF rollouts dataset, in the E038 format.
Usage (repo root): .venv/bin/python experiments/E065_image_harness_rollouts/scripts/hf_build.py OUT_DIR
Writes OUT_DIR/rollouts/e065_{images,text}_<batch>.parquet, OUT_DIR/tasks/e065_pairs_*.txt, OUT_DIR/run_config_e065.json.
Image-set `messages` keep the images inline (OpenAI `image_url` parts with base64 PNG data URLs)."""
import gzip, json, shutil, sys
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
ROOT = Path("/home/vu/Workspace/ARC-AGI-2"); R = ROOT / "evals/results"; HERE = ROOT / "experiments/E065_image_harness_rollouts"
OUT = Path(sys.argv[1]); E025 = ROOT / "experiments/E025_train_rollouts/data"
(OUT / "rollouts").mkdir(parents=True, exist_ok=True); (OUT / "tasks").mkdir(exist_ok=True)
man = {json.loads(l)["task_id"]: json.loads(l) for l in open(E025 / "manifest.jsonl")}
P = "tools_persistent_grid_lean2sub_deepseek-flash_low_"
BATCHES = ["arc2_train", "nvarc_b0", "nvarc_b1"]
configs = {}
for kind, tag in [("images", "e065"), ("text", "e065t")]:
    for b in BATCHES:
        name, recs, rows = f"e065_{kind}_{b}", {}, []
        for r in map(json.loads, open(R / f"{P}{tag}_{b}.jsonl")):
            recs[(r["task_id"], r["pair"], r["attempt"])] = r  # resumed attempts: last record wins
        for r in recs.values():
            m, u = man.get(r["task_id"], {}), r.get("usage") or {}
            tr = json.load(gzip.open(r["trace_path"], "rt")) if r.get("trace_path") and Path(r["trace_path"]).exists() else {}
            rows.append(dict(task_id=r["task_id"], batch=m.get("batch", b), source=m.get("source", "kaggle_arc2_training"),
                parent=m.get("parent"), pair=r["pair"], attempt=r["attempt"], correct=r.get("correct"),
                train_pass=r.get("train_pass"), train_frac=r.get("train_frac"), answer_source=r.get("answer_source"),
                stop_reason=r.get("stop_reason"), error=r.get("error"), pred=json.dumps(r.get("pred")), turns=r.get("turns"),
                tool_calls=r.get("tool_calls"), prompt_tokens=u.get("prompt"), cache_hit_tokens=u.get("cache_hit"),
                completion_tokens=u.get("completion"), seconds=r.get("seconds"), est_cost_usd=r.get("est_cost_usd"),
                model=r.get("model"), provider="DeepSeek API", harness="E058 (E046 best + images)" if kind == "images"
                else "E046 best (text)", messages=json.dumps(tr.get("messages")), events=json.dumps(tr.get("events"))))
            cfg = dict(r["config"]); cfg["challenges"] = str(Path(cfg["challenges"]).relative_to(ROOT))
            if cfg.get("solutions"): cfg["solutions"] = str(Path(cfg["solutions"]).relative_to(ROOT))
            configs[name] = cfg
        pq.write_table(pa.Table.from_pylist(rows), OUT / "rollouts" / f"{name}.parquet", compression="zstd")
        print(name, len(rows), "correct", sum(bool(x["correct"]) for x in rows), "errors", sum(bool(x["error"]) for x in rows),
              "no-trace", sum(x["messages"] == "null" for x in rows))
json.dump(configs, open(OUT / "run_config_e065.json", "w"), indent=1)
for f in ["pairs_arc2_train.txt", "pairs_nvarc_b0.txt", "pairs_nvarc_b1_first3570.txt"]:
    shutil.copy(HERE / f, OUT / "tasks" / f"e065_{f}")
