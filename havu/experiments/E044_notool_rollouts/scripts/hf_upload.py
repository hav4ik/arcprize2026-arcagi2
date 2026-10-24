"""E044: package the no-tool rollouts collected so far and push them to the HF dataset as `notool_*` configs.
Usage: .venv/bin/python experiments/E044_notool_rollouts/scripts/hf_upload.py [--loop SECONDS] [--until-done PIDFILE]
Each cycle: build parquet from completed (non-error) rows, secret-scan the results and traces, upload only on 0 hits.
The eval120 check run is never uploaded (eval contamination)."""
import argparse, gzip, json, os, subprocess, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "evals"))
from run_deepseek_arcprize import EFFORT_HEADERS, NOTOOL_SUMMARY_INTRO, build_prompt, load_kaggle  # noqa: E402
from task_summary import INTRO as SUMMARY_INTRO, task_summary  # noqa: E402

REPO = "livobereshka/arc-agi2-deepseek-flash1017-rollouts-e031-e036"
HERE = ROOT / "experiments/E044_notool_rollouts"; R = ROOT / "evals/results"; E025 = ROOT / "experiments/E025_train_rollouts/data"
A2 = (ROOT / "data/arc-agi_training_challenges.json", ROOT / "data/arc-agi_training_solutions.json")
B0 = (E025 / "nvarc_b0_challenges.json", E025 / "nvarc_b0_solutions.json")
B1 = (E025 / "nvarc_b1_challenges.json", E025 / "nvarc_b1_solutions.json")
SPLITS = {  # config -> (source name, results files in priority order, task files, pair list or None = every row)
    "notool_e044_arc2_train": ("arc2_train", ["deepinfra"], A2, "pairs_arc2_train.txt"),
    "notool_e044_nvarc_b0": ("nvarc_b0", ["deepinfra", "deepseek"], B0, "pairs_nvarc_b0.txt"),
    "notool_e044_nvarc_b1": ("nvarc_b1", ["deepinfra", "inferencenet", "deepseek"], B1, "pairs_nvarc_b1.txt"),
    # a second, independent sample of every arc2_train test pair via the DeepSeek API (from an empty-split bug, kept)
    "notool_e044_arc2_train_sample2": ("arc2_train", ["deepseek"], A2, None),
}
man = {json.loads(l)["task_id"]: json.loads(l) for l in open(E025 / "manifest.jsonl")}
_tasks = {}


def rows_for(name, src, provs, ch, so, pairs_file):
    files = [f for f in (R / f"e044_notool_{p}_{src}.jsonl" for p in provs) if f.exists()]
    if not files:
        return [], []
    keep = set(open(HERE / pairs_file).read().strip().split(",")) if pairs_file else None
    recs = {}
    for f in files:  # in priority order; a pair already filled by an earlier file is never replaced
        mine = {}
        for r in map(json.loads, open(f)):
            if "error" not in r and (keep is None or f"{r['task_id']}:{r['pair']}" in keep):
                mine[(r["task_id"], r["pair"], r["attempt"])] = r  # resumed: last success wins within a file
        for k, r in mine.items():
            recs.setdefault(k, r)
    recs = list(recs.values())
    if name not in _tasks:
        _tasks[name] = load_kaggle(ch, so)
    tasks, rows, traces = _tasks[name], [], []
    for r in recs:
        t = tasks[r["task_id"]]; test_in = t["test"][r["pair"]]["input"]
        prompt = build_prompt(t["train"], test_in)
        if r.get("task_summary"):
            prompt += task_summary(t["train"], test_in).replace(SUMMARY_INTRO, NOTOOL_SUMMARY_INTRO, 1)
        msgs = ([{"role": "system", "content": EFFORT_HEADERS[r["effort_header"]]}] if r.get("effort_header") else [])
        reasoning = gzip.open(ROOT / r["reasoning_path"], "rt").read()
        traces.append(str(ROOT / r["reasoning_path"]))
        msgs += [{"role": "user", "content": prompt},
                 {"role": "assistant", "reasoning_content": reasoning, "content": r["content"]}]
        m, u = man.get(r["task_id"], {}), r.get("usage") or {}
        rows.append(dict(task_id=r["task_id"], batch=m.get("batch", src), source=m.get("source", "kaggle_arc2_training"),
            parent=m.get("parent"), pair=r["pair"], attempt=r["attempt"], correct=r["correct"], pred=json.dumps(r.get("pred")),
            finish_reason=r.get("finish_reason"), prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
            reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"), seconds=r.get("seconds"),
            est_cost_usd=r.get("est_cost_usd"), model=r["model"], provider=f"{r['openrouter_provider']} (OpenRouter, fp8)" if r.get("openrouter_provider") else "DeepSeek API", effort=r["effort"],
            effort_header=r.get("effort_header"), task_summary=r.get("task_summary"), messages=json.dumps(msgs)))
    return rows, traces + [str(f) for f in files]


def card(stats, final):
    s = (HERE / "scripts/base_card.md").read_text()
    cfg = "".join(f"- config_name: {n}\n  data_files: rollouts/{n}.parquet\n" for n in SPLITS if n in stats)
    s = s.replace("---\n\n# ", f"{cfg}---\n\n# ", 1) if cfg else s
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tot = sum(v["n"] for v in stats.values())
    lines = [f"\n## No-tool rollouts (`notool_*`, E044) — {'complete' if final else f'IN PROGRESS, updated {now}'}\n",
             "Same tasks as the tool rollouts (`e031_*` pairs, then all 7,500 `nvarc_b1` tasks in parent-balanced order), one",
             "attempt each, **no tools**: Flash-1017 via OpenRouter → DeepInfra (fp8) and, from 2026-10-23 20:10 UTC, every third",
             "remaining pair via the DeepSeek API, and `nvarc_b1` split three ways (DeepInfra, InferenceNet fp8, DeepSeek API;",
             "`provider` column; parity checked), effort low, the minimal effort header as a",
             "system message (no-tool wording) and the E017 host-computed grid summary appended to the official ARC Prize prompt",
             "(E035 arm D: 75.4% pass@2 / 62.6% pass@1 on the 120 public-eval tasks via the DeepSeek API).",
             "Each row's `messages` is [system header, user prompt incl. summary, assistant with `reasoning_content` and `content`].",
             "`finish_reason` = `length` marks a truncated answer (context limit 131k).",
             "`notool_e044_arc2_train_sample2` is a second, independent DeepSeek-API sample of every arc2_train test pair.\n",
             "| Config | Rollouts | Correct | Output tokens mean / median | Cost |", "|---|---|---|---|---|"]
    lines += [f"| `{n}` | {v['n']:,} | {v['c']:,} ({100*v['c']/max(1,v['n']):.1f}%) | {v['mean']/1e3:.1f}k / {v['med']/1e3:.1f}k | ${v['cost']:.2f} |"
              for n, v in stats.items()]
    lines.append(f"\nTotal so far: {tot:,} of 10,000 planned." + ("" if final else " Rows are appended as the run proceeds; "
                 "partial `nvarc_b1` is a prefix of the parent-balanced order."))
    return s.rstrip() + "\n" + "\n".join(lines) + "\n"


def cycle(final=False):
    with tempfile.TemporaryDirectory(dir="/tmp/claude-1000") as td:
        td = Path(td); (td / "rollouts").mkdir()
        stats, scan = {}, []
        for name, (src, provs, (ch, so), pairs_file) in SPLITS.items():
            rows, files = rows_for(name, src, provs, ch, so, pairs_file)
            if not rows:
                continue
            scan += files
            pq.write_table(pa.Table.from_pylist(rows), td / "rollouts" / f"{name}.parquet", compression="zstd")
            out = sorted(r["completion_tokens"] or 0 for r in rows)
            stats[name] = dict(n=len(rows), c=sum(bool(r["correct"]) for r in rows), mean=sum(out) / len(out),
                               med=out[len(out) // 2], cost=sum(r["est_cost_usd"] or 0 for r in rows))
        if not stats:
            print("nothing to upload yet", flush=True); return
        (td / "README.md").write_text(card(stats, final))
        lst = td / "scan.txt"; lst.write_text("\n".join(scan + [str(td / "README.md")]))
        sc = subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "data_tools/secret_scan.py"), str(lst)],
                            capture_output=True, text=True)
        print(sc.stdout.strip().splitlines()[-1] if sc.stdout.strip() else sc.stderr[-300:], flush=True)
        if sc.returncode != 0:
            print("secret scan failed: NOT uploading", flush=True); return
        from huggingface_hub import HfApi
        os.environ.setdefault("HF_TOKEN", next(l.split(":", 1)[1].strip().strip("'\"") for l in
                                               (ROOT / "SECRETS.yaml").read_text().splitlines() if l.startswith("huggingface_api_key:")))
        HfApi().upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(td), allow_patterns=["README.md", "rollouts/notool_*"],
                              commit_message=f"no-tool rollouts: {sum(v['n'] for v in stats.values())} ({'final' if final else 'partial'})")
        print(f"{datetime.now(timezone.utc):%H:%M} uploaded", {k: v["n"] for k, v in stats.items()}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0)
    ap.add_argument("--until-done", default="", help="file holding the launcher pid; exit after a final upload once it ends")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    while True:
        alive = a.until_done and Path(a.until_done).exists() and subprocess.run(
            ["kill", "-0", Path(a.until_done).read_text().strip()], capture_output=True).returncode == 0
        try:
            cycle(final=a.final)
        except Exception as e:  # noqa: BLE001 - keep looping; next cycle retries
            print("upload cycle failed:", type(e).__name__, str(e)[:300], flush=True)
        if not a.loop or (a.until_done and not alive):
            break
        time.sleep(a.loop)
