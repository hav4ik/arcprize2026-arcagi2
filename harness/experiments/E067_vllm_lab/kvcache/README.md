# E067 kvcache lab: prefix caching, KV pressure, CPU KV offload (upstream vLLM 0.20, RTX 3090)

Write-up: `notes/2026-10-30_vllm_lab_kvcache.md` (tables, mechanism, sizing rules, metric traps, replay-bench smoke).

## Code
- `loadgen.py`: multi-turn agentic load generator. It speaks token ids to `/v1/completions` with forced reply
  lengths. Optional gates on new attempts: `--gate-kv`, `--gate-hit`, `--gate-ctx`.
- `kvlab_sched.py`: observer subclass of vLLM's `AsyncScheduler` (`--scheduler-cls kvlab_sched.LabScheduler`,
  `KVLAB_SCHED_LOG=...`). It logs prefill, decode and recompute tokens, head-of-line-blocked steps and evictable cached
  blocks.
- `run_suite.py`: one fresh server per run on GPU 1 under the lab GPU lock (suites exp1, exp2, exp3, exp3b, exp3c,
  exp1b). `bench_smoke.py`: smoke test of `evals/serve_replay_bench.py`.
- `analyze.py`: `analyze.py RUN... [--window 60 300] [--series 30] [--md] [--cols a,b]`.

## Data
- `runs/` keeps, per run: `*.turns.jsonl`, `*.sched.jsonl`, `*.server.log`, `*.config.json`, `*.server.json`, the
  suite logs, and `bench_smoke_try1_longctx_crash/`.
- Files over 1 MB (the ten `*.metrics.jsonl`, 1 Hz `/metrics` samples) are in the private HF dataset
  `livobereshka/arc-agi2-vllm-serving-lab` under `kvcache/runs/`, together with a full copy of `runs/`.
- Restore them before running `analyze.py`:

```bash
HF_TOKEN="$(sed -n 's/^huggingface_api_key:[[:space:]]*//p' SECRETS.yaml | tr -d "\"' ")" .venv/bin/python -c "
import os; from huggingface_hub import snapshot_download
snapshot_download('livobereshka/arc-agi2-vllm-serving-lab', repo_type='dataset', allow_patterns='kvcache/runs/*',
                  local_dir='experiments/E067_vllm_lab', token=os.environ['HF_TOKEN'])"
# -> files land in experiments/E067_vllm_lab/kvcache/runs/
```

## Status
- Runs completed: Exp 1 (C = 8, 16, 32, 64), Exp 2 (offload 4 and 12 GiB at C = 32 and 64), and the Exp 3 KV-usage
  gate.
- The Exp 3 hit-rate gate is partial (to t = 165 s).
- Not run, because local GPU use was stopped to protect the cards: the projected-context gate (`exp3c`), the gate +
  offload run (`exp3b`), max-num-seqs and reserve-full-isl (`exp1b`), and the replay-bench retry with YaRN.
