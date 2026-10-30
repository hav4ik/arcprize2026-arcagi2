# Raw data of the B32-M V3 replica (Kaggle-matched VRAM, pod 3, 2026-10-30)

The files over 1 MB are in the private HF dataset `livobereshka/arc-agi2-vllm-serving-lab` (commit 20a6c0e9774d), under
`replicas/b32m_v3_kaggle_matched/`. Sizes were checked after upload.

| path in the dataset | sha256 (first 16) | bytes | contents |
|---|---|---|---|
| `dsv41_traces.tar.gz` | 6fbe85e47bb69114 | 12,574,460 | per-attempt traces (`traces/*.json.gz`), written by the notebook's cell 4 |
| `submission.json` | 31bca516e544df85 | 1,059,088 | the final submission (score 50.42 over the 240 x2t tasks) |
| `pod/mon.jsonl` | 3e9176285f37ccab | 7,767,125 | 60 s / 5 s host and GPU monitor samples (`mon_sampler.py`) |
| `pod/metrics_scrape.txt` | e8e3c628af7a2c0d | 1,853,563 | the server's /metrics every 60 s, across restarts |
| `logs_bundle.tgz` | c7944e1ff3416b0a | 619,571 | every .log: server.log, server_restart1-3.log, solver.log, stage.log, run.log, mem.log, gpu_hold.log, setup.log, verify_e128.log, try1/ |

Download (the HF token goes in an env var, never on a command line):

```python
from huggingface_hub import snapshot_download
snapshot_download("livobereshka/arc-agi2-vllm-serving-lab", repo_type="dataset",
                  allow_patterns=["replicas/b32m_v3_kaggle_matched/*"], local_dir="b32m_v3_raw")
```

Still here (small files):
- `dsv41_logs/`: attempts.jsonl, metrics.jsonl, status.json, server_cmd.json.
- `pod/`: nb_rerun.ipynb (the notebook as run, sha 771da13fd94d) and nb_rerun_out.ipynb (executed, with outputs), gpu_hold.out, status_floor.json.
- `pod/try1/`: try 1 (410 MiB hold, stopped before any attempt finished).

The timeline and analysis are in `../README.md`; the dashboard files are in `../data/`.
