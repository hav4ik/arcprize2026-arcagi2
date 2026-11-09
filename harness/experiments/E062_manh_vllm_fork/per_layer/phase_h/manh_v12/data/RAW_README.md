# Raw data of the Manh v12 replica (notebook 363638159, pod 2, 2026-10-30)

The large files are in the private HF dataset `livobereshka/arc-agi2-vllm-serving-lab` (commits ccbb4c16f45c and
e54042936828), under `replicas/manh_v12/`. Sizes and sha256 were checked after upload; the dataset was confirmed private and every file
was secret-scanned (traces unpacked) before upload.

| path in the dataset | sha256 (first 16) | bytes | contents |
|---|---|---|---|
| `dsv41_traces.tar.gz` | 0f87e095bcb39d6b | 24,781,752 | per-attempt traces (`traces/*.json.gz`, 480 attempts), written by the notebook's cell 4 |
| `gpumem.log` | 1a0309e9b6889a75 | 1,827,034 | the dashboard's "GPU memory:" / allocator-cap / throughput lines from server.log (was `data/gpumem.log`) |
| `samp.jsonl` | 2989e236e7fd33cf | 1,535,124 | the 60 s host-RAM / VRAM sampler, 609 samples with the server up (was `data/nbruns/m12_rerun/samp.jsonl`) |
| `attempts.jsonl` | 6d3cf6491eefb359 | 1,173,214 | the solver's attempt records, 480 (was `data/dsv41_logs/attempts.jsonl`) |
| `submission.json` | ba4b3779ac7f8166 | 964,179 | the final submission: Kaggle-style score 80.83 over the 240 x2t tasks, 277/344 pairs (was `data/submission.json`) |
| `logs_bundle.tgz` | f8632760e8444eab | 363,360 | every .log: dsv41_logs/server/server.log (start 0, the only start), solver.log, stage.log, supervisor.log, gpumem.log, nbruns/m12_rerun/{mem.log, run_nb.log} |

Download (the HF token goes in an env var, never on a command line):

```python
from huggingface_hub import snapshot_download
snapshot_download("livobereshka/arc-agi2-vllm-serving-lab", repo_type="dataset",
                  allow_patterns=["replicas/manh_v12/*"], local_dir="manh_v12_raw")
```

Still here (files under ~1 MB):
- **Run files:** `dsv41_logs/` (metrics.jsonl, status.json, solver.log, supervisor.json/.log, stage.log).
- **Notebook and sampler:** `nbruns/m12_rerun/` holds the executed notebook with every cell output (nb.ipynb, nb_outputs.txt), the timeline, the sampler's alerts and tps rows (samp.alerts, samp.tps.tsv) and mem.log.
- **Dashboard files:** ram.tsv, tps.tsv, status.json, workload.txt.

The write-up is `../README.md`; the Kaggle-clock report is `../replica_report.md`.
