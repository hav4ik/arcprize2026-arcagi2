# Raw data of the "SFT 025ep on V12" replica (pod 2, 2026-10-30, stopped by the user at Kaggle 4.76 h)

The files over 1 MB are in the private HF dataset `livobereshka/arc-agi2-vllm-serving-lab` (commit 7a5678c6a28d), under
`replicas/sft_025ep_v12/`. Before upload the dataset was confirmed private, and every file, with the traces unpacked,
was secret-scanned (284 files, 0 hits). After upload, sizes were checked on the Hub; the LFS file's sha256 was checked too.

| path in the dataset | sha256 (first 16) | bytes | contents |
|---|---|---|---|
| `dsv41_traces.tar.gz` | 099207ab73fe689f | 11,794,613 | per-attempt traces (`traces/*.json.gz`, 249 attempts), tarred from the pod after the stop (the notebook's cell 4 never ran) |
| `server.log` | e9d4f258b1312436 | 2,126,486 | the server log of start 0, the only start (was `data/dsv41_logs/server/server.log`) |
| `submission.json` | 928d21a2d9ef1b4f | 1,019,290 | the solver's submission at the stop, Kaggle-style 66.32 over the 240 x2t tasks (222/344 pairs). It was `data/submission.json`; the pre-stop copy in `nbruns/s025_rerun/stop_snapshot/` was byte-identical |

Download (the HF token goes in an env var, never on a command line):

```python
from huggingface_hub import snapshot_download
snapshot_download("livobereshka/arc-agi2-vllm-serving-lab", repo_type="dataset",
                  allow_patterns=["replicas/sft_025ep_v12/*"], local_dir="sft_025ep_v12_raw")
```

Still here (files under 1 MB):
- **Dashboard files:** gpumem.log, ram.tsv, tps.tsv, status.json (verdict rewritten after the stop), workload.txt.
- **Run files:** `dsv41_logs/` (attempts.jsonl, solver.log, metrics.jsonl, status.json, supervisor.json/.log), b32m_env.json (installed versions), b32m_gpu_gate.json.
- **Runner and sampler:** `nbruns/s025_rerun/`: nb.timeline, run_nb.log, the sampler's samp.jsonl / samp.alerts / samp.tps.tsv, mem.log, tree.txt (input mapping), notebook.sha256, uvcache_cleanup.log, and `stop_snapshot/attempts.jsonl` with the time it was taken.
- The executed notebook was not saved: nbclient writes it only at the end, and the runner was killed.

The write-up is `../README.md`.
