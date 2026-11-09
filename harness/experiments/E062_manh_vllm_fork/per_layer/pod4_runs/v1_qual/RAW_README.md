# Raw data of the pod 4 V1 qualification replica (b32m-fix V1, 2026-10-30)

The files over 1 MB are in the private HF dataset `livobereshka/arc-agi2-vllm-serving-lab`, under `replicas/v1_qual/`
(commit 9dfd4ecbf38f). The dataset was confirmed private before the upload; every file was secret-scanned first
(`data_tools/secret_scan.py`, 0 hits); sizes were checked after the upload.

| path in the dataset | sha256 (first 16) | bytes | contents |
|---|---|---|---|
| `pod4_v1_qual_bundle.tgz` | b6ca97aa7cb4bbc4 | 13,526,999 | everything copied from the pod at the stop: runs/v1_qual (timeline, nb.log, nb.ipynb, mon.jsonl, fault.jsonl, alerts, metrics_scrape.txt, stop.log), kwork_v1_qual/dsv41_logs (attempts, solver.log, metrics, status, supervisor json, server logs of starts 0 / 1 / 2, traces), submission.json, b32m_env.json, p4/logs |
| `dsv41_traces.tar.gz` | b98226c2bc929924 | 12,772,174 | per-attempt traces (`traces/*.json.gz`, 332 records' traces), as in the bundle |
| `mon.jsonl` | a20864b23766b462 | 3,595,277 | the kit sampler's samples (876; 529 at 5 s) |
| `server_start0.log` | afe404616460357e | 2,503,257 | vLLM server log, start 0 (09:22-12:21 UTC) |
| `server_start1.log` | 2e729a50927464bd | 1,416,318 | start 1 (after the TP3 kill, 12:21-15:46) |
| `server_start2.log` | c1eef767eded9edd | 82,331 | start 2 (after the hang, 15:47-15:51) |
| `gpumem.log` | c4235c1e407934d7 | 1,065,011 | the dashboard's allocator-cap / GPU-memory / throughput lines (also in `data/`) |

Download (the HF token goes in an env var, never on a command line):

```python
from huggingface_hub import snapshot_download
snapshot_download("livobereshka/arc-agi2-vllm-serving-lab", repo_type="dataset",
                  allow_patterns=["replicas/v1_qual/*"], local_dir="v1_qual_raw")
```

The prompts in the traces contain ARC-AGI-2 evaluation tasks (workload evalx2t_s2026): do not train on them.
