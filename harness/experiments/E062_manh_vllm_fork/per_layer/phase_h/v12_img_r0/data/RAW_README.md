# Raw data of the "V12 + images (causal)" interim run r0 (pod 2, 2026-10-30, stopped at Kaggle 0.82 h)

The files over 1 MB are in the private HF dataset `livobereshka/arc-agi2-vllm-serving-lab` under `replicas/v12_img_r0/`.
Before upload the dataset was confirmed private, and every file, with the traces unpacked, was secret-scanned (117 files, 0 hits). Sizes were checked on the Hub after upload.

| path in the dataset | contents |
|---|---|
| `dsv41_traces.tar.gz` | per-attempt traces (`traces/*.json.gz`), tarred from the pod after the stop |
| `submission.json` | the solver's submission at the stop: Kaggle-style 10.90 over the 240 x2t tasks. It was `data/submission.json`; the pre-stop copy in `nbruns/v12img_rerun/stop_snapshot/` was byte-identical |

Everything else (under 1 MB) is here; see `../README.md`.
