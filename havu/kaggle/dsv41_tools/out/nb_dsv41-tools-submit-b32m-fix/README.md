# B32-M fix (2026-10-29 pivot): tonight's candidate

B32-M V3 (LB 46.5) with the causes found today removed. The model is the E128 text model already on Kaggle
(`chankhavu/dsv41-flash-1017-arc-tool-e128-2bit`). The harness is B32's (`882c817f`). The server is Manh's v6 cell.
Builder: `kaggle/dsv41_tools/build_img_manh.py --text` (command below). The diff against his v6 is in `review/`.

| B32-M V3 | This build | Why |
|---|---|---|
| DCP4 (lossy fp8 combine, 0.72x speed on Kaggle) | TP4, no DCP, his production flags | his tuned Kaggle config |
| wheel g5e1bccd07c (no prefill-memory fix) | v10 gf506afa210 (has 7ab756940a "budget prefill memory the startup profile misses") | likely OOM trigger: re-prefill of long requests |
| util 0.938, ~1.1 GiB VRAM free | util **0.90**, ~1.9 GiB free | room for long-prefill spikes |
| offload 32 GiB, 80 workers | offload **10 GiB/rank**, **64** workers/seqs | host RAM ~150 of 173 GiB |
| B32's solver: launches into a dead port, burns retries (S24), 19 min restarts, no hang detection | solver `324ea6ee` (S25 probe, S26 follows a supervisor) + **cell 2s restart supervisor** (death or 25 min stall, max 2 restarts, waits for GPUs and RAM to clear) | B32-M's 46.5 fits a run that stopped producing answers at hour 4-6 |
| exec timeout 300 s | **900 s** | a slow step no longer kills the engine |
| 80-task load-test commit | light smoke (2 tasks) + checks: answers, crash lines, RAM headroom, file sizes, supervisor log | quota; fail closed on a bad config |

Context stays at 163,840. In B32-M's replica, 1 of 480 attempts hit it, but outputs reach 95k (p90) to 128k (max).

```
python3 kaggle/dsv41_tools/build_img_manh.py --text --manh-nb <Manh v6 pull> --slug dsv41-tools-submit-b32m-fix \
  --max-num-seqs 64 --gpu-util 0.90 --kv-offload-gib 10 --exec-timeout-s 900
```
Optional, if the decode-numerics test incriminates his decode-only paths:
`--server-env VLLM_DSA_DECODE_INDEXER_ROW_SHARD=0 --server-env VLLM_TRITON_SPARSE_INDEXER_LOGITS=0 --server-env VLLM_TP_HOST_GATHER=0`.
