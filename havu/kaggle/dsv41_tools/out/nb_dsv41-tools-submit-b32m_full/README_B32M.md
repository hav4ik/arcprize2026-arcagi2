# B32-M notebook, NO-SMOKE build (no smoke cells; a commit run would use the solver default smoke deadlines, ~95 min). The smoke build in ../nb_dsv41-tools-submit-b32m/ is also submission-ready (reruns ignore smoke settings, as B32 V2).

Proposed kernel: **chankhavu/dsv41-tools-submit-b32m** (new, private, 4xL4, internet off). NOT pushed.
Rebuild: `.venv/bin/python kaggle/dsv41_tools/build_b32m.py --manh-nb <kernels pull -m of nguyennguyen599/arc2-2026-harness-deepseek>`
(`--manh-flags` after E062b; `--smoke-minutes 0` for the no-smoke variant in `../nb_dsv41-tools-submit-b32m_full/`).

## What it is
B32 (out/nb_candb32 = dsv41-tools-submit-candb V2) with only the environment layer swapped.

**Unchanged from B32:**
- Code, byte-identical with cell 0b's checks: solver 77fe80a3, harness 882c817f, train_check 8fe36175, auto_repair 2292369c / 571f47eb.
- B32's emergency writer, the v3 fallback in reruns, the model integrity check and the smoke report.

**Environment from Manh:**
- Image: **Kaggle's latest** (`docker_image_pinning_type: latest`, no pinned sha). Manh says his code needs the newest environment, and smoke V1 on his pinned sha saw no GPU (NVML not found, torch 0 devices).
- Inputs: his datasets nguyennguyen599/{arc2-2026-src, cu13-install, unsloth-main, vllm-wheels}, plus chankhavu/dsv41-tools-kit, the E128 model and the competition. B32's fork-wheel kernel is dropped.
- Install: his cells 1-5, verbatim plus a timing line, except:
  - `!rm -r /kaggle/working/*` is dropped;
  - his CUDA-install raise happens in commit runs only.

**Plan B (cell 2r):** his Kaggle wheel `vllm-0.32.1rc1.dev0+g5e1bccd07c` cannot load the Kaggle copy of E128 (KeyError 'layers.1.engram.embed.weight.rows_0'). Cell 2r patches his installed `vllm/models/deepseek_v41/nvidia/model.py` with exactly his GitHub fix a6c3ee8c2b (= our E062 patch 01). The diff is embedded, in `b32m_patches/a6c3ee8c2b_engram_row_chunks.diff` (sha256 14032fab…). The cell is guarded by the full sha256 of model.py:
- `ead1e087c29b7d9e…` (5e1bccd07c): apply, check the result is `b2a04fc2a251d570…` (a6c3ee8c2b), drop the file's .pyc;
- already `b2a04fc2…` (his rebuilt wheel): skip;
- anything else: commit runs RAISE; reruns leave the file as is and continue.

The path taken is in `b32m_env.json` → `rowchunk_patch`. The Python diff applier was checked locally: 5e1bccd07c's file plus the diff equals a6c3ee8c2b's file.

**Server:** the solver launches his vLLM (wrapper: `python -m vllm.entrypoints.cli.main serve`) with the E062 L4 replica's config:
- DCP4, interleave 64, retention 512, embed_tokens UVA (1 GiB);
- OffloadingConnector 32 GiB with offload_prompt_only false;
- FULL graphs up to 96, max-num-seqs 96, util 0.945 (Kaggle's equivalent of the pods' 0.938);
- Humming pad 64,64, driver reserve 512 MiB, VLLM_PLUGINS="".

**Solver:** 80 workers, KV gate 0.85 + queue hold, multi-test 0.5, context 163840, auto-repair.

**Manh's extra flags (cell 2m):** OFF.

## GPU gate (cell 0a, right after the clock cell; every run)
- Records `b32m_gpu_gate.json`: /dev/nvidia*, `nvidia-smi -L`, ldconfig's NVIDIA libraries, NVIDIA_* / LD_LIBRARY_PATH, every libnvidia-ml.so* / libcuda.so* under /usr and /opt, and the Python version.
- If the device nodes exist and NVML is not found but a libnvidia-ml.so.1 exists, its folder goes on LD_LIBRARY_PATH.
- It retries for up to 90 s. Without 4 GPUs, or on a Python other than 3.13: commit runs RAISE at once; reruns keep B32's fallback submission and turn our later cells off.

## Smoke (commit run only; reruns ignore all of it)
- **Deadlines from the first cell:** stop-new 32:00, force 34:00, hard stop 35:00, **kill switch 36:30**.
- **Tasks:** the 2 smallest: 00576224 (1 test input) and 25ff71a9 (2 test inputs). Server metrics every 30 s.
- **Records and checks:**
  - env sampler: cgroup memory, /dev/shm, GPU power and clock, disk;
  - `b32m_env.json`: install cells' start times, package versions, his wheel name, row-chunk path, sha256 of the weight-loading files;
  - model integrity in the background (the report waits ≤ 90 s; an unfinished hash is reported, a mismatch fails);
  - tool sandbox on the new Python, 16 sessions;
  - API contract in the background, `smoke_contract.json`:
    - required: dsv41 with max_model_len ≥ 163840, /tokenize + /detokenize, token-id /v1/completions returns token_ids, /metrics has the KV-usage metric;
    - recorded only: previous-turn thinking via `reasoning_content` / `reasoning`, chat tool calls.
- **Report cell (4b):** the timeline (install cells, solver start, server start/up/failure, first attempt done, force cut, final write), the server's windows (gen tok/s, running, waiting, KV, prefix hit), the env record and the contract. If the server never came up, it also shows how far the server got: weight load, KV pool, CUDA graphs, errors and the log tail.
- **Verdict:** the last cell raises "B32-M SMOKE FAILED" on any failure, including a server that never came up, a row-chunk patch that did not end patched or already fixed, or no final write.
- **Estimated wall time:** ~38-39 min inside the notebook (the kill switch at 36:30, then the final check and reports, with waits of ≤ 90 s for integrity and ≤ 30 s for the contract), about 40 min of session: ~1.3-1.4 h of GPU quota at 2x. A failed GPU gate ends the commit run in ~2 min.

## Weight loading depends on (the smoke records the version that ran)
- His vllm wheel only: `vllm/models/deepseek_v41/**` (nvidia/model.py patched by 2r; expert_gptq.py, common/engram.py, quant_config.py), `vllm/model_executor/model_loader/*`, `vllm/model_executor/layers/quantization/humming.py`, `layers/fused_moe/experts/fused_humming_moe.py`;
- humming_kernels 0.1.18;
- the model itself.

His `dsv41_vllm.py` runs plain `vllm serve`: no checkpoint prep and no runtime patches, so arc2-2026-src is copied but not used by this server.

## When Manh ships a new wheel
1. Re-list vllm-wheels and update `kaggle/dsv41_tools/b32m_manh_versions.json`.
2. Re-pull his notebook if his install cells changed.
3. Rebuild. Cell 2r skips itself if the new wheel's model.py has the fixed hash, and fails loudly if it is a third version: then add its hashes or drop 2r.

## Known differences from the L4 replica (b9ad670a67 + our patches 01-03)
- His wheel is main 5e1bccd07c: it has the OOM fix, the indexer row shard (off by default) and the AVX host all-reduce bit.
- With 2r it also has patch 01. Patch 02 is missing: full re-renders (~0.1% of turns) drop previous-turn thinking, which the contract records.
- Patch 03 (DeepGEMM guard) is missing. If his build vendors no DeepGEMM, the server fails at start ("requires DeepGEMM support"), and the report shows it.
