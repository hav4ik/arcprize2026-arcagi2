# Candidate E066-IMG (2026-10-29): image harness + E72-152 on Manh's server

What ships tonight, if the user picks it. The serving side is Manh's code and config (source of truth). The harness
side is ours. The only serving-side addition is our patch that lets his vLLM load per-layer expert counts.

## What it is

- **Notebook:** `build_img_manh.py` takes Manh's notebook as input (`nguyennguyen599/dsv41-tools-submit-b32m-external-server`,
  pulled with `kaggle kernels pull -m`; built on his **v6**, scriptVersionId 363607395: v5 + the server's output root in `dsv41_logs`; notebook sha256 1e773bbc42e8, ~20:30 UTC. Review package for Manh: `out/nb_dsv41-tools-submit-img-manh/README.md`). His install cells,
  production server cell 2m (his `dsv41_vllm.py` launcher, his flags and env: TP4, no DCP, util 0.94, --concurrency 80,
  --context 163840, 12 GiB/rank CPU KV offload, lossless host all-reduce MODE 556, row shard, AR overlap), solver cell 3
  (ARC_EXTERNAL_SERVER) and stop cell 4 stay byte-identical. His v5 dropped the GPU gate (0a, commented out) and his
  row-chunk cell 2r (his newest wheel loads row chunks). We change or add only these cells:
  - **settings (00):** `ARC_GRID_FORMAT=text+image`, `MANH_VISION=1`, `MANH_VISION_MAX_IMAGES=32`. His launcher accepts
    2-32 images; ARC's worst case is ~22-24.
    - The commit run is a **light smoke** (`--commit light`, the default): the 2 smallest tasks (00576224, 25ff71a9 with 2
      test inputs), deadlines capped at ~50 min, ending when both are done. This saves quota versus his 80-task load
      test (`--commit load`). The competition rerun ignores all of it.
  - **0b:** our pinned files from `pinned_img/`:
    - solver `e96f7cb7` = B32's 77fe80a3 + the image switch + S25 (= his S23 ARC_EXTERNAL_SERVER; the probe tolerates
      non-refused failures, 3 in a row);
    - harness `1d1a281e` (live evals/ frozen 2026-10-29: E058 text+image, chat-path `--context-limit`, `--token-exact`
      refused with images);
    - `arc_harness/grid_image.py` `18ed78fb`; B32's train_check `8fe36175` and auto-repair `2292369c`/`571f47eb`.
    - The checks are his, plus ARC_GRID_FORMAT / MANH_VISION and the features we need.
  - **1b:** commit-run input integrity against the E066 export's sha256 manifest (`--model-manifest`).
  - **2r:** our vLLM patch (`cells/img_patches.py`), sha256-guarded and multi-variant. It replaces his 2r on his v4 and
    is inserted after his install cells on v5:
    - `model.py`: per-layer `n_routed_experts_per_layer`; for wheel g5e1bccd07c also the Engram row chunks
      (ead1e087 -> 75f3725a); for gf506afa210 / g71dca8bc24 029aa62e -> ad4ac0fd;
    - `dsv4_topk.py`: Triton router whitelist 6..512 (7e1e101a -> ab32ea7e).
    - Branch `dynamic-count-moe` (66cf71a0b7 on Manh's main 71dca8bc24) in `scratchpad/e062_port/manh_vllm`, not pushed.
  - **2c:** the API contract also requires parsed chat tool calls (probe max_tokens 4000) and an accepted image.
  - **3w (new, after his solver cell 3):** commit runs wait up to 15 min for the contract thread before his cell 4 stops
    the server (audit 4).
  - **4b:** the report regex accepts the external solver's `server up after Ns (external);`; it reads MANH_SERVER_LOG;
    with images, contract failures are fatal. Audit 4 added: 2r's outcome record (`e066_vllm_patch.json`) and 1b's
    names-and-sizes pass are fatal checks; the light smoke fails on a crash line, zero answered attempts or no
    submission.json.
  - **metadata:** our slug `chankhavu/dsv41-tools-submit-img-manh`, `docker_image_pinning_type: latest` (user), the
    E066 Kaggle Model, his datasets (vllm-wheels latest = v10, gf506afa210, frozen through 0 UTC per Manh).
- **Model:** E72-152 (`DeepSeek-V4.1-Flash-1017-arc-tool-img-E72-152-avg120-2bit`, floor 72 / cap 152 / step 16, 4,800
  experts, vision kept) or C1 (uniform E128 + vision) as the fallback. Kaggle's quota fits one; the user picks.

## Build

```
python3 build_img_manh.py --manh-nb <pulled Manh notebook dir> \
  --model-source chankhavu/<kaggle-model-slug>/PyTorch/default/1 \
  --model-manifest experiments/E066_dsv41_image_pruning/results/<e72c152|c1>_gptq_sha256_manifest.json
```
Output goes to `out/nb_dsv41-tools-submit-img-manh/`. Tests: `.venv/bin/python -m unittest discover -s kaggle/dsv41_tools/tests -p 'test_img.py'`
(15 tests, built against his v5). Also run `test_b32m.py`, `test_solver.py` and the evals tests.

## Tested

- **Local:**
  - test_img.py (15): build, pins, settings and 0b execution in commit and rerun, the patch variants on the real wheel
    files, the solver switch, the S25 probe, the report-regex fix, contract.
  - Manh's 4 ExternalServer tests against our solver; test_solver 41, test_b32m, evals suites.
  - Three Opus audits (notebook, chat path, vLLM patch). Their findings are fixed or accepted; see the 2026-10-29
    section of experiments/E062_manh_vllm_fork/README.md.
- **Pod 2 (Runpod 4xL4), real E72-152 weights:**
  - sha256 of all 137 files verified;
  - KV pool 3,259,129 (DCP4) / 1,336,475 (no DCP, util 0.92); image sanity coherent (5,681 image tokens = the harness's
    count);
  - Manh's REAL launcher with the notebook's exact args (arc2-2026-src v22): launch.json matches, contract passes;
  - prior reasoning kept and prefix stable on the chat path (render check with tools);
  - load run on v8 with his older cell (no DCP, util 0.92, 120 workers, KV 1.34M):
    - 0 errors, 0 preemptions, 21/27 first-attempt pairs correct at 30 min;
    - gen 742-912 tok/s;
    - GPU prefix hits fell to ~3% after 16 min, but ~90% of prompt tokens were restored from his CPU KV offload and
      only 3.5-8% were recomputed; the cost is fewer running streams (120 -> 60);
  - **his production cell on the v10 wheel** (gf506afa210 + our f506 patch): KV pool **2,264,661** tokens (Kaggle-eq
    ~2.35M), server up in 125 s, contract passes, greedy parity with v8 within noise. A 90-min load run started 19:35 UTC.
- **Running (2026-10-29 ~19:45 UTC), ~90 min each, same 240-task image workload, 80 workers:**
  - pod 2: E72-152, his production cell as is (no DCP);
  - pod 3: C1, his v5 production cell + DCP4 (`--decode-context-parallel-size=4 --cp-kv-cache-interleave-size=64
    --prefix-cache-retention-interval=512`);
  - pod 4 (requested from the user): E72-152 + DCP4, for a clean DCP A/B against pod 2;
  - dsv4-pruning's ARC evals (E72-152 vs C1), for the model choice.
- The DCP-or-not decision (his cell as is, or + the 3 DCP4 flags) is taken from these runs; it changes only flags in
  his cell 2m, not the Kaggle Model.

## Open / accepted risks

- No server restart: his design, so a mid-run engine death ends new attempts.
- No Kaggle smoke before the submission's own commit run (user: low quota). Kaggle-only risks: image, mounts, slower
  hardware.
- No DCP in his config: GPU KV pool 2.26M (pod; DCP4 3.26M), and his 48 GiB CPU KV offload absorbs evictions (~90%
  of prompt tokens restored from CPU in the older-cell run). Steady-state throughput is being measured.
- The per-layer layout carries ~0.2 GiB/GPU of allocator waste (~0.19M KV), measured and included in the 3.38M.

## Submission steps (each Kaggle action needs the user's OK)

1. The user picks E72-152 or C1.
2. dsv4-pruning uploads that export as a private Kaggle Model, plain files like the E128 Kaggle Model, with
   e043_manifest.json at the root.
3. Verify with `verify_kaggle_model.py <handle> <manifest>`: names and sizes.
4. Rebuild the notebook with that handle and manifest, rerun the tests, and diff against the previous build.
5. Push and commit the notebook. The commit run is the light smoke (2 tasks; ~25-35 min incl. ~20 min of install,
   model load and compile = ~1 h of 4xL4 quota). Read its verdict in cell 4b.
6. Submit that version.
