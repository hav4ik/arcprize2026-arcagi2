# E056: candidate B overlay kit = kit overlay + E054 #156 part 1 (indexer skip) + E051 KV CPU offload

Status (2026-10-27 05:20 UTC): **kit built (`vllm_overlay_rw.tgz` sha256 `1a3a6679`, 16 members); packaging
through the notebook's cell 0b + cell 2 works with the builder unchanged; GPU on Vast 4x4090 box 2 (57817226): install
GO, server start GO, byte-level offload parity GO (skip on, private memory, 3k-120k, 0 mismatches), late-phase A/B
candB vs offload-only = tie (618.5 vs 617.0 gen tok/s, +0.2%, n = 1; preemptions 155 vs 24).** §4 has the verdicts and
§5 what remains before a Kaggle submission.

Candidate B is the Kaggle serving stack for E128 (pruned DeepSeek-V4.1-Flash-1017, 2-bit, TP4 on 4xL4) plus two
changes that were each validated on their own but never together:

1. **E054, fork PR #156 part 1** (decode indexer grid-stride context skip): one file,
   `vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py`, sha `9f13485b` (replaces the kit's `3da076e1`).
   E054 verdict GO: bit-identical logits, decode step -5.7% / -6.5% at B = 36 / 48. Kill switch `DSV41_INDEXER_SKIP=0`
   (default on), which runs the old kernel byte for byte.
2. **E051, KV CPU offload**: 4 files (offloading `config.py` / `scheduler.py`, kv_offload cpu `manager.py` /
   `spec.py`). E051: byte-level KV parity up to 120k tokens, 3.4-4x gen tok/s at 56-72 streams on the Kaggle-sized pool.
   Inert without `--kv-transfer-config`.

E054's `engram.py` change (upstream #58678, NO-GO) is **not** included; `engram.py` stays the kit version `5f1e7230`.

## 1. The kit (`kit/`)

| file | sha256 |
|---|---|
| `kit/vllm_overlay_rw.tgz` | `1a3a667904d435964dd4ce1f9b5c205e2b148411ffa3b8f5e00d9527cda03306` |
| `kit/vllm_overlay_rw.sha256` | `624e7754fc322e76416425c9a3660306d137e9d1c43db444f766c3f4e117f04e` |

A drop-in for the kit's pair (`kaggle/dsv41_tools/kit/vllm_overlay_rw.{tgz,sha256}`, tgz `377d83e0`): 16 members,
paths relative to site-packages, the kit's 12 in the kit's order, then the 4 E051 files appended. The sha list has the
same `<sha256>  <path>` format, in the same order.

| # | member | sha256 | status |
|---|---|---|---|
| 1 | `vllm/e043_humming.py` | `74dcbb14` | kit |
| 2 | `vllm/models/deepseek_v4_1/synth_init.py` | `c2bf82d4` | kit |
| 3 | `vllm/model_executor/kernels/attention/dsa/dsv41_indexer.py` | `9f13485b` | **replaced** (kit `3da076e1`; E054 file) |
| 4 | `vllm/model_executor/kernels/linear/__init__.py` | `9fa77d64` | kit |
| 5 | `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` | `27a01ecc` | kit |
| 6 | `vllm/model_executor/layers/quantization/humming.py` | `58c3519f` | kit |
| 7 | `vllm/model_executor/layers/quantization/utils/humming_utils.py` | `9ae7acb8` | kit |
| 8 | `vllm/model_executor/model_loader/dummy_loader.py` | `2948cac0` | kit |
| 9 | `vllm/models/deepseek_v4_1/common/engram.py` | `5f1e7230` | kit (E054's #58678 not taken) |
| 10 | `vllm/models/deepseek_v4_1/nvidia/model.py` | `96b95b08` | kit |
| 11 | `vllm/models/deepseek_v4_1/nvidia/router.py` | `88f2f05f` | kit |
| 12 | `vllm/models/deepseek_v4_1/quant_config.py` | `da3b8dd7` | kit |
| 13 | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py` | `664b4b01` | **appended** (E051; wheel original `04a31474`) |
| 14 | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | `dffd7d86` | **appended** (E051; wheel `cd43cf5f`) |
| 15 | `vllm/v1/kv_offload/cpu/manager.py` | `eda86f25` | **appended** (E051; wheel `3b8b9f6a`) |
| 16 | `vllm/v1/kv_offload/cpu/spec.py` | `a0dbbafc` | **appended** (E051; wheel `240d6863`) |

- **None of the 4 E051 files was a kit member**, so all 4 are appended (the builder would replace in place and say so
  if one were).
- **E051 shas** equal the "sha256(overlay)" column of E051's `gpu/install_overlay.sh` (which `setup_offload_venv.sh`
  runs), i.e. the files E051 tested on the GPU. The builder pins all 5 source shas and the kit tgz sha and refuses any
  other version.
- **Deterministic:** GNU tar, gzip mtime 0, member mtime 2026-10-25 00:00 UTC, uid/gid 0 root; kit members keep their
  mode, new members 0644. Two builds are byte-identical; `--check` rebuilds in memory and compares bytes.

```bash
.venv/bin/python experiments/E056_candidate_b/scripts/build_kit.py          # build + check
.venv/bin/python experiments/E056_candidate_b/scripts/build_kit.py --check  # verify kit/ (members, shas, byte rebuild)
```

## 2. Packaging: the kit goes through the notebook's cell 0b and cell 2 unchanged

Read in `kaggle/dsv41_tools/build_frozen_mt.py` (not edited) and the V5 notebook it builds on:

- `--overlay DIR` adds `vllm_overlay_rw.tgz` and `vllm_overlay_rw.sha256` from DIR to cell 0b's embedded files. They
  are **not** in `_MT_NEW` (both names already exist in the kit), so cell 0b backs up the kit's pair as `*.v3orig`,
  extracts the embedded pair over it, and asserts both shas (`_MT_SHA`: `1a3a6679` / `624e7754`).
- `_MT_NEW` is only for files new to the **kit directory** (auto-repair files). Members new to the **tarball** need
  nothing from cell 0b: cell 2 runs `tarfile.open(KIT/vllm_overlay_rw.tgz).extractall(SP)`, which creates or overwrites
  every member (the 4 E051 paths already exist in the fork wheel, so they overwrite the wheel's originals), then
  asserts every line of `KIT/vllm_overlay_rw.sha256` against `SP`. With this kit that is 16 lines.
- `_MT_NEED` has no overlay entry; the feature checks cover the solver and harness only. The overlay's integrity rests
  on the pinned pair shas (cell 0b) plus the per-member sha check (cell 2), which is enough.
- **Fallback (competition rerun, any cell-0b failure):** the `*.v3orig` pair is restored, so cell 2 installs the kit's
  12-file overlay (wheel offload files untouched) and `ARC_SERVER_EXTRA_ARGS` is reset to Version 5's flags (no
  `--kv-transfer-config`, which the stock connector would reject).
- **`PYTORCH_CUDA_ALLOC_CONF`:** when `--server-extra` contains `--kv-transfer-config`, the settings cell strips
  `expandable_segments` (vLLM rejects a KV connector with it). The solver's `server_env()` starts from `os.environ`,
  so the server inherits the stripped value.
- **Stale bytecode:** cell 2 does not delete `__pycache__`. On Kaggle the venv is fresh (`uv pip install` does not
  compile bytecode). On the box copy, old `.pyc` files are invalidated by source mtime and size (the import check
  after install loads the new scheduler).

Tests (CPU, local):

- `kaggle/dsv41_tools/tests` (the builder's own suite, unchanged): **54/54 OK**.
- `tests/test_notebook_install.py` (E056, 3 tests, **OK**): builds a scratch notebook with
  `build_frozen_mt.py --overlay kit/ --retention 512 --server-extra '["--kv-transfer-config", <candidate B JSON>]'`
  and runs `scripts/notebook_install.py` against a fake site-packages holding stand-ins for the 4 wheel files:
  - commit run: cell 0b OK, 16/16 members installed and sha-checked, the 4 stand-ins replaced, the offload flag kept,
    `expandable_segments` stripped;
  - commit run with a wrong setting: cell 0b raises, nothing installed;
  - rerun with a wrong setting: fallback; the kit's 12-file overlay (`377d83e0`) installed, the 4 wheel files
    untouched, `--kv-transfer-config` dropped.

`scripts/notebook_install.py` is the real packaging path, not a hand install. It takes the code from the generated
`.ipynb` and runs, in one process and in the notebook's order: cell 00 (settings), cell 0 (kit unpack + MANIFEST
check), cell 0b, each verbatim, then cell 2's overlay block verbatim (its own `import` line plus the
`extractall` / sha-assert lines, dedented). It skips cell 1 (model staging) and cell 2's wheel install, CUDA shim and
import checks (the box venv already has the fork wheel set). Input: the V5 Kaggle kit `kaggle_kit_v5/dsv41_kit.tgz`
and a 1-task challenges file.

Scratch notebook used on the box (not committed; rebuild with the command in §5): `e056-candb-installtest.ipynb`,
sha256 `831cb9ca`, embedded files solver `0a6f7406`, harness `882c817f`, overlay pair `1a3a6679` / `624e7754`.

## 3. GPU tests on Vast 4x RTX 4090 box 2 (57817226), 2026-10-27 02:52 - 05:17 UTC

### Box layout and rules followed

- Everything is under `/root/e056/` (`kit/`, `kit_base/`, `scripts/`, `nb/`, `input/`, `gpu/` = this repo's `gpu/`,
  `e051/gpu` + `e051/tests` = E051's scripts and CPU tests, unchanged copies; `shapes/tools_e043_e128t.json` = the
  converted E042 shapes, sha `86bc24ce`; `runs/`).
- **Venv:** `/root/dsv41/venv_candb` is a `cp -a` copy of `/root/dsv41/venv` (which resolves to
  `/workspace/dsv41/venv`), with every `bin/` shebang rewritten (`gpu/setup_candb_venv.sh`, the loop of E051's
  `setup_offload_venv.sh`). `/root/dsv41/venv` was checked before and after: its 12 kit files and the 4 wheel originals
  are unchanged (`results/runs/src_venv_{before,after}.sha256`). `venv_e054` was left in place (disk was enough).
- **Servers** run in their own session and are stopped by process group (`gpu/run_arm.sh` trap), never by name.
- **Host RAM:** the plan was 94 (Engram) + 32 (offload) + 20 = 146 GiB, under half of 503 GiB, and the launcher
  refuses to start above that.
- **Secrets:** `HF_TOKEN` and similar are unset in the server env; `env.txt` holds only `DSV41_* VLLM_* E043_*
  CUDA_HOME PYTORCH_* PYTHONPATH`, and the image's `VLLM_*` variables (Qwen defaults, no tokens) show up in it.
  `PYTORCH_CUDA_ALLOC_CONF` is unset. Every copied file was scanned for `hf_...`, `HF_TOKEN=`, `api_key` and
  `Bearer`: none found.
- **Server flags** (`gpu/serve_candb.sh`, both arms): the Kaggle command (TP4, `--kv-cache-dtype fp8`, block 128,
  max-model-len 163840, `--max-num-seqs 64`, MBT 2048, prefix caching, `--prefix-cache-retention-interval 512`,
  Engram `cpu_offload`, `--linear-backend humming`, FULL CUDA graphs capped at 64), `--gpu-memory-utilization 0.92`
  (E054's value on this box), `--num-gpu-blocks-override 21490` (Kaggle's pool), and
  `--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":34359738368,"offload_prompt_only":false,"swa_store_policy":"checkpoint","shared_region":false,"eviction_policy":"lru"}}'`,
  plus `--worker-extension-cls e056_ext.E056Ext` (E051's parity methods plus two read-only probes; it adds no serving
  behaviour). Env: E051/E054's (`VLLM_USE_V2_MODEL_RUNNER=1`, Humming expert JSON, `DSV41_INDEXER_Q_PREDECODE=1`,
  `VLLM_SERVER_DEV_MODE=1`, CUDA 13 toolkit), and a separate `VLLM_CACHE_ROOT` per arm.

### (a) Install through the notebook path + E051 CPU tests: **GO**

`gpu/setup_candb_venv.sh` (34 s; `results/runs/setup_candb_venv.log`, `notebook_install.log`, `e051_cpu_tests.log`):
- `scripts/notebook_install.py` on the scratch notebook (cells 00 / 0 / 0b verbatim + cell 2's overlay block) printed
  `E046 CHECK OK` in cell 0b and installed `1a3a6679` / `624e7754`: **16/16 sha lines match**. The settings cell left
  `--kv-transfer-config` in `ARC_SERVER_EXTRA_ARGS` and `PYTORCH_CUDA_ALLOC_CONF` empty.
- Checked again by the script: all 16 files match. `import vllm` and the offload scheduler resolve to the copy, and the
  scheduler has `SWA_STORE_CHECKPOINT` (no stale bytecode). `vllm --version` reports
  `0.28.1rc1.dev517+glm53.dsv41.vision11.sm89sm120.cu130`.
- **E051's 26 CPU tests against the installed tree: 26/26 OK** (0.67 s).

### (b) Server start with the candidate-B flags: **GO**

From `results/runs/late_candb/` (`server.log`, `startup.txt`, `indexer_state_{start,end}.json`, `loaded_files.json`,
`metrics.jsonl`); the start is the same for the offload arm (`late_offload/`):
- **Ready after 180 s** (cold `VLLM_CACHE_ROOT`; the offload arm took 110 s).
- `KV offloading: swa_store_policy=checkpoint, hits aligned to 128 tokens, skipped scratch groups [8]` is logged once
  (EngineCore).
- **KV pool:** `Overriding num_gpu_blocks=29122 with num_gpu_blocks_override=21490`, so `GPU KV cache size:
  1,591,736 tokens`, the same as E054's pinned arms. The KV layout reported by the extension is 9 groups (7 SWA at
  32-token blocks, MLA + indexer at 128, the compressor ring at 8) in one packed storage of 2,448,140,800 B = 21,490
  pages of 113,920 B.
- **Tracebacks: 0 while serving**, in both arms. candB's log has one traceback, the `AsyncLLM output_handler failed`
  raised after `EngineCore: trigger received signal=SIGTERM` (our stop with requests in flight, as in E054's `both`
  arm). The offload arm's log has none. The stack's usual `CUDAGraphMode.FULL is not supported with
  DeepseekV41IndexerBackend ... FULL_DECODE_ONLY` warning appears here as in E054's arms (4 lines each).
- **Host memory** (`metrics.jsonl`, 5 s): Shmem went 0.2 → 91.1 GiB while Engram loaded (23.6 GiB per rank logged), then
  **91.1 → 123.1 GiB (+32.0 GiB) at 02:56:15-21, when the offload pool was allocated**. That is exactly
  4 × 8 GiB, as predicted for 34,359,738,368 B (PyTorch rounds each rank's pinned buffer up to a power of two; 8 GiB per
  rank is already one). `free` used went 104 → 224 GiB, server PSS was 133.5 GiB when ready and at most 137.2 GiB during
  the replay window, and box MemAvailable stayed ≥ 270 GiB.
- **Indexer skip is active (3 independent checks):**
  1. `E054 DSv4.1 indexer: grid-stride context skip on (candidates on), target programs 16384` in 6 processes (API
     server, EngineCore, 4 workers), 0 `off` lines; the offload arm shows 6 `off` lines and 0 `on` lines;
  2. the worker probe `e056_indexer_state` (dev-mode `/collective_rpc`) reads `_INDEXER_SKIP = True` and file sha
     `9f13485b` on all 4 ranks;
  3. the probe also lists the compiled Triton variants per GPU. In candB, `_mxfp4_indexer_logits_stride_kernel` had 6
     after startup and 14 after the 70-min run, while the old `_mxfp4_indexer_logits_kernel` had **0** throughout. The
     offload arm (`DSV41_INDEXER_SKIP=0`) is the reverse: old kernel 6, stride 0. So under candB the old kernel never
     ran, neither in graphs nor in eager mixed steps.
  - `e056_loaded_files` shows the workers import all 5 changed modules from `venv_candb` with the kit shas.

### (c) E051 byte-level offload parity (private memory, `shared_region=false`, indexer skip on): **GO**

`e051/gpu/kv_parity.py --lengths 3000 20000 60000 120000 --max-tokens 300` on the candB server, before the replay
(`results/runs/late_candb/kv_parity.{out,json}`, 5 min 49 s). Per case, the same request is answered once from a GPU
prefix hit and once restored from host RAM into a KV pool poisoned with 0xA5. Every restored page (MLA + indexer
blocks below the hit, each SWA group's 4-block checkpoint) is compared by sha1 on all 4 ranks. "GPU hit" here means
the offload-off path; the indexer skip is on in both paths:

| case | GPU-hit cached | host-restored (= external hits) | pages checked per rank (×4 ranks) | mismatches per rank | poison pages per rank |
|---|---|---|---|---|---|
| prompt 3,000 | 2,944 | 2,944 | 51 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| resume 3,000 (+300 decode, +200 tool) | (expected 3,200) | 3,200 | 53 | 0 | 0 |
| prompt 20,000 | 19,968 | 19,968 | 184 | 0 | 0 |
| resume 20,000 | (20,224) | 20,224 | 186 | 0 | 0 |
| prompt 60,000 | 59,904 | 59,904 | 496 | 0 | 0 |
| resume 60,000 | (60,288) | 60,288 | 499 | 0 | 0 |
| prompt 120,000 | 119,936 | 119,936 | 965 | 0 | 0 |
| resume 120,000 | (120,192) | 120,192 | 967 | 0 | 0 |

**KV PARITY PASS**: 0 tracebacks during the test, hits 128-aligned, and the same numbers as E051's gate 3 (shared
memory, box 1) and its private-memory rerun.

### (d) Late-phase throughput A/B on this box: **tie; no regression, no measurable indexer gain at this load**

- **Arms.** Each arm had one server start, the same flags and venv (`venv_candb`), and one extension.
  - **offload** = `DSV41_INDEXER_SKIP=0`, i.e. the E051 files with the old indexer kernel. E054 showed this switch
    reproduces the old kernel exactly, and the probe confirmed only the old kernel compiled.
  - **candB** = `DSV41_INDEXER_SKIP` unset.
  - Order: candB first (after its parity test and a reset of both prefix caches), then offload.
- **Workload:** E051's hour-8 arms exactly, `e051/gpu/tool_replay_kv.py --shapes tools_e043_e128t.json --conc 36
  --tool-delay 20 --warm-s 600 --window-s 1500 --max-wall-s 5400 --scale 1.25 --tail-frac 0.3 --min-start-ctx 30000
  --seed 0`, after `POST /reset_prefix_cache?reset_external=true` and E051's 4-stream smoke (both arms 249.7 tok/s).
  - Setup (36 cold histories of ≥ 30k tokens, prefill-bound at ~1.7k tok/s) took 1,617 s / 1,622 s, so each arm ran
    for 62 min.
  - Table: `scripts/ab_table.py results/runs late --md`, using the replay JSONs and `metrics.jsonl` over the replay's
    own window (start + setup + drain + 600 s).

| (window = 1,500 s) | offload (skip off) | candB (skip on) | candB vs offload |
|---|---|---|---|
| gen tok/s | 617.0 | **618.5** | +0.2% |
| attempts / h (in window) | 43.2 (18) | 43.2 (18) | = |
| turns / h (in window) | 825.6 (344) | 832.8 (347) | +0.9% |
| prompt tokens queried | 17.42M | 17.33M | |
| GPU prefix hit | 57.7% | 55.7% | |
| host-restored tokens (`vllm:external_prefix_cache_hits_total`) | 6.94M (39.8% of queried) | 7.23M (41.7%) | |
| true uncached share (queried - GPU hit - host) | 2.5% (291.5 tok/s) | 2.5% (290.8 tok/s) | = |
| turn uncached mean / p99 | 2,186 / 55,445 | 2,175 / 55,445 | |
| turns > 20k uncached / full re-prefills (≥ 50k, ≤ 640 cached) | 4 / 4 | 4 / 4 | = |
| preemptions (window; whole run) | 2; 24 | 80; 155 | see below |
| TTFT p50 / p99 | 0.88 / 240 s | 1.01 / 297 s | |
| turn p50 / p99 | 57.0 / 2,311 s | 60.8 / 2,280 s | |
| KV mean / peak (window) | 0.782 / 0.999 | 0.786 / 1.000 | |
| running mean / waiting mean (max) | 31.3 / 0.2 (4) | 31.3 / 0.3 (10) | |
| host load / store | 18.9 / 68.5 MB/s (1,088 loads, 3.9 ms, 26 MB each) | 34.4 / 68.7 MB/s (1,404 loads, 5.7 ms, 37 MB each) | |
| host memory: Shmem max / server PSS max | 123.2 / 136.7 GiB | 123.2 / 137.2 GiB | |
| 5-min gen tok/s inside the window | 615, 532, 625, 661, 652 | 608, 510, 646, 671, 654 | |

- **Throughput: a tie.** The seeded workload is identical in both arms (same turn-prompt p50 / p90 of 37,808 / 100,917
  and the same uncached p99), and the 5-min windows track each other within ±3% in both directions. So
  throughput here is set by the workload's phase, not by the indexer.
  - E054's +6-7% decode gain (B = 36-48, contexts 8-40k, no offload) and its +7.7% replay gain (1 s tool idle) do not
    show at this late-phase, offload-heavy load: long contexts (p50 38k, p90 101k), 2,048-token chunks for tool
    results and host restores in most steps.
  - Two candidate reasons, not tested here. First, at 60-100k contexts the stride kernel saves less per row (E054
    kernel A/B: 1.4-1.6x at 128k vs 4.5x at 32k). Second, the engine loop may be CPU-bound: E051's CPU benchmark puts
    the connector's scheduler side at 3.3 ms per step at 36 × 40k, growing with context (`update_offload_keys`
    slices from 0 every step), and the box's load average was 11-14.
  - One run per arm, no repeat.
- **Preemptions:** higher with candB (155 vs 24 over the run, 80 vs 2 in the window). With offload they are cheap:
  preempted streams come back from host (candB had more and larger loads), throughput and attempts/h are unchanged, and
  TTFT p99 is 297 vs 240 s. #156 changes neither the KV pool nor any allocation (the pool is pinned and identical), so
  the gap most likely comes from timing, since preemption is a threshold effect on a stack that is not run-to-run
  deterministic. With n = 1 per arm this is a flag, not a finding. The L4 soak should log preemptions per hour.
- **Context, E051's hour-8 arms with the same workload** (other boxes; retention interval 128, ours 512): offload off
  thrashes at ~93-96 gen tok/s at 36-48 streams (box 1); offload on at 36 streams gives 602.7 (box 1, 28 GiB) and
  874.0 (box 3, 32 GiB). Box 2 here sits with box 1. Our GPU hit (56-58% vs box 1's 37.4%) is higher, plausibly
  from interval 512.

## 4. Verdicts

| test | result | verdict |
|---|---|---|
| Step 1: kit build (16 members, deterministic, pinned inputs) | `1a3a6679` / `624e7754`; rebuild byte-identical | **GO** |
| Step 1: packaging through cell 0b + cell 2 (builder unchanged) | builder suite 54/54; E056 cell tests 3/3 (commit ok, commit fail raises, rerun fallback to `377d83e0`) | **GO** |
| (a) install via the extracted notebook cells on the box + E051 CPU tests | 16/16 shas; source venv unchanged; 26/26 tests | **GO** |
| (b) server start, candidate-B flags | offload log line, pool 21,490 blocks / 1,591,736 tokens, 0 serving tracebacks, +32.0 GiB pinned (4 × 8 GiB), skip active (log, module flag, only the stride kernel compiled) | **GO** |
| (c) E051 byte-level offload parity, skip on | 8/8 cases, 3k-120k, all 4 ranks: 0 mismatches, 0 poison | **GO** |
| (d) late-phase A/B, candB vs offload only (36 streams) | 618.5 vs 617.0 gen tok/s (+0.2%), 43.2 attempts/h both; preemptions 155 vs 24 (n = 1) | **GO on safety (no regression); speed gain not shown at this load** |

**Overall: GO for the combined kit as candidate B's overlay.** The two changes coexist: offload restores the exact KV
bytes with the skip on, and the skip runs exclusively under offload. At hour-8 load the indexer gain is not visible, so
candidate B's throughput case rests on offload (E051: 6x at this load), and #156 is free and exact.

## 5. Using this kit in a Kaggle notebook: blockers and checks

**Nothing in the packaging blocks it.** `build_frozen_mt.py --overlay experiments/E056_candidate_b/kit` plus
`--server-extra '["--kv-transfer-config", <JSON>]'` (and `--retention 512`) needs no builder or solver change. The
notebook grows by about 132 KB of base64 for the 99 KB tarball. Still open before a submission (not tested here):

1. **L4 / Kaggle conditions:** PCIe Gen3 host transfers (E051 measured Gen4 at ~4.3 GB/s), 50 W prefill, and a 12 h
   soak with preemptions per hour logged. E051 recommends a gate that also holds while requests wait (S17) rather than
   running ungated.
2. **Host RAM:** 32 GiB private pinned costs exactly 32 GiB (measured +32.0 GiB). E051's Kaggle budget is ~139-157 GiB
   against the 173 GiB cgroup. Not measured on Kaggle.
3. **KV pool without the override:** both arms pinned the pool with `--num-gpu-blocks-override 21490`; the notebook
   sizes it by util (0.945). Check that the commit run logs `GPU KV cache size: 1,554,183 tokens` with the connector
   present, in case it takes GPU memory before profiling.
4. **Fallback:** a competition-rerun failure of cell 0b restores the kit's 12-file overlay and Version 5's flags (no
   `--kv-transfer-config`), as tested. The commit run raises instead.
5. `DSV41_INDEXER_SKIP` is the kill switch: set it to `"0"` in the settings cell; it reaches the server through
   `server_env()`. The offload connector's kill switch is dropping `--kv-transfer-config`, which makes the 4 files inert.

## 6. Commands

Local (repo root):

```bash
.venv/bin/python experiments/E056_candidate_b/scripts/build_kit.py            # kit/ (+ check)
.venv/bin/python -m unittest discover -s kaggle/dsv41_tools/tests               # builder suite: 54 OK
.venv/bin/python -m unittest discover -s experiments/E056_candidate_b/tests     # E056 cell tests: 3 OK
KVT='{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":34359738368,"offload_prompt_only":false,"swa_store_policy":"checkpoint","shared_region":false,"eviction_policy":"lru"}}'
EXTRA=$(.venv/bin/python -c 'import json,sys; print(json.dumps(["--kv-transfer-config", sys.argv[1]]))' "$KVT")
.venv/bin/python kaggle/dsv41_tools/build_frozen_mt.py --slug e056-candb-installtest --workers 36 \
  --overlay experiments/E056_candidate_b/kit --retention 512 --server-extra "$EXTRA" --out <scratch>/nb
```

Box bundle `/root/e056/`: `kit/` (this kit), `kit_base/vllm_overlay_rw.sha256` (the kit's 12 lines),
`scripts/notebook_install.py`, `nb/e056-candb-installtest.ipynb`, `input/{dsv41_kit.tgz (kaggle_kit_v5),
arc-agi_test_challenges.json (1 dummy task)}`, `gpu/` (this repo's), `e051/gpu/{e051_ext,kv_parity,tool_replay_kv,
metrics_logger,summarize_capacity}.py`, `e051/tests/{test_dsv41_kv_offload,dsv41_fixture}.py`,
`shapes/tools_e043_e128t.json` (from E042 `metadata/trace_shapes/`). Then on the box:

```bash
bash /root/e056/gpu/setup_candb_venv.sh                                         # (a), 34 s
ARM=candb RUN=late_candb PARITY=1 setsid -f bash /root/e056/gpu/run_arm.sh > /root/e056/runs/late_candb.driver.out 2>&1 < /dev/null   # (b) (c) (d)
ARM=offload RUN=late_offload setsid -f bash /root/e056/gpu/run_arm.sh > /root/e056/runs/late_offload.driver.out 2>&1 < /dev/null       # (d), after candB stopped
python3 scripts/ab_table.py results/runs late --md                              # locally, on the copied runs
```

Incident: the first chained launch of the offload arm ran twice. A `setsid nohup ... &` waiter inside `ssh '...'`
that looked dead was in fact alive, and I relaunched it with `gpu/chain_offload.sh`. Both servers started at 04:08 and
both OOMed at load. Their logs are kept in `results/runs/failed_dup_late_offload/`, and the progress file has a NOTE
line. The arm was rerun once, alone, at 04:09:53. Lesson: launch with `setsid -f` and check with `ps -o args` on the
full command line.

## 7. Files

- `scripts/build_kit.py` (kit builder, `--check`), `scripts/notebook_install.py` (cells 00 / 0 / 0b + cell 2's overlay
  block from a generated notebook), `scripts/ab_table.py` (the A/B table).
- `tests/test_notebook_install.py` (CPU: commit ok / commit fail / rerun fallback through the notebook cells).
- `gpu/setup_candb_venv.sh`, `gpu/serve_candb.sh`, `gpu/run_arm.sh`, `gpu/e056_ext.py` (worker probes),
  `gpu/chain_offload.sh` (see the incident above).
- `kit/vllm_overlay_rw.{tgz,sha256}`.
- `results/runs/` (copied from `/root/e056/runs/`; scanned for tokens, none found):
  - `late.progress` (driver timeline);
  - `setup_candb_venv.log`, `notebook_install.log`, `e051_cpu_tests.log`, `src_venv_{before,after}.sha256`;
  - `late_candb/` and `late_offload/`: `server.log`, `env.txt`, `startup.txt`, `indexer_state_{start,end}.json`,
    `loaded_files.json`, `metrics.jsonl`, `smoke.{json,out}`, `replay_C36_D20_S1.25_T0.3_M30000.{json,out}`, and
    for candB also `kv_parity.{json,out}`;
  - `failed_dup_late_offload/` (the duplicate launch's OOM logs).

## Kaggle smoke (2026-10-27, user-approved): `chankhavu/dsv41-tools-submit-candb` Version 1 — PASS
Pushed 06:20 UTC, COMPLETE by ~06:42; notebook built by `build_frozen_mt.py --slug dsv41-tools-submit-candb --workers 56
--budget 0.5 --gate 0.85 --retention 512 --util 0.945 --auto-repair --admit-max-waiting 1 --smoke-minutes 30 --overlay
experiments/E056_candidate_b/kit --solver/--harness/--train-check kaggle/dsv41_tools/pinned_candb/... --server-extra
[--kv-transfer-config ... cpu_bytes_to_use 17179869184 (16 GiB), shared_region false, checkpoint, lru]`.
Output: `kaggle_smoke_v1/` (kernel log, `kernel_log.txt`, dsv41_logs, smoke_env.jsonl, submission.json).
- **Code:** "E046 CHECK OK" with solver 77fe80a3, harness 882c817f, train_check 8fe36175, auto-repair, overlay 1a3a6679;
  "vLLM overlay installed and checked" (16 files).
- **Server:** up after 675 s (V6 bench: 590 s); GPU KV cache 1,554,183 tokens; "KV offloading: swa_store_policy=checkpoint,
  hits aligned to 128 tokens, skipped scratch groups [8]"; PYNCCL all-reduce (custom all-reduce off); 0 tracebacks, 0 OOM.
- **Run:** 3 tasks / 6 pairs, all submitted (5/6 train_pass), submission valid, done at +0.35 h (21 min).
- **Allocator:** `PYTORCH_CUDA_ALLOC_CONF` is unset in Kaggle's image, so V5/V6 also ran without expandable segments:
  B's allocator setting equals V6's and the pods'.
- **Host memory (cgroup v2, limit 173.0 GiB):** peak anon 10.8 GiB, peak shmem 106.9 GiB (Engram ~91 + 16 GiB pinned
  pool), unevictable 0, swap 0. memory.current / memory.peak reach 173.0 GiB only through page cache (file up to
  166 GiB: the model read), which is reclaimable. VM MemTotal 188.7 GiB, min MemAvailable 64.8 GiB. Non-reclaimable
  ~118 GiB at startup; with the pods' 12 h anon growth (~+5-10 GiB) the projected peak is ~125-130 GiB, ~43 GiB below
  the limit.
- **GPUs:** power limit **72 W** (not 50 W), peak draw 72.1-72.6 W, SM up to 2,040 MHz, peak memory 22.06-22.12 / 23.03 GB,
  75-80 C. The pods also run at 72 W, so power is not a pod-vs-Kaggle difference (the toolkit's 1.45x prefill gap has
  another cause).
- **Disk:** /kaggle/working 19.5 GiB free, /tmp 1,041 GiB, /dev/shm 85.5 GiB.
- Benign, also in earlier Kaggle runs: "Error in sitecustomize ... No module named 'wrapt'".
