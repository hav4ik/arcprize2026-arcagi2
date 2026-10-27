# E056: candidate B overlay kit = kit overlay + E054 #156 part 1 (indexer skip) + E051 KV CPU offload

Status: **(GPU results below)**

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

## 3. GPU tests on Vast 4x RTX 4090 box 2 (57817226), 2026-10-27 02:52 - 05:2x UTC

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
- **Tracebacks: 0 while serving.** The single traceback in candB's log (and in the offload arm's) is the
  `AsyncLLM output_handler failed` raised after `EngineCore: trigger received signal=SIGTERM`, i.e. our stop with
  requests in flight, as in E054's `both` arm. The stack's usual `CUDAGraphMode.FULL is not supported with
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

(d follows.)
