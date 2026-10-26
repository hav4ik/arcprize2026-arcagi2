# E051: KV CPU offloading for DeepSeek-V4.1 in the SM89 vLLM fork

Status (2026-10-26 21:00 UTC): **GPU gates 1-3 passed on Vast 57816747 (4×4090)**: install, server start, byte-level
KV parity up to 120k tokens. Gate 4 (capacity A/B) running. Earlier: designed, patched and tested on CPU (26 unit
tests, mini-engine over the fork's real KV manager, mutation check).
mutation check); GPU parity and capacity untested. Not on the 24-h critical path for the projected-75 goal
(`experiments/E046_multi_test_rollout/goal75/`): once thrash is removed, more workers add ~0.2 pts in the simulator.

- Why it failed on 10-24: the 8-row compressor ring group trips the block-size divisibility check
  (`offloading/config.py:57-64`, fixed upstream in #57145 after the fork's 2 Sep snapshot) and the scheduler's
  `FullAttentionSpec` assertion (`offloading/scheduler.py:125`).
- Design (`DESIGN.md`): never offload the ring; MLA+indexer dense (one 128-token block per host slot, 3,584 B/token for
  all 4 ranks); SWA as checkpoints only (`swa_store_policy=checkpoint`: the 4-chunk window at the prompt end and a
  rolling latest 128-aligned boundary; dense SWA would be 28x the MLA bytes); hits rounded down to 128 so a restore
  resumes where a GPU hit would. ~205 MB per 50k-token conversation; a 28 GiB budget holds ~3x the GPU pool.
- Patch: `overlay/vllm/` (4 full replacement files: offloading config, scheduler, cpu manager, cpu spec) and
  `patch/dsv41_kv_offload.diff`; byte-identical job logs with offloading off or at default options; installer
  `gpu/install_overlay.sh` (sha-checked, `--revert`).
- GPU plan (`GPU_TEST_PLAN.md`): parity (greedy ids + top-5 logprobs identical for GPU hit vs host restore), then
  capacity (tool replay at 36/56/72 streams with 1 s / 20 s tool idle, offload off vs on), then a 3-4 h pod soak.
- Effort to a submit-ready kit: ~4-7 days. Main risk: untested combination of the offload connector with async
  scheduling, Model Runner V2 and FULL CUDA graphs on a hybrid model. Phase 2: upstream #57652 single-copy layout
  (4x less host memory).

Handoff for a separate agent: `HANDOFF.md` (2026-10-26).

## GPU results (box 1 = Vast 57816747, 4×RTX 4090, PCIe Gen4, owned by E051 from 2026-10-26 19:50 UTC)

Box layout: `/root/dsv41/venv_offload` is a copy of the shared venv with only the 4 overlay files installed
(`gpu/install_overlay.sh`, `--revert` ready). The shared `/root/dsv41/venv` is untouched. **Gotcha:** a copied venv's
`bin/` scripts keep `#!/root/dsv41/venv/bin/python`, so `venv_offload/bin/vllm` would silently run the unpatched code.
The 53 shebangs were rewritten in the copy. All E051 files and runs are under `/root/e051/`. Host RAM is shared, so the
plan is at most half of 503 GB; `gpu/box_serve.sh` refuses to start above that.

- **Gate 1 (install), 19:58 UTC:** 26/26 CPU tests pass against the installed tree.
- **Gate 2 (server start + smoke):**
  - Kaggle server command plus `--kv-transfer-config` (28 GiB, `swa_store_policy=checkpoint`) and
    `--num-gpu-blocks-override 21490`; ready in 210-230 s.
  - Log: `KV offloading: swa_store_policy=checkpoint, hits aligned to 128 tokens, skipped scratch groups [8]`.
  - Engram pins 23.6 GiB per rank; the 30.06 GB shm region is mapped by all 4 ranks.
  - A 3,000-token prompt after a GPU reset hits 2,944 tokens from the host. 0 tracebacks, 0 preemptions.
- **Run-to-run nondeterminism (blocks the logprob gate):** two GPU-prefix-hit runs of the same prompt already differ at
  token 0 (logprob −4.436 vs −4.495). Two cold runs differ too, also with `VLLM_BATCH_INVARIANT=1`. So the planned
  bit-identical logprob check (`gpu/parity_test.py`) cannot pass on this stack, offloading or not
  (`gpu/diag_parity.py`).
- **Gate 3 (parity), replaced by a byte-level check: PASS.**
  - Tools: `gpu/kv_parity.py` + `gpu/e051_ext.py`, a worker extension loaded with
    `--worker-extension-cls e051_ext.E051Ext` and called via dev-mode `/collective_rpc`, test servers only.
  - Method: a host restore must put back exactly the KV pages a GPU prefix hit uses. The KV pool is filled with 0xA5
    before every restore, so a load that silently writes nothing cannot pass.
  - All 4 ranks, every compressed-attention block below the hit and each sliding-window group's 4-block checkpoint,
    byte-identical, 0 poison pages:

    | L | prompt case: hit (pages per rank) | resume case after a 300-token decode: hit (pages per rank) |
    |---|---|---|
    | 3,000 | 2,944 (51) | 3,200 (53) |
    | 20,000 | 19,968 (184) | 20,224 (186) |
    | 60,000 | 59,904 (496) | 60,288 (499) |
    | 120,000 | 119,936 (965) | 120,192 (967) |

  - Equivalence argument: a GPU hit and a host restore give the same kernels identical KV bytes and the same fresh ring
    and per-request state, so the remaining difference is the stack's own nondeterminism.
- **Transfers (Gen4):**
  - loads 1.81 GB in 0.42 s (~4.3 GB/s, ~7.5 ms per load), so a 120k-token restore takes ~0.12 s;
  - stores ~0.77 GB/s in the background;
  - lookup sync delay 0.09 ms mean.
- **Host memory:** Shmem 119 GiB (Engram 94 + offload 28); box-wide used 166 GB.
- **Gate 4 (capacity A/B), started 20:59 UTC:** `gpu/run_capacity.sh` runs `gpu/tool_replay_kv.py`, an E042 replay
  copy with offload accounting. Workload: E128 tool shapes, 600 s warm-up, 2,400 s window, pool 21,490 blocks.
  Arms: offload on vs off at 56 streams / 20 s tool idle, 72 / 20 and 36 / 1. Results: pending.

