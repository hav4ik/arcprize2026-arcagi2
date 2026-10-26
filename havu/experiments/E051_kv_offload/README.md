# E051: KV CPU offloading for DeepSeek-V4.1 in the SM89 vLLM fork

Status (2026-10-26): designed, patched and tested on CPU (26 unit tests, mini-engine over the fork's real KV manager,
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
