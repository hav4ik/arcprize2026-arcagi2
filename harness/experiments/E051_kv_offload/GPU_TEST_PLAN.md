# GPU test plan: DSv4.1 KV CPU offload on the 4×RTX 4090 dev box

Run these steps in order; each one gates the next. The box matches the Kaggle L4s where it matters most: SM89, 24 GB per GPU, TP4, and
504 GB of RAM. It differs in two ways: PCIe Gen4 instead of Kaggle's Gen3, and about 450 W per GPU instead of 50 W. So compare **counts**
(re-prefilled tokens, loads, preemptions) and **ratios** (offload on vs off), not absolute tok/s. Keep the GPU pool at Kaggle's size with
`--num-gpu-blocks-override 21490`. Kaggle has 2.28 GiB per GPU in 113,920-B blocks (`server.log:343`).

Never stop servers or runs by process name; other agents share hosts. Use the PID file the serve script writes.

## 0. Install

```bash
# venv = fork wheel + the 12-file kit overlay (as on the pods), model = E128 2-bit
K=<this kvoffload dir>; VENV=/opt/dsv41/venv; MODEL_DIR=/data/dsv41-flash-1017-arc-tool-e128-2bit
$K/gpu/install_overlay.sh $VENV            # sha256-checks the 4 targets, keeps *.orig-dsv41off backups
# CPU unit tests against the installed tree (no GPU needed; ~1 s):
cd $K/tests && PYTHONPATH=$K/tests $VENV/bin/python -m unittest -v test_dsv41_kv_offload
df -h /dev/shm                             # must have >= OFFLOAD_GIB free, else SHARED_REGION=false
```

To revert, run `$K/gpu/install_overlay.sh $VENV --revert`.

## 1. Server flags

The script is `gpu/serve_dsv41_offload.sh`: the Kaggle command from `server_cmd.json`, plus offload and `VLLM_SERVER_DEV_MODE=1`. The
relevant flags:

```
--num-gpu-blocks-override 21490
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"cpu_bytes_to_use":30064771072,"offload_prompt_only":false,
  "swa_store_policy":"checkpoint","shared_region":true,"eviction_policy":"lru"}}'
```

- **Total host budget.** `cpu_bytes_to_use` is the total over all 4 ranks: 28 GiB here, 7 GiB per rank.
- **Shorter form.** `--kv-offloading-size 28` merges `cpu_bytes_to_use` into a given `--kv-transfer-config`
  (`config/vllm.py:986-1013`).
- **Kaggle form.** With the Kaggle solver, pass the same flag through
  `ARC_SERVER_EXTRA_ARGS='[..., "--kv-transfer-config", "<json>"]'` (`kaggle/dsv41_tools/solver_dsv41.py:175,302`).

Startup checks in the log:
- `KV offloading: swa_store_policy=checkpoint, hits aligned to 128 tokens, skipped scratch groups [<ring idx>]`;
- `Allocating 1 CPU tensors` / `cudaHostRegister rank=... 28.xx GB`;
- the GPU KV size unchanged against the run without offload;
- the time to "ready" against the baseline (expect +10-30 s for populating and pinning 28 GiB).

Do the same with `OFFLOAD_GIB=0` (baseline) and `POLICY=all`.

## 2. Smoke (15 min)

1. Start: `OFFLOAD_GIB=28 POLICY=checkpoint gpu/serve_dsv41_offload.sh`, then `python gpu/metrics_logger.py --out smoke.jsonl &`.
2. Send one 3k-token completion, then the same prompt again after `POST /reset_prefix_cache`. Expect
   `usage.prompt_tokens_details.cached_tokens == 2944` on the second call, `vllm:external_prefix_cache_hits` +2944, and
   `vllm:kv_offload_load_bytes` > 0.
3. **Fail** if there is any traceback in the server log, any `kv_offload_allocation_failure`, or `cpu_cache_usage_perc` stuck at 1.0.

## 3. Correctness: restored prefix vs GPU hit (bit-identical)

Run it alone, with no other traffic, since batch composition changes numerics:

```bash
$VENV/bin/python $K/gpu/parity_test.py --url http://127.0.0.1:8100 --lengths 3000 20000 60000 120000 --json parity.json
```

Per length L, the script runs two cases:

- **prompt case.** The resume point is `round_down(L-1, 128)`:
  1. reset everything;
  2. cold run (this stores);
  3. GPU-hit run;
  4. reset the GPU only;
  5. host-restore run.
- **resume case.** The resume point is the finished request's rolling checkpoint at `round_down(L+63, 128)`. The next-turn prompt is
  `P + sampled ids + 200 tool ids`:
  - host path: cold P, reset the GPU, run the next turn;
  - GPU path: cold P again, run the next turn.

**Pass** needs all of these:
- `cached_tokens` is equal to the expected boundary on both paths;
- the external hit count and load bytes are above 0 on the host path;
- **token ids and all top-5 logprobs are exactly equal**;
- the cold run is deterministic.

Repeat once with `POLICY=all` (dense SWA) and once with `SHARED_REGION=false`.

**If it fails, use `first_diff`:**
- A difference at token 0 points to state that was not restored: check the ring and indexer buffers.
- A late, tiny difference points to numerics: confirm the request ran alone, then repeat.

## 4. Capacity: more streams than the GPU KV holds

**Workload.** Use the repo's E128-shaped replay, which sends token ids and appends the model's own ids:

```bash
python experiments/E042_dsv41_l4_synthetic/scripts/tool_replay.py \
  --shapes experiments/E043_dsv41_real_pruning/results/tool_shapes_e128t_w10_u2_gptq_eng64_sub80.json \
  --url http://127.0.0.1:8100 --model dsv41 --conc C --tool-delay D --warm-s 600 --window-s 2400 --json run_C_D.json
```

Run `metrics_logger.py` alongside every run. The matrix, each run about 50 min:

| run | offload | C (streams) | D (s idle per turn) | purpose |
|---|---|---|---|---|
| A0 / A1 | off / on | 36 | 1 | overhead at today's operating point: tok/s within noise, store traffic |
| B0 / B1 | off / on | 56 | 20 | idle streams lose GPU prefixes: re-prefill vs restore |
| C0 / C1 | off / on | 72 | 20 | beyond capacity: preemption-recompute vs restore |
| C2 | on, `POLICY=all` | 72 | 20 | host footprint and churn of dense SWA (expect allocation failures or evictions) |

**Report per run:**
- generated tok/s;
- uncached prompt tokens per turn (`prompt_tokens − cached_tokens`), mean and p99;
- the number of turns that re-prefill more than 20k tokens (full re-prefills);
- turn latency p50/p99 and TTFT;
- preemptions;
- external hits divided by external queries;
- loads per minute and load MB/s, with `load_time` per load (ms per 50 MB);
- stores in MB/s, and `cpu_cache_usage_perc`;
- host `Shmem` / cgroup memory;
- the mean `lookup_sync_delay`.

**Decision.** Offload is worth shipping if B1 and C1 cut uncached prompt tokens by more than 50% against B0 and C0 and gain at least
5% generated tok/s at equal or better p99 latency, while A1 stays within 2% of A0.

## 5. Soak with the real pipeline (3-4 h)

Run the pod pipeline (`experiments/E046_multi_test_rollout/pods/podrun.sh`, multi-test, budget 0.5, context limit 163840) on the eval
set, at the current candidate (36 workers, gate 0.6) and at 44 workers with gate 0.75. Add the offload flags through
`ARC_SERVER_EXTRA_ARGS`.

**Pass** needs all of these:
- 0 server restarts;
- 0 `Traceback` / `AssertionError` in the server log (watch `assert transfer_result.success` and `update_state_after_alloc`);
- KV peak no worse than the run without offload;
- attempts per hour better than without offload;
- accuracy on the finished attempts within the noise band of the same config without offload (`bench_accuracy` is flawed, so match by
  test input as in findings §10).

## 6. Kaggle-shaped check (L4 pod; you rent it)

Repeat steps 3 and 4 (A and B only) on a 4×L4 pod with KV pinned to Kaggle's pool (`ARC_GPU_UTIL=0.938`). This is where the real Gen3
PCIe load times and the 50-72 W prefill cost appear.

Also measure on Kaggle itself, in a commit run before any submission:
- `df -h /dev/shm`;
- cgroup `memory.max` and `memory.current` with Engram loaded;
- whether `cudaHostRegister` of 25-30 GiB succeeds next to Engram's 94 GiB.

If `/dev/shm` is too small, use `"shared_region": false`.

## 7. Metrics to log (all runs)

From `/metrics`, every 5 s via `gpu/metrics_logger.py`:
- `vllm:kv_cache_usage_perc`, `num_requests_running/waiting`, `num_preemptions`;
- `prefix_cache_queries/hits` (GPU) and `external_prefix_cache_queries/hits` (host);
- `kv_offload_load_bytes/time`, `kv_offload_store_bytes/time`, and the load/store size histograms;
- `kv_offload_cpu_cache_usage_perc` (with write and read), `kv_offload_allocation_failure`;
- `kv_offload_lookup_sync_delay_seconds`, `prompt_tokens`, `generation_tokens`.

From the host: `Shmem`, `Mlocked` and `MemAvailable` from `/proc/meminfo`, and cgroup `memory.current`.

From the client, per turn: `prompt_tokens`, `cached_tokens`, TTFT, duration.

Optional: `nvidia-smi dmon -s pucvmet` for PCIe rx/tx during restores.
