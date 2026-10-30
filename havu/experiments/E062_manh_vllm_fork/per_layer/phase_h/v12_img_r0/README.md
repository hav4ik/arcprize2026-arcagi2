# "V12 + images (causal)" r0: interim run, stopped by the user at 0.82 h

**Interim.** Notebook nb_v12_img **790b1bd9d7b1**: no `--watermark-reload-only`, with V12's recovery.
- Model: the staged HF causal E128 image model (@209c810a747c) plus E128's Engram.
- Workload, runner and wheels (g22df163d2d) as in `../v12_img/README.md`. That README holds the setup, the checks and the per-file sha256 of the staged model.
- Launched 17:55:56 UTC; server up at +263 s; solving from 18:00:20.
- At 18:45:07 UTC (Kaggle 0.82 h) the user stopped it, to rerun with the final code (Manh's flag plus the new solver recovery).
- The run was killed by PGID: 60 groups. The supervisor got SIGTERM first; the server exited in 16 s and GPUs went back to 0 MiB.
- The 1 "server died" line and the 4 errored records (re-queued attempts) are the stop itself. Before it: 0 deaths, retries or OOMs.

## Last state vs V12 text at the same clock (0.82 h)

| measure | r0 (images) | V12 (text) |
|---|---|---|
| score so far over 240 (= the submission.json score, 10.90) | 10.9 | 14.4 |
| tasks answered | 36 | 41 |
| first-attempt accuracy so far | 25/36 = 69.4% | 34/41 = 82.9% |
| gen tok/s, running (0-0.82 h) | 667, 59.9 | 742, 61.7 |
| prefix-cache hit | 0.909 | 0.950 |
| uncached prompt tok/s | 576 | 286 |
| uncached prompt tokens per turn | 1,655 | 1,007 |
| prompt tokens per turn | 19.1k | 16.3k |
| preemptions | 0 | 0 |
| server running / solver in-flight | 61.4 / 62.1 | 61.8 / 62.3 |
| MM processor-cache hit rate | 97.0% of 5,966 image queries | - |
| API-server + EngineCore RSS (MM caches) | 6.2 GiB | 2.8 GiB |
| worst cap − reserved | 1.34 GiB | 1.45 GiB |
| Kaggle-eq RAM (cgroup) | 136.7 GiB | ~134 |

- **Paired per task at 0.82 h** (`../scripts/paired_at_hour.py`), on the 34 tasks both runs answered: images 72.1% vs text 83.8%.
  - 0 gained; 4 lost (4c3d4a41, 4cccce55, 6e453dd6, e376de54). Paired −11.8 pp, 95% CI −23.5 to −2.9.
  - These are the earliest, fastest tasks, so n is small and the sample is not representative of the whole run.
- **Images cost throughput and prefill:** −10% gen tok/s, +64% uncached prompt tokens per turn, and +17% prompt per turn. The KV gate began holding at ~0.7 h; V12 began at ~0.6 h.
- **MM processor caches:** +3.4 GiB host RAM (API server + EngineCore).

## Notes as the run went

### Load (18:00 UTC)
- Server up at +263 s (V12: +297 s): his supervisor, start 0. Solver from 18:00:20: 240 tasks, 344 pairs, the usual deadlines.
- **The image path is on:**
  - non-default args include `disable_chunked_mm_input: True` and `limit_mm_per_prompt: {image: 32}`;
  - the vision tower is loaded: MMEncoderAttention initialised, and the encoder cache "profiled with 2 image items";
  - 0 "bidirectional" lines;
  - the solver is 9ebf372e (the image solver; not the v3 fallback), with env ARC_GRID_FORMAT=text+image and MANH_VISION_MAX_IMAGES=32;
  - images reach the server: 476 MM-cache queries in the first minute (~7.4 per attempt).
- **KV pool 1,597,824 tokens** (V12: 1,608,578, −0.7%).
- Model loading 14.56 GiB per rank. Allocator caps 21.15-21.20 GiB with 19.15 reserved (as V12).

### State at 0.76 h (18:41 UTC; run "r0")
This run is "r0": V12 + images without Manh's later `--watermark-reload-only` flag. It stays up as an interim speed and quality reference until the final notebook arrives.

Stable: 0 events, retries or OOMs.

| at 0.76 h | V12 + images r0 | V12 text |
|---|---|---|
| score so far (tasks answered) | 9.7 (32) | 13.5 (39) |
| first-attempt accuracy so far | 22/32 = 68.8% | 32/39 = 82.1% |
| gen tok/s, running | 671, 60.9 | 743, 62.5 |
| prefix-cache hit | 0.903 | 0.945 |
| uncached prompt tok/s | 590 | 311 |
| uncached prompt tokens per turn | 1,624 | 986 |
| prompt tokens per turn | 18.7k | 16.0k |
| preemptions | 0 | 0 |
| server running / solver in-flight | 63.0 / 63.3 | 63.1 / 63.3 |
| MM processor-cache hits | 96.6% of 5,287 image queries | - |
| API-server + EngineCore RSS | 6.2 GiB | 2.8 GiB |
| worst cap − reserved | 1.36 GiB | 1.49 GiB |
| Kaggle-eq RAM (cgroup) | 136.8 GiB | ~134 (SFT text run 132.4) |

- The KV gate began holding at ~0.7 h; V12 began at ~0.6 h.
- Alerts were startup artifacts only:
  - supervisor starting → up;
  - "server waiting 51" from the initial burst;
  - the load ramp of cap − peak;
  - "RAM rising 259 GiB/h", whose window included the pre-server samples.


## Data
`data/`: the same files as the other replicas.
- Dashboard files: gpumem.log, ram.tsv, tps.tsv, status.json, workload.txt.
- `dsv41_logs/`: attempts, metrics, solver log, supervisor, server log.
- `nbruns/v12img_rerun/`: samp.jsonl, alerts, timeline, mem.log, `stop_snapshot/`.

The per-attempt traces (`dsv41_traces.tar.gz`) and the files over 1 MB are in the private HF dataset `livobereshka/arc-agi2-vllm-serving-lab` under `replicas/v12_img_r0/` (`data/RAW_README.md`).
