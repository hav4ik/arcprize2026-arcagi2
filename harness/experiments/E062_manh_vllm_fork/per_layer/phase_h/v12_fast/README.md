# V12-fast ("V12 + Manh's watermark-reload-only + replay CUDA graphs + recovery A+B"), pod ccn5ns0b8emb3o, stopped at Kaggle h 4.06

Run `v12fast2_rerun`, launched 22:39:44 UTC 2026-10-30, stopped ~02:45 UTC by PGID at the coordinator's request (verdict clear). Notebook `kaggle/dsv41_tools/out/nb_v12_fast` (276100286588) with ONE pod-side change: cell 0b's `_MT_EXPECT` ARC_GRID_FORMAT/MANH_VISION_MAX_IMAGES set to json/16 (the shipped notebook expected text+image, so its rerun fell back to the v3 kit and the solver crashed at +364 s; the pod copy is sha 9f7e7074d206). Model: V12's E128, 136/136 files sha256-verified against cell 1b's manifest; arc2-2026-src v23; wheels g22df163d2d; 4xL4 Runpod host. Raw data: HF `livobereshka/arc-agi2-vllm-serving-lab` `replicas/v12_fast/` (attempts, metrics, samp, traces, logs bundle); `data/` here is the 5-min sync.

| Kaggle h | score (fast / V12) | tasks answered | gen tok/s | running | worst cap-reserved GiB | prefix hit |
|---|---|---|---|---|---|---|
| 1 | 11.9 / 18.1 | 35 / 50 | 639 / 729 | 59.9 / 58.7 | 1.23 / 1.37 | 0.875 / 0.889 |
| 2 | 26.5 / 29.8 | 83 / 87 | 499 / 559 | 35.8 / 36.3 | 0.98 / 1.19 | 0.876 / 0.558 |
| 3 | 35.0 / 40.4 | 112 / 126 | 518 / 537 | 37.6 / 33.0 | 0.96 / 1.08 | 0.958 / 0.499 |
| 4 | 42.5 / 53.3 | 139 / 164 | 478 / 554 | 34.2 / 36.4 | 0.96 / 1.08 | 0.445 / 0.973 |

- **Score:** behind V12 at every hour (first-attempt accuracy 69.7% vs 73.2% at 4.06 h; no second attempts yet in either).
- **Decode at matched load:** 0.89-0.93x pod 2 (running 58-64: 672 vs 723; 40-57: 570 vs 622; 25-39: 481 vs 541 tok/s, KV >= 60%). The host is ~7-11% slower; the flags do not show a gain.
- **Memory:** 0 allocator retries/OOMs, host RAM 133 GiB Kaggle-eq (flat). cap-reserved ~0.1-0.2 GiB below V12 at the same hours (0.96 vs 1.08 plateau). KV pool 1,536,192 vs 1,608,578 tokens (-4.5%).
- **Prefix cache:** the two runs swap regimes (ours 0.88 vs V12's 0.56 in hour 2; ours drops to 0.41-0.45 with 4.0k tok/s host-KV restores from hour 4), so neither run is cache-stable under this load.
- **Verdict:** replay graphs + watermark-reload-only showed no speed gain on our workload and cost ~0.2 GiB of margin; this host is ~7-11% slower, which explains part of the score gap.
