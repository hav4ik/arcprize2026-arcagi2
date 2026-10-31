# "V12 + images FINAL" r1 (notebook 5f4cfe5d1694): stopped at Kaggle 0.98 h (19:49 UTC)
- **Setup:** `../v12_img/README.md` (staged causal model, g22df163d2d wheels) with notebook 5f4cfe5d1694: watermark-reload-only, solver 0c2f81e8, no replay CUDA graphs. Launched 18:49:43; stopped by PGID at 19:49 (49 groups, GPUs to 0) to relaunch with Manh's replay graphs. 0 events before the stop.
- **At 0.97 h vs V12:** score 17.1 vs 16.9, tasks 47 vs 47, paired +2.6 pp on 38 shared tasks (CI 0.0 to +7.9).
  - Serving: gen tok/s 680 vs 756; prefix hit 0.854 vs 0.954; uncached prompt per turn 1,844 vs 997; preemptions 4 vs 0; host-KV restore 681 tok/s vs 0.
  - Images: MM-cache hits 97.5% (6,943 queries).
  - Memory: API+Engine RSS 6.1 vs 2.8 GiB; worst cap − reserved 1.28 vs 1.41 GiB; Kaggle-eq RAM 136.4 GiB.
- **Data:** `data/` (traces left on the pod at /root/kwork_v12img2_rerun).
