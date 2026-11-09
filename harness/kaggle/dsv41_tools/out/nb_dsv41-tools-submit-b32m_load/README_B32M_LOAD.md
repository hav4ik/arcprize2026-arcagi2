# B32-M Kaggle LOAD TEST: not a submission candidate (built 2026-10-28 ~22:10 UTC)

This folder is the same notebook as smoke V2 (`../nb_dsv41-tools-submit-b32m/`, the candidate) except for its commit run. A push makes it a new version of `chankhavu/dsv41-tools-submit-b32m` (Version 3). **Do not submit that version.** Submit V2.

Rebuild: `.venv/bin/python kaggle/dsv41_tools/build_b32m.py --manh-nb <pull dir> --load-test --smoke-minutes 43 --out kaggle/dsv41_tools/out/nb_dsv41-tools-submit-b32m_load`

## Commit run
- **Tasks:** 80 typical-size eval tasks, the middle 80 of the 120 by prompt size (4,044-11,375 characters; 31 have several test inputs; 113 pairs). The challenges come from the competition's `arc-agi_evaluation_challenges.json`, so all 80 workers are busy. If that file is not mounted, it falls back to the test file and V2's 2 tasks.
- **Deadlines from the first cell:** stop-new 38:00, force 40:00, hard stop 41:00, kill switch 42:30. Session ≤ ~45 min, ~1.5 h GPU quota at 2x.
- **Report:**
  - the server's gen tok/s for every 30-s solver window;
  - vLLM's 10-s server windows: pure-decode windows (prompt 0 tok/s, ≥ 70 running) with n / median / p90 / max, and the mean gen tok/s while running;
  - the timeline (install, server start and up) and KV / running.
- **Verdict:** fails only on a server error or crash, or no valid submission.json:
  - fatal: the server never came up, died or exited, or the solver crashed; the environment or row-chunk patch failed;
  - a warning only: few finished tasks, contract results, no final write.
- **Everything else is as V2:** GPU gate first, latest image, plan B row-chunk patch, B32's code byte-identical, Manh's extra flags OFF.
- **Competition rerun:** identical to V2 (the load settings live in the commit-only block).
