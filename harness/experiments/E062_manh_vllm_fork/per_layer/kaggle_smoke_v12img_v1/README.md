# Kaggle commit-run smoke of `chankhavu/dsv41-tools-v12-img` v1 (2026-10-30)

Notebook `kaggle/dsv41_tools/out/nb_v12_img_submit` (sha256 5bea5b7f…), pushed 22:59:26 UTC. Status COMPLETE at 23:27:40 UTC, so every commit-run verdict passed (a failed check raises).
- Install: vLLM 0.32.1rc1.dev0+g22df163d2d on Python 3.13 / torch 2.14+cu130, Kaggle's latest environment.
- Server: up after 458.8 s. Non-default args include FULL cudagraphs, decoder replay sizes 32-256, watermark_reload_only, disable_chunked_mm_input and image limit 32.
- Tasks: 4 smoke tasks × 2 attempts, all 8 correct and train-passing, 0 errors and 0 allocator retries.
- Memory: KV pool 1,527,155 tokens; reserved 19.59 GiB, free 1.68-1.74 GiB per rank at 23:26.
- Shutdown: the traceback at 23:27:02 is the notebook stopping the server after `done` (EngineDeadError, plus 500s for in-flight probe requests); the supervisor exited with code 0.
- Files: solver/supervisor/stage logs (`.txt`, since the repo ignores `*.log`), attempts.jsonl, status/supervisor json, b32m_env.json, and server.log without progress-bar lines (`server.log.head`).
