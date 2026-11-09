# Serve replay bench: our tool workload, replayed deterministically (2026-10-30)

`evals/serve_replay_bench.py` replays recorded harness attempts against an OpenAI-compatible server (vLLM) at a fixed
concurrency. It reports serving performance per 30 s window and overall. The point is to compare serving stacks (vLLM
forks, server flags) on our real workload in ~20 min per configuration instead of 12 h end-to-end runs.
Tests: `evals/tests/test_serve_replay_bench.py` (fake streaming server).

## What it does
- **Input:** harness traces (`{"messages", "events"}`). It accepts `dsv41_logs/traces/*.json.gz`,
  `dsv41_traces.tar.gz`, `evals/results/*/*.messages.json.gz`, dirs (recursive), tarballs and globs. Identical files
  (a dir and its tarball) are kept once.
- **Turns:** turn k = the k-th assistant message, with the k-th `usage` event. Its prompt is every message before it.
  Each request generates exactly the recorded `completion_tokens` (`max_tokens = min_tokens`, `ignore_eos`,
  temperature 0). Assistant messages without usage (a turn cut by a restart or kill) are dropped.
- **`--mode tokens` (default):** the harness's `--token-exact` path. Prompt ids are rendered once, turn k+1 = turn k's
  prompt + the ids the server generated + the rendered new tool results, sent to `/v1/completions`. Prompt growth and
  the prefix-cache pattern follow the real run, including the server's cached decode blocks. After a turn the real run
  cut at max_tokens or rewrote, the next prompt is a fresh render, as in the harness.
  - **Rendering:** through the server's `/tokenize`, or `--local-render MODEL_DIR` with no server. That uses the
    model's `encoding/encoding.py` + `tokenizer.json`, applied like the fork's `vllm/tokenizers/deepseek_v41.py`.
  - **Prep cache:** `--prep-cache F` keeps the renders for reuse.
  - **Tools:** `--tools auto` picks, per trace source, the `run_deepseek_tools.py` tool set whose turn-1 render has the
    recorded length.
- **`--mode chat`:** recorded messages to `/v1/chat/completions`. Each turn re-prefills the previous assistant turn
  (the recorded text, not what the server generated), so hits come out lower than in reality. This is the only mode for
  image traces.
- **`--mode synthetic`:** random ids with the recorded lengths, no tokenizer.
- **Order and passes:** N attempts in flight; a finished one is replaced by the next. The order is a seeded shuffle
  (`--order heavy-first` for late-run load). Each pass over the traces gets its own `cache_salt`, and each run a fresh
  salt (`--cache-salt auto`), so nothing cached earlier on the same server is reused.
- **Measured per window:**
  - client side: generated / prompt tok/s, TTFT, turn latency and inter-token latency percentiles, attempts / requests
    in flight;
  - server side, from `/metrics`: gen / prompt tok/s, prefix-cache hit rate, uncached prompt tok/s, external (CPU
    offload) hits, KV usage mean / max, running / waiting, preemptions;
  - with `--server-log`: per-rank `GPU memory: ... (peak P) ...` and `Capped the PyTorch allocator at X` lines, giving
    headroom = min over ranks of cap - peak, max peak, min free, allocator retries, OOMs. The KV pool size comes from
    the `GPU KV cache size` line.
- **Outputs in `--out`:** `windows.jsonl` (written as it runs), `requests.jsonl` (one line per turn: served vs
  recorded prompt and cached tokens), `summary.json` (steady = after `--warmup-s` and before the tail where fewer than N
  attempts remain; all), `report.md`, `tools.json`.

## Validation (offline, 2026-10-30)
- **Fidelity, tokens mode:**
  - Renders: the real V4.1 encoder + tokenizer (`data/dsv4_prune/model_meta`; `tokenizer.json` sha c90dfa01, the
    Kaggle Model copy's file).
  - Full b32m_x2t set (480 attempts): the planned prompt lengths equal the recorded `prompt_tokens` on
    **8,703 of 8,703 turns**.
    - 5 of those turns are sent as full renders, as the harness did. One is after a max_tokens cut; four follow
      non-canonical output (`parse_fallbacks`), detected from the recorded lengths.
    - Before that detection, 4 attempts were off by 1-120 tokens.
  - Also checked: 40 attempts of `E062_manh_vllm_fork/per_layer/phase_h/runs/r1_rerun_bec71fb1`.
  - Tools: auto-detection picked `--answer-tool --multi-test`, the B32-M solver's setting.
  - `--local-render` and an emulated server `/tokenize` give byte-identical plans (40 attempts).
- **Prep cost:** the full set renders locally in 4.4 min (13.5 cores busy, 3.3 GB peak RSS). Through a
  single-process `/tokenize` it is ~4x slower, since tokenization dominates at ~64 ms per 30k-token render. The cache
  file is 4.4 MB.
  - The b32m_x2t cache is on Hugging Face: private dataset `livobereshka/arc-agi2-vllm-serving-lab`, file
    `replay/b32m_x2t.tokens.json.gz`. How to download it, plus its `tools.json` and `prepare.json`:
    `experiments/E067_vllm_lab/replay/`.
- **Run-side footprint:** about 250 MB RSS with the full set loaded.
- **Client overhead:** the whole set in synthetic mode at concurrency 64 against the fake server ran at 3.2k tok/s,
  with client and fake server together using ~0.35-0.4 CPU cores. Client and `/metrics` token counts agreed within
  0.1%.
- **Not yet run against a real vLLM server:** no pod was used. The first pod run should start with an A/A pair.

## Running it on a 4xL4 pod
1. **On the dev box (no GPU):** build the prep cache once per trace set. It is valid for every server running the
   same tokenizer and template.
   ```bash
   .venv/bin/python evals/serve_replay_bench.py --prepare-only \
       --traces experiments/E064_manh_notebook_b32_harness/l4e2e/runs/b32m_x2t/out/dsv41_traces.tar.gz \
       --local-render data/dsv4_prune/model_meta --prep-cache prep/b32m_x2t.tokens.json.gz --out prep/b32m_x2t
   ```
   - Check that `prep/b32m_x2t/prepare.json` shows `prompt_len_exact_frac` 1.0.
   - For b32m_x2t this step is done: download the cache from HF instead (`experiments/E067_vllm_lab/replay/README.md`).
     Its `tools.json` and `prepare.json` are in that folder.
   - Copy to the pod: `evals/serve_replay_bench.py`, the trace tarball, the cache and `prep/b32m_x2t/tools.json`. The
     bench needs only the standard library.
2. **On the pod:** start the server as the candidate does (e.g. B32-M's command from `server_cmd.json`, prefix caching
   on), logging to a file.
3. **Run:**
   ```bash
   python serve_replay_bench.py --base-url http://127.0.0.1:8000 --traces dsv41_traces.tar.gz \
       --prep-cache b32m_x2t.tokens.json.gz --tools tools.json --concurrency 64 --duration 1200 --warmup-s 300 \
       --seed 0 --server-log server.log --out runs/base
   ```
   - The run first checks one cached plan against the server's own `/tokenize` (`fidelity.server_render_check`:
     "mismatch" means the cache does not fit this tokenizer or template).
   - Concurrency: the b32m_x2t replica (`ARC_WORKERS=80`, KV gate 0.85) held 80 attempts in flight for most of the
     run, at KV 0.2-0.6. Use the concurrency the candidate's gate actually holds (`metrics.jsonl` `running`).

## Flag ablation (~20 min per arm)
- **Hold fixed:** the same traces, prep cache, `--seed`, `--concurrency`, `--duration 1200 --warmup-s 300` and
  `--order`. Change exactly one server flag between arms.
- **Restart the server for every arm.** The per-run salt already prevents cache reuse, but a restart also resets KV,
  the allocator and the counters.
- **Run an A/A pair first.** The stack is not run-to-run deterministic at the server level, so the A/A gap is the
  noise floor. Then run A, B, A (or ABBA) to see drift.
- **Compare from `summary.json` → `steady`:**
  - throughput: `gen_tok_s`, `uncached_tok_s`;
  - cache and KV: `prefix_hit`, `kv_usage_mean/max`, `preemptions`, `external_hit_tok_s`;
  - latency (`requests_steady`): `ttft_p90`, `latency_p90`;
  - memory: `mem_headroom_min_gib`, `mem_ooms`.
  ```bash
  for r in runs/*/; do python -c "import json,sys; s=json.load(open(sys.argv[1]+'summary.json')); t=s['steady']; \
  print(sys.argv[1], t['gen_tok_s'], t['uncached_tok_s'], t['prefix_hit'], t['kv_usage_max'], t['preemptions'], \
  s['requests_steady']['ttft_p90'], t['mem_headroom_min_gib'])" $r; done
  ```
- **Late-run pressure** (long contexts, KV-bound): add `--order heavy-first`.
- **Cheap stacks that can reach steady state:** `--duration 2400`, or `--attempts` with no duration for one full pass.

## Caveats
- **Forced output lengths vs natural EOS:**
  - Every turn generates exactly the recorded length, so the workload is fixed. A change that would make the model
    write more or less (pruning, quantization, effort, sampling) is not measured here; use accuracy runs for that.
  - At temperature 0 with `ignore_eos` the generated text is degenerate, often repetitive. This flatters speculative
    decoding (MTP acceptance) and may shift MoE expert load. `--temperature 1.0` matches the real sampling but the text
    is still garbage.
  - The tool parser and detokenizer see junk too.
- **Tool time:**
  - The traces do not record per-turn tool latency, and the default `--think-time 0` keeps every attempt's request in
    the server all the time. That is more pressure than a real run, where attempts sit in the sandbox or train check.
  - `--think-time S` adds S seconds per tool result between turns.
- **Admission:** concurrency is fixed. The solver's KV gate, deadlines and smallest-first ordering are not modelled,
  and all N attempts start at turn 1 at t = 0 (a prefill burst). Exclude it with `--warmup-s`.
  - In a 20-min run contexts are shorter than in hours 6-11 of a Kaggle run, so compare arms with each other, not with
    the leaderboard.
- **Generated-token cache hits:** in tokens mode, hits on the previous turn's output exist only if the server caches
  decode blocks (the fork's retention interval). That is real behaviour, and an ablation target.
- **Chat mode** under-counts prefix hits by roughly each turn's output. **Synthetic mode** has faithful lengths and
  prefix structure but random token content.
- **/metrics is server-wide:** run nothing else against the server during a measurement.
- **Memory headroom resolution:** the fork logs `GPU memory:` about once a minute per rank, so headroom is resolved to
  about a minute.
