# E019 gateway: one OpenAI-compatible endpoint over the Vast 1×RTX PRO 6000 SGLang boxes

Set up 2026-10-12. The E019 Qwen harness talks to a local **SGLang Model Gateway** (`smg`, shipped in `sglang-router` 0.3.2), which routes to several single-GPU Vast boxes. Every conversation is pinned to one worker by a routing key, so the worker's radix cache holds its prefix on every turn.

## Clients

- URL: `http://127.0.0.1:30100/v1` (`ENDPOINTS.yaml` entry `local-sglang-router-3x6000`). It is bound to localhost only.
- Model id: `nvidia/Qwen3.8-27B-NVFP4` (max_model_len 262,144).
- Auth: none. The gateway ignores the client's `Authorization` header (checked). Pass any dummy key, e.g. `api_key="none"`, which is the entry's token.
- **Stickiness header (required):** send the same `x-smg-routing-key: <conversation or attempt id>` on every turn of a conversation, for example:

  ```python
  await client.completions.create(..., extra_headers={"x-smg-routing-key": f"{task}_{pair}_{attempt}"})
  ```

  - A request without the header is routed to a random worker, so its cache hits are lost.
  - `x-smg-target-worker: http://127.0.0.1:3010N` forces a specific worker; use it for debugging only.
- Endpoints: `/v1/completions` (the harness's raw-template path), `/v1/chat/completions` and `/generate` all route the same way. `/generate` returns `meta_info.cached_tokens`; `/v1/*` reports `prompt_tokens_details: null` because `--enable-cache-report` is off on the workers.

## Gateway

- **Version:** `sglang-router==0.3.2` (released 2026-01-15) in `.venv`. It is the last release under that name; the gateway inside reports `sgl-model-gateway 0.3.2`. The successor package `smg` 1.11.0 was tested in a throwaway venv and has the same auth limitation (below), so it was not adopted.
- **Command** (`run_router.sh`, `router_cmd`):

  ```bash
  .venv/bin/smg launch --host 127.0.0.1 --port 30100 \
    --policy manual --assignment-mode min_load --max-idle-secs 14400 \
    --request-timeout-secs 3600 --retry-max-retries 3 \
    --cb-failure-threshold 10 --cb-timeout-duration-secs 60 \
    --health-check-interval-secs 30 --health-failure-threshold 3 --health-check-timeout-secs 10 \
    --prometheus-host 127.0.0.1 --prometheus-port 29100 --log-level info
  ```

- **Policy:** `manual` + `min_load`.
  - The first request with a new routing key goes to the worker with the fewest in-flight requests; every later request with that key goes to the same worker.
  - Keys idle for more than 4 h are forgotten.
  - This replaced `cache_aware`, on the user's decision that per-conversation KV stickiness is the priority.
- **Files** (all in this directory):
  - `router.log`: gateway log.
  - `supervisor.log`: restarts and re-registrations.
  - `router.pid`, `supervisor.pid`.
  - `workers.json`, `tunnels.conf`: the worker registry. It holds IPs and ports only, never tokens.
  - `tunnel_<name>.log`, `tunnel_<name>.pid`.

## Workers

| ENDPOINTS name | Vast id | GPU / location | gateway URL (tunnel) | ssh | state 15:35 UTC |
|---|---|---|---|---|---|
| `vastai-1x6000-au-qwen27b` | 55572174 | RTX PRO 6000 Max-Q, Australia | `http://127.0.0.1:30101` | 162.43.170.108:28326 | registered, healthy |
| `vastai-1x6000-jp-qwen27b` | 55574668 | RTX PRO 6000 Max-Q, Japan | `http://127.0.0.1:30102` | 43.235.149.172:41266 | registered, healthy |
| (CN, to add) | 55572178 | RTX PRO 6000 WS, Jiangsu | – | – | image still pulling |
| (US, to add) | 55572201 | RTX PRO 6000 WS, US | – | – | image still pulling |

- **Serving setup:** every box runs the Vast `vastai/sglang:v0.5.20-cuda-13.0` template, which starts SGLang from the env vars `SGLANG_MODEL` / `SGLANG_ARGS` under supervisor (`/opt/supervisor-scripts/sglang.sh`, log `/var/log/portal/sglang.log`):

  ```
  sglang serve --model-path nvidia/Qwen3.8-27B-NVFP4 --attention-backend flashinfer --kv-cache-dtype fp8_e4m3 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --download-dir /workspace/models \
    --host 127.0.0.1 --port 18000 --tensor-parallel-size 1
  ```

- **Radix cache:** on (`disable_radix_cache: false`).
- **Metrics:** `enable_metrics: false`, so the workers have no `/metrics`. See "Cache hit rates".
- **Caddy proxy:** Caddy (container port 8000, auth by bearer token) proxies to 127.0.0.1:18000.
- **First boot:** the box downloads the ~20 GB of weights to `/workspace/models`; AU took ~7 min from "running" to serving.

## Auth: why SSH tunnels

- **Caddy requires the token everywhere:** Vast's Caddy edge requires the instance token on every path, including `/health`.
- **Per-worker keys don't cover the probes:** the gateway does accept a per-worker key via `POST /workers {"url", "api_key"}`. A fake-worker test showed it sends `Authorization: Bearer <key>` on completions, `/server_info` and `/model_info`, overriding the client's header. But its `detect_connection_mode` step and its health checks call `/health` **without** the key.
  - Caddy answers 401, and the worker is never registered.
  - `disable_health_check: true` and `connection_mode: "http"` in the POST body do not skip detection.
  - The same happens with `smg` 1.11.0.
- **What we do instead:** each worker is reached through an SSH local forward `127.0.0.1:3010N -> box:127.0.0.1:18000`. This bypasses Caddy, so health checks work and the gateway needs no tokens.
  - The tunnels use our `~/.ssh/id_ed25519` key, which is registered on the Vast account, over the box's direct `22/tcp` mapping.
  - `tunnels.sh` runs one restart loop per worker: `ssh -N`, ServerAlive 15 s × 4, `ExitOnForwardFailure`, retry after 5 s. autossh is not installed.
- **The `--direct` mode** of `register_worker.py` (Caddy URL + `api_key`) is kept for servers whose `/health` is open.
- **Token gotcha:** the Vast API's `extra_env.OPEN_BUTTON_TOKEN` is a placeholder (`"1"`). The real Caddy token is the instance's `jupyter_token` field; it equals `OPEN_BUTTON_TOKEN` inside the container, which was checked by hash. ENDPOINTS entries must use `jupyter_token`.

## Operating it

```bash
R=experiments/E019_qwen_program_answers/router
# register a new box. It must be in ENDPOINTS.yaml (name, token = instance jupyter_token, addr = public_ip:<8000/tcp host port>).
# This health-checks the box directly (/v1/models + a 4-token completion), looks up its ssh port with vastai,
# starts its tunnel loop and POSTs the tunnel URL to /workers. Additive: no gateway or tunnel restart.
.venv/bin/python $R/register_worker.py vastai-1x6000-us-qwen27b
.venv/bin/python $R/register_worker.py --list                      # gateway view + registry
.venv/bin/python $R/register_worker.py --remove vastai-1x6000-cn-qwen27b   # DELETE /workers/<id> + stop its tunnel
.venv/bin/python $R/register_worker.py --restore                   # re-register everything in workers.json
$R/tunnels.sh status            # per-tunnel loop state and /health through it
$R/tunnels.sh start             # (re)start any tunnel loop that is not running; running ones are untouched
$R/run_router.sh status         # gateway + supervisor pids, worker health/load
$R/run_router.sh start | stop   # the supervisor restarts the gateway if it dies, then runs --restore
```

- **Routing state is lost on a gateway restart:** workers are re-added automatically, but key-to-worker assignments are not. In-flight conversations then get new assignments, and their next turn re-prefills once.
- **Never restart the gateway during an eval.**
- **Removing a worker is safe:** a key whose worker was removed is reassigned on its next request (`smg_manual_policy_branch_total{branch="occupied_miss"}`), with no error. That turn misses the cache.

## Verification (2026-10-12, 15:20–15:35 UTC)

- **Health:**

  | | AU | JP |
  |---|---|---|
  | `/v1/models` | `nvidia/Qwen3.8-27B-NVFP4`, 262k | same |
  | KV pool (`max_total_num_tokens`, fp8) | 939,671 | 944,285 |
  | chunked prefill / page size | 8192 / 1 | 8192 / 1 |
  | chat + `python` tool call (low effort) | `finish=tool_calls`, args `{"code": "print(1234*5678)"}`, 1.4 s | same, 1.2 s |
  | single-stream decode | 43 tok/s | 54 tok/s |
  | TCP connect / `/v1/models` RTT through tunnel | 0.21 s / 0.39 s | 0.11 s / 0.21 s |

  - `/health` itself takes 1.2–1.5 s on SGLang 0.5.20; it runs a generation.
  - Raw results: `health_au.json`, `health_jp.json`.
- **Stickiness:** `routing_probe.py sticky` runs keyed conversations of 3–4 turns, one request at a time and interleaved across conversations. Each turn re-sends the whole transcript (ARC-task prefixes of 3.2–14k tokens).
  - AU + JP, 4 conversations × 3 turns: every conversation stayed on one worker (1 on AU, 3 on JP).
  - Cached fraction on turns 2–3 was 0.98–0.995, with one exception noted below.
  - Earlier runs on AU alone (3 × 4 turns) gave 0.93–0.99 on turns 2–4.
- **Spread:** 20 new keys sent at once went 10 / 10 across two worker URLs.
  - This was tested with a temporary second tunnel to the AU box (`127.0.0.1:30109`), which has since been removed.
  - Its stale series still appear in the gateway's Prometheus output.
- **Reassignment:** after `DELETE /workers/<id>` of the worker that held a key, the key's next turn was served by the remaining worker (HTTP 200, `occupied_miss`).
- **Throughput:** the aggregate decode probe (e.g. 3×32 concurrency) was **not run**, because the E019 eval started on the gateway first.
  - Reference: E008 measured ~1.4–1.9k tok/s for this model on 4×RTX 6000 at 48 concurrency.
  - To measure later on an idle pool: `.venv/bin/python evals/bench_throughput.py --endpoint local-sglang-router-3x6000 --model nvidia/Qwen3.8-27B-NVFP4 64:150`. Note that the bench sends no routing key, so it spreads randomly, which is fine for decode throughput.

### Caveats

- **`min_load` ties:** `min_load` counts in-flight requests. When requests arrive one at a time (all workers at 0), new keys tie and pile onto one worker: 3 of 4 in one sequential test.
  - Under real concurrent load they balance.
  - If key counts drift apart (conversations waiting on tool calls hold no in-flight request), `--assignment-mode min_group`, the fewest keys per worker, is the alternative. It needs a gateway restart.
- **Cache misses on the same worker:** one conversation (`probe-aujp-2`) had a turn-2 cached fraction of only 0.02 even though it stayed on the same worker. Qwen3.8-27B is a hybrid-attention model, and SGLang can reuse its linear-attention state only at stored checkpoints. If the re-sent text re-tokenizes differently from the generated tokens at a turn boundary, the match can fall back to an early checkpoint. This is a server-side cache effect, not a routing one. Watch the hit rate during the eval.
- **Max-Q GPUs:** AU and JP are Max-Q (300 W) cards, so expect lower decode throughput than on the WS boxes.

## Cache hit rates per worker

- **Workers:** they run without `--enable-metrics`, so there is no `/metrics`. Use either of these:
  - the SGLang log on the box, which has one line per prefill batch with `#new-token` and `#cached-token`:

    ```bash
    ssh -p <ssh_port> root@<ip> "grep 'Prefill batch' /var/log/portal/sglang.log | tail -200" | \
      python3 -c "import re,sys; n=c=0
    for l in sys.stdin:
        m=re.search(r'#new-token: (\d+), #cached-token: (\d+)', l)
        if m: n+=int(m[1]); c+=int(m[2])
    print(f'hit rate {c/max(1,n+c):.3f} over {n+c} prompt tokens')"
    ```

    The decode lines in the same log give `token usage`, `#running-req` and `#queue-req`.
  - a `/generate` probe with `max_new_tokens=1` on a transcript prefix, which returns `meta_info.cached_tokens`. `routing_probe.py` does this.
  - Enabling `--enable-metrics` (append it to `SGLANG_ARGS` on the box and `supervisorctl restart sglang`) would make `evals/sglang_cache_monitor.py --url http://127.0.0.1:3010N --out …` work. It needs a server restart, so do it only between runs.
- **Gateway:** `curl -s 127.0.0.1:29100/metrics | grep -E 'smg_worker_(requests_active|cb_outcomes_total)|smg_manual_policy_branch'` gives:
  - per-worker in-flight and completed requests;
  - `occupied_hit` (sticky), `vacant` (new key) and `occupied_miss` (reassigned) counts.

## Scripts

- `register_worker.py`: register / remove / list / restore workers (tunnel or `--direct`).
- `tunnels.sh`: SSH tunnel restart loops from `tunnels.conf`.
- `run_router.sh`: gateway supervisor.
- `health_check.py <ENDPOINTS names…>`: direct per-worker health (`/health` RTT, `/v1/models`, server info, tool-call chat).
- `routing_probe.py sticky|spread|turn`: routing-key stickiness, spread and reassignment probes, using gateway Prometheus counters and `/generate` cached_tokens.
- `evals/sglang_cache_monitor.py`: `--endpoint` is now repeatable, so one monitor can poll several ENDPOINTS entries.

## Harness-side worker slots (E020, 2026-10-12)

**Problem:** `min_load` counts in-flight HTTP requests. A conversation waiting on a tool call has none, so a box whose attempts are mid-tool looks idle and gets new keys while its KV still holds their prefixes. AU went 5 → 22 running, then KV 0.99, a retraction, and hit 0.46. Under that load AU's `/health`, which runs a generation, also timed out (10 s), and the gateway marked it unhealthy (keys reassigned: `occupied_miss`).

**`x-smg-target-worker` does not work here:** with `--policy manual`, sgl-model-gateway 0.3.2 ignores it. Six requests targeted at `http://127.0.0.1:30103` all went to other workers (per-worker `smg_worker_cb_outcomes_total`).

**Fix (`evals/arc_harness/slots.py`, `run_qwen_tools.py --worker-slots ... [--slot-state F]`):**
- Each attempt holds one slot on one worker for its whole lifetime: the worker with the most free slots, ties to the least used.
- Every turn goes **directly** to that worker's tunnel URL (`http://127.0.0.1:3010N/v1`), not through the gateway. The routing-key header is still sent, and is harmless there.
- Every slot-aware process shares one state file: `router/slots.json`, guarded by `fcntl.flock` on `slots.json.lock`.
  - Format: `{"slots": {url: cap}, "leases": {id: {url, pid, run, t}}}`.
  - Leases of dead pids are dropped on every acquire.
  - Caps are re-read on every acquire, so they can be changed live with no restart:

    ```bash
    cd evals && ../.venv/bin/python -m arc_harness.slots --state ../experiments/E019_qwen_program_answers/router/slots.json --set http://127.0.0.1:30101=12
    cd evals && ../.venv/bin/python -m arc_harness.slots --state ../experiments/E019_qwen_program_answers/router/slots.json --show
    ```

- `--worker-slots auto:N` gives every worker listed by the gateway's `GET /workers` N slots. `url=n,url=n` sets them explicitly.
- **Sizing:** slots per box ≈ KV tokens (~939k) ÷ the p90 context per attempt. For xhigh-96k program attempts that is about 12–14.
- **Caveat:** processes that still go through the gateway are not counted. Lower the caps while they run.
- **Kaggle:** the same pool works for a two-server layout: two local vLLM URLs, `url=n,url=n`.
- **Records:** each attempt records `worker`. Gateway retries and circuit breaking do not apply on the direct path; the harness client retries transient errors itself, 8 times with backoff.
