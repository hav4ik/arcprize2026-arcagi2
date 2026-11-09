# Serve replay bench: out

- Server: http://127.0.0.1:8067 model `m`, vLLM None, KV cache 195312 tokens (server log)
- Workload: 480 attempts, 8703 turns from 1 trace spec(s); mode `synthetic`, concurrency 8, seed 0, order shuffle, think time 2.0 s per tool result, temperature 0.0
- Run: 102.7 s, 56 attempts started, 2 completed, 74 turns, 54 failed attempts; warm-up 60.0 s
- Steady interval: t = 60-103 s (after warm-up, before the tail where fewer than 8 attempts remain)

| Metric | Steady | All |
|---|---|---|
| Generated tok/s (server) | 1057.9 | 1148.6 |
| Generated tok/s (client) | 743.3 | 1006.7 |
| Prompt tok/s (server) | 11057.2 | 12791.7 |
| Uncached prompt tok/s | 925.5 | 1338.7 |
| Prefix-cache hit rate | 0.9163 | 0.8953 |
| External (offload) hit tok/s | 0.0 | 0.0 |
| KV usage mean | 0.797 | 0.627 |
| KV usage max | 0.797 | 0.797 |
| Running / waiting (mean) | 7.0 / 0.0 | 6.3 / 0.0 |
| Preemptions | 0 | 0 |
| Turns / attempts done | 19 / 2 | 74 / 2 |
| TTFT p50 / p90 / p99 (s) | 0.07 / 0.17 / 0.36 | 0.07 / 0.47 / 1.90 |
| Turn latency p50 / p90 / p99 (s) | 3.36 / 15.80 / 24.84 | 2.84 / 14.01 / 30.38 |
| Inter-token latency p50 / p90 (ms) | 6.1 / 6.5 | 5.2 / 6.5 |

## Fidelity

- Tools: 
- Planned prompt lengths vs recorded: {"turns": 8703, "prompt_len_exact_frac": 1.0, "prompt_len_mean_abs_diff": 0.0, "prompt_len_mean_diff": 0.0, "segments": {"delta": 8221, "full": 0, "rebase": 2}}
- Served prompt tokens vs recorded: exact 1.0000, mean |diff| 0.0 tokens
- Prefix-cache hit: recorded in the traces 0.9023, served (usage) 0.9605, server metrics 0.8953
- Output length mismatches: 0; turns without returned token ids: 0; request errors 54, aborted at the end 0; connection retries 80

## Windows

| t (s) | gen tok/s | client gen | prompt tok/s | uncached tok/s | hit | KV | run/wait | preempt | TTFT p50/p90 (s) | turn p50 (s) | attempts | headroom GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 (warm-up) | 1193.3 | 1193.4 | 14923.8 | 2574.0 | 0.828 | 0.366 | 5/0 | 0 | 0.05/1.20 | 2.6 | 8 | - |
| 60 (warm-up) | 1194.7 | 1194.8 | 12394.0 | 516.6 | 0.958 | 0.717 | 7/0 | 0 | 0.07/0.10 | 3.9 | 8 | - |
| 90 | 1057.9 | 1057.7 | 11057.2 | 925.5 | 0.916 | 0.797 | 7/0 | 0 | 0.07/0.17 | 3.4 | 1 | - |
| 103 | - | 0.0 | - | - | - | - | -/- | - | -/- | - | 0 | - |
