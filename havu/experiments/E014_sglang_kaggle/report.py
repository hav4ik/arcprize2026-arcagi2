"""Markdown tables from E014 replay results (kaggle/runs/<slug>/replay_bench.json).

    .venv/bin/python experiments/E014_sglang_kaggle/report.py kaggle/runs/e014-replay-vllm-v1 kaggle/runs/e014-replay-sglang-v1
"""
import json
import os
import sys

COLS = [("agents", "agents"), ("decode_tok_s", "decode tok/s"), ("client_new_prefill_tok_s", "new prefill tok/s"),
        ("client_hit_rate", "hit (client)"), ("server_hit_rate", "hit (server)"), ("cont_hit_vs_ideal", "cont. hit/ideal"),
        ("cont_miss_frac", "cont. miss frac"), ("running_mean", "running mean"), ("running_max", "running max"),
        ("waiting_max", "waiting max"), ("kv_usage_mean", "KV use mean"), ("kv_usage_max", "KV use max"),
        ("live_ctx_mean", "live ctx (client)"), ("preemptions", "preempt"), ("retracted_queue_max", "retract q max"),
        ("ttft_p50", "TTFT p50 s"), ("ttft_p90", "TTFT p90 s"), ("errors", "errors")]


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}" if v < 2 else f"{v:.0f}"
    return str(v)


for d in sys.argv[1:]:
    r = json.load(open(os.path.join(d, "replay_bench.json")))
    print(f"\n## {d}\nphases {r.get('phases')} fatal {r.get('fatal')}")
    for name, c in r["configs"].items():
        print(f"\n### {name}\nstarts: " + json.dumps([{k: v for k, v in s.items() if k != 'log_tail'}
                                                       for s in c.get("starts", [])]))
        print("capacity:", c.get("capacity"), "| skipped:", c.get("skipped"), "| errors:", c.get("errors"))
        lv = c.get("levels") or []
        if not lv:
            continue
        print("| " + " | ".join(h for _, h in COLS) + " |")
        print("|" + "---|" * len(COLS))
        for x in lv:
            print("| " + " | ".join(fmt(x.get(k)) for k, _ in COLS) + " |")
