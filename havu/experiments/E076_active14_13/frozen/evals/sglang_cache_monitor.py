"""Log SGLang cache health during a run (E019): poll each worker's Prometheus /metrics and /v1/loads.

    .venv/bin/python evals/sglang_cache_monitor.py --url http://localhost:30000 --out evals/results/cache_<run>.jsonl
    .venv/bin/python evals/sglang_cache_monitor.py --endpoint <ENDPOINTS.yaml name> ... (bearer token from the file)

One JSON line per worker per poll: running / queued requests, token and mamba usage, the server's cache_hit_rate
gauge, cumulative prompt tokens, retracted requests, and the window hit rate computed from the prefill log counters
when available. Tokens are never printed.
"""
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAUGES = ("num_running_reqs", "num_queue_reqs", "token_usage", "full_token_usage", "mamba_usage", "cache_hit_rate",
          "gen_throughput", "num_retracted_reqs", "prompt_tokens_total", "generation_tokens_total",
          "cached_tokens_total", "kv_used_tokens", "kv_evictable_tokens", "mamba_evictable_tokens", "max_total_num_tokens")
_LINE = re.compile(r"^sglang:([a-zA-Z_]+)(\{[^}]*\})?\s+([-0-9.eE+naninf]+)$")


def fetch(url, token, timeout=10):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def parse_metrics(text):
    out = {}
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if m and m.group(1) in GAUGES and 'tp_rank="' in (m.group(2) or "") and 'tp_rank="0"' not in m.group(2):
            continue  # one TP rank only (gauges are replicated per rank)
        if m and m.group(1) in GAUGES:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", action="append", default=[], help="worker base URL (repeatable)")
    ap.add_argument("--endpoint", action="append", default=[], help="ENDPOINTS.yaml entry (addr + token; repeatable)")
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    targets = [(u.rstrip("/"), None) for u in args.url]
    if args.endpoint:
        import yaml
        entries = yaml.safe_load((ROOT / "ENDPOINTS.yaml").read_text())
        for name in args.endpoint:
            e = next(e for e in entries if e.get("name") == name)
            targets.append((f"http://{e['addr']}".rstrip("/"), e.get("token")))
    prev = {}
    while True:
        for url, token in targets:
            rec = {"t": round(time.time()), "url": url}
            try:
                m = parse_metrics(fetch(url + "/metrics", token))
                rec.update(m)
                p = prev.get(url)
                if p and "cached_tokens_total" in m and m.get("prompt_tokens_total", 0) > p.get("prompt_tokens_total", 0):
                    rec["window_hit"] = round((m["cached_tokens_total"] - p["cached_tokens_total"])
                                              / (m["prompt_tokens_total"] - p["prompt_tokens_total"]), 4)
                prev[url] = m
            except Exception as exc:  # keep polling; never include the token in the record
                rec["error"] = type(exc).__name__
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
