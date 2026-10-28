#!/usr/bin/env python3
"""E060: log vLLM serving health during a run. Polls the server's Prometheus /metrics every --every seconds and
writes one JSON line per poll: running / waiting requests, KV-cache usage, cumulative prompt / generation tokens,
prefix-cache queries / hits, preemptions, speculative-decoding drafts / accepted tokens, plus window rates since the
previous poll (generation tok/s, prompt tok/s, prefix hit rate) and host memory (MemAvailable, cgroup usage).

    python vllm_monitor.py --url http://127.0.0.1:8000 --out runs/monitor_<run>.jsonl [--every 15]
"""
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

_LINE = re.compile(r"^(vllm:[a-zA-Z_:]+)(\{[^}]*\})?\s+([-0-9.eE+naninf]+)$")
KEYS = {
    "vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
    "vllm:kv_cache_usage_perc": "kv_usage", "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:generation_tokens_total": "gen_tokens", "vllm:prefix_cache_queries_total": "pc_queries",
    "vllm:prefix_cache_hits_total": "pc_hits", "vllm:num_preemptions_total": "preemptions",
    "vllm:spec_decode_num_drafts_total": "sd_drafts", "vllm:spec_decode_num_draft_tokens_total": "sd_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens_total": "sd_accepted", "vllm:request_success_total": "req_success",
    "vllm:prompt_tokens_cached_total": "prompt_cached",
}


def scrape(url):
    with urllib.request.urlopen(url + "/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        name, val = m.group(1), float(m.group(3))
        key = KEYS.get(name)
        if key:
            out[key] = out.get(key, 0.0) + val  # sum over label sets (engines, finish reasons)
    return out


def host_mem():
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemAvailable", "Cached", "Shmem"):
                info[k] = round(int(v.split()[0]) / 2**20, 1)  # GiB
        info["cgroup_GiB"] = round(int(Path("/sys/fs/cgroup/memory.current").read_text()) / 2**30, 1)
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=float, default=15)
    args = ap.parse_args()
    prev, prev_t = None, None
    with open(args.out, "a") as f:
        while True:
            t = time.time()
            try:
                cur = scrape(args.url)
            except Exception as exc:
                f.write(json.dumps({"t": round(t, 1), "error": f"{type(exc).__name__}"}) + "\n")
                f.flush()
                time.sleep(args.every)
                continue
            rec = {"t": round(t, 1), **{k: round(v, 4) for k, v in cur.items()}, **host_mem()}
            if prev is not None:
                dt = t - prev_t
                d = {k: cur.get(k, 0) - prev.get(k, 0) for k in cur}
                rec["gen_tps"] = round(d.get("gen_tokens", 0) / dt, 1)
                rec["prompt_tps"] = round(d.get("prompt_tokens", 0) / dt, 1)
                if d.get("pc_queries", 0) > 0:
                    rec["pc_hit_window"] = round(d.get("pc_hits", 0) / d["pc_queries"], 4)
                if d.get("sd_drafts", 0) > 0:
                    rec["sd_accept_len_window"] = round(1 + d.get("sd_accepted", 0) / d["sd_drafts"], 3)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            prev, prev_t = cur, t
            time.sleep(max(0.0, args.every - (time.time() - t)))


if __name__ == "__main__":
    main()
