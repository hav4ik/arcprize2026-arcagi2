#!/usr/bin/env python3
"""Poll vLLM /metrics and host memory every --interval s into JSONL (stdlib only).

    python gpu/metrics_logger.py --url http://127.0.0.1:8100 --out offload_metrics.jsonl

Logged per sample: KV usage, running/waiting, preemptions, GPU prefix-cache
queries/hits, external (host) prefix-cache queries/hits, offload load/store
bytes and seconds, CPU cache usage (total/write/read), allocation failures,
lookup delay sums, prompt/generation tokens; plus /proc/meminfo Shmem,
Mlocked, MemAvailable and the cgroup's memory.current when readable.
Counters are cumulative; diff consecutive samples for rates.
"""

import argparse
import json
import time
import urllib.request

KEEP = (
    "vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting",
    "vllm:num_preemptions", "vllm:prefix_cache_queries", "vllm:prefix_cache_hits",
    "vllm:external_prefix_cache_queries", "vllm:external_prefix_cache_hits",
    "vllm:kv_offload_load_bytes", "vllm:kv_offload_load_time", "vllm:kv_offload_store_bytes",
    "vllm:kv_offload_store_time", "vllm:kv_offload_cpu_cache_usage_perc",
    "vllm:kv_offload_cpu_cache_write_usage_perc", "vllm:kv_offload_cpu_cache_read_usage_perc",
    "vllm:kv_offload_allocation_failure", "vllm:kv_offload_lookup_sync_delay_seconds_sum",
    "vllm:kv_offload_lookup_async_delay_seconds_sum", "vllm:prompt_tokens",
    "vllm:generation_tokens",
)


def scrape(url):
    with urllib.request.urlopen(f"{url}/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        base = name.removesuffix("_total")
        if base in KEEP or name in KEEP:
            key = base if base in KEEP else name
            out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def host_mem():
    out = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            if k in ("MemAvailable", "Shmem", "Mlocked", "Unevictable"):
                out[k + "_GiB"] = round(int(v.split()[0]) / 2**20, 2)
    except OSError:
        pass
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            out["cgroup_GiB"] = round(int(open(path).read()) / 2**30, 2)
            break
        except OSError:
            continue
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=float, default=5.0)
    a = p.parse_args()
    with open(a.out, "a") as f:
        while True:
            row = {"t": round(time.time(), 1)}
            try:
                row.update(scrape(a.url))
            except Exception as e:  # server restarting: keep logging host memory
                row["error"] = str(e)[:120]
            row.update(host_mem())
            f.write(json.dumps(row) + "\n")
            f.flush()
            time.sleep(a.interval)


if __name__ == "__main__":
    main()
