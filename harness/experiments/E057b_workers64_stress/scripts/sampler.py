#!/usr/bin/env python3
"""E057b: every --interval s, one JSONL row with the vLLM server's /metrics (every non-bucket series summed by name;
the TTFT and queue-time histogram buckets kept), the solver's status.json (running / queued / done), and host memory
(/proc/meminfo, the cgroup's memory.current). Stdlib only; read-only; exits at --until (epoch s).

    python3 sampler.py --run-dir /root/runs/NAME --out /root/runs/NAME/metrics10.jsonl --until EPOCH
"""
import argparse
import json
import math
import time
import urllib.request

HIST = ("vllm:time_to_first_token_seconds", "vllm:request_queue_time_seconds")


def scrape(url):
    with urllib.request.urlopen(f"{url}/metrics", timeout=8) as r:
        text = r.read().decode(errors="replace")
    out, buckets = {}, {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        try:
            head, val = line.rsplit(" ", 1)
            v = float(val)
        except ValueError:
            continue
        if not math.isfinite(v):
            continue
        name = head.split("{", 1)[0]
        if name.endswith("_bucket"):
            base = name[:-7]
            if base in HIST and 'le="' in head:
                le = head.split('le="', 1)[1].split('"', 1)[0]
                b = buckets.setdefault(base, {})
                b[le] = b.get(le, 0.0) + v
            continue
        out[name] = out.get(name, 0.0) + v
    if buckets:
        out["_buckets"] = buckets
    return out


def meminfo():
    out = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            if k in ("MemAvailable", "Shmem", "AnonPages", "Cached", "Mlocked", "Unevictable"):
                out[k + "_GiB"] = round(int(v.split()[0]) / 2 ** 20, 2)
    except OSError:
        pass
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            out["cgroup_GiB"] = round(int(open(path).read()) / 2 ** 30, 2)
            break
        except (OSError, ValueError):
            continue
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=float, default=10.0)
    p.add_argument("--until", type=float, required=True)
    a = p.parse_args()
    status = f"{a.run_dir}/out/dsv41_logs/status.json"
    with open(a.out, "a") as f:
        while time.time() < a.until:
            t = time.time()
            row = {"t": round(t, 1)}
            try:
                row["m"] = scrape(a.url)
            except Exception as e:  # server not up yet / restarting / stopped
                row["err"] = type(e).__name__
            try:
                st = json.load(open(status))
                row["st"] = {k: st.get(k) for k in ("elapsed_h", "phase", "running", "queued", "done", "errors",
                                                    "pairs_with_answer", "restarts", "server_ok")}
            except Exception:
                pass
            row["mem"] = meminfo()
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            f.flush()
            time.sleep(max(0.5, a.interval - (time.time() - t)))


if __name__ == "__main__":
    main()
