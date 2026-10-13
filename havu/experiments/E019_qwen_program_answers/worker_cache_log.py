"""Per-worker cache health for the E019 gateway runs, from each box's SGLang log over ssh (the workers have no
/metrics). Every --interval seconds, for each worker in router/workers.json: prefill #new-token / #cached-token in
the new log lines (window hit rate), the last decode line (running, queue, token usage, gen throughput) and the
number of retraction lines. One JSON line per worker per poll.

    .venv/bin/python experiments/E019_qwen_program_answers/worker_cache_log.py --out evals/results/cache_gw_dev56.jsonl
    .venv/bin/python experiments/E019_qwen_program_answers/worker_cache_log.py --summary evals/results/cache_gw_dev56.jsonl
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = "/var/log/portal/sglang.log"
PRE = re.compile(r"Prefill batch.*?#new-token: (\d+), #cached-token: (\d+)")
DEC = re.compile(r"Decode batch, #running-req: (\d+), #(?:full )?token: (\d+), (?:full )?token usage: ([\d.]+).*?"
                 r"gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)")


def ssh(host, port, cmd, timeout=60):
    return subprocess.run(["ssh", "-p", str(port), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                           "-o", "ConnectTimeout=15", f"root@{host}", cmd], capture_output=True, text=True,
                          timeout=timeout).stdout


def poll(name, w, pos, since=None):
    out = ssh(w["ssh_host"], w["ssh_port"], f"echo LINES=$(wc -l < {LOG})")  # the box prints a banner first
    n_lines = int(re.findall(r"LINES=(\d+)", out)[-1])
    start = pos[name] if name in pos else max(1, n_lines - 20000)  # first poll: look back (filtered by --since)
    text = ssh(w["ssh_host"], w["ssh_port"], f"sed -n '{start},{n_lines}p' {LOG} | grep -E 'Prefill batch|Decode batch|etract'")
    pos[name] = n_lines + 1
    new = cached = 0
    last = None
    retract = 0
    decs = []
    for line in text.splitlines():
        if since and line[1:20] < since:
            continue
        m = PRE.search(line)
        if m:
            new += int(m[1])
            cached += int(m[2])
        m = DEC.search(line)
        if m:
            last = m
            decs.append((int(m[1]), float(m[3]), float(m[4]), int(m[5])))
        if "etract" in line and "server_args" not in line:
            retract += 1
    rec = {"t": round(time.time()), "worker": name, "new_tokens": new, "cached_tokens": cached,
           "window_hit": round(cached / (new + cached), 4) if new + cached else None, "retract_lines": retract}
    if last:
        rec.update(running=int(last[1]), tokens=int(last[2]), token_usage=float(last[3]),
                   gen_tps=float(last[4]), queue=int(last[5]))
    if decs:  # window means over decode log lines (every 40 decode steps)
        rec.update(mean_running=round(sum(d[0] for d in decs) / len(decs), 1),
                   mean_gen_tps=round(sum(d[2] for d in decs) / len(decs), 1),
                   max_token_usage=max(d[1] for d in decs), max_queue=max(d[3] for d in decs),
                   per_stream_tps=round(sum(d[2] for d in decs) / max(1, sum(d[0] for d in decs)), 1))
    return rec


def summary(path):
    tot = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if "error" in r:
            continue
        t = tot.setdefault(r["worker"], {"new": 0, "cached": 0, "retract": 0, "polls": 0, "usage_max": 0, "hits": []})
        t["new"] += r["new_tokens"]
        t["cached"] += r["cached_tokens"]
        t["retract"] += r["retract_lines"]
        t["polls"] += 1
        t["usage_max"] = max(t["usage_max"], r.get("token_usage", 0))
        if r["window_hit"] is not None and r["new_tokens"] + r["cached_tokens"] > 50000:
            t["hits"].append(r["window_hit"])
    for w, t in tot.items():
        hits = sorted(t["hits"])
        print(f"{w}: hit {t['cached'] / max(1, t['new'] + t['cached']):.3f} over {(t['new'] + t['cached']) / 1e6:.1f}M "
              f"prompt tokens; window hit min {hits[0] if hits else None} median {hits[len(hits) // 2] if hits else None}; "
              f"max token usage {t['usage_max']:.2f}; retraction lines {t['retract']}; polls {t['polls']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--interval", type=float, default=300)
    ap.add_argument("--summary")
    ap.add_argument("--since", help="ignore log lines before this UTC time, e.g. '2026-10-12 15:38:00'")
    args = ap.parse_args()
    if args.summary:
        summary(args.summary)
        return
    pos = {}
    while True:
        workers = json.loads((HERE / "router/workers.json").read_text())
        for name, w in workers.items():
            if "ssh_host" not in w:
                continue
            try:
                rec = poll(name, w, pos, args.since)
            except Exception as exc:
                rec = {"t": round(time.time()), "worker": name, "error": type(exc).__name__}
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
