"""Sidecar monitor for a qwen38_pipeline run: one sample per interval of the vLLM server, GPUs, CPUs and the
pipeline's progress, appended to <output-root>/monitor/timeline.jsonl with a one-line summary per sample.

Reads only files the pipeline writes and the server's /metrics; it never talks to the pipeline. Exits one
sample after --pid exits."""

import argparse
import glob
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

SAMPLE = re.compile(r"^(vllm:[a-zA-Z0-9_:]+?)(?:\{[^}]*\})?\s+([-+0-9.eE]+|NaN|[-+]Inf)$")
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc")


def scrape(port):
    """Every vllm: sample except histogram buckets, summed over its label sets."""
    try:
        text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
    except Exception:  # noqa: BLE001 - the server may still be loading or has exited
        return None
    out = {}
    for line in text.splitlines():
        m = SAMPLE.match(line.strip())
        if m and not m[1].endswith(("_bucket", "_created")):
            try:
                out[m[1]] = out.get(m[1], 0.0) + float(m[2])
            except ValueError:
                pass
    return out


def gpus():
    try:
        text = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return [[float(v) for v in line.split(",")] for line in text.strip().splitlines()]
    except Exception:  # noqa: BLE001
        return []


def cpu_times():
    fields = [int(v) for v in open("/proc/stat").readline().split()[1:]]
    return sum(fields), fields[3] + fields[4]  # total, idle + iowait


def sandboxes():
    count = 0
    for cmdline in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            count += b"qwen38_repl_runtime.py" in open(cmdline, "rb").read()
        except OSError:
            pass
    return count


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def progress(root):
    summary = load(root / "summary.json") or {}
    rows = summary.get("results", [])
    tasks = {}
    for r in rows:
        tasks.setdefault(r["task"], []).append(r)
    trajectories = []
    for path in glob.glob(str(root / "programs" / "*" / "traj_*.json")):
        t = load(path)
        if t:
            log = t.get("turn_log", [])
            trajectories.append(dict(task=Path(path).parent.name, k=t.get("trajectory"), turns=t.get("turns"),
                                     tokens=t.get("completion_tokens"), stop=t.get("stop"), seconds=t.get("seconds"),
                                     gen_s=sum(x.get("gen_s", 0) for x in log),
                                     tools_s=sum(x.get("tools_s", 0) for x in log),
                                     slot_wait_s=(t.get("session") or {}).get("slot_wait_s", 0),
                                     accepted=t.get("accepted") is not None))
    return dict(status=summary.get("status"), done=len(rows), active=len(summary.get("active", {})),
                planned=summary.get("planned_outputs"), valid=sum(bool(r.get("valid")) for r in rows),
                correct=sum(bool(r.get("correct")) for r in rows),
                correct_top1=sum(bool(r.get("correct_top1")) for r in rows),
                # Task-weighted, as the competition scores: each task's outputs share one point.
                score=sum(sum(bool(r.get("correct")) for r in v) / len(v) for v in tasks.values()),
                score_top1=sum(sum(bool(r.get("correct_top1")) for r in v) / len(v) for v in tasks.values()),
                tasks_done=len(tasks),
                tir_status={s: sum(r.get("tir_status") == s for r in rows) for s in
                            {r.get("tir_status") for r in rows}},
                running_trajectories=sum(t["stop"] is None for t in trajectories),
                trajectories=len(trajectories), turns=sum(t["turns"] or 0 for t in trajectories),
                completion_tokens=sum(t["tokens"] or 0 for t in trajectories),
                gen_s=sum(t["gen_s"] for t in trajectories), tools_s=sum(t["tools_s"] for t in trajectories),
                slot_wait_s=sum(t["slot_wait_s"] for t in trajectories),
                stops={s: sum(t["stop"] == s for t in trajectories) for s in {t["stop"] for t in trajectories}})


def rate(now, before, key, seconds):
    if not now or not before or key not in now or key not in before or now[key] < before[key]:
        return None
    return (now[key] - before[key]) / seconds


def ratio(now, before, num, den):
    if not now or not before or num not in now or den not in now:
        return None
    d = now[den] - before.get(den, 0)
    return (now[num] - before.get(num, 0)) / d if d > 0 else None


def fmt(v, digits=0):
    return "n/a" if v is None else f"{v:.{digits}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--port", type=int, default=18083)
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--pid", type=int, help="Stop one sample after this process exits.")
    args = p.parse_args()
    folder = args.output_root / "monitor"
    folder.mkdir(parents=True, exist_ok=True)
    started, before, before_t, cpu_before = time.time(), None, None, cpu_times()
    while True:
        alive = args.pid is None or os.path.exists(f"/proc/{args.pid}")
        time.sleep(args.interval)
        now_t, metrics, cpu_now = time.time(), scrape(args.port), cpu_times()
        dt = now_t - before_t if before_t else None
        busy = 1 - (cpu_now[1] - cpu_before[1]) / max(1, cpu_now[0] - cpu_before[0])
        gpu = gpus()
        prog = progress(args.output_root)
        row = dict(t=round(now_t, 1), elapsed_s=round(now_t - started), metrics=metrics, gpus=gpu,
                   cpu_busy=round(busy, 3), loadavg=os.getloadavg()[0], sandboxes=sandboxes(), progress=prog)
        if dt:
            row.update(gen_tps=rate(metrics, before, "vllm:generation_tokens_total", dt),
                       prompt_tps=rate(metrics, before, "vllm:prompt_tokens_total", dt),
                       prefix_hit=ratio(metrics, before, "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"),
                       accept_per_draft=ratio(metrics, before, "vllm:spec_decode_num_accepted_tokens_total",
                                              "vllm:spec_decode_num_drafts_total"),
                       preemptions=rate(metrics, before, "vllm:num_preemptions_total", dt / 60))
        with open(folder / "timeline.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        m = metrics or {}
        kv = m.get("vllm:kv_cache_usage_perc")
        busy_time = prog["gen_s"] + prog["tools_s"]
        line = (f"MON {time.strftime('%H:%M:%S')} +{(now_t - started) / 60:.0f}m | gen {fmt(row.get('gen_tps'))} t/s"
                f" prompt {fmt(row.get('prompt_tps'))} t/s | run {fmt(m.get('vllm:num_requests_running'))}"
                f" wait {fmt(m.get('vllm:num_requests_waiting'))} KV {fmt(kv and 100 * kv)}%"
                f" prefix-hit {fmt(row.get('prefix_hit') and 100 * row['prefix_hit'])}%"
                f" accept/draft {fmt(row.get('accept_per_draft'), 2)} preempt/min {fmt(row.get('preemptions'))}"
                f" | GPU {'/'.join(fmt(g[0]) for g in gpu)}% | CPU {100 * busy:.0f}% sandboxes {row['sandboxes']}"
                f" | outputs {prog['done']}/{prog['planned']} active {prog['active']} correct {prog['correct']}"
                f" (top1 {prog['correct_top1']}) score {prog['score']:.2f}/{prog['tasks_done']} done-tasks"
                f" | traj {prog['running_trajectories']} running, {prog['turns']} turns,"
                f" tools {fmt(100 * prog['tools_s'] / busy_time if busy_time else None)}% of turn time")
        print(line, flush=True)
        with open(folder / "monitor.log", "a") as f:
            f.write(line + "\n")
        before, before_t, cpu_before = metrics or before, now_t if metrics else before_t, cpu_now
        if not alive:
            break


if __name__ == "__main__":
    main()
