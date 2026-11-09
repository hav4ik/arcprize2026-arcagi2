"""Newest vLLM GPU memory state over all server logs of the run: max allocator retries over ranks, total OOMs, and
min (cap - peak) GiB over ranks (cap from "Capped the PyTorch allocator", peak from the newest "GPU memory:" line of the
newest server log). Prints: <retries_max> <ooms_total> <min_cap_minus_peak or 9>"""
import glob, os, re
logs = sorted(glob.glob("/kaggle/working/dsv41_logs/**/server*.log", recursive=True), key=os.path.getmtime)
cap, last = {}, {}
for f in logs[-1:]:
    for line in open(f, errors="replace"):
        w = re.search(r"\((Worker\S*) pid=\d+\)", line); who = w[1] if w else "?"
        c = re.search(r"Capped the PyTorch allocator at ([\d.]+) GiB", line)
        if c: cap[who] = float(c[1])
        m = re.search(r"GPU memory: [\d.]+ GiB allocated \(peak ([\d.]+) GiB\).* (\d+) allocator retries, (\d+) OOMs", line)
        if m: last[who] = (float(m[1]), int(m[2]), int(m[3]))
r = max([v[1] for v in last.values()] or [0]); o = sum(v[2] for v in last.values())
d = min([cap[k] - v[0] for k, v in last.items() if k in cap] or [9])
print(r, o, round(d, 2))
