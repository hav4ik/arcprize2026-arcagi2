"""Markdown table over E067 memory-lab runs: table.py LOGDIR name1 name2 ..."""
import glob
import json
import re
import sys

d = sys.argv[1]
print("| case | util | mbt | cap | compile | peak_act | KV GiB | KV tok | ready a/r/free | nvsmi ready MiB"
      " | run peak a | max r | max isp | retries | OOM | ok req |")
print("|" + "---|" * 16)
for n in sys.argv[2:]:
    f = [x for x in glob.glob(f"{d}/{n}.*.jsonl") if not x.endswith(".req.jsonl")][0]
    rs = [json.loads(line) for line in open(f)]
    ev = {r["ev"]: r for r in rs if r["ev"] not in ("step", "frag")}
    st = [r for r in rs if r["ev"] == "step" and "a" in r]
    busy = [r for r in st if r.get("ntok", 0) > 0]
    log = open(f"{d}/{n}.server.log").read()
    util = re.search(r"gpu_memory_utilization=([0-9.]+)|Desired GPU memory utilization is \(([0-9.]+)", log)
    util = next(g for g in util.groups() if g) if util else "?"
    mbt = re.search(r"_b(\d+)", n)
    kvt = re.search(r"GPU KV cache size: ([0-9,]+) tokens", log)
    comp = "warm" if "Directly load the compiled graph" in log else "cold"
    p, rd = ev["profiled"], ev["ready"]
    cap = f"{ev['cap']['cap']:.2f}" if "cap" in ev else "-"
    if "expandable_segments:True" in ev["patched"].get("alloc_conf", ""):
        cap += " +exp(launch)"
    elif "expandable_on" in ev:
        cap += " +exp(after)"
    oom = ev.get("oom")
    oomtxt = "-"
    if oom:
        m = re.search(r"Tried to allocate ([0-9.]+ [MG]iB)", oom["msg"])
        oomtxt = f"step {oom['s']} ntok {oom.get('ntok')} ({m.group(1) if m else '?'})"
    try:
        ok = sum(1 for line in open(f"{d}/{n}.req.jsonl") if '"status": 200' in line)
    except FileNotFoundError:
        ok = "-"
    try:
        nvr = open(f"{d}/{n}.nvsmi_ready.txt").read().split(",")[1].strip().split()[0]
    except Exception:
        nvr = "?"
    pk = f"{max(r['ap'] for r in busy):.2f}" if busy else "-"
    allst = st or [rd]
    print(f"| {n} | {util} | {mbt.group(1) if mbt else '?'} | {cap} | {comp} | {p['peak_act']:.3f} | {p['kv_budget']:.2f}"
          f" | {kvt.group(1) if kvt else '?'} | {rd['a']:.2f}/{rd['r']:.2f}/{rd['free']:.2f} | {nvr} | {pk}"
          f" | {max(r['r'] for r in allst + ([oom] if oom else [])):.2f} | {max(r['isp'] for r in allst):.2f}"
          f" | {(oom or allst[-1])['ret']} | {oomtxt} | {ok} |")
