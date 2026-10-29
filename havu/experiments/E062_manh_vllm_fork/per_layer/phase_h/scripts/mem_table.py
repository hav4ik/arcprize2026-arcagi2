"""Tabulate mem_probe.py output (GiB). Usage: mem_table.py <probe.jsonl>"""
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1])]
G = 1024 * 1024
for r in rs:
    if r["kind"] == "vmstat":
        print(f"vmstat {r['label']}: compact_stall {r['compact_stall']:,} fail {r['compact_fail']:,} success {r['compact_success']:,}")
    if r["kind"] == "log":
        for k in ("engram", "model_loading", "madv_collapse", "thp_config"):
            print(f"{k}: {len(r[k])} lines"); [print("   ", l[:230]) for l in r[k][:12]]
t0 = next((r["t"] for r in rs if r["kind"] == "sample"), 0)
print("| min | MemAvail | Anon | AnonHuge | Mlocked | Unevict | Cached | Shmem | PageTables | tree RSS |")
for r in rs:
    if r["kind"] != "sample":
        continue
    m = r["meminfo"]
    print(f"| {(r['t'] - t0) / 60:.0f} | " + " | ".join(f"{m.get(k, 0) / G:.1f}" for k in
          ("MemAvailable", "AnonPages", "AnonHugePages", "Mlocked", "Unevictable", "Cached", "Shmem")) +
          f" | {m.get('PageTables', 0) / 1024:.0f} MiB | {r['tree_rss_kib'] / G:.1f} |")
last = [r for r in rs if r["kind"] == "sample"]
for r in (last[:1] + last[-1:]) if last else []:
    print(f"processes at +{(r['t'] - t0) / 60:.0f} min: | pid | name | VmRSS | RssAnon | RssShmem | VmPTE MiB | AnonHuge |")
    for p in r["procs"]:
        print(f"| {p['pid']} | {p['name']} | {p.get('VmRSS', 0) / G:.1f} | {p.get('RssAnon', 0) / G:.1f} | "
              f"{p.get('RssShmem', 0) / G:.1f} | {p.get('VmPTE', 0) / 1024:.0f} | {p.get('AnonHugePages', 0) / G:.1f} |")
