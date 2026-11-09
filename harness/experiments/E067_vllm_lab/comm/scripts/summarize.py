"""Markdown tables from results/*.json (ar_*, har_proto, e2e_*)."""
import glob
import json
import os

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def kb(n):
    return f"{n >> 10} KiB" if n < (1 << 20) else f"{n >> 20} MiB"


def ar_tables():
    runs = {}
    for f in sorted(glob.glob(os.path.join(R, "ar_*.json"))):
        d = json.load(open(f))
        runs[d["label"]] = {r["bytes"]: r for r in d["rows"]}
    order = [k for k in ["default", "p2p_disable", "proto_ll", "proto_ll128", "proto_simple", "shm_memcpy",
                         "nch8", "nch8_ll", "shm_disable"] if k in runs]
    sizes = sorted(next(iter(runs.values())).keys())
    for key, title in [("graph_us", "graph-captured, us per call"), ("eager_us", "eager back-to-back, us per call")]:
        print(f"\nNCCL all-reduce, 2 ranks, {title}\n")
        print("| size | " + " | ".join(order) + " |")
        print("|---|" + "---|" * len(order))
        for s in sizes:
            print(f"| {kb(s)} | " + " | ".join(f"{runs[o][s][key]:.1f}" if s in runs[o] else "-" for o in order) + " |")
    print("\nbus bandwidth GB/s (graph)\n")
    print("| size | " + " | ".join(order) + " |")
    print("|---|" + "---|" * len(order))
    for s in sizes:
        if s >= 256 << 10:
            print(f"| {kb(s)} | " + " | ".join(f"{runs[o][s]['graph_busbw_GBps']:.2f}" if s in runs[o] else "-"
                                             for o in order) + " |")


def har_table(path):
    d = json.load(open(path))
    rows = d["rows"]
    keys = [k for k in rows[0] if k.endswith("_us")]
    print(f"\n{os.path.basename(path)}: us per call (slowest rank); args {d['args']}\n")
    print("| size | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for r in rows:
        print(f"| {kb(r['bytes'])} | " + " | ".join(f"{r[k]:.1f}" for k in keys) + " |")
    mm = {k: max(r[k] for r in rows) for k in rows[0] if k.endswith("_mismatch")}
    print("\nmax mismatching bf16 values vs NCCL:", mm)


def e2e_table():
    fs = sorted(glob.glob(os.path.join(R, "e2e_*.json")))
    if not fs:
        return
    print("\nvLLM decode (gen tok/s; step ms)\n")
    print("| config | load s | " + " | ".join(f"c={c}" for c in (1, 8, 32)) + " | first ids (c=32 req 0) |")
    print("|---|---|---|---|---|---|")
    for f in fs:
        d = json.load(open(f))
        rr = {r["conc"]: r for r in d["rows"]}
        cells = [f"{rr[c]['gen_tok_s']:.0f} ({rr[c]['step_ms']:.2f})" if c in rr else "-" for c in (1, 8, 32)]
        ids = rr.get(32, {}).get("first_ids", [])[:8]
        print(f"| {d['label']} | {d['load_s']} | " + " | ".join(cells) + f" | {ids} |")


if __name__ == "__main__":
    if glob.glob(os.path.join(R, "ar_*.json")):
        ar_tables()
    for p in sorted(glob.glob(os.path.join(R, "har_*.json"))):
        har_table(p)
    e2e_table()
