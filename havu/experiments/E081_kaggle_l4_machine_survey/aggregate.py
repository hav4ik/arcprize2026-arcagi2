"""E081: aggregate runs/probe-*/probe.json into a per-run table and summary statistics (markdown on stdout)."""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "/kaggle/input/models/chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pytorch/default/1"
WHEELS = "/kaggle/input/datasets/nguyennguyen599/vllm-wheels"
DEBS = "/kaggle/input/datasets/nguyennguyen599/cu13-install"
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "runs/probe-*/probe.json")) + glob.glob(os.path.join(HERE, "runs/probeb-*/probe.json"))):
    r = json.load(open(f))
    idn, g = r.get("identity", {}), r.get("gpucheck", {})
    copy = g.get("copy", {})
    alone = {p["pair"]: p for p in g.get("pairs", []) if p["mode"] == "alone"}
    m, w, d = r.get("mounts", {}).get(MODEL, {}), r.get("mounts", {}).get(WHEELS, {}), r.get("mounts", {}).get(DEBS, {})
    rows.append(dict(
        run=os.path.basename(os.path.dirname(f)), host=idn.get("hostname"),
        machine=",".join(sorted(x["uuid"][-6:] for x in idn.get("gpus", []))),
        bus=",".join(x["pci.bus_id"][-7:] for x in idn.get("gpus", [])),
        power=",".join(sorted({x["power.limit"] for x in idn.get("gpus", [])})),
        h2d_min=min((v["h2d"] for v in copy.values()), default=None), d2h_min=min((v["d2h"] for v in copy.values()), default=None),
        slow_gpus=sorted(int(k) for k, v in copy.items() if v["h2d"] < 6 or v["d2h"] < 6),
        ag516=({k: v.get("ag516K") for k, v in sorted(alone.items())}),
        model_single=m.get("cold_single_MBps"), model_par8=m.get("cold_par8_MBps"),
        wheels_par8=w.get("cold_par8_MBps"), debs_par8=d.get("cold_par8_MBps"),
        work_write=r.get("disks", {}).get("/kaggle/working", {}).get("write_MBps"),
        shm_write=r.get("disks", {}).get("/dev/shm", {}).get("write_MBps"),
        total_s=r.get("t_total_s"), errors=[k for k in r if k.endswith("_error")]))

print(f"## Per run ({len(rows)} probes)\n")
print("| run | machine (GPU uuid tails) | H2D/D2H min GB/s | slow GPUs | NCCL ag 516 KB µs: best / worst pair | model mount MB/s single / 8-par | wheels 8-par | debs 8-par | /kaggle/working write | power |")
print("|---|---|---|---|---|---|---|---|---|---|")
for x in rows:
    ag = {k: v for k, v in x["ag516"].items() if v}
    best = min(ag.items(), key=lambda kv: kv[1]) if ag else ("-", None)
    worst = max(ag.items(), key=lambda kv: kv[1]) if ag else ("-", None)
    print(f"| {x['run']} | {x['machine'][:20]} | {x['h2d_min']} / {x['d2h_min']} | {x['slow_gpus'] or '-'} | "
          f"{best[1]:.0f} ({best[0]}) / {worst[1]:.0f} ({worst[0]}) | {x['model_single']} / {x['model_par8']} | "
          f"{x['wheels_par8']} | {x['debs_par8']} | {x['work_write']} | {x['power']} |" if ag else f"| {x['run']} | (no gpucheck) |")


def dist(name, vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return f"- {name}: n/a"
    q = lambda p: vals[min(len(vals) - 1, int(p * (len(vals) - 1) + 0.5))]
    return f"- {name}: min {vals[0]:.0f}, p25 {q(.25):.0f}, median {q(.5):.0f}, p75 {q(.75):.0f}, max {vals[-1]:.0f} (n={len(vals)})"


print("\n## Summary\n")
machines = {x["machine"] for x in rows}
print(f"- runs: {len(rows)}; distinct machines (by GPU UUIDs): {len(machines)}; distinct hosts: {len({x['host'] for x in rows})}")
bad = [x["run"] for x in rows if x["slow_gpus"]]
print(f"- runs with a slow host link (any GPU H2D or D2H < 6 GB/s): {len(bad)} {bad}")
worst_ag = [max(v for v in x["ag516"].values() if v) for x in rows if any(x["ag516"].values())]
best_ag = [min(v for v in x["ag516"].values() if v) for x in rows if any(x["ag516"].values())]
print(dist("NCCL all-gather 516 KB, best pair µs", best_ag))
print(dist("NCCL all-gather 516 KB, worst pair µs", worst_ag))
print(dist("model mount cold single-stream MB/s", [x["model_single"] for x in rows]))
print(dist("model mount cold 8-parallel MB/s", [x["model_par8"] for x in rows]))
print(dist("vllm-wheels cold 8-parallel MB/s", [x["wheels_par8"] for x in rows]))
print(dist("cu13-install cold 8-parallel MB/s", [x["debs_par8"] for x in rows]))
print(dist("/kaggle/working write MB/s", [x["work_write"] for x in rows]))
print(dist("/dev/shm write MB/s", [x["shm_write"] for x in rows]))
if rows and all(x["model_par8"] for x in rows):
    t = [131.09e3 / x["model_par8"] / 60 for x in rows]
    print(dist("implied minutes to read the whole 131 GB model at the 8-parallel rate", t))
print(f"- probe errors: {[(x['run'], x['errors']) for x in rows if x['errors']]}")
