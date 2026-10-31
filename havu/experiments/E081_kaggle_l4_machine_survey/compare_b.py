"""E081b: per probe, both E128 models' cold read speed on the same machine, plus zone and GPU/NCCL health."""
import json, glob, os
X = os.path.dirname(os.path.abspath(__file__))
TXT = "/kaggle/input/models/chankhavu/dsv41-flash-1017-arc-tool-e128-2bit/pytorch/default/1"
IMG = "/kaggle/input/models/arazzd/deepseek-v41-flash-1017-arc-tool-img-e128-causal/pytorch/default/1"
for f in sorted(glob.glob(f"{X}/runs/probeb-*/probe.json")):
    r = json.load(open(f)); m = r.get("mounts", {}); idn = r.get("identity", {}); g = r.get("gpucheck", {})
    first = "IMG" if list(m)[:1] == [IMG] else "TXT"
    copy = g.get("copy", {}); ag = {p["pair"]: p.get("ag516K") for p in g.get("pairs", []) if p["mode"] == "alone"}
    print(os.path.basename(os.path.dirname(f)), "| zone", idn.get("gce"), "| read first:", first,
          "| H2D/D2H min", min((v["h2d"] for v in copy.values()), default=None), min((v["d2h"] for v in copy.values()), default=None),
          "| NCCL ag516 best/worst", min(ag.values(), default=None), max(ag.values(), default=None))
    for name, k in (("TXT E128 (V12)", TXT), ("IMG E128 causal", IMG)):
        v = m.get(k, {})
        print(f"   {name:16s} single {v.get('cold_single_MBps')}  par8 {v.get('cold_par8_MBps')} MB/s  ({v.get('files')} files, {v.get('total_gb')} GB)")
    print("   errors:", [k for k in r if k.endswith('_error')])
