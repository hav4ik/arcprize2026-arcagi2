import json, gzip, re, sys
sys.path.insert(0, "/home/vu/Workspace/ARC-AGI-2/experiments/E064_vision_on_demand/scripts/trace_review")
from traces import ARMS, load
from refs import REF
arm, tasks, per = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
D = load(ARMS[arm])
for t in tasks:
    for k, r in sorted(D.items()):
        if k[0] != t or r.get("multi_share") or k[2] != 0: continue
        m = json.load(gzip.open(r["trace_path"]))["messages"]
        rs = [(i, x.get("reasoning_content") or x.get("reasoning") or "") for i, x in enumerate([y for y in m if y["role"] == "assistant"])]
        out = []
        for i, x in rs:
            for mm in REF.finditer(x):
                out.append(f"t{i}: ..." + x[max(0, mm.start()-220):mm.end()+220].replace("\n", " ") + "...")
        print(f"### {t} p{k[1]} a0 correct={r['correct']} refs={len(out)}")
        for s in out[:per]: print("  ", s)
