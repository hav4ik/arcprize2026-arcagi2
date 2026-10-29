import json, gzip, re, sys, statistics as st
sys.path.insert(0, "/home/vu/Workspace/ARC-AGI-2/experiments/E064_vision_on_demand/scripts/trace_review")
from traces import ARMS, load
WANT = re.compile(r"(visuali[sz]e|visually (?:confirm|check|verify|inspect)|(?:look|inspect) (?:at )?(?:the )?(?:test|my|our|predicted|candidate) (?:output|answer|prediction|result)s? visually|render(?:ed)? (?:it|the (?:output|answer|grid))|plot (?:it|the grid))", re.I)
VIS_CODE = re.compile(r"\bmatplotlib\b|\bimshow\b|\bfrom PIL\b|\bimport PIL\b")
for a in ("DS text", "DS img", "Q text", "Q img"):
    D = load(ARMS[a]); n = want = code = 0; p0 = []
    for k, r in D.items():
        if r.get("multi_share"): continue
        t = json.load(gzip.open(r["trace_path"]))
        m = t["messages"]; ev = t.get("events") or []
        asst = [x for x in m if x["role"] == "assistant"]
        txt = " ".join((x.get("reasoning_content") or x.get("reasoning") or "") + (x.get("content") or "") for x in asst)
        calls = " ".join(c["function"]["arguments"] for x in asst for c in (x.get("tool_calls") or []))
        n += 1; want += bool(WANT.search(txt)); code += bool(VIS_CODE.search(calls))
        u = next((e.get("usage") for e in ev if isinstance(e, dict) and e.get("usage")), None)
        if u: p0.append(u.get("prompt") or u.get("prompt_tokens") or 0)
    print(f"{a:8} attempts {n}: wants to see a grid (visualize/visually check) {100*want/n:4.1f}%; tries matplotlib/PIL {100*code/n:4.1f}%; first-turn prompt tokens median {st.median(p0) if p0 else 'n/a'}")
# context of "want to see" phrases in DS text
D = load(ARMS["DS text"]); c = 0
for k, r in sorted(D.items()):
    if r.get("multi_share"): continue
    m = json.load(gzip.open(r["trace_path"]))["messages"]
    for x in m:
        if x["role"] != "assistant": continue
        s = x.get("reasoning_content") or ""
        for mm in WANT.finditer(s):
            if c < 12: print("   ", k[0], "..." + s[max(0, mm.start()-140):mm.end()+100].replace("\n", " ")); c += 1
