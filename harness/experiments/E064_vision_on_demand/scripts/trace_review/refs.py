"""Explicit image references in reasoning, where they occur, and per-task wins/losses of image vs text."""
import json, gzip, re, sys, collections, random
sys.path.insert(0, "/home/vu/Workspace/ARC-AGI-2/experiments/E064_vision_on_demand/scripts/trace_review")
from traces import ARMS, load
REF = re.compile(r"(?:the|in the|from the|these|both|each|first|second|output|input) images?\b|\bimages? (?:show|shows|suggest|look|looks|appear)|\b(?:picture|visually|i see in the)\b", re.I)
WRONG = re.compile(r"(images? (?:may|might|could) be (?:wrong|misleading|inaccurate)|rely on (?:the )?json|json (?:is|grids are) authoritative|image (?:is|was) (?:wrong|misleading|inaccurate)|doesn't match the image|contradict)", re.I)
pair = [("DS text", "DS img"), ("Q text", "Q img")]
D = {a: load(ARMS[a]) for a in ARMS}
def reasoning(r):
    m = json.load(gzip.open(r["trace_path"]))["messages"]
    return [(x.get("reasoning_content") or x.get("reasoning") or "") for x in m if x["role"] == "assistant"]
for a in ARMS:
    n = ref = wrong = 0; refturn = collections.Counter()
    for k, r in D[a].items():
        if r.get("multi_share"): continue
        rs = reasoning(r); n += 1
        hits = [i for i, x in enumerate(rs) if REF.search(x)]
        ref += bool(hits); wrong += bool(WRONG.search(" ".join(rs)))
        for i in hits[:1]: refturn["t%d" % min(i, 3)] += 1
    print(f"{a:11} explicit image refs in {100*ref/n:5.1f}% of attempts; first ref turn {dict(sorted(refturn.items()))}; distrust/contradiction phrases {100*wrong/n:4.1f}%")
for t_, i_ in pair:
    T, I = D[t_], D[i_]
    tasks = sorted({k[0] for k in I} & {k[0] for k in T})
    def sc(d, t): 
        rs = [r for k, r in d.items() if k[0] == t]; return sum(bool(r["correct"]) for r in rs) / max(1, len(rs))
    win = [t for t in tasks if sc(I, t) > sc(T, t) + 0.25]; loss = [t for t in tasks if sc(T, t) > sc(I, t) + 0.25]
    print(i_, "vs", t_, "tasks", len(tasks), "image better", len(win), win, "image worse", len(loss), loss)
