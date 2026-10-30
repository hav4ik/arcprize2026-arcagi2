"""Does the server keep earlier turns' reasoning in the prompt? For consecutive chat turns k-1, k:
delta = prompt_k - prompt_{k-1} - completion_{k-1} (= tool result + template if the reasoning is kept,
that minus reasoning_{k-1} if it is dropped)."""
import json, gzip, sys, statistics as st
def turns(path):
    ev = json.load(gzip.open(path, "rt"))["events"]
    return [e for e in ev if "usage" in e and e.get("usage")]
def u(e, k):
    x = e["usage"]
    if k == "p": return x.get("prompt_tokens", x.get("prompt"))
    if k == "c": return x.get("completion_tokens", x.get("completion"))
    if k == "r": return ((x.get("completion_tokens_details") or {}).get("reasoning_tokens"))
for tag in sys.argv[1:]:
    rows, neg, big = [], 0, 0
    for l in open(f"evals/results/tools_persistent_grid_lean2sub_{tag}.jsonl"):
        r = json.loads(l)
        if r.get("error") or r["pair"] != 0 or not r.get("trace_path"): continue
        t = turns(r["trace_path"])
        for a, b in zip(t, t[1:]):
            if None in (u(a, "p"), u(b, "p"), u(a, "c")): continue
            d = u(b, "p") - u(a, "p") - u(a, "c"); rs = u(a, "r")
            rows.append((d, rs))
    ds = [d for d, _ in rows]; withr = [(d, rr) for d, rr in rows if rr and rr > 2000]
    print(f"{tag}: {len(rows)} turn pairs; delta median {st.median(ds):.0f}, min {min(ds)}, share<0 {100*sum(d<0 for d in ds)/len(ds):.1f}%")
    if withr:
        print(f"   after turns with >2000 reasoning tokens ({len(withr)}): delta median {st.median(d for d,_ in withr):.0f}, "
              f"median reasoning {st.median(rr for _,rr in withr):.0f}, share delta<0 {100*sum(d<0 for d,_ in withr)/len(withr):.1f}%")
    else:
        print("   no reasoning_tokens in usage")
