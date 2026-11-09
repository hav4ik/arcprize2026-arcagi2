"""Control: a concurrency-1 decode pass vs the concurrent decode pass and vs the prefill pass, on each prompt's common
greedy prefix (the prefill pass scored the concurrent pass's tokens). Usage: ab_control.py <dec32.json> <dec1.json> <prefill.json>"""
import json, sys
d32 = {r["i"]: r for r in json.load(open(sys.argv[1]))["results"]}
d1 = {r["i"]: r for r in json.load(open(sys.argv[2]))["results"]}
pre = {(x["i"], x["k"]): x for x in json.load(open(sys.argv[3]))["results"]}
def stats(v):
    a = sorted(abs(t) for t in v)
    return (f"n {len(v)} mean|d| {sum(a) / len(a):.4f} p99 {a[min(len(a) - 1, int(0.99 * len(a)))]:.4f} max {a[-1]:.4f} "
            f">0.5: {sum(t > 0.5 for t in a)} mean d {sum(v) / len(v):+.4f}") if v else "n 0"
dd, d1p, d32p, common = [], [], [], []
for i, r in d1.items():
    a = d32[i]; n = 0
    while n < min(len(a["ids"]), len(r["ids"])) and a["ids"][n] == r["ids"][n]:
        dd.append(a["lps"][n] - r["lps"][n])
        x = pre.get((i, n))
        if x and x["lp"] is not None:
            d1p.append(r["lps"][n] - x["lp"]); d32p.append(a["lps"][n] - x["lp"])
        n += 1
    common.append(n)
print("common greedy prefix lengths (conc 1 vs conc 32):", common)
print("decode conc 32 vs decode conc 1:", stats(dd))
print("decode conc 1  vs prefill     :", stats(d1p))
print("decode conc 32 vs prefill (same positions):", stats(d32p))
