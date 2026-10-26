import sys
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/goal75")
from kvcap import run
S = 1.04
rows = [("V6 as submitted (x36 g0.6)", 1.0, 36, 0.6, S),
        ("next candidate (x44 g0.7, interval 512 ~ KV x1.07)", 1.07, 44, 0.7, S),
        ("+ #58678 Engram split (KV +9%) + #156 indexer (+7% decode)", 1.17, 44, 0.7, S * 1.07),
        ("+ #56227 bounded replay (KV headroom ~+10%, fewer re-prefills +3%)", 1.29, 48, 0.75, S * 1.07 * 1.03),
        ("+ E051 offload (effective KV x2.5), x56 ungated", 2.5 * 1.17, 56, 0, S * 1.07 * 1.03),
        ("+ E051 offload, x64 ungated", 2.5 * 1.17, 64, 0, S * 1.07 * 1.03)]
for label, kv, w, g, s in rows:
    a, b = run(kv, w, g, s=s)
    print(f"{label:68s} {a:5.1f}  (2nd attempts {b:4.1f}%)")
