import sys
sys.path.insert(0, "/tmp/claude-1000/-home-vu-Workspace-ARC-AGI-2/81e68821-e1f4-4b05-bad6-9a606f77ad54/scratchpad/goal75")
from kvcap import run
for kvx in (1.14, 1.24):
    for scale in (1.0, 0.9):
        row = []
        for w, gate in ((36, 0.6), (40, 0.65), (44, 0.7)):
            a, b = run(kvx, w, gate, scale=scale)
            row.append(f"x{w} g{gate}: {a:5.1f}")
        print(f"KV x{kvx} tokens x{scale}: " + " | ".join(row))
