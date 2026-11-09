"""Per-window server metrics from solver.log "server:" lines, by minutes since the first such line (attempts start),
plus gate-hold log lines and attempts done. Usage: windows.py <solver.log> [window_min=10] [max_min=40] [pool_tokens]"""
import re, sys
path = sys.argv[1]; W = float(sys.argv[2]) if len(sys.argv) > 2 else 10; M = float(sys.argv[3]) if len(sys.argv) > 3 else 40
pool = float(sys.argv[4]) if len(sys.argv) > 4 else 0
pat = re.compile(r"\+(\d+\.\d+)h\] server: ([\d.]+) gen tok/s, prefix hits ([\d.]+), uncached ([\d.]+) tok/s, KV ([\d.]+), "
                 r"running ([\d.]+), waiting ([\d.]+), preemptions ([\d.]+).*?\(last (\d+) s\); (\d+) attempts done")
rows, holds = [], []
for line in open(path, errors="replace"):
    m = pat.search(line)
    if m:
        rows.append([float(x) for x in m.groups()])
    elif "KV gate:" in line:
        h = re.search(r"\+(\d+\.\d+)h\]", line)
        if h:
            holds.append(float(h.group(1)))
if not rows:
    print("no server lines"); sys.exit()
t0 = rows[0][0] - rows[0][8] / 3600
print(f"{'win(min)':>9} {'gen/s':>7} {'hits':>5} {'uncached/s':>10} {'KV':>6} {'run':>5} {'wait':>5} {'preempt':>7} {'holds':>5} {'done':>5}" + ("  KV/stream" if pool else ""))
start = 0.0
while start < M:
    sel = [r for r in rows if start <= (r[0] - t0) * 60 < start + W]
    if sel:
        dur = sum(r[8] for r in sel)
        avg = lambda i: sum(r[i] * r[8] for r in sel) / dur
        nh = sum(1 for h in holds if start <= (h - t0) * 60 < start + W)
        kvps = (avg(4) * pool / avg(5)) if pool and avg(5) else 0
        print(f"{start:4.0f}-{start + W:<4.0f} {avg(1):7.1f} {avg(2):5.2f} {avg(3):10.1f} {avg(4):6.3f} {avg(5):5.1f} {avg(6):5.1f} {sel[-1][7]:7.0f} {nh:5d} {int(sel[-1][9]):5d}"
              + (f"  {kvps / 1000:6.1f}k" if pool else ""))
    start += W
