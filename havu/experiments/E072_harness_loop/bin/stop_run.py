"""Stop one replica run by process group: the runner's tree (kernel, solver, sandboxes), the server supervisor and the
server's group, the sampler and memlog. Every group leader's cmdline is checked before signalling. The supervisor gets
SIGTERM first (it stops its server, as the notebook's _manh_stop does), then every group gets SIGKILL. Waits until no
GPU process remains. usage: stop_run.py RUN_NAME [--dry-run]"""
import json, os, signal, subprocess, sys, time
run = sys.argv[1]; dry = "--dry-run" in sys.argv; R = f"/root/nbruns/{run}"; W = f"/root/kwork_{run}"
def cmd(p):
    try: return open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
    except OSError: return ""
def table():
    P = {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                st = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split(); P[int(p)] = (int(st[1]), int(st[2]))
            except OSError: pass
    return P
P = table(); kids = {}
for p, (pp, pg) in P.items(): kids.setdefault(pp, []).append(p)
def desc(r):
    out, todo = [], [r]
    while todo:
        x = todo.pop(); out.append(x); todo += kids.get(x, [])
    return out
runner = int(open(f"{R}/run_nb.pid").read()); assert "run_nb.py" in cmd(runner), cmd(runner)
sup = json.load(open(f"{W}/dsv41_logs/supervisor.json")); sp, srv = sup["supervisor_pid"], sup["pid"]
assert "dsv41_vllm.py" in cmd(sp) and run in cmd(sp), cmd(sp); assert "vllm.entrypoints" in cmd(srv), cmd(srv)
sampler = int(open(f"{R}/sampler.pid").read()); assert "v12_sampler.py" in cmd(sampler) and run in cmd(sampler)
memlog = int(open(f"{R}/memlog.pid").read()); assert "memlog.sh" in cmd(memlog) and run in cmd(memlog)
groups = sorted({P[x][1] for x in desc(runner) + desc(sp) + desc(srv) if x in P} | {sampler, memlog})
print("groups", len(groups), groups)
for g in groups:
    print("  pgid", g, cmd(g)[:110] or "(leader gone)")
if dry: sys.exit(0)
os.kill(sp, signal.SIGTERM); t = time.time()
while time.time() - t < 90 and os.path.exists(f"/proc/{srv}"): time.sleep(1)
print(f"supervisor SIGTERM: server gone after {time.time() - t:.0f}s" if not os.path.exists(f"/proc/{srv}") else "server still up after 90 s")
for g in groups:
    try: os.killpg(g, signal.SIGKILL)
    except ProcessLookupError: pass
t = time.time()
while time.time() - t < 180:
    q = subprocess.run("nvidia-smi --query-compute-apps=pid --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.split()
    if not q: break
    time.sleep(2)
print(f"GPU processes left: {q} after {time.time() - t:.0f}s")
print(subprocess.run("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader", shell=True, capture_output=True, text=True).stdout)
left = [x for g in groups for x, (pp, pg) in table().items() if pg == g]
print("processes left in those groups:", left)
