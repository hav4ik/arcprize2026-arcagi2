"""E081: fire probe notebooks 02..N on Kaggle 4xL4, at most 2 running (Kaggle's batch GPU limit), stop launching once
the estimated GPU time (sum of push->complete wall time) reaches BUDGET_MIN. Downloads each run's outputs to runs/."""
import os, subprocess, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
KAG = os.path.join(ROOT, ".venv/bin/kaggle")
FIRST, LAST, BUDGET_MIN = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
TMP = os.environ.get("E081_TMP", "/tmp/e081")
used_min = float(os.environ.get("E081_USED_MIN", "0"))  # already spent (probe 01)
log = open(os.path.join(HERE, "launch.log"), "a")
def say(*a):
    m = time.strftime("%H:%M:%S ", time.gmtime()) + " ".join(map(str, a)); print(m, flush=True); log.write(m + "\n"); log.flush()
def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout + subprocess.run(["true"]).stdout
def status(slug):
    out = subprocess.run([KAG, "kernels", "status", slug], capture_output=True, text=True).stdout
    return out.strip().splitlines()[-1] if out.strip() else ""
running, todo = {}, list(range(FIRST, LAST + 1))
while todo or running:
    for n, (slug, t0) in list(running.items()):
        s = status(slug)
        if any(k in s for k in ("COMPLETE", "ERROR", "CANCEL")):
            mins = (time.time() - t0) / 60; used_min += mins
            d = os.path.join(HERE, "runs", f"probe-{n:02d}"); os.makedirs(d, exist_ok=True)
            subprocess.run([KAG, "kernels", "output", slug, "-p", d], capture_output=True)
            say(f"probe-{n:02d} {s.split()[-1]} after {mins:.1f} min; est. used {used_min:.0f} min")
            del running[n]
    while todo and len(running) < 2:
        if used_min + 5 * (len(running) + 1) > BUDGET_MIN:
            say(f"budget: est. {used_min:.0f} min used; not launching {todo}"); todo = []; break
        n = todo[0]; nb = os.path.join(TMP, f"nb{n:02d}")
        subprocess.run([sys.executable, os.path.join(HERE, "build.py"), str(n), nb], capture_output=True)
        out = subprocess.run([KAG, "kernels", "push", "-p", nb], capture_output=True, text=True).stdout
        if "successfully pushed" in out:
            running[n] = (f"chankhavu/e081-l4-probe-{n:02d}", time.time()); todo.pop(0); say(f"pushed probe-{n:02d}")
        else:
            say(f"push probe-{n:02d} refused: {out.strip()[-160:]}"); break
    time.sleep(30)
say(f"ALL_DONE est. GPU {used_min:.0f} min")
