"""Run a Kaggle notebook's code cells in order in one namespace (like "Run All"); stop at the first exception.
Usage: run_nb.py <notebook.ipynb>   (environment = the Kaggle environment to emulate)"""
import json
import sys
import time
import traceback

nb = json.load(open(sys.argv[1]))
ns = {"__name__": "__main__"}
t0 = time.time()
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    print(f"##### RUN_NB cell {i} start +{time.time() - t0:.1f}s", flush=True)
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    except BaseException:
        print(f"##### RUN_NB cell {i} RAISED +{time.time() - t0:.1f}s\n" + traceback.format_exc(), flush=True)
        print("##### RUN_NB notebook FAILED", flush=True)
        sys.exit(1)
    print(f"##### RUN_NB cell {i} done +{time.time() - t0:.1f}s", flush=True)
print(f"##### RUN_NB notebook OK +{time.time() - t0:.1f}s", flush=True)
