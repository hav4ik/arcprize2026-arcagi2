"""Execute a Kaggle notebook end to end on the pod, as "Run All" would: nbclient, one kernel (the current Python, i.e.
the Kaggle-like venv), cells in order, stop at the first error. Logs each cell's start/end (UTC, seconds) to
<out>.timeline and saves the executed notebook with outputs to <out>.ipynb.
Usage: run_nb.py <notebook.ipynb> <out prefix> [cell_timeout_s=50000]"""
import json, sys, time
import nbformat
from nbclient import NotebookClient
nb_path, out = sys.argv[1], sys.argv[2]
tmo = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
nb = nbformat.read(nb_path, as_version=4)
tl = open(out + ".timeline", "a", buffering=1)
t0 = time.time()
def first_line(cell):
    return (cell.source.splitlines() or [""])[0][:110]
def on_start(cell, cell_index, **kw):
    tl.write(f"{time.strftime('%H:%M:%S', time.gmtime())} +{time.time() - t0:7.0f}s START cell {cell_index}: {first_line(cell)}\n")
def on_done(cell, cell_index, execute_reply=None, **kw):
    st = (execute_reply or {}).get("content", {}).get("status")
    tl.write(f"{time.strftime('%H:%M:%S', time.gmtime())} +{time.time() - t0:7.0f}s END   cell {cell_index}: status {st}\n")
client = NotebookClient(nb, timeout=tmo, kernel_name="python3", allow_errors=False, record_timing=True,
                        on_cell_start=on_start, on_cell_executed=on_done, resources={"metadata": {"path": "/kaggle/working"}})
try:
    client.execute()
    tl.write(f"NOTEBOOK_DONE ok +{time.time() - t0:.0f}s\n")
except Exception as exc:
    tl.write(f"NOTEBOOK_DONE error +{time.time() - t0:.0f}s: {type(exc).__name__}: {str(exc)[-1500:]}\n")
finally:
    nbformat.write(nb, out + ".ipynb")
