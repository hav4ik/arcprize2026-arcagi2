"""Run the notebook's "Manh's server cell" (extracted verbatim from scratchpad/nb_img_ext/dsv41-tools-submit-img.ipynb)
on the pod as the notebook would: /kaggle/working -> WORK, the Python part executed as is (his DSV41_MOE_TUNING code and
os.environ.update block), then the `!python ...` line run as a subprocess with the venv's Python, {E066_MODEL} and
{server_root} filled in, plus optional extra launcher args (e.g. --vllm-arg=--decode-context-parallel-size=4).
Usage: run_manh_cell.py <cell.py> <work dir> <model dir> <python> [extra launcher args ...]
Writes <work>/last_server_root.txt."""
import os, shlex, subprocess, sys
cell, work, model, py = sys.argv[1:5]
extra = sys.argv[5:]
src = open(cell).read().replace("/kaggle/working", work)
lines = src.splitlines()
i = next(k for k, l in enumerate(lines) if l.startswith("!python"))
j = i
while lines[j].rstrip().endswith("\\"):
    j += 1
pre, bang, post = "\n".join(lines[:i]), " ".join(l.rstrip().rstrip("\\") for l in lines[i:j + 1]), lines[j + 1:]
ns = {"E066_MODEL": model, "__name__": "manh_cell"}
exec(compile(pre, "manh_server_cell", "exec"), ns)  # his env + DSV41_MOE_TUNING, server_root
cmd = bang[1:].replace("{E066_MODEL}", model).replace("{server_root}", ns["server_root"])
argv = shlex.split(cmd)
assert argv[0] == "python", argv[0]
argv = [py] + argv[1:] + extra
open(os.path.join(work, "last_server_root.txt"), "w").write(ns["server_root"] + "\n")
print("SERVER_ROOT", ns["server_root"], flush=True)
print("ARGV", " ".join(shlex.quote(a) for a in argv), flush=True)
print("ENV", {k: (v if "TUNING" not in k else f"<{len(v)} chars>") for k, v in os.environ.items() if k.startswith("VLLM_")}, flush=True)
rc = subprocess.call(argv, cwd=work)
print("LAUNCHER_RC", rc, flush=True)
print("ARC_EXTERNAL_SERVER_LOG", f"{ns['server_root']}/server/server.log", flush=True)
