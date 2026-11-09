"""reap.py <run>: after a notebook ended, stop what it left: the run's sampler/memlog (pid files, cmdline checked) and a server
whose pids are in dsv41_logs/supervisor.json (cmdline checked: dsv41_vllm.py / vllm.entrypoints). Never kills by name."""
import json, os, signal, sys, time
run = sys.argv[1]; R = f"/root/nbruns/{run}"; W = f"/root/kwork_{run}"
def cmd(p):
    try: return open(f"/proc/{p}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
    except OSError: return ""
def kill(pid, must, sig=signal.SIGKILL):
    c = cmd(pid)
    if c and any(m in c for m in must):
        try: os.killpg(os.getpgid(pid), sig); print("killed group of", pid, c[:80])
        except OSError: pass
for f, must in (("sampler.pid", ["v12_sampler.py"]), ("memlog.pid", ["memlog.sh"]), ("run_nb.pid", ["run_nb.py"])):
    try: kill(int(open(f"{R}/{f}").read()), must)
    except (OSError, ValueError): pass
try:
    sup = json.load(open(f"{W}/dsv41_logs/supervisor.json"))
    if cmd(sup.get("supervisor_pid", 0)) and "dsv41_vllm.py" in cmd(sup["supervisor_pid"]):
        os.kill(sup["supervisor_pid"], signal.SIGTERM); time.sleep(20)
    for k in ("supervisor_pid", "pid"):
        kill(sup.get(k, 0), ["dsv41_vllm.py", "vllm.entrypoints"])
except (OSError, ValueError, KeyError): pass
