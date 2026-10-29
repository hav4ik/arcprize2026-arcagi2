# NOT USED (E064, 2026-10-28: superseded by B32-M; kept for reference)
# Cell M3 (E064): the solver against Manh's server (tool harness, KV gate, deadlines, incremental submission).
# Only the solver's own session is killed by the kill switch; the server is never touched from here.
if MODE != "fallback":
  try:
    import signal, subprocess, threading
    env = dict(BASE_ENV)
    if MODE == "smoke":  # commit runs: deadlines from now (the server is up), so the smoke stays short
        _now = time.time() - T0
        for _k, _d in (("ARC_STOP_NEW_S", 600), ("ARC_FORCE_AT_S", 1500), ("ARC_HARD_STOP_S", 1650),
                       ("ARC_KILL_AT_S", 1800)):
            env[_k] = str(int(_now + float(os.environ.get(f"E064_SMOKE_{_k}", _d))))
    p = subprocess.Popen([sys.executable, "-u", f"{KIT}/solver_dsv41.py"], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)

    def _kill():
        try:
            os.killpg(p.pid, signal.SIGKILL)  # the solver's session only (it started no server)
            print("kill switch: solver session killed", flush=True)
        except Exception:
            pass
    _timer = threading.Timer(max(0.0, T0 + float(env["ARC_KILL_AT_S"]) - time.time()), _kill)
    _timer.daemon = True
    _timer.start()
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    _timer.cancel()
    print("solver exit code", p.returncode, flush=True)
  except Exception as exc:
    print("solver cell failed:", type(exc).__name__, exc, flush=True)
    if not RERUN:
        raise
