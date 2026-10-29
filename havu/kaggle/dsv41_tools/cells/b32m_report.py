# Cell 4b (B32-M smoke report, commit runs only): the timeline (install cells, row-chunk patch, solver start, server start /
# up), the server's throughput windows (gen tok/s, running, KV every 30 s), the environment record (b32m_env.json) and
# the API contract (smoke_contract.json). If the server never came up, the timeline and the server log's tail say how
# far it got (install, weight load, KV/graph setup). It only prints; the verdict cell at the end raises on a failure.
import re
if not os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):
    _b32m_bad = []
    _w = os.environ.get("ARC_OUT", "/kaggle/working")
    _ev = {}
    try:
        _ev = json.load(open(f"{_w}/b32m_env.json"))
        print("B32-M env:", {k: _ev.get(k) for k in ("python", "packages", "vllm_wheels_in_inputs", "rowchunk_patch",
                                                      "error")}, flush=True)
        if _ev.get("error"):
            _b32m_bad.append(f"environment: {_ev['error']}")
        if (_ev.get("rowchunk_patch") or {}).get("action") not in ("patched", "already_fixed"):
            _b32m_bad.append(f"row-chunk patch: {(_ev.get('rowchunk_patch') or {}).get('action')}")
    except Exception as exc:
        _b32m_bad.append(f"b32m_env.json: {type(exc).__name__}: {exc}")
    _sl = ""
    try:
        _sl = open(f"{_w}/dsv41_logs/solver.log").read()
    except Exception as exc:
        _b32m_bad.append(f"solver.log: {type(exc).__name__}: {exc}")

    def _at(pattern):  # pod-clock hours of the first solver.log line matching `pattern`, in minutes
        m = re.search(r"\+(\d+\.\d+)h\] [^\n]*" + pattern, _sl)
        return round(float(m.group(1)) * 60, 1) if m else None
    _tl = {"manh_install_cells_start_s": _ev.get("install_cell_start_s"), "env_cell_start_s": _ev.get("env_cell_start_s"),
           "solver_start_min": _at("fallback submission written"), "server_start_min": _at("server start \\(first"),
           "server_up_min": _at("server up after"), "server_failed_min": _at("server (exited during startup|not up after)"),
           "first_attempt_done_min": _at(r"\[1\] "), "force_cut_min": _at("force deadline"),
           "final_write_min": _at("submission written \\(final\\)")}
    _up = re.findall(r"server up after (\d+)s; (.*)", _sl)
    _tl["server_up"] = _up[0] if _up else None
    print("B32-M timeline (minutes after the first cell; install cells in seconds):", _tl, flush=True)
    if not _up:
        _b32m_bad.append("server never came up (timeline above)")
        try:  # how far the server got: weight load, KV pool, CUDA graphs
            _sv = open(f"{_w}/dsv41_logs/server.log", errors="replace").read()
            for _pat in ("Loading weights took", "Model loading took", "GPU KV cache size", "Capturing CUDA graph",
                         "Graph capturing finished", "Starting vLLM API server", "Error", "Traceback", "KeyError"):
                _hits = [l for l in _sv.splitlines() if _pat in l]
                if _hits:
                    print(f"  server.log [{_pat}] x{len(_hits)}: {_hits[-1][-240:]}", flush=True)
            print("  server.log tail:", " | ".join(_sv.splitlines()[-12:])[-2500:], flush=True)
        except Exception as exc:
            print("  no server.log:", type(exc).__name__, exc, flush=True)
    else:
        print("B32-M attempts done:", _sl.count(" stop="), flush=True)
    try:
        _mt = [json.loads(l) for l in open(f"{_w}/dsv41_logs/metrics.jsonl") if l.strip()]
        print("B32-M server windows (min, gen tok/s, server running, waiting, KV, prefix hit):",
              [(round(m.get("elapsed_h", 0) * 60, 1), m.get("gen_tok_s"), m.get("srv_running"), m.get("srv_waiting"),
                m.get("kv"), m.get("prefix_hit")) for m in _mt], flush=True)
        _gt = [m["gen_tok_s"] for m in _mt if m.get("gen_tok_s")]
        print("B32-M gen tok/s: max", max(_gt, default=None), "| mean of windows", round(sum(_gt) / len(_gt), 1)
              if _gt else None, "| KV max", max((m.get("kv") or 0 for m in _mt), default=None), flush=True)
    except Exception as exc:
        print("B32-M metrics unavailable:", type(exc).__name__, exc, flush=True)
    if "submission written (final)" not in _sl:
        _b32m_bad.append("no final submission write")
    try:
        import time as _ct
        for _ in range(6):  # the contract thread writes its file once the server answered
            if os.path.exists(f"{_w}/smoke_contract.json"):
                break
            _ct.sleep(5)
        _ck = json.load(open(f"{_w}/smoke_contract.json"))
        print("B32-M contract:", _ck, flush=True)
        _b32m_bad += [f"contract: {x}" for x in _ck.get("fail", [])]
    except Exception as exc:
        _b32m_bad.append(f"smoke_contract.json: {type(exc).__name__}: {exc}")
    if os.environ.get("B32M_LOAD_TEST") == "1":  # load test: throughput is the point, finished tasks are not
        try:  # vLLM's 10-s stats lines: pure-decode windows = no prompt tokens and >= 70 running
            _sv = open(f"{_w}/dsv41_logs/server.log", errors="replace").read()
            _wins = [(float(a), float(b), int(c)) for a, b, c in re.findall(
                r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs",
                _sv)]
            _pure = sorted(g for pr, g, r in _wins if pr == 0 and r >= 70)
            _q = lambda xs, f: xs[min(len(xs) - 1, int(f * len(xs)))] if xs else None
            _act = [g for pr, g, r in _wins if r > 0]
            print("B32-M LOAD pure-decode windows (prompt 0 tok/s, running >= 70):", {"n": len(_pure), "median": _q(_pure, .5),
                  "p90": _q(_pure, .9), "max": _pure[-1] if _pure else None}, flush=True)
            print("B32-M LOAD server windows (10 s):", {"n": len(_wins), "mean gen tok/s while running": round(
                sum(_act) / len(_act), 1) if _act else None, "running max": max((r for _, _, r in _wins), default=None),
                "windows with prompt tokens": sum(1 for pr, _, _ in _wins if pr > 0)}, flush=True)
            _m30 = [m.get("gen_tok_s") for m in _mt if m.get("gen_tok_s") is not None] if "_mt" in dir() else []
            print("B32-M LOAD solver windows (30 s) gen tok/s:", _m30, "| mean", round(sum(_m30) / len(_m30), 1)
                  if _m30 else None, flush=True)
        except Exception as exc:
            print("B32-M LOAD stats unavailable:", type(exc).__name__, exc, flush=True)
        _crash = [x for x in ("server died", "server exited during startup", "solver crashed", "server not up after")
                  if x in _sl]
        _fatal = [x for x in _b32m_bad if x.startswith(("server never", "environment", "row-chunk"))]
        _fatal += [f"solver.log: {x}" for x in _crash]
        try:
            _sub = json.load(open(f"{_w}/submission.json"))
            assert isinstance(_sub, dict) and _sub, "empty"
        except Exception as exc:
            _fatal.append(f"no valid submission.json ({type(exc).__name__}: {exc})")
        for _x in _b32m_bad:
            if _x not in _fatal:
                print("B32-M LOAD warning (not fatal in a load test):", _x, flush=True)
        _b32m_bad = _fatal
    B32M_BAD = _b32m_bad  # the verdict cell at the end raises on it (after B32's smoke report has printed)
