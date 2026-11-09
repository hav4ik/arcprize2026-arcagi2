"""E051 in-process stack sampler (containers without ptrace, so no py-spy). Test servers only.

Put this directory on PYTHONPATH and set E051_SAMPLE=1. Every Python process of the server then starts a daemon
thread that sleeps E051_SAMPLE_DELAY seconds, and, if its process title (VLLM::EngineCore, VLLM::Worker_TP0, ...) or
argv contains one of the comma-separated E051_SAMPLE_MATCH strings, samples all threads' stacks at E051_SAMPLE_HZ for
E051_SAMPLE_DUR seconds. It writes <E051_SAMPLE_OUT>/sample_<title>_<pid>.txt: per thread, the share of samples in
which each function is on the stack (inclusive) and on top (self), plus the hottest collapsed stacks.
"""
import os

if os.environ.get("E051_SAMPLE") == "1":
    import collections
    import sys
    import threading
    import time

    def _title():
        try:
            import setproctitle  # vLLM sets VLLM::EngineCore / VLLM::Worker_TPn

            return setproctitle.getproctitle()
        except Exception:
            return " ".join(sys.argv)[:120]

    def _sample():
        time.sleep(float(os.environ.get("E051_SAMPLE_DELAY", "900")))
        title = _title()
        match = [m for m in os.environ.get("E051_SAMPLE_MATCH", "EngineCore").split(",") if m]
        if match and not any(m in title for m in match):
            return
        hz = float(os.environ.get("E051_SAMPLE_HZ", "200"))
        dur = float(os.environ.get("E051_SAMPLE_DUR", "120"))
        me = threading.get_ident()
        incl = collections.defaultdict(collections.Counter)
        top = collections.defaultdict(collections.Counter)
        stacks = collections.defaultdict(collections.Counter)
        nsamp = collections.Counter()
        t_end = time.time() + dur
        while time.time() < t_end:
            names = {t.ident: t.name for t in threading.enumerate()}
            for tid, frame in sys._current_frames().items():
                if tid == me:
                    continue
                tname = names.get(tid, str(tid))
                fs = []
                f = frame
                while f is not None and len(fs) < 60:
                    c = f.f_code
                    fs.append(f"{os.path.basename(c.co_filename)}:{c.co_name}")
                    f = f.f_back
                nsamp[tname] += 1
                top[tname][fs[0]] += 1
                for fn in set(fs):
                    incl[tname][fn] += 1
                stacks[tname][";".join(reversed(fs[:25]))] += 1
            time.sleep(1.0 / hz)
        out = os.environ.get("E051_SAMPLE_OUT", "/tmp")
        safe = "".join(ch if ch.isalnum() else "_" for ch in title)[:40]
        with open(os.path.join(out, f"sample_{safe}_{os.getpid()}.txt"), "w") as fh:
            fh.write(f"title={title} pid={os.getpid()} hz={hz} dur={dur}\n")
            for tname, n in nsamp.most_common():
                fh.write(f"\n=== thread {tname}: {n} samples\n-- inclusive\n")
                for fn, c in incl[tname].most_common(40):
                    fh.write(f"{100 * c / n:6.1f}% {fn}\n")
                fh.write("-- self\n")
                for fn, c in top[tname].most_common(25):
                    fh.write(f"{100 * c / n:6.1f}% {fn}\n")
                fh.write("-- stacks\n")
                for st, c in stacks[tname].most_common(12):
                    fh.write(f"{100 * c / n:6.1f}% {st}\n")

    threading.Thread(target=_sample, name="e051-sampler", daemon=True).start()
