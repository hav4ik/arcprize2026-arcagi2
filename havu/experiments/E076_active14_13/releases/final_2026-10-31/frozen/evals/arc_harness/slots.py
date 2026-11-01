"""E020 worker-slot scheduler: per-box limits on concurrent attempts, shared by every process that uses it.

Why: the gateway's min_load policy counts in-flight HTTP requests, and an attempt waiting on a tool call has none,
so a box whose attempts are mid-tool looks idle and receives new conversations while its KV still holds all their
prefixes (AU went from 5 to 22 running and then retracted). Here each attempt holds one slot on one worker for its
whole lifetime, and every turn goes straight to that worker.

The gateway (sgl-model-gateway 0.3.2, manual policy) ignores `x-smg-target-worker` (tested 2026-10-12: six targeted
requests all went elsewhere), so the scheduler talks to the workers' own URLs (the SSH-tunnel ports 3010N, no auth),
which the gateway lists at GET /workers.

State file (JSON, guarded by fcntl.flock on a sibling .lock file):
    {"slots": {url: n}, "leases": {lease_id: {"url", "pid", "run", "t"}}}
- `slots` is re-read on every acquire, so caps can be changed live by editing the file (or `--set`).
- Leases of dead processes (pid not alive) are dropped on every acquire.

    .venv/bin/python -m arc_harness.slots --state F --show            # from evals/
    .venv/bin/python -m arc_harness.slots --state F --set http://127.0.0.1:30101=12
"""
import fcntl
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def gateway_workers(gateway="http://127.0.0.1:30100"):
    import httpx
    r = httpx.get(gateway.rstrip("/") + "/workers", timeout=10, trust_env=False)
    r.raise_for_status()
    return [w["url"] for w in r.json()["workers"]]


def parse_spec(spec, gateway="http://127.0.0.1:30100"):
    """'auto:14' (every gateway worker, 14 slots) or 'url=n,url=n'."""
    if spec.startswith("auto:"):
        n = int(spec.split(":", 1)[1])
        return {u: n for u in gateway_workers(gateway)}
    out = {}
    for item in spec.split(","):
        url, n = item.rsplit("=", 1)
        out[url.strip()] = int(n)
    return out


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class SlotPool:
    def __init__(self, state, initial=None, run=""):
        self.state, self.lock_path, self.run = Path(state), Path(str(state) + ".lock"), run
        self.state.parent.mkdir(parents=True, exist_ok=True)
        if initial:
            with self._locked() as st:
                for url, n in initial.items():
                    st["slots"].setdefault(url, n)  # an existing cap (possibly edited live) wins

    @contextmanager
    def _locked(self):
        with open(self.lock_path, "a+") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                st = json.loads(self.state.read_text()) if self.state.exists() and self.state.stat().st_size else {}
                st.setdefault("slots", {})
                st.setdefault("leases", {})
                yield st
                tmp = self.state.with_suffix(".tmp")
                tmp.write_text(json.dumps(st, indent=1))
                tmp.replace(self.state)
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def try_acquire(self):
        """(lease_id, url) on the worker with the most free slots, or None if every worker is full."""
        with self._locked() as st:
            st["leases"] = {k: v for k, v in st["leases"].items() if _alive(v["pid"])}
            used = {u: 0 for u in st["slots"]}
            for v in st["leases"].values():
                used[v["url"]] = used.get(v["url"], 0) + 1
            free = sorted(((n - used.get(u, 0), -used.get(u, 0), u) for u, n in st["slots"].items()), reverse=True)
            if not free or free[0][0] <= 0:
                return None
            url = free[0][2]
            lease = uuid.uuid4().hex
            st["leases"][lease] = {"url": url, "pid": os.getpid(), "run": self.run, "t": round(time.time())}
            return lease, url

    async def acquire(self, poll=5.0):
        import asyncio
        while True:
            got = self.try_acquire()
            if got:
                return got
            await asyncio.sleep(poll)

    def release(self, lease):
        with self._locked() as st:
            st["leases"].pop(lease, None)

    def show(self):
        with self._locked() as st:
            st["leases"] = {k: v for k, v in st["leases"].items() if _alive(v["pid"])}
            used = {}
            for v in st["leases"].values():
                used.setdefault(v["url"], {}).setdefault(v["run"], 0)
                used[v["url"]][v["run"]] += 1
            return {u: {"slots": n, "used": sum(used.get(u, {}).values()), "by_run": used.get(u, {})}
                    for u, n in st["slots"].items()}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True)
    ap.add_argument("--set", help="url=n[,url=n]: change caps live")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    pool = SlotPool(args.state)
    if args.set:
        with pool._locked() as st:
            st["slots"].update(parse_spec(args.set))
    print(json.dumps(pool.show(), indent=1))


if __name__ == "__main__":
    main()
