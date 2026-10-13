"""E020: keep worker-slot caps so every box runs <= PER_BOX xhigh-96k attempts, counting legacy gateway processes.

Every 2 minutes (gateway load sampled every 20 s; the max over 10 min is used):
- legacy processes (they go through the gateway, not slot-aware; given as file:concurrency:jobs) have
  min(concurrency, unfinished jobs) attempts in flight, spread over the LEGACY_WORKERS boxes (assumed even);
- cap(legacy box) = max(0, PER_BOX - ceil(legacy in flight / n legacy boxes));
- cap(any other worker in slots.json, e.g. a newly added box) = PER_BOX;
- with --raise: a box whose KV peak (evals/results/cache_gw_dev56.jsonl, worker_cache_log) stayed < 0.75 over the
  last 30 min may go to RAISE_TO.

    cap_controller.py --per-box 10 --legacy-workers http://127.0.0.1:30101,... file:conc:jobs ...
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "experiments/E019_qwen_program_answers/router/slots.json"
CACHE_LOG = ROOT / "evals/results/cache_gw_dev56.jsonl"
PORT_NAMES = {}  # worker URL -> worker name in the cache log, filled from router/workers.json when available
sys.path.insert(0, str(ROOT / "evals"))
from arc_harness.slots import SlotPool, gateway_workers  # noqa: E402


def kv_ok(url, since):
    """True if the cache log has records for this worker in the window and all KV peaks < 0.75."""
    name = PORT_NAMES.get(url)
    if not name or not CACHE_LOG.exists():
        return False
    recs = [json.loads(line) for line in CACHE_LOG.read_text().splitlines() if line.strip()]
    rs = [r for r in recs if r.get("worker") == name and r.get("t", 0) >= since and "error" not in r]
    return len(rs) >= 5 and all(r.get("max_token_usage", r.get("token_usage", 1)) < 0.75 for r in rs)


def latest_kv(url):
    """KV usage of the box's latest cache-log record (within 12 min), or None."""
    name = PORT_NAMES.get(url)
    if not name or not CACHE_LOG.exists():
        return None
    last = None
    for line in CACHE_LOG.read_text().splitlines()[-200:]:
        if line.strip():
            r = json.loads(line)
            if r.get("worker") == name and "error" not in r:
                last = r
    return last.get("token_usage") if last and last["t"] > time.time() - 720 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("legacy", nargs="*", help="file:concurrency:jobs of gateway (non-slot) processes")
    ap.add_argument("--per-box", type=int, default=10)
    ap.add_argument("--state", default=str(STATE), help="slot state file whose caps this controller sets")
    ap.add_argument("--other-state", action="append", default=[],
                    help="other slot state files: their live leases count as load on each box (repeatable)")
    ap.add_argument("--raise-to", type=int, default=0, help="allow this cap on boxes with KV peak < 0.75 for 30 min")
    ap.add_argument("--legacy-workers", default="gateway",
                    help="boxes the legacy processes use: 'gateway' (all gateway workers, re-read) or url,url")
    args = ap.parse_args()
    legacy = [(Path(f), int(c), int(n)) for f, c, n in (x.rsplit(":", 2) for x in args.legacy)]
    lw = gateway_workers() if args.legacy_workers == "gateway" else args.legacy_workers.split(",")
    try:
        reg = json.loads((ROOT / "experiments/E019_qwen_program_answers/router/workers.json").read_text())
        for name, w in reg.items():  # {name: {"gateway_url": ...}}
            if isinstance(w, dict) and w.get("gateway_url"):
                PORT_NAMES[w["gateway_url"]] = name
    except Exception:
        pass
    pool = SlotPool(args.state)
    others = [SlotPool(f) for f in args.other_state]
    last, history = None, {}
    while True:
        inflight = 0
        for f, c, n in legacy:
            done = sum(1 for line in f.read_text().splitlines() if line.strip()) if f.exists() else 0
            inflight += min(c, max(0, n - done))
        if args.legacy_workers == "gateway":  # legacy attempts may land on any gateway worker (min_load)
            try:
                lw = gateway_workers()
            except Exception:
                pass
        per_legacy = math.ceil(inflight / max(1, len(lw))) if inflight else 0
        # per-box gateway load: the MAX over the last 10 min of samples every 20 s (an attempt waiting on a tool
        # call holds no request, so single samples undercount and caps would oscillate)
        for _ in range(6):
            try:
                import httpx
                ws = httpx.get("http://127.0.0.1:30100/workers", timeout=10, trust_env=False).json()["workers"]
                now = time.time()
                for w in ws:
                    history.setdefault(w["url"], []).append((now, w["load"]))
            except Exception:
                pass
            time.sleep(20)
        cutoff = time.time() - 600
        loads = {u: max(l for t, l in h if t >= cutoff) for u, h in history.items() if any(t >= cutoff for t, _ in h)}
        other_used = {}
        for o in others:
            for u, v in o.show().items():
                other_used[u] = other_used.get(u, 0) + v["used"]
        with pool._locked() as st:
            caps = {}
            for url in st["slots"]:
                # legacy attempts on this box: the gateway's in-flight requests there (slot attempts bypass the
                # gateway), +10% for attempts waiting on a tool call; never below the even share on legacy boxes
                on_box = math.ceil(1.1 * loads.get(url, 0)) if loads else per_legacy
                if url in lw and loads == {}:
                    on_box = per_legacy
                base = max(0, args.per_box - on_box - other_used.get(url, 0)) if inflight else \
                    max(0, args.per_box - other_used.get(url, 0))
                if args.raise_to and kv_ok(url, time.time() - 1800):
                    base = max(base, (args.raise_to - per_legacy) if url in lw else args.raise_to)
                kv = latest_kv(url)
                if kv is not None and kv > 0.85:  # KV guard: no new attempts on a box above 0.85
                    base = 0
                caps[url] = base
            st["slots"].update(caps)
        if caps != last:
            print(time.strftime("%H:%M", time.gmtime()), f"legacy in flight {inflight} -> caps {caps}", flush=True)
            last = caps


if __name__ == "__main__":
    main()
