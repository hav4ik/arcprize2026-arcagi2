"""Watch our Kaggle competition submissions and time them (Kaggle does not show how long a rerun has been going).

Read-only towards Kaggle: it only lists submissions. It never submits, pushes or cancels anything.

    .venv/bin/python kaggle/monitor_submissions.py --once                       # table, then exit
    nohup .venv/bin/python kaggle/monitor_submissions.py --until-done --interval 300 \
        >> kaggle/runs/submission_watch.log 2>&1 &  echo $! > kaggle/runs/submission_watch.pid

State: kaggle/runs/submission_watch.jsonl, one line per observed status change (submission id, description,
submitted_at UTC, status, observed_at, elapsed since submitted, score). Elapsed time is measured from Kaggle's
submission time, so it includes any queueing before the notebook starts; the notebook's own guard (11.75 h after
its first cell) therefore fires at ~11.75 h elapsed or a little later.

Auth: KAGGLE_API_TOKEN from the environment; if unset, it is read from SECRETS.yaml (`kaggle_api_token`) into
this process's environment only. The token is never printed or written anywhere.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "arc-prize-2026-arc-agi-2"
STATE = ROOT / "kaggle" / "runs" / "submission_watch.jsonl"
PENDING = "PENDING"
UTC = dt.timezone.utc


# ---------------------------------------------------------------- parsing (pure; unit-tested offline)

def _get(obj, *names, default=None):
    for n in names:
        v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
        if v is not None:
            return v
    return default


def parse_status(v) -> str:
    """SubmissionStatus.PENDING / enum / 'pending' / 0 -> 'PENDING' (also COMPLETE, ERROR)."""
    if v is None:
        return "UNKNOWN"
    name = getattr(v, "name", None)
    if isinstance(name, str):
        return name.upper()
    if isinstance(v, int):
        return {0: "PENDING", 1: "COMPLETE", 2: "ERROR"}.get(v, str(v))
    s = str(v).strip()
    return s.rsplit(".", 1)[-1].upper() or "UNKNOWN"


def parse_time(v) -> dt.datetime | None:
    """Kaggle returns naive datetimes (or strings) in UTC."""
    if v in (None, ""):
        return None
    if isinstance(v, dt.datetime):
        t = v
    else:
        s = str(v).strip().replace("Z", "+00:00")
        try:
            t = dt.datetime.fromisoformat(s)
        except ValueError:
            t = dt.datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def parse_score(v):
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def normalize(sub) -> dict:
    t = parse_time(_get(sub, "date", "submitted_at"))
    return {
        "id": str(_get(sub, "ref", "id", default="")),
        "description": str(_get(sub, "description", default="") or ""),
        "submitted_at": t.isoformat() if t else None,
        "status": parse_status(_get(sub, "status")),
        "public_score": parse_score(_get(sub, "public_score", "publicScore")),
        "private_score": parse_score(_get(sub, "private_score", "privateScore")),
        "error": str(_get(sub, "error_description", "errorDescription", default="") or ""),
    }


def hours_since(iso: str | None, now: dt.datetime) -> float | None:
    if not iso:
        return None
    return (now - dt.datetime.fromisoformat(iso)).total_seconds() / 3600


def fmt_h(h) -> str:
    return "?" if h is None else f"{h:.2f} h"


# ---------------------------------------------------------------- state

class Watch:
    """Tracks last observed status per submission; appends one JSONL record per change."""

    def __init__(self, state_path: Path, interval: float, warn_hours: float, guard_hours: float,
                 out=None, write=True):
        self.path = Path(state_path)
        self.interval = interval
        self.warn_hours = warn_hours
        self.guard_hours = guard_hours
        self.out = out or sys.stdout
        self.write = write
        self.last: dict[str, dict] = {}         # id -> last record written (from file + this run)
        self.last_pending_seen: dict[str, str] = {}  # id -> last observed_at while PENDING (this run)
        self.warned: set[str] = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.last[r["id"]] = r
                    if r.get("status") == PENDING:
                        self.last_pending_seen[r["id"]] = r["observed_at"]

    def say(self, msg: str, now: dt.datetime):
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%SZ')}] {msg}", file=self.out, flush=True)

    def update(self, subs: list[dict], now: dt.datetime) -> list[dict]:
        """Process one poll's normalized submissions. Returns the change records (also appended to the file)."""
        changes = []
        for s in subs:
            sid, st = s["id"], s["status"]
            el = hours_since(s["submitted_at"], now)
            prev = self.last.get(sid)
            if prev is None or prev["status"] != st or prev.get("public_score") != s["public_score"]:
                rec = {"id": sid, "description": s["description"], "submitted_at": s["submitted_at"],
                       "status": st, "observed_at": now.isoformat(),
                       "elapsed_h": None if el is None else round(el, 4),
                       "public_score": s["public_score"], "private_score": s["private_score"],
                       "error": s["error"] or None, "prev_status": prev["status"] if prev else None}
                if prev and prev["status"] == PENDING and st != PENDING:
                    lo = hours_since(s["submitted_at"],
                                     dt.datetime.fromisoformat(self.last_pending_seen.get(sid, prev["observed_at"])))
                    rec["wall_h_upper"] = rec["elapsed_h"]
                    rec["wall_h_lower"] = None if lo is None else round(lo, 4)
                changes.append(rec)
                self.last[sid] = rec
                if self.write:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                self.say(self.describe(rec), now)
            if st == PENDING:
                self.last_pending_seen[sid] = now.isoformat()
                if el is not None and el >= self.warn_hours and sid not in self.warned:
                    self.warned.add(sid)
                    self.say(f"WARNING {sid} still PENDING at {fmt_h(el)} (> --warn-hours {self.warn_hours}); "
                             f"notebook guard at ~{self.guard_hours} h", now)
        return changes

    def describe(self, r: dict) -> str:
        d = r["description"][:60]
        if r.get("prev_status") == PENDING and r["status"] != PENDING:
            lo, hi = r.get("wall_h_lower"), r.get("wall_h_upper")
            wall = f"wall {fmt_h(hi)}" if lo is None else \
                f"wall {lo:.2f}-{hi:.2f} h (done between last PENDING poll and this one; +-{self.interval / 60:.0f} min)"
            extra = f" error='{r['error']}'" if r.get("error") else ""
            return (f"DONE {r['id']} {r['status']} public={r['public_score']} private={r['private_score']} "
                    f"{wall}; submitted {r['submitted_at']}{extra} | {d}")
        if r["status"] == PENDING:
            return f"NEW {r['id']} PENDING {fmt_h(r['elapsed_h'])} since submit ({r['submitted_at']}) | {d}"
        return (f"SEEN {r['id']} {r['status']} public={r['public_score']} (prev={r.get('prev_status')}, "
                f"submitted {r['submitted_at']}) | {d}")

    def heartbeat(self, subs: list[dict], now: dt.datetime):
        pend = [s for s in subs if s["status"] == PENDING]
        if not pend:
            self.say("heartbeat: no pending submissions", now)
        for s in pend:
            el = hours_since(s["submitted_at"], now)
            warn = " WARNING over --warn-hours" if el is not None and el >= self.warn_hours else ""
            self.say(f"heartbeat {s['id']} pending {fmt_h(el)}; notebook guard at ~{self.guard_hours} h{warn} "
                     f"| {s['description'][:60]}", now)


def table(subs: list[dict], now: dt.datetime) -> str:
    rows = [("id", "submitted_at (UTC)", "elapsed", "status", "public", "private", "description / error")]
    for s in subs:
        el = hours_since(s["submitted_at"], now)
        rows.append((s["id"], (s["submitted_at"] or "?")[:19].replace("T", " "),
                     fmt_h(el) if s["status"] == PENDING else "-", s["status"],
                     "" if s["public_score"] is None else str(s["public_score"]),
                     "" if s["private_score"] is None else str(s["private_score"]),
                     s["description"][:70] + (f" [error: {s['error']}]" if s["error"] else "")))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(r[:-1])) + "  " + r[-1] for r in rows)


# ---------------------------------------------------------------- Kaggle API (network)

def load_token():
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    p = ROOT / "SECRETS.yaml"
    if p.exists():
        m = re.search(r"^kaggle_api_token:\s*(.+)$", p.read_text(), re.M)
        if m:
            os.environ["KAGGLE_API_TOKEN"] = m.group(1).strip().strip("\"' ")


def make_api():
    load_token()
    from kaggle.api.kaggle_api_extended import KaggleApi  # does not auto-authenticate (unlike `import kaggle`)
    api = KaggleApi()
    api.authenticate()
    return api


def fetch(api, competition: str, page_size: int = 50) -> list[dict]:
    return [normalize(s) for s in (api.competition_submissions(competition, page_size=page_size) or []) if s]


def fetch_with_retry(api_factory, state: dict, competition: str, log, max_backoff: float = 1800):
    """Retry with exponential backoff (30 s, 60 s, ... capped) until a poll succeeds."""
    delay = 30.0
    while True:
        try:
            if state.get("api") is None:
                state["api"] = api_factory()
            return fetch(state["api"], competition)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001  (network / auth / rate limit; message never contains the token)
            msg = str(e).replace(os.environ.get("KAGGLE_API_TOKEN", "\0"), "<token>")
            log(f"API error ({type(e).__name__}: {msg[:200]}); retry in {delay:.0f} s")
            if "401" in msg or "Unauthorized" in msg:
                state["api"] = None
            time.sleep(delay)
            delay = min(delay * 2, max_backoff)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--competition", default=COMPETITION)
    ap.add_argument("--once", action="store_true", help="print the current table and exit (no state written)")
    ap.add_argument("--interval", type=float, default=300, help="poll interval in seconds (min 60)")
    ap.add_argument("--until-done", action="store_true", help="exit when no submission is pending")
    ap.add_argument("--warn-hours", type=float, default=11.0)
    ap.add_argument("--guard-hours", type=float, default=11.75, help="notebook self-stop, for messages only")
    ap.add_argument("--heartbeat-min", type=float, default=30)
    ap.add_argument("--state", type=Path, default=STATE)
    a = ap.parse_args(argv)
    a.interval = max(a.interval, 60.0)
    now = lambda: dt.datetime.now(UTC)  # noqa: E731

    if a.once:
        subs = fetch(make_api(), a.competition)
        print(table(subs, now()))
        return 0

    w = Watch(a.state, a.interval, a.warn_hours, a.guard_hours)
    w.say(f"watching {a.competition} every {a.interval:.0f} s (until_done={a.until_done}, "
          f"warn {a.warn_hours} h, pid {os.getpid()})", now())
    api_state: dict = {}
    last_hb = 0.0
    while True:
        subs = fetch_with_retry(make_api, api_state, a.competition, lambda m: w.say(m, now()))
        t = now()
        w.update(subs, t)
        if time.time() - last_hb >= a.heartbeat_min * 60:
            w.heartbeat(subs, t)
            last_hb = time.time()
        if a.until_done and not any(s["status"] == PENDING for s in subs):
            w.say("no pending submissions; exiting (--until-done)\n" + table(subs, t), t)
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
