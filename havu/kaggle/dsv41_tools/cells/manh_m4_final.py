# NOT USED (E064, 2026-10-28: superseded by B32-M; kept for reference)
# Cell M4 (E064): final check of submission.json (the fallback if it is invalid); log archive; smoke verdict.
import shutil


def _ok(g):
    return (isinstance(g, list) and 0 < len(g) <= 30 and all(isinstance(r, list) and 0 < len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1 and all(type(v) is int and 0 <= v <= 9 for r in g for v in r))


try:
    sub = json.load(open(f"{WORK}/submission.json"))
    assert set(sub) == set(_tasks), f"task ids differ ({len(sub)} vs {len(_tasks)})"
    for _t, _v in _tasks.items():
        assert len(sub[_t]) == len(_v["test"]), f"{_t}: {len(sub[_t])} rows for {len(_v['test'])} test inputs"
        assert all(_ok(r["attempt_1"]) and _ok(r["attempt_2"]) for r in sub[_t]), f"{_t}: invalid grid"
    print("submission.json valid", flush=True)
except Exception as exc:
    print("submission.json invalid, writing the fallback:", exc, flush=True)
    json.dump(_fb, open(f"{WORK}/submission.json", "w"))
try:
    _logs = f"{WORK}/dsv41_logs"
    if os.path.isdir(f"{_logs}/traces"):
        with tarfile.open(f"{WORK}/dsv41_traces.tar.gz", "w:gz") as _tf:
            _tf.add(f"{_logs}/traces", arcname="traces")
        shutil.rmtree(f"{_logs}/traces", ignore_errors=True)
except Exception as exc:
    print("log archive failed:", exc, flush=True)
if MODE == "smoke":  # commit runs: the smoke must have finished attempts through the harness, without errors
    _recs = []
    try:
        _recs = [json.loads(l) for l in open(f"{WORK}/dsv41_logs/attempts.jsonl")]
    except OSError:
        pass
    _done = [r for r in _recs if not r.get("error")]
    _te = [r.get("token_exact") or {} for r in _done]
    _v = {"attempts": len(_recs), "ok": len(_done), "with_pred": sum(r.get("pred") is not None for r in _done),
          "token_exact_turns": sum(t.get("turns", 0) for t in _te), "resyncs": sum(t.get("resyncs", 0) for t in _te),
          "errors": [r.get("error", "")[:120] for r in _recs if r.get("error")][:5]}
    print("E064 SMOKE:", _v, flush=True)
    json.dump(_v, open(f"{WORK}/e064_smoke.json", "w"), indent=1)
    if not (_v["ok"] and _v["token_exact_turns"]):  # a cut attempt may have no answer yet; it must have run turns
        raise RuntimeError(f"E064 SMOKE FAILED: {_v}")
print(f"done at +{(time.time() - T0) / 3600:.2f}h", flush=True)
