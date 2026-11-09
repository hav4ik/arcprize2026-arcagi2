# B32-M smoke verdict (commit runs only): RAISES on any failure the B32-M report cell found, so the commit run errors.
import os
if not os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):
    _bad = globals().get("B32M_BAD", ["the B32-M report cell did not run"])
    if _bad:
        print("=" * 100, f"B32-M SMOKE FAILED: {_bad}", "=" * 100, sep="\n", flush=True)
        raise RuntimeError(f"B32-M SMOKE FAILED: {_bad}")
    print("B32-M SMOKE OK", flush=True)
