"""Log DeepSeek balance next to the harness's own cost estimates, every N minutes (never prints the key).

    .venv/bin/python evals/balance_log.py --every 10 >> evals/results/balance_log.tsv
Compare balance drops with est_cost_usd sums (off-peak list prices); a drop ~2x the estimate means peak billing.
The balance lags usage, so compare over long windows only.
"""
import argparse
import glob
import json
import time
from datetime import datetime, timezone

import httpx

from run_deepseek_arc2 import api_key


def estimated_spend():
    total = 0.0
    for path in glob.glob(str(__import__("pathlib").Path(__file__).parent / "results/tools_persistent_*.jsonl")):
        for line in open(path):
            try:
                total += json.loads(line).get("est_cost_usd", 0)
            except json.JSONDecodeError:
                pass
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=10, help="minutes")
    args = ap.parse_args()
    key = api_key()
    while True:
        try:
            r = httpx.get("https://api.deepseek.com/user/balance",
                          headers={"Authorization": f"Bearer {key}"}, timeout=30).json()
            bal = r["balance_infos"][0]["total_balance"]
        except Exception as exc:
            bal = f"error:{type(exc).__name__}"
        print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M}\t{bal}\t{estimated_spend():.2f}", flush=True)
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
