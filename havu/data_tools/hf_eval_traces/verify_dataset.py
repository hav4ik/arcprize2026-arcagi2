"""Verify the local eval-traces dataset before upload: scope, residual sensitive patterns, text dumps for secret_scan.py.

    .venv/bin/python data_tools/hf_eval_traces/verify_dataset.py --out <dataset dir> --dump <tmp dir>
    then: ls <tmp dir>/*.txt <dataset files> | .venv/bin/python data_tools/secret_scan.py -

Prints counts only (never matched values, except for the benign residual-pattern classes listed below).
"""
import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
TEXT_COLS = ["model_detail", "final_content", "error", "messages", "turn_events", "trace_text", "run_config", "record_extra", "trace_file",
             "source_file", "predicted_output"]
CHECKS = {
    "home_path": re.compile(r"/home/\w+"),
    "root_path": re.compile(r"(?<![\w.~/-])/root/"),
    "workspace_path": re.compile(r"(?<![\w.~/-])/workspace"),
    "kaggle_path": re.compile(r"(?<![\w.~/-])/kaggle/"),
    "url_local": re.compile(r"https?://(?:localhost|127\.|\d{1,3}(?:\.\d{1,3}){3})"),
    "ip_port": re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}:\d{2,5}\b"),
    "private_ip": re.compile(r"(?<![\d.])(?:172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}(?![\d.])|127\.0\.0\.1"),
    "vast": re.compile(r"vast\.ai"),
    "ssh_p": re.compile(r"ssh\s+-p"),
    "email": re.compile(r"[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|org|net|edu|io|ai|gov|co|uk|de|ru|cn|me|dev)\b"),
    "user_email": re.compile(r"chankhavu@"),
    "username": re.compile(r"chankhavu"),
    "any_ipv4_meta": None,  # only counted in run_config/record_extra/turn_events
}
IPV4 = re.compile(r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", required=True)
    a = ap.parse_args()
    out, dump = Path(a.out), Path(a.dump)
    dump.mkdir(parents=True, exist_ok=True)
    ev = {Path(p).stem for p in glob.glob(str(ROOT / "data/arc-agi-2-github/data/evaluation/*.json"))}
    test = set(json.loads((ROOT / "data/arc-agi_test_challenges.json").read_text()))
    hits, where = Counter(), Counter()
    vast_ctx = Counter()
    n_rows, ids = 0, set()
    shards = sorted(out.glob("data/*/*.parquet"))
    for sh in shards:
        t = pq.read_table(sh)
        n_rows += t.num_rows
        tids = t.column("task_id").to_pylist()
        ids |= set(tids)
        name = sh.relative_to(out).as_posix().replace("/", "__").replace(".parquet", ".txt")
        with open(dump / name, "w") as fh:
            for col in TEXT_COLS:
                for v in t.column(col).to_pylist():
                    if v is None:
                        continue
                    fh.write(v)
                    fh.write("\n")
                    for k, rx in CHECKS.items():
                        if rx is None:
                            if col in ("run_config", "record_extra", "turn_events"):
                                n = len(IPV4.findall(v))
                                if n:
                                    hits[k] += n
                                    where[(k, col)] += n
                            continue
                        n = len(rx.findall(v))
                        if n:
                            hits[k] += n
                            where[(k, col)] += n
                            if k == "vast":
                                for m in re.finditer(r".{0,25}vast\.ai.{0,10}", v):
                                    vast_ctx[m.group(0)] += 1
    # also the small text files
    for f in ("runs.csv", "README.md"):
        p = out / f
        if p.exists():
            s = p.read_text()
            for k, rx in CHECKS.items():
                if rx is not None and rx.findall(s):
                    where[(k, f)] += len(rx.findall(s))
    print(f"rows {n_rows}, shards {len(shards)}, distinct task ids {len(ids)}")
    print(f"task ids outside the 120 eval tasks: {len(ids - ev)}; in hidden test set ids: {len(ids & test)}")
    print("residual pattern counts by (pattern, column):")
    for (k, c), n in sorted(where.items()):
        print(f"  {k:16s} {c:16s} {n}")
    if vast_ctx:
        print("vast.ai contexts (should be model_detail labels only):", dict(vast_ctx.most_common(5)))
    print("dump files:", len(list(dump.glob("*.txt"))))


if __name__ == "__main__":
    main()
