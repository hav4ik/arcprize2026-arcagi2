"""Show one attempt of an eval run: reasoning, tool calls/results, final answer, marked CORRECT/WRONG against the right key.

  .venv/bin/python data_tools/show_trace.py RUN_ID --task 16de56c4 --attempt 0 [--pair 0]
  .venv/bin/python data_tools/show_trace.py RUN_ID --wrong-only          # list failed attempts of the run
  .venv/bin/python data_tools/show_trace.py RUN_ID --task T --download   # fetch from livobereshka/arc-agi2-eval-traces-private

RUN_ID is the run_id column of reports/eval_runs.csv (a unique prefix is enough). The key follows the run's `version` column
(G = GitHub key, K = Kaggle key). Without --download the local workspace files are used. Uses HF_TOKEN or SECRETS.yaml
(huggingface_readonly_api_key, else huggingface_api_key) for --download; the token is never printed.
"""
import argparse
import csv
import glob
import gzip
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "livobereshka/arc-agi2-eval-traces-private"


def P(*a):
    return os.path.join(ROOT, *a)


def find_run(rid):
    rows = list(csv.DictReader(open(P("reports/eval_runs.csv"))))
    ex = [r for r in rows if r["run_id"] == rid]
    cand = ex or [r for r in rows if r["run_id"].startswith(rid)]
    if len(cand) != 1:
        sys.exit("run_id %r matches %d runs: %s" % (rid, len(cand), [r["run_id"] for r in cand[:10]]))
    return cand[0]


def token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    txt = open(P("SECRETS.yaml")).read()
    for k in ("huggingface_readonly_api_key", "huggingface_api_key"):
        m = re.search(r"^%s:\s*(\S+)" % k, txt, re.M)
        if m:
            return m.group(1).strip("\"'")


def run_dir(row):
    idx = {r["source_record_file"]: r["path"] for r in csv.DictReader(open(P("reports/hf_private_index.csv"))) if r["source_record_file"]}
    url = row["hf_traces"]
    if "eval-traces-private/tree/main/" in url:
        return url.split("tree/main/")[1]
    return idx.get(row["path"])


def eq(a, b):
    return json.dumps(a) == json.dumps(b)


def keys():
    gh, kg = {}, {}
    for f in glob.glob(P("data/arc-agi-2-github/data/evaluation/*.json")):
        gh[os.path.basename(f)[:-5]] = [t["output"] for t in json.load(open(f))["test"]]
    f = P("data/arc-agi_evaluation_solutions.json")
    if os.path.exists(f):
        kg = json.load(open(f))
    return gh, kg


def load_records(row, local_root):
    path = os.path.join(local_root, row["path"]) if local_root else P(row["path"])
    if local_root and not os.path.exists(path):
        path = os.path.join(local_root, run_dir(row), os.path.basename(row["path"]))
    return path, [json.loads(l) for l in open(path) if l.strip()]


def verdict(rec, version, gh, kg):
    key = (gh if version.lower().startswith("g") else kg)
    t, p = rec["task_id"], int(rec.get("pair", 0) or 0)
    pred = rec.get("pred")
    if pred is None:
        return "NO ANSWER"
    if t in key and p < len(key[t]):
        return "CORRECT" if eq(pred, key[t][p]) else "WRONG"
    return "%s (recorded; key not found)" % ("CORRECT" if rec.get("correct") else "WRONG")


def show_msgs(m):
    for x in m["messages"]:
        role = x["role"]
        if role == "assistant":
            if x.get("reasoning_content"):
                print("--- assistant reasoning ---\n" + x["reasoning_content"])
            if x.get("content"):
                print("--- assistant says ---\n" + str(x["content"]))
            for c in x.get("tool_calls") or []:
                f = c.get("function", {})
                a = f.get("arguments", "")
                try:
                    a = json.loads(a)
                    a = a.get("code") or a.get("expr") or json.dumps(a)
                except Exception:
                    pass
                print("--- tool call %s ---\n%s" % (f.get("name"), a))
        elif role == "tool":
            print("--- tool result ---\n" + str(x.get("content"))[:4000])
        elif role == "user":
            c = x.get("content")
            c = c if isinstance(c, str) else json.dumps(c)
            print("--- %s (first 600 chars) ---\n%s" % (role, c[:600]))
        else:
            print("--- %s ---\n%s" % (role, str(x.get("content"))[:300]))


def show_attempt(row, rec, tdir, version, gh, kg):
    t, p, a = rec["task_id"], rec.get("pair", 0), rec.get("attempt", 0)
    print("=" * 100)
    print("run %s | task %s pair %s attempt %s | %s | tokens %s | turns %s" % (
        row["run_id"], t, p, a, verdict(rec, version, gh, kg), (rec.get("usage") or {}).get("completion"), rec.get("turns")))
    pat = "%s_p%s_a%s" % (t, p, a)
    files = sorted(glob.glob(os.path.join(tdir, pat + "[._]*"))) if tdir else []
    if not files:
        print("(no trace file for %s in %s)" % (pat, tdir))
    for f in files:
        if f.endswith("messages.json.gz"):
            show_msgs(json.load(gzip.open(f)))
        elif f.endswith(".gz"):
            print("--- reasoning ---\n" + gzip.open(f, "rt", errors="replace").read())
    print("--- final answer (pred) ---\n" + json.dumps(rec.get("pred")))
    if rec.get("content") and not files:
        print("--- final content ---\n" + str(rec["content"])[:3000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--task")
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--attempt", type=int, default=0)
    ap.add_argument("--wrong-only", action="store_true")
    ap.add_argument("--download", action="store_true", help="fetch from the private HF repo into --cache first")
    ap.add_argument("--cache", default="/tmp/arc_traces_cache")
    a = ap.parse_args()
    row = find_run(a.run_id)
    version = row["version"]
    gh, kg = keys()
    local_root = None
    if a.download:
        from huggingface_hub import snapshot_download
        d = run_dir(row)
        if not d:
            sys.exit("no private-repo directory for this run (serving-lab replica?): " + row["hf_traces"])
        pat = ["%s/*" % d] if not a.task else ["%s/*.jsonl" % d, "%s/*/%s_p*" % (d, a.task), "%s/%s_p*" % (d, a.task), "%s/traces.parquet" % d]
        snapshot_download(REPO, repo_type="dataset", allow_patterns=pat, local_dir=a.cache, token=token())
        local_root = a.cache
    if "traces.parquet" in " ".join(os.listdir(os.path.join(local_root, run_dir(row)))) if local_root and run_dir(row) and os.path.isdir(os.path.join(local_root, run_dir(row))) else False:
        sys.exit("this run is a parquet copy: open %s/traces.parquet with pandas (columns task_id, pair, attempt, correct, messages, trace_text)" % os.path.join(local_root, run_dir(row)))
    path, recs = load_records(row, local_root)
    stem = path[:-6] if path.endswith(".jsonl") else path
    tdir = stem if os.path.isdir(stem) else os.path.dirname(path)
    print("run %s (version %s), %d records, records %s" % (row["run_id"], version, len(recs), path), file=sys.stderr)
    if a.wrong_only:
        n = 0
        for r in recs:
            v = verdict(r, version, gh, kg)
            if v != "CORRECT" and not v.startswith("CORRECT"):
                n += 1
                print("%s p%s a%s %s  tokens=%s" % (r["task_id"], r.get("pair", 0), r.get("attempt", 0), v, (r.get("usage") or {}).get("completion")))
        print("%d of %d attempts not correct" % (n, len(recs)), file=sys.stderr)
        return
    if not a.task:
        sys.exit("give --task (and --attempt), or --wrong-only")
    sel = [r for r in recs if r["task_id"] == a.task and int(r.get("pair", 0) or 0) == a.pair and int(r.get("attempt", 0) or 0) == a.attempt]
    if not sel:
        sys.exit("no record for task %s pair %d attempt %d" % (a.task, a.pair, a.attempt))
    show_attempt(row, sel[-1], tdir, version, gh, kg)


if __name__ == "__main__":
    main()
