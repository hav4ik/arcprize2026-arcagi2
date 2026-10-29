"""E066 full-model reference: task-weighted pass@1/pass@2 and per-attempt stats of an E058-harness result file.

Usage: summarize.py RESULTS.jsonl [MORE.jsonl ...]
Latest non-error record per (task, pair, attempt); an attempt with only error records counts as wrong. Multi-test attempts write one record per pair with usage on pair 0
and the same turns/tool counts on every pair, so per-attempt stats use the lowest pair's record.
"""
import json
import sys
from collections import Counter, defaultdict


def main(paths):
    recs, errs = {}, Counter()
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            k = (r["task_id"], int(r["pair"]), int(r["attempt"]))
            if r.get("error"):
                errs[k] += 1
                continue
            recs[k] = r
    pairs = defaultdict(set)
    for t, p, a in list(recs) + list(errs):  # an attempt that only errored counts as wrong
        pairs[t].add(p)
    missing = [k for k in errs if k not in recs]
    s1, s2 = [], []
    for t in sorted(pairs):
        ps = sorted(pairs[t])
        ok = lambda p, a: bool(recs.get((t, p, a), {}).get("correct"))
        s2.append(sum(ok(p, 0) or ok(p, 1) for p in ps) / len(ps))
        s1.append(sum((ok(p, 0) + ok(p, 1)) / 2 for p in ps) / len(ps))
    att = defaultdict(dict)
    for (t, p, a), r in recs.items():
        att[t, a][p] = r
    tok, turns, calls, terr, stop, src, secs, tpass = [], [], [], [], Counter(), Counter(), [], 0
    for (t, a), d in att.items():
        r0 = d[min(d)]
        tok.append(sum(r["usage"]["completion"] for r in d.values()))
        turns.append(r0.get("turns") or 0)
        calls.append(r0.get("tool_calls") or 0)
        terr.append(r0.get("tool_errors") or 0)
        stop[r0.get("stop_reason")] += 1
        src[r0.get("answer_source")] += 1
        secs.append(r0.get("seconds") or 0)
        tpass += bool(r0.get("train_pass"))
    n = len(att)
    full = sum(all((t, p, a) in recs for p in pairs[t] for a in (0, 1)) for t in pairs)
    print(f"records {len(recs)} (error records {sum(errs.values())}, keys only errored {len(missing)}); "
          f"tasks {len(pairs)} ({full} with both attempts of every pair); attempts {n}")
    print(f"task-weighted pass@1 {sum(s1) / len(s1):.1%}  pass@2 {sum(s2) / len(s2):.1%}  "
          f"(pass@2 sum {sum(s2):.2f} / {len(s2)}; errored attempts count as wrong)")
    print(f"per attempt: output tokens mean {sum(tok) / n / 1000:.1f}k, turns {sum(turns) / n:.2f}, "
          f"tool calls {sum(calls) / n:.2f}, tool errors {sum(terr) / n:.2f} "
          f"(tool-error rate {sum(terr) / max(1, sum(calls)):.1%} of calls), seconds mean {sum(secs) / n:.0f}, "
          f"train_pass {tpass / n:.1%}")
    print("stop reasons", dict(stop.most_common()))
    print("answer source", dict(src.most_common()))


if __name__ == "__main__":
    main(sys.argv[1:])
