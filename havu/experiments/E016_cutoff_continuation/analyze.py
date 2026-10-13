"""E016 summary: per-attempt table, per-arm totals, and s20x60 pass@1/pass@2 with rescued answers substituted."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import continue_cutoffs as cc  # noqa: E402

ARMS = ["toolnote", "usermsg", "toolnote_think"]


def main():
    rows = [json.loads(line) for line in (HERE / "results/continuations.jsonl").read_text().splitlines()]
    best = {}
    for r in rows:
        if "error" not in r:
            best[(r["task_id"], r["pair"], r["attempt"], r["variant"])] = r
    recs = [json.loads(line) for line in cc.RESULTS.read_text().splitlines() if line.strip()]
    latest = {(r["task_id"], r["pair"], r["attempt"]): r for r in recs}
    tasks = cc.load_tasks(Path(recs[0]["config"]["challenges"]))
    cut = cc.cutoff_records()

    print("| task.pair#att | cells replayed (diverged) | probe: answer var / best fn train fit / test ok | "
          + " | ".join(f"{a}: action, answer, correct, in/out tok, $" for a in ARMS) + " |")
    print("|" + "---|" * (3 + len(ARMS)))
    for c in cut:
        k = (c["task_id"], c["pair"], c["attempt"])
        b = best.get(k + ("baseline",), {})
        bf = b.get("best_function") or {}
        cells = [f"{k[0]}.{k[1]}#{k[2]}", f"{b.get('replay', {}).get('cells')} ({b.get('replay', {}).get('diverged')})",
                 f"{'yes' if b.get('answer_present') else 'no'} / {bf.get('name', '-')} {bf.get('train_ok', 0)}/"
                 f"{b.get('n_train_train', '?')} / {'yes' if bf.get('test_correct') or b.get('any_function_correct') else 'no'}"]
        for a in ARMS:
            r = best.get(k + (a,))
            if r is None:
                cells.append("-")
                continue
            acts = "+".join(x.split("[")[0] for x in r["actions"])
            cells.append(f"{acts}, {r['source'] or 'none'}, {'**yes**' if r['correct'] else 'no'}, "
                         f"{r['usage']['prompt'] / 1000:.1f}k/{r['usage']['completion']}, {r['cost_usd']:.4f}")
        print("| " + " | ".join(cells) + " |")
    print()
    for a in ARMS + ["baseline"]:
        rs = [best[(c["task_id"], c["pair"], c["attempt"], a)] for c in cut if (c["task_id"], c["pair"], c["attempt"], a) in best]
        if a == "baseline":
            print(f"baseline probe: {len(rs)} done; `answer` set in {sum(r['answer_present'] for r in rs)}; "
                  f"fn fitting all train in {sum(bool(r['functions_fitting_all_train']) for r in rs)}; "
                  f"correct (answer or best fn): {sum(r['answer_correct'] or (r['best_function'] or {}).get('test_correct', False) for r in rs)}; "
                  f"any fn correct: {sum(r['any_function_correct'] for r in rs)}; replay divergences "
                  f"{sum(r['replay']['diverged'] for r in rs)}/{sum(r['replay']['cells'] for r in rs)} cells")
            continue
        u = {k: sum(r["usage"][k] for r in rs) for k in ("prompt", "cache_hit", "completion")}
        print(f"{a}: {len(rs)} done, answered {sum(r['pred'] is not None for r in rs)} "
              f"(submit {sum(r['source'] == 'submit' for r in rs)}, session {sum(r['source'] == 'session' for r in rs)}, "
              f"parse {sum(r['source'] == 'parse' for r in rs)}), correct {sum(r['correct'] for r in rs)}, "
              f"input {u['prompt'] / 1e3:.0f}k (hit {u['cache_hit'] / 1e3:.0f}k), output {u['completion'] / 1e3:.1f}k, "
              f"${sum(r['cost_usd'] for r in rs):.4f}")
    print(f"total recorded spend ${sum(r.get('cost_usd', 0) for r in rows):.4f} (all rows incl. the 1-record test)")

    def scores(override):
        correct = {k: bool(r.get("correct")) for k, r in latest.items()}
        correct.update(override)
        out = {}
        for name, attempts in (("pass@1 a0", [0]), ("pass@1 a1", [1]), ("pass@2", [0, 1])):
            per_task, hits, n = [], 0, 0
            for t, task in tasks.items():
                ok = [any(correct.get((t, p, a), False) for a in attempts) for p in range(len(task["test"]))]
                per_task.append(sum(ok) / len(ok))
                hits, n = hits + sum(ok), n + len(ok)
            out[name] = (sum(per_task) / len(per_task), hits / n)
        out["pass@1 mean"] = tuple((out["pass@1 a0"][i] + out["pass@1 a1"][i]) / 2 for i in range(2))
        return out

    base = scores({})
    print("\n| arm | pass@1 mean (task-wt / pair-wt) | pass@2 (task-wt / pair-wt) |")
    print("|---|---|---|")
    for name, override in [("s20x60 as recorded", {})] + [
            (f"+ {a} rescues", {(c["task_id"], c["pair"], c["attempt"]): best[(c["task_id"], c["pair"], c["attempt"], a)]["correct"]
                                for c in cut if (c["task_id"], c["pair"], c["attempt"], a) in best}) for a in ARMS]:
        s = scores(override) if override else base
        print(f"| {name} | {s['pass@1 mean'][0]:.2%} / {s['pass@1 mean'][1]:.2%} | {s['pass@2'][0]:.2%} / {s['pass@2'][1]:.2%} |")
    for a in ARMS:
        for c in cut:
            k = (c["task_id"], c["pair"], c["attempt"])
            r = best.get(k + (a,))
            if r and r["correct"]:
                other = latest.get((k[0], k[1], 1 - k[2]), {}).get("correct")
                print(f"{a} rescued {k}; other attempt on this pair correct: {other}")


if __name__ == "__main__":
    main()
