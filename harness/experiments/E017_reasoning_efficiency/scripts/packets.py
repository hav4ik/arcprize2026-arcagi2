"""Build per-attempt context packets for the annotators: rule references, reference program, answer diff, replay timeline."""
import json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/E017_reasoning_efficiency"
TASKS = ROOT / "data/arc-agi-2-github/data/evaluation"
E7, E15 = {}, {}
for l in open(ROOT / "experiments/E007_failure_analysis/classifications.jsonl"):
    r = json.loads(l); E7.setdefault((r["task_id"]), []).append(r)
for f in ["solved", "hard"]:
    for l in open(ROOT / f"experiments/E015_dsl_analysis/review/{f}.jsonl"):
        r = json.loads(l); E15[r["task"]] = r
REC = {}
for run, fn in {"sub80s40": "tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40.jsonl",
                "sub96n": "tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96n.jsonl"}.items():
    for l in open(ROOT / "evals/results" / fn):
        r = json.loads(l)
        if "error" not in r:
            REC[(run, r["task_id"], r["pair"], r["attempt"])] = r
REPLAY = {}
for fn in ("replay.jsonl", "replay_sample.jsonl"):
    if (OUT / fn).exists():
        for l in open(OUT / fn):
            r = json.loads(l); REPLAY[(r["run"], r["task_id"], r["pair"], r["attempt"])] = r

def g2s(g):
    return "\n".join("".join(str(v) for v in row) for row in g)

def main():
    (OUT / "packets").mkdir(exist_ok=True)
    for l in open(OUT / "sample.jsonl"):
        s = json.loads(l)
        k = (s["run"], s["task_id"], s["pair"], s["attempt"])
        rec = REC[k]
        task = json.loads((TASKS / f"{s['task_id']}.json").read_text())
        truth = task["test"][s["pair"]]["output"]
        pred = rec.get("pred")
        if isinstance(pred, str):
            pred = json.loads(pred)
        lines = [f"# Context for {s['sid']}: task {s['task_id']} test pair {s['pair']} ({len(task['test'])} test pairs), attempt {s['attempt']}, run {s['run']}",
                 f"bucket: {s['bucket']}; outcome: {s['outcome']}; output tokens {s['tokens']}; turns {s['turns']}; tool calls {rec['tool_calls']}; stop {rec['stop_reason']}; answer_source {rec.get('answer_source')}",
                 f"Task JSON (train + test incl. outputs): {TASKS / (s['task_id'] + '.json')}",
                 f"Rendered trace: {OUT / 'rendered' / (s['sid'] + '.txt')}",
                 f"Raw trace (gzip JSON messages): {s['trace_path']}"]
        lines.append("\n## Rule references (from earlier analyses; may be imperfect)")
        for r in E7.get(s["task_id"], [])[:1]:
            lines.append(f"- E007 true_rule: {r['true_rule']}")
        if s["task_id"] in E15:
            lines.append(f"- E015 rule: {E15[s['task_id']].get('rule')}")
        prog = ROOT / f"experiments/E015_dsl_analysis/programs/{s['task_id']}.py"
        if prog.exists():
            lines.append(f"- A correct DeepSeek program for this task (solved at least one test pair): {prog}")
        lines.append("\n## Submitted answer vs truth")
        lines.append(f"truth shape {len(truth)}x{len(truth[0])}")
        if pred:
            lines.append(f"pred shape {len(pred)}x{len(pred[0])}")
            if len(pred) == len(truth) and len(pred[0]) == len(truth[0]):
                diff = sum(a != b for ra, rb in zip(pred, truth) for a, b in zip(ra, rb))
                lines.append(f"cells wrong: {diff} of {len(truth) * len(truth[0])}")
            if s["outcome"] != "correct":
                lines.append("TRUTH (one string per row):\n" + g2s(truth))
                lines.append("PRED:\n" + g2s(pred))
        else:
            lines.append("no answer")
        rp = REPLAY.get(k)
        if rp:
            lines.append("\n## Replay timeline (each run_python cell re-executed offline; after each cell every solve*/transform*/apply*/predict* function was probed)")
            lines.append("cell | turn | cum tokens (% of attempt) | status | best #train pairs reproduced by a probed fn | fns passing ALL train | such fn also right on test | `answer` var == truth")
            for i, t in enumerate(rp["timeline"]):
                if not t.get("probe"):
                    lines.append(f"{i} | {t['turn']} | {t['cum_tok']} | {t['status']} | probe failed"); continue
                lines.append(f"{i} | {t['turn']} | {t['cum_tok']} ({100*t['cum_tok']/max(1,s['tokens']):.0f}%) | {t['status']} | {t['best_train']}/{t['n_train']} | {','.join(t['full_train']) or '-'} | {t['full_train_test_ok']} | {t['answer_ok']}")
            lines.append("(Functions with other names, or inline checks, are not probed: a model can pass train without showing here.)")
        (OUT / "packets" / f"{s['sid']}.md").write_text("\n".join(lines) + "\n")
main()
