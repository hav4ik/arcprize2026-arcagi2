"""Direct-answer mode: the model reasons (e.g. Qwen3.8 at reasoning_effort xhigh) and
writes the test output grid(s) itself -- no tools. Two samples per task give the two
attempts. This is how headline ARC-AGI-2 numbers for chat models are usually measured,
so it is the comparison the REPL agent has to beat.

    python run_direct.py '<json config>'

Config: base_url, model, task_ids, samples (2), concurrency, max_tokens, deadline_epoch,
temperature/top_p/top_k, template_kwargs, out, data_dir.
Writes <out> (JSON) and <out>.jsonl (one line per finished sample).
"""
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import ChatClient                       # noqa: E402
from grids import show                              # noqa: E402

# grids.SYSTEM tells the model it solves puzzles "by writing Python programs" -- right for
# the REPL agent, wrong when the answer is the grid itself.
DIRECT_SYSTEM = (
    "You are an expert at solving ARC-AGI puzzles. Each puzzle has a few input->output "
    "example pairs that share one hidden transformation rule. Grids are 2D arrays of digits "
    "0-9 (colours); 0 is usually the background."
)

# A reply cut off before any grid (the thinking budget missing or ignored, or the answer
# itself cut) gets ONE short follow-up without thinking: the tail of its own reasoning,
# and an order to commit. An unanswered task scores nothing; a committed guess may score.
RESCUE = """{prompt}

## Your notes so far (they were cut off)
{notes}

Stop analysing now. Apply the rule you were converging on and give the final test output grid(s): each as rows of digits in its own ```grid code block, in test order, with nothing else in the block."""

DIRECT_INSTRUCTIONS = """Solve this ARC puzzle. The examples share one hidden transformation rule. Work out the rule, check it against every example, then apply it to the test input(s).

At the very end of your answer, give each test output grid as rows of digits inside its own ```grid code block, in test order, with nothing else in the block."""


def direct_prompt(task):
    parts = [DIRECT_INSTRUCTIONS, ""]
    for i, p in enumerate(task["train"], 1):
        parts += [f"## Example {i}", show(p["input"], "Input"), show(p["output"], "Output"), ""]
    for j, t in enumerate(task["test"], 1):
        parts += [f"## Test input {j}", show(t["input"], "Input"), ""]
    return "\n".join(parts).rstrip()


def parse_grids(text, n_tests):
    """The last n_tests digit-only code blocks (```grid preferred, any fence accepted)."""
    blocks = re.findall(r"```(?:grid)?[^\n]*\n(.*?)```", text or "", re.S)
    grids = []
    for b in blocks:
        rows = [r.strip() for r in b.strip().splitlines() if r.strip()]
        rows = [r.replace(" ", "").replace(",", "") for r in rows]
        if rows and all(r.isdigit() for r in rows) and len({len(r) for r in rows}) == 1:
            grids.append([[int(c) for c in r] for r in rows])
    return grids[-n_tests:] if len(grids) >= n_tests else None


def answer(client, task, seed, deadline, max_tokens=None, rescue_tokens=8192, notes_chars=16000):
    """One direct answer to `task`: reason, then write the test grid(s). Returns a record;
    rec["grids"] is the list of test grids, or None when nothing parseable came back."""
    n = len(task["test"])
    prompt = direct_prompt(task)

    def timeout():
        return max(60.0, min(4 * 3600, deadline - time.time() + 60))

    mt = max_tokens or client.max_tokens
    for _ in range(3):
        r = client.chat([{"role": "system", "content": DIRECT_SYSTEM},
                         {"role": "user", "content": prompt}],
                        seed=seed, max_tokens=mt, timeout=timeout())
        if not (r.get("context_overflow") and mt > 8192):
            break
        mt = int(mt * 0.75)                    # a long prompt: shrink the reply, never give up
    if "error" in r:
        return {"status": "error: " + r["error"][:200], "grids": None, "gen_tokens": 0}
    rec = {"finish": r["finish_reason"], "gen_tokens": r["completion_tokens"],
           "reasoning_chars": len(r["reasoning"] or ""),
           "content_tail": (r["content"] or "")[-1500:],
           "reasoning_tail": (r["reasoning"] or "")[-3000:]}
    grids = parse_grids(r["content"], n)
    if grids:
        return dict(rec, status="answered", grids=grids)
    notes = ((r["reasoning"] or "") + "\n" + (r["content"] or "")).strip()
    if notes and deadline - time.time() > 120:
        r2 = client.chat([{"role": "system", "content": DIRECT_SYSTEM},
                          {"role": "user", "content": RESCUE.format(prompt=prompt,
                                                                    notes=notes[-notes_chars:])}],
                         seed=seed + 1, max_tokens=rescue_tokens, timeout=timeout(),
                         template_kwargs={"enable_thinking": False, "reasoning_effort": "low"})
        if "error" in r2:
            rec["rescue_error"] = r2["error"][:200]
        else:
            rec["gen_tokens"] += r2["completion_tokens"]
            rec["rescue_finish"] = r2["finish_reason"]
            rec["rescue_tail"] = ((r2["content"] or "") or (r2["reasoning"] or ""))[-1500:]
            # Without thinking, a reasoning parser may still file the reply as reasoning.
            grids = parse_grids(r2["content"], n) or parse_grids(r2["reasoning"], n)
            if grids:
                return dict(rec, status="rescued", grids=grids)
    return dict(rec, status="no grid", grids=None)


def main(cfg):
    t0 = time.time()
    root = cfg.get("data_dir", "/kaggle/input")
    import glob
    ch = json.load(open(glob.glob(f"{root}/**/arc-agi_evaluation_challenges.json", recursive=True)[0]))
    sol = json.load(open(glob.glob(f"{root}/**/arc-agi_evaluation_solutions.json", recursive=True)[0]))
    ids = cfg["task_ids"]
    client = ChatClient(cfg["base_url"], cfg["model"], max_tokens=cfg.get("max_tokens", 49152),
                        temperature=cfg.get("temperature", 1.0), top_p=cfg.get("top_p", 0.95),
                        top_k=cfg.get("top_k", 20), thinking=cfg.get("thinking", True),
                        thinking_token_budget=cfg.get("thinking_token_budget"),
                        template_kwargs=cfg.get("template_kwargs"), timeout=4 * 3600)
    out = cfg.get("out", "/kaggle/working/direct_results.json")
    jsonl = out.replace(".json", ".jsonl")
    deadline = cfg.get("deadline_epoch", float("inf"))
    lock = threading.Lock()
    recs = []
    jobs = [(t, k) for k in range(cfg.get("samples", 2)) for t in ids]   # sample-major

    def work(job):
        t, k = job
        rec = {"task_id": t, "sample": k}
        if time.time() > deadline:
            rec["status"] = "skipped: deadline"
        else:
            ts = time.time()
            rec.update(answer(client, ch[t], 1000 * k + 7, deadline,
                              rescue_tokens=cfg.get("rescue_tokens", 8192)))
            rec["wall"] = round(time.time() - ts, 1)
            grids = rec["grids"]
            rec["correct"] = [g == s for g, s in zip(grids, sol[t])] if grids else None
        with lock:
            recs.append(rec)
            with open(jsonl, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{time.time() - t0:6.0f}s] {t}#{k} {rec['status']:10s} "
                  f"tok {rec.get('gen_tokens', 0):6d} finish {rec.get('finish')} "
                  f"correct {rec.get('correct')}", flush=True)

    with ThreadPoolExecutor(max_workers=cfg.get("concurrency", 16)) as ex:
        list(ex.map(work, jobs))

    # pass@2 per task: any test output matched by either sample (fractional per test).
    per = {}
    for t in ids:
        n = len(sol[t])
        hit = [False] * n
        for r in recs:
            if r["task_id"] == t and r.get("correct"):
                hit = [h or c for h, c in zip(hit, r["correct"])]
        per[t] = sum(hit) / n
    tok = [r["gen_tokens"] for r in recs if "gen_tokens" in r]
    summary = {"config": cfg, "per_task": per, "score": sum(per.values()), "n_tasks": len(ids),
               "samples": len(recs), "mean_gen_tokens": sum(tok) / max(len(tok), 1),
               "cut_at_max_tokens": sum(1 for r in recs if r.get("finish") == "length"),
               "wall_s": time.time() - t0, "records": recs}
    json.dump(summary, open(out, "w"))
    print(f"\nDIRECT SCORE {summary['score']:.2f} / {len(ids)}  mean tokens {summary['mean_gen_tokens']:.0f}  "
          f"cut {summary['cut_at_max_tokens']}  wall {summary['wall_s']:.0f}s", flush=True)
    print("per task:", {t: round(v, 2) for t, v in per.items()}, flush=True)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
