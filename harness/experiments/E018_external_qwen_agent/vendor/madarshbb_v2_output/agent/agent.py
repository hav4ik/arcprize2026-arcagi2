"""Batched program-synthesis agent for ARC tasks.

Every task gets K independent attempts (conversations). Each ROUND sends every
live conversation to the model in ONE batch -- that is what keeps the GPUs
busy -- then runs each extracted `transform` on the task's train and test inputs
in the sandbox. An attempt whose program reproduces every train output is
VERIFIED and stops. Otherwise the model is shown where its program failed and
asked again, up to R rounds.

The verification against the demos is the point of the whole approach: a
program that reproduces all examples is a far stronger selector than any
likelihood score (exp06: the NLL scorer prefers a wrong grid over the injected
truth for 83 % of missed outputs).

Answers per test output are voted over verified programs; if fewer than two
distinct verified answers exist, the gap is filled from the best unverified
programs (most train examples right), and the source of each answer is kept so
a fallback solver can be preferred over weak guesses.
"""
import time
from collections import Counter, defaultdict

from grids import SYSTEM, task_prompt, feedback_prompt
from backend import extract_code
from sandbox import run_many


def visible_reply(text, keep=6000):
    """What goes back into the conversation: the answer without the reasoning.

    Qwen3-style models wrap reasoning in <think>; gpt-oss via raw harmony output
    puts the answer after a 'final' channel marker. Either way the program is in
    the tail, and re-sending long reasoning would crowd out the context.
    """
    t = text or ""
    if "</think>" in t:
        t = t.split("</think>")[-1]
    for marker in ("<|channel|>final<|message|>", "assistantfinal"):
        if marker in t:
            t = t.split(marker)[-1]
    t = t.strip()
    return t[-keep:] if len(t) > keep else t


def _key(g):
    return tuple(tuple(r) for r in g)


class Attempt:
    __slots__ = ("task_id", "idx", "messages", "rounds", "status", "programs",
                 "tokens", "note")

    def __init__(self, task_id, idx, prompt):
        self.task_id, self.idx = task_id, idx
        self.messages = [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}]
        self.rounds = 0
        self.status = "active"          # active | verified | failed | dropped
        self.programs = []              # dicts: round, code, n_ok, n_train, test
        self.tokens = 0
        self.note = ""


def solve(tasks, backend, samples=4, rounds=3, max_tokens=None, deadline=None,
          stop_after=2, max_prompt_tokens=None, sandbox_timeout=10.0,
          sandbox_workers=None, log=print):
    """tasks: {task_id: {"train": [...], "test": [...]}}.

    Returns (answers, report). answers[task_id] is a list over test inputs of
    [(grid, source), ...] best first; report holds per-attempt diagnostics.
    """
    t0 = time.time()
    deadline = deadline or float("inf")
    attempts = [Attempt(tid, i, task_prompt(task))
                for tid, task in tasks.items() for i in range(samples)]
    verified_count = Counter()
    rounds_log = []

    for rnd in range(1, rounds + 1):
        live = [a for a in attempts if a.status == "active"
                and verified_count[a.task_id] < stop_after]
        for a in attempts:
            if a.status == "active" and verified_count[a.task_id] >= stop_after:
                a.status, a.note = "failed", "task already verified"
        if not live:
            break
        if time.time() > deadline:
            log(f"round {rnd}: deadline reached with {len(live)} attempts live")
            break
        if max_prompt_tokens:
            keep = []
            for a in live:
                if backend.prompt_tokens(a.messages) > max_prompt_tokens:
                    a.status, a.note = "dropped", "prompt too long"
                else:
                    keep.append(a)
            live = keep
            if not live:
                break

        tg = time.time()
        replies = backend.chat([a.messages for a in live], max_tokens=max_tokens,
                               metas=[{"task_id": a.task_id, "attempt": a.idx} for a in live])
        gen_s = time.time() - tg

        jobs, owners = [], []
        for a, (text, ntok, finish) in zip(live, replies):
            a.rounds += 1
            a.tokens += ntok
            code = extract_code(text)
            a.messages.append({"role": "assistant", "content": visible_reply(text)})
            if code is None:
                why = ("Your reply was cut off before the program." if finish == "length"
                       else "No ```python block defining transform() was found.")
                a.messages.append({"role": "user", "content":
                                   why + " Give the complete program in ONE ```python code block."})
                if a.rounds >= rounds:
                    a.status, a.note = "failed", "no code"
                continue
            task = tasks[a.task_id]
            inputs = [p["input"] for p in task["train"]] + [t["input"] for t in task["test"]]
            jobs.append((code, inputs))
            owners.append((a, code))

        ts = time.time()
        results = run_many(jobs, timeout=sandbox_timeout, workers=sandbox_workers)
        run_s = time.time() - ts

        n_ver = 0
        for (a, code), res in zip(owners, results):
            task = tasks[a.task_id]
            ntr = len(task["train"])
            train_res = []
            for r, pair in zip(res[:ntr], task["train"]):
                ok = "output" in r and r["output"] == pair["output"]
                train_res.append({"ok": ok, "output": r.get("output"), "error": r.get("error")})
            n_ok = sum(t["ok"] for t in train_res)
            test_out = [r.get("output") for r in res[ntr:]]
            a.programs.append({"round": a.rounds, "code": code, "n_ok": n_ok,
                               "n_train": ntr, "test": test_out,
                               "test_errors": [r.get("error") for r in res[ntr:]]})
            if n_ok == ntr and all(o is not None for o in test_out):
                a.status = "verified"
                verified_count[a.task_id] += 1
                n_ver += 1
            elif a.rounds >= rounds:
                a.status, a.note = "failed", f"best {max(p['n_ok'] for p in a.programs)}/{ntr}"
            else:
                a.messages.append({"role": "user",
                                   "content": feedback_prompt(train_res, task["train"])})

        toks = sum(r[1] for r in replies)
        rounds_log.append({"round": rnd, "live": len(live), "programs": len(jobs),
                           "verified_now": n_ver, "gen_s": gen_s, "run_s": run_s,
                           "tokens": toks, "tok_per_s": toks / max(gen_s, 1e-9)})
        log(f"round {rnd}: {len(live)} live, {len(jobs)} programs, {n_ver} newly verified, "
            f"tasks verified {sum(1 for t in tasks if verified_count[t])}/{len(tasks)}, "
            f"gen {gen_s:.0f}s ({toks / max(gen_s, 1e-9):.0f} tok/s) run {run_s:.1f}s")

    answers = {tid: aggregate(tid, tasks[tid], attempts) for tid in tasks}
    report = {
        "rounds": rounds_log,
        "wall_s": time.time() - t0,
        "attempts": [{"task_id": a.task_id, "idx": a.idx, "status": a.status,
                      "rounds": a.rounds, "tokens": a.tokens, "note": a.note,
                      "best_ok": max((p["n_ok"] for p in a.programs), default=-1),
                      "n_train": len(tasks[a.task_id]["train"]),
                      "programs": a.programs} for a in attempts],
    }
    return answers, report


def aggregate(tid, task, attempts):
    """Per test input: candidate grids best first, each tagged with its source.

    Verified programs vote; unverified programs are ranked by train examples
    solved, then by how many programs agree.
    """
    mine = [a for a in attempts if a.task_id == tid]
    out = []
    for j in range(len(task["test"])):
        ver = Counter()
        unv = defaultdict(lambda: [0, 0])          # key -> [best n_ok, votes]
        grids = {}
        for a in mine:
            for p in a.programs:
                g = p["test"][j] if j < len(p["test"]) else None
                if g is None:
                    continue
                k = _key(g)
                grids[k] = g
                if p["n_ok"] == p["n_train"]:
                    ver[k] += 1
                else:
                    u = unv[k]
                    u[0] = max(u[0], p["n_ok"])
                    u[1] += 1
        ranked = [(grids[k], f"verified x{n}") for k, n in ver.most_common()]
        seen = {k for k in ver}
        rest = sorted((k for k in unv if k not in seen),
                      key=lambda k: (-unv[k][0], -unv[k][1]))
        ranked += [(grids[k], f"unverified {unv[k][0]}/{len(task['train'])}") for k in rest]
        out.append(ranked)
    return out


def to_submission(answers, tasks, fallback=None):
    """Kaggle format. `fallback(task_id, j)` supplies grids when the agent has
    fewer than two; otherwise the test input is echoed."""
    sub = {}
    for tid, task in tasks.items():
        rows = []
        for j, t in enumerate(task["test"]):
            cands = [g for g, _ in answers.get(tid, [[]] * len(task["test"]))[j]]
            if fallback and len(cands) < 2:
                for g in fallback(tid, j):
                    if g not in cands:
                        cands.append(g)
            while len(cands) < 2:
                cands.append(t["input"])
            rows.append({"attempt_1": cands[0], "attempt_2": cands[1]})
        sub[tid] = rows
    return sub


def score(sub, solutions):
    """ARC Prize scoring: a task scores the fraction of its test outputs matched
    by either attempt; the total is the sum over tasks (and / n as a percent)."""
    total, per = 0.0, {}
    for tid, sols in solutions.items():
        rows = sub.get(tid, [])
        hit = 0
        for j, sol in enumerate(sols):
            if j < len(rows) and sol in (rows[j]["attempt_1"], rows[j]["attempt_2"]):
                hit += 1
        per[tid] = hit / len(sols)
        total += per[tid]
    return total, 100.0 * total / max(len(solutions), 1), per
