import importlib, os, sys

# 1. Enable 4-attempt gated ensemble + 24k min retry budget for E032-SFT
os.environ["ARC_ATTEMPT2"] = "gated"
os.environ["ARC_MAX_ATTEMPTS"] = "4"
os.environ["ARC_B_MIN2"] = "24000"

# 2. Write 4-attempt majority-voting selection.py
open("/kaggle/working/arc_solver/evals/arc_harness/selection.py", "w").write('"""Kaggle answer selection (E021, notebook v3 + v5_e032 4-attempt ensemble): attempt_1 / attempt_2 for one test pair,\ncross-attempt behavioural majority voting, and adaptive multi-wave scheduling (up to 4 attempts per pair).\n\nPure functions over attempt records, shared by kaggle/src/solver.py and offline simulations. An attempt record\nneeds only `pred`, `train_pass`, `train_frac`, `answer_source`, `tokens` and optionally `alternatives` (the\nagent_qwen `record_alternatives` list: other test grids of the same attempt, [{\'src\', \'pred\', \'train_frac\'}]).\n\nRanking (tiers, then cross-attempt behavioural majority voting inside each tier):\n  0. train-passing grids (the attempt\'s answer or an alternative with train_frac 1.0):\n     sorted by (verified_votes DESC, primary_votes DESC, total_votes DESC, is_primary DESC, attempt ASC, rank ASC)\n  1. attempts\' own answers that are not known to fail train (e.g. submit_answer of `answer`, no check possible):\n     sorted by (non_weak_primary_votes DESC, primary_votes DESC, total_votes DESC, is_primary DESC, attempt ASC, rank ASC)\n  2. the rest (answers known to fail train, alternatives below 1.0, unknown alternatives):\n     sorted by (max_train_frac DESC, primary_votes DESC, total_votes DESC, is_primary DESC, attempt ASC, rank ASC).\nWith a single attempt this matches v3\'s exact tier and tie-break order.\n"""\nfrom .grids import valid_grid\n\n# answer sources that come from a fallback rather than a submission the model chose (E012/E019: ~0-20% correct)\nWEAK_SOURCES = {"forced", "session", "salvage", "parse", "lenient", "program_best_fail", "session_solve"}\n\n\ndef _grid_key(grid):\n    return tuple(tuple(r) for r in grid)\n\n\ndef candidates(attempts):\n    """attempts: {attempt_index: record}. Valid grids only; annotated with cross-attempt vote counts."""\n    raw = []\n    for a in sorted(attempts):\n        r = attempts[a] or {}\n        if valid_grid(r.get("pred")):\n            raw.append({\n                "grid": r["pred"], "attempt": a, "rank": 0, "primary": True,\n                "src": r.get("answer_source"),\n                "train_pass": r.get("train_pass"), "train_frac": r.get("train_frac"),\n            })\n        if valid_grid(r.get("first_tp_pred")) and r.get("first_tp_pred") != r.get("pred"):\n            raw.append({\n                "grid": r["first_tp_pred"], "attempt": a, "rank": 0, "primary": True,\n                "src": "first_tp_pred",\n                "train_pass": True, "train_frac": 1.0,\n            })\n        for i, alt in enumerate(r.get("alternatives") or []):\n            if not isinstance(alt, dict) or not valid_grid(alt.get("pred")):\n                continue\n            f = alt.get("train_frac")\n            raw.append({\n                "grid": alt["pred"], "attempt": a, "rank": i + 1, "primary": False,\n                "src": f"alt:{alt.get(\'src\')}",\n                "train_pass": (f == 1.0) if f is not None else None,\n                "train_frac": f,\n            })\n\n    by_grid = {}\n    for c in raw:\n        k = _grid_key(c["grid"])\n        g = by_grid.setdefault(k, {\n            "verified_attempts": set(),\n            "primary_attempts": set(),\n            "non_weak_primary_attempts": set(),\n            "all_attempts": set(),\n            "max_train_frac": 0.0,\n            "any_train_pass": False,\n        })\n        a = c["attempt"]\n        g["all_attempts"].add(a)\n        if c["primary"]:\n            g["primary_attempts"].add(a)\n            if c.get("src") not in WEAK_SOURCES and c.get("train_pass") is not False:\n                g["non_weak_primary_attempts"].add(a)\n        if c.get("train_pass") is True:\n            g["verified_attempts"].add(a)\n            g["any_train_pass"] = True\n        frac = c.get("train_frac")\n        if frac is not None and frac > g["max_train_frac"]:\n            g["max_train_frac"] = frac\n\n    out = []\n    for c in raw:\n        k = _grid_key(c["grid"])\n        g = by_grid[k]\n        cc = dict(c)\n        cc["verified_votes"] = len(g["verified_attempts"])\n        cc["primary_votes"] = len(g["primary_attempts"])\n        cc["non_weak_primary_votes"] = len(g["non_weak_primary_attempts"])\n        cc["total_votes"] = len(g["all_attempts"])\n        cc["max_train_frac"] = 1.0 if g["any_train_pass"] else g["max_train_frac"]\n        if g["any_train_pass"]:\n            cc["train_pass"] = True\n            cc["train_frac"] = 1.0\n        out.append(cc)\n    return out\n\n\ndef tier(c):\n    if c.get("train_pass") is True:\n        return 0\n    if c["primary"] and c.get("train_pass") is not False:\n        return 1\n    return 2\n\n\ndef rank_key(c):\n    t = tier(c)\n    v_votes = c.get("verified_votes", 1 if c.get("train_pass") is True else 0)\n    nw_votes = c.get("non_weak_primary_votes", 1 if (c.get("primary") and c.get("src") not in WEAK_SOURCES) else 0)\n    p_votes = c.get("primary_votes", 1 if c.get("primary") else 0)\n    t_votes = c.get("total_votes", 1)\n    m_frac = c.get("max_train_frac") if c.get("max_train_frac") is not None else (c.get("train_frac") or 0.0)\n    if t == 0:\n        return (0, -v_votes, -p_votes, -t_votes, not c["primary"], c["attempt"], c["rank"])\n    if t == 1:\n        return (1, -nw_votes, -p_votes, -t_votes, not c["primary"], c["attempt"], c["rank"])\n    return (2, -m_frac, -p_votes, -t_votes, not c["primary"], c["attempt"], c["rank"])\n\n\ndef select_two(cands):\n    """The top-2 distinct candidates (0, 1 or 2 of them) in rank order."""\n    out = []\n    for c in sorted(cands, key=rank_key):\n        if all(c["grid"] != o["grid"] for o in out):\n            out.append(c)\n        if len(out) == 2:\n            break\n    return out\n\n\ndef retry_category(rec, include_unknown=True):\n    """Single-attempt retry category (0=failed train, 1=unknown train, 2=no/weak answer, None=train_pass)."""\n    alts = [a for a in rec.get("alternatives") or [] if isinstance(a, dict) and valid_grid(a.get("pred"))]\n    if rec.get("train_pass") is True or any(a.get("train_frac") == 1.0 for a in alts):\n        return None\n    if not valid_grid(rec.get("pred")) or rec.get("answer_source") in WEAK_SOURCES:\n        return 2\n    if rec.get("train_pass") is False:\n        return 0\n    return 1 if include_unknown else None\n\n\ndef retry_priority(rec, include_unknown=True):\n    """Sort key for the gated second attempt (smaller first), or None if the pair should not be retried."""\n    cat = retry_category(rec, include_unknown)\n    if cat is None:\n        return None\n    fracs = [rec.get("train_frac") or 0.0] + [a.get("train_frac") or 0.0 for a in rec.get("alternatives") or []\n                                               if isinstance(a, dict)]\n    return (cat, -max(fracs), rec.get("tokens") or 0)\n\n\ndef multi_attempt_priority(attempts, include_unknown=True):\n    """Sort key for scheduling the next gated attempt (Attempt 1, 2, or 3, up to 4 attempts total) given all\n    completed attempts `{attempt_idx: rec}` for a test pair, or None if consensus is already reached:\n\n    Wave ordering `(wave_group, sub_cat, -max_frac, total_tokens)`:\n      - wave_group 0: Urgent Wave 2 (n_done == 1, Attempt 0 failed train [0], had unknown train [1], or weak/no ans [2])\n      - wave_group 1: Urgent Wave 3 (n_done == 2, 1-vs-1 tie between top candidates [0], or still 0 verified & no 2-vote consensus [1..3])\n      - wave_group 2: Consensus & Diversity Wave (verified_votes == 1: confirm single train-pass & populate attempt_2)\n      - wave_group 3: Wave 4 Tie-breaker / Deep Retry (n_done == 3, 3-way tie or still unsolved/unverified)\n      - None: Early stop! Consensus reached (verified_votes >= 2 with >=2 distinct candidates or >=3 attempts,\n              or non_weak_primary_votes >= 3).\n    """\n    if not attempts:\n        return None\n    n_done = len(attempts)\n    cands = sorted(candidates(attempts), key=rank_key)\n    distinct = []\n    for c in cands:\n        if all(c["grid"] != d["grid"] for d in distinct):\n            distinct.append(c)\n\n    total_tok = sum((r or {}).get("tokens") or 0 for r in attempts.values())\n    max_frac = max([c.get("max_train_frac") or 0.0 for c in distinct] or [0.0])\n\n    best = distinct[0] if distinct else None\n    second = distinct[1] if len(distinct) > 1 else None\n\n    # Early stop check: strong cross-attempt consensus reached\n    if best is not None:\n        if best.get("verified_votes", 0) >= 2 and (len(distinct) >= 2 or n_done >= 3):\n            return None\n        if best.get("non_weak_primary_votes", 0) >= 3:\n            return None\n\n    # After 1st attempt (scheduling Attempt 1 / Wave 2)\n    if n_done == 1:\n        rec0 = next(iter(attempts.values()))\n        cat = retry_category(rec0, include_unknown=include_unknown)\n        if cat is not None:\n            return (0, cat, -max_frac, total_tok)\n        # Attempt 0 passed train (verified_votes == 1): schedule in wave_group 2 (after all urgent retries)\n        return (2, 0, -max_frac, total_tok)\n\n    # Check if top 2 distinct candidates are tied in tier and votes\n    is_tied = False\n    if best is not None and second is not None and tier(best) == tier(second):\n        if tier(best) == 0 and best.get("verified_votes", 0) == second.get("verified_votes", 0):\n            is_tied = True\n        elif tier(best) == 1 and best.get("primary_votes", 0) == second.get("primary_votes", 0):\n            is_tied = True\n\n    # After 2nd attempt (scheduling Attempt 2 / Wave 3)\n    if n_done == 2:\n        if is_tied:\n            return (1, 0, -max_frac, total_tok)  # High-priority tie-breaker!\n        if best is None or tier(best) == 2:\n            return (1, 1, -max_frac, total_tok)  # Both attempts failed train or gave weak/no answer\n        if tier(best) == 1:\n            if best.get("non_weak_primary_votes", 0) < 2:\n                return (1, 2, -max_frac, total_tok) if include_unknown else None\n            # 2 unverified primary votes agree, but if no second candidate for attempt_2, low-priority Wave 3\n            return (2, 2, -max_frac, total_tok) if len(distinct) < 2 else None\n        # tier(best) == 0 and verified_votes == 1 (only 1 of the 2 attempts passed train)\n        return (2, 1, -max_frac, total_tok)\n\n    # After 3rd attempt (scheduling Attempt 3 / Wave 4)\n    if n_done == 3:\n        if is_tied:\n            return (3, 0, -max_frac, total_tok)  # 3-way tie-breaker!\n        if best is None or tier(best) == 2:\n            return (3, 1, -max_frac, total_tok)  # Still no train-passing or valid primary answer\n        if tier(best) == 1 and best.get("non_weak_primary_votes", 0) < 2 and include_unknown:\n            return (3, 2, -max_frac, total_tok)\n        if tier(best) == 0 and best.get("verified_votes", 0) == 1 and len(distinct) < 2:\n            return (3, 3, -max_frac, total_tok)\n        return None\n\n    return None\n')

# 3. Patch agent_qwen.py so submit_answer("answer") inherits train_pass/train_frac from solve(test_input) / salvage
aq_path = "/kaggle/working/arc_solver/evals/arc_harness/agent_qwen.py"
aq = open(aq_path).read()
t1_old = "train_check = await _submit_train_check(ex, answer_code if source in (\"submit\", \"forced\") else None, train)"
t1_new = "train_check = await _submit_train_check(ex, answer_code if source in (\"submit\", \"forced\") else None,\n                                                    train, pred=pred)"
t2_old = """async def _submit_train_check(ex, code, train):
    \"\"\"Submit mode, bookkeeping only: re-evaluate the answer expression with test_input bound to each training
    input (E011 check). train_pass None when the expression does not read test_input (e.g. `answer`).\"\"\"
    if not code:
        return {"train_pass": None, "train_frac": None}
    try:
        check = await ex.run(check_answer_on_train, code)
    except Exception:
        check = None
    if check is None:
        return {"train_pass": None, "train_frac": None}
    return {"train_pass": not check[0], "train_frac": 1 - len(check[0]) / len(train)}"""
t2_new = """async def _submit_train_check(ex, code, train, pred=None):
    \"\"\"Submit mode, bookkeeping only: re-evaluate the answer expression with test_input bound to each training
    input (E011 check). If the expression does not read test_input (e.g. `answer`), check whether solve(test_input)
    or salvage_answer in the session produces `pred` and use its train check.\"\"\"
    if code:
        try:
            check = await ex.run(check_answer_on_train, code)
        except Exception:
            check = None
        if check is not None:
            return {"train_pass": not check[0], "train_frac": 1 - len(check[0]) / len(train)}
    if pred is not None and valid_grid(pred):
        try:
            _, s_grid = await ex.run(submit_answer, "solve(test_input)")
            if s_grid == pred:
                check = await ex.run(check_answer_on_train, "solve(test_input)")
                if check is not None:
                    return {"train_pass": not check[0], "train_frac": 1 - len(check[0]) / len(train)}
        except Exception:
            pass
        try:
            info, sal_grid = await ex.run(salvage_answer)
            if sal_grid == pred and info and info.get("train_pairs"):
                frac = info.get("train_matched", 0) / info["train_pairs"]
                return {"train_pass": frac == 1.0, "train_frac": frac}
        except Exception:
            pass
    return {"train_pass": None, "train_frac": None}"""
if t1_old in aq:
    aq = aq.replace(t1_old, t1_new, 1)
if t2_old in aq:
    aq = aq.replace(t2_old, t2_new, 1)
assert "pred=pred" in aq and "async def _submit_train_check(ex, code, train, pred=None):" in aq
open(aq_path, "w").write(aq)

# 4. Patch solver.py for 4-attempt multi-wave scheduling
sol_path = "/kaggle/working/arc_solver/solver.py"
sol = open(sol_path).read()
s1_old = "ATTEMPT2_SAFETY = envf(\"ARC_ATTEMPT2_SAFETY\", 0.85)  # gated: budget <= safety x per-attempt speed x time to force"
s1_new = s1_old + "\nMAX_ATTEMPTS = int(envf(\"ARC_MAX_ATTEMPTS\", 4))     # max attempts per test pair across adaptive waves (default 4)"
s2_old = """        self.priority = {}  # gated attempt 2: job -> selection.retry_priority key
        self.gated = {"eligible": 0, "started": 0, "dropped_time": 0}

    def may_get_work(self):
        \"\"\"Gated mode: a worker with nothing to do now waits while first attempts can still queue retries.\"\"\"
        if ATTEMPT2 != "gated" or time.time() >= self.force_at - self.launch_margin:
            return False
        return bool(self.pending1 or self.pending2 or any(j[2] == 0 for j in self.inflight))"""
s2_new = """        self.priority = {}  # gated retries: job -> selection.multi_attempt_priority key
        self.pair_records = {}  # (tid, p) -> {attempt_idx: rec}
        self.gated = {"eligible": 0, "started": 0, "dropped_time": 0}

    def may_get_work(self):
        \"\"\"Gated mode: a worker with nothing to do now waits while running attempts can still queue retries.\"\"\"
        if ATTEMPT2 != "gated" or time.time() >= self.force_at - self.launch_margin:
            return False
        return bool(self.pending1 or self.pending2 or any(j[2] + 1 < self.attempts for j in self.inflight))"""
s3_old = """            tid, p, a = job
            meta = {k: rec.get(k) for k in ("train_pass", "train_frac", "answer_source", "alternatives")}
            if any(v is not None for v in meta.values()):
                self.sub.meta.setdefault((tid, p), {})[a] = meta
            if rec["answered"]:
                self.sub.answers.setdefault((tid, p), {})[a] = rec["pred"]
            if rec["answered"] or rec.get("alternatives"):
                self.sub.write()
            if a == 0 and self.attempts > 1 and ATTEMPT2 == "legacy":
                self.pending2.append((tid, p, 1))
            elif a == 0 and self.attempts > 1 and ATTEMPT2 == "gated":
                from arc_harness.selection import retry_priority
                key = retry_priority(rec, include_unknown=bool(ATTEMPT2_UNKNOWN))
                if key is not None:
                    self.priority[(tid, p, 1)] = key
                    self.pending2.append((tid, p, 1))
                    self.gated["eligible"] += 1"""
s3_new = """            tid, p, a = job
            self.pair_records.setdefault((tid, p), {})[a] = rec
            meta = {k: rec.get(k) for k in ("train_pass", "train_frac", "answer_source", "alternatives")}
            if any(v is not None for v in meta.values()):
                self.sub.meta.setdefault((tid, p), {})[a] = meta
            if rec["answered"]:
                self.sub.answers.setdefault((tid, p), {})[a] = rec["pred"]
            if rec["answered"] or rec.get("alternatives"):
                self.sub.write()
            if a + 1 < self.attempts and ATTEMPT2 == "legacy":
                self.pending2.append((tid, p, a + 1))
            elif a + 1 < self.attempts and ATTEMPT2 == "gated":
                from arc_harness.selection import multi_attempt_priority
                key = multi_attempt_priority(self.pair_records[(tid, p)], include_unknown=bool(ATTEMPT2_UNKNOWN))
                if key is not None:
                    self.priority[(tid, p, a + 1)] = key
                    self.pending2.append((tid, p, a + 1))
                    self.gated["eligible"] += 1"""
s4_old = "sch = Scheduler(tasks, sub, client, force_at, hard_stop, conc, attempts=2, tag=mode)"
s4_new = "sch = Scheduler(tasks, sub, client, force_at, hard_stop, conc, attempts=MAX_ATTEMPTS, tag=mode)"
s5_old = """                          "free_attempt2": FREE_ATTEMPT2, "attempt2": ATTEMPT2, "session_mem_mb": SESSION_MEM_MB,
                          "tool_desc": TOOLS_DESC or "default", "fallback_solve": TOOLS_FALLBACK_SOLVE,
                          "model_source": MODEL_SOURCE or "official", "dropped": DROPPED, "exec_style": EXEC_STYLE,
                          "strict_features": STRICT_FEATURES, "effort_header": EFFORT_HEADER or "-",
                          "turns_note": TURNS_NOTE, "note_cap": NOTE_CAP, "soft_wording": SOFT_WORDING,
                          "gated": sch.gated, "attempts_by_index": {str(i): sum(r["attempt"] == i for r in recs)
                                                                    for i in (0, 1)},
                          "think_nudges": sum(r.get("think_nudges") or 0 for r in recs),
                          "think_forced": sum(r.get("think_forced") or 0 for r in recs),
                          "train_pass": sum(r.get("train_pass") is True for r in recs),
                          "with_alternatives": sum(bool(r.get("alternatives")) for r in recs),
                          "slot2_sources": {k: sum(1 for v in sub.sources.values() if (v[1] or "").startswith(k))
                                            for k in ("a0:alt", "a1:alt", "a0:", "a1:", "fallback", "dup")}}"""
s5_new = """                          "free_attempt2": FREE_ATTEMPT2, "attempt2": ATTEMPT2, "max_attempts": MAX_ATTEMPTS,
                          "session_mem_mb": SESSION_MEM_MB,
                          "tool_desc": TOOLS_DESC or "default", "fallback_solve": TOOLS_FALLBACK_SOLVE,
                          "model_source": MODEL_SOURCE or "official", "dropped": DROPPED, "exec_style": EXEC_STYLE,
                          "strict_features": STRICT_FEATURES, "effort_header": EFFORT_HEADER or "-",
                          "turns_note": TURNS_NOTE, "note_cap": NOTE_CAP, "soft_wording": SOFT_WORDING,
                          "gated": sch.gated, "attempts_by_index": {str(i): sum(r["attempt"] == i for r in recs)
                                                                    for i in range(max(2, MAX_ATTEMPTS))},
                          "think_nudges": sum(r.get("think_nudges") or 0 for r in recs),
                          "think_forced": sum(r.get("think_forced") or 0 for r in recs),
                          "train_pass": sum(r.get("train_pass") is True for r in recs),
                          "with_alternatives": sum(bool(r.get("alternatives")) for r in recs),
                          "slot2_sources": {k: sum(1 for v in sub.sources.values() if (v[1] or "").startswith(k))
                                            for k in tuple(f"a{i}:alt" for i in range(max(2, MAX_ATTEMPTS)))
                                            + tuple(f"a{i}:" for i in range(max(2, MAX_ATTEMPTS)))
                                            + ("fallback", "dup")}}"""
if "MAX_ATTEMPTS =" not in sol:
    sol = sol.replace(s1_old, s1_new, 1)
for old, new in [ (s2_old, s2_new), (s3_old, s3_new), (s4_old, s4_new), (s5_old, s5_new)]:
    if old in sol:
        sol = sol.replace(old, new, 1)
assert "MAX_ATTEMPTS" in sol and "multi_attempt_priority" in sol and "self.pair_records" in sol
open(sol_path, "w").write(sol)

sys.path[:0] = ["/kaggle/working/arc_solver", "/kaggle/working/arc_solver/evals"]
for m in ["arc_harness.grids", "arc_harness.arcprize", "arc_harness.prompts", "arc_harness.executor", "arc_harness.audit", "arc_harness.selection", "arc_helpers", "task_summary", "arc_harness.agent_qwen", "solver"]:
    mod = importlib.import_module(m)
    print("ok", m, getattr(mod, "HARNESS_VERSION", ""))
print("4-ATTEMPT ENSEMBLE PATCH APPLIED & VERIFIED: ATTEMPT2=", mod.ATTEMPT2, "MAX_ATTEMPTS=", mod.MAX_ATTEMPTS, "B_MIN2=", mod.B_MIN2)
