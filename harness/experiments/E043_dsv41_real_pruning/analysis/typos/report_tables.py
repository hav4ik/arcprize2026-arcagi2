#!/usr/bin/env python3
"""Step 7: aggregate everything into markdown tables (out/tables.md) and a JSON of key numbers (out/key_numbers.json).
"""
import json
import math
import os
import pickle
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT  # noqa: E402
from ident import OCC_COLS  # noqa: E402

EVAL = ["API-full", "SH-full", "E128", "E112", "E96", "E72"]
PRUNED = ["E128", "E112", "E96", "E72"]
DEV = ["DEV-E128-t1", "DEV-E128-minp05", "DEV-E128-topp95"]
ALL = EVAL + DEV + ["TRAIN"]
IDENT_EXC = {"NameError", "AttributeError", "ImportError", "ModuleNotFoundError"}
HIGH_CONF = {"fix=vocab", "fix", "vocab", "vocab-prefix", "fix-span"}
md = []
key = {}


def P(s=""):
    md.append(s)


def table(df, floatfmt="{:.1f}"):
    cols = list(df.columns)
    P("| " + " | ".join(str(c) for c in cols) + " |")
    P("|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|")
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(floatfmt.format(v) if not math.isnan(v) else "")
            else:
                cells.append(str(v))
        P("| " + " | ".join(cells) + " |")
    P()


def binom_two_sided(k, n):
    if n == 0:
        return float("nan")
    p = 0.0
    obs = math.comb(n, k) * 0.5 ** n
    for i in range(n + 1):
        q = math.comb(n, i) * 0.5 ** n
        if q <= obs * (1 + 1e-9):
            p += q
    return min(1.0, p)


def load(m, kind):
    f = f"{OUT}/{kind}_{m}.pkl"
    return pickle.load(open(f, "rb")) if os.path.exists(f) else None


def att_df(m):
    A = load(m, "attempts")
    rows = []
    for a in A:
        calls = [c for t in a["turns"] for c in t["calls"] if c["name"] == "run_python"]
        rows.append({"model": m, "task_id": a["task_id"], "pair": a["pair"], "attempt": a["attempt"],
                     "correct": a["correct"], "turns": a["turns_n"], "completion": a["completion"] or 0,
                     "stop": a["stop_reason"], "calls": len(calls),
                     "err_calls": sum(c["status"] == "error" for c in calls)})
    return pd.DataFrame(rows)


def main():
    E = {m: pd.DataFrame(load(m, "ident_errors")) for m in ALL if load(m, "ident_errors") is not None}
    ATT = {m: att_df(m) for m in ALL}
    syn = json.load(open(f"{OUT}/synerr.json"))

    # ---------------- 1. taxonomy ----------------
    P("## T1. Error taxonomy (per 100 run_python calls)")
    rows = []
    for m in ALL:
        A = load(m, "attempts")
        calls = [c for a in A for t in a["turns"] for c in t["calls"] if c["name"] == "run_python"]
        n = len(calls)
        ex = Counter(c["exc_type"] if c["status"] == "error" else c["status"] for c in calls)
        kw = sum(1 for c in calls if c["status"] == "error" and c["exc_type"] == "TypeError"
                 and re.search(r"unexpected keyword argument|invalid keyword argument", c["exc_msg"] or ""))
        ident = sum(ex[k] for k in IDENT_EXC) + kw
        r = {"model": m, "attempts": len(A), "calls": n,
             "any error": 100 * sum(1 for c in calls if c["status"] == "error") / n,
             "identifier (Name/Attr/Import/Module/kwarg)": 100 * ident / n,
             "NameError": 100 * ex["NameError"] / n, "AttributeError": 100 * ex["AttributeError"] / n,
             "Import+ModuleNotFound": 100 * (ex["ImportError"] + ex["ModuleNotFoundError"]) / n,
             "kwarg TypeError": 100 * kw / n,
             "SyntaxError": 100 * (ex["SyntaxError"] + ex["IndentationError"]) / n,
             "other TypeError": 100 * (ex["TypeError"] - kw) / n,
             "Value+Index+Key": 100 * (ex["ValueError"] + ex["IndexError"] + ex["KeyError"]) / n,
             "timeout": 100 * ex["timeout"] / n,
             "'if False else' self-repair": syn[m]["markers_per100"].get("if-False-else", 0.0),
             "'?' in code": syn[m]["markers_per100"].get("?-in-code", 0.0)}
        rows.append(r)
    t1 = pd.DataFrame(rows)
    key["taxonomy"] = t1.to_dict("records")
    table(t1, "{:.2f}")

    P("### SyntaxError classes (per 100 calls) and bracket pairs")
    rows = []
    for m in ALL:
        s = syn[m]
        r = {"model": m}
        for k in ["invalid-syntax", "bracket-mismatch", "unmatched-closer", "never-closed", "forgot-comma", "string",
                  "incomplete-input", "indentation"]:
            r[k] = s["syntax_per100"].get(k, 0.0)
        r["{ closed by ] or )"] = (s["bracket_pairs"].get("{..]", 0) + s["bracket_pairs"].get("{..)", 0))
        r["all mismatches"] = sum(s["bracket_pairs"].values())
        r["caret on merged bracket token"] = s["caret_merged_share"] if s["caret_merged_share"] is not None else float("nan")
        r["baseline merged"] = s["baseline_closer_in_merged_token"]
        rows.append(r)
    table(pd.DataFrame(rows), "{:.2f}")

    # ---------------- 2. identifier classes ----------------
    P("## T2. Identifier errors: classification")
    rows = []
    for m in ALL:
        if m not in E:
            continue
        d = E[m]
        if "kind" not in d:
            continue
        d = d[d.kind.notna()]
        c = d.cls.value_counts()
        rows.append({"model": m, "identifier errors": len(d), "typo": c.get("typo", 0),
                     "typo, high-confidence intended": int(((d.cls == "typo") & d.intended_src.isin(HIGH_CONF)).sum()),
                     "unresolved": c.get("unresolved", 0), "state/scope": c.get("state-or-scope", 0),
                     "missing import": c.get("missing-import", 0), "wrong API": c.get("wrong-api", 0),
                     "unavailable module": c.get("unavailable-module", 0), "use-before-def": c.get("use-before-def", 0),
                     "typo share": c.get("typo", 0) / max(1, len(d))})
    table(pd.DataFrame(rows), "{:.2f}")

    # ---------------- 3. edit types ----------------
    P("## T3. Character-level edit type of typos (share of typos with a recovered intended name)")
    rows = []
    T = {m: E[m][(E[m].get("cls") == "typo")] if "cls" in E[m] else pd.DataFrame() for m in E}
    for m in ALL:
        t = T.get(m)
        if t is None or len(t) == 0:
            continue
        vc = t.edit.value_counts(normalize=True)
        r = {"model": m, "typos": len(t)}
        for k in ["deletion", "insertion(doubled)", "segment-repeat", "insertion", "substitution", "transposition",
                  "truncation", "chunk-deletion", "tail-replaced"]:
            r[k] = 100 * vc.get(k, 0)
        r["multi-edit"] = 100 * sum(v for k, v in vc.items() if k.startswith("multi"))
        r["len intended (median)"] = float(t.intended.str.len().median())
        rows.append(r)
    table(pd.DataFrame(rows), "{:.0f}")

    P("### Most frequent typos (pruned models pooled)")
    pool = pd.concat([T[m] for m in PRUNED])
    top = (pool.groupby(["typo", "intended"]).size().sort_values(ascending=False).head(25)).reset_index()
    top.columns = ["typo", "intended", "count"]
    top["intended tokens (in context)"] = [
        " | ".join(pool[(pool.typo == r.typo) & (pool.intended == r.intended)].i_toks.iloc[0]) for r in top.itertuples()]
    top["models"] = [",".join(sorted(set(pool[(pool.typo == r.typo) & (pool.intended == r.intended)].model)))
                     for r in top.itertuples()]
    table(top)

    # ---------------- 4. token-level ----------------
    P("## T4. Token-level locus of typos vs per-position opportunity")
    rows = []
    for m in ALL:
        t = T.get(m)
        occ = load(m, "occ")
        if t is None or len(t) == 0 or occ is None:
            continue
        o = pd.DataFrame(occ, columns=["task_id", "pair", "attempt"] + OCC_COLS)
        o = o[o.kind != "attr_dummy"]
        n_occ = len(o)
        n_mid = int(o.n_mid.sum())
        n_sym = int((o.n_bnd - o.n_mid).sum())
        multitok = (o.n_tok >= 2).mean()
        bt = t.btype.value_counts()
        r = {"model": m, "typos": len(t),
             "intended multi-token %": 100 * (t.n_tok_i >= 2).mean(),
             "base: occurrences multi-token %": 100 * multitok,
             "first token correct %": 100 * (t.div_tok >= 1).mean(),
             "diverges at a token boundary %": 100 * t.at_boundary.mean(),
             "at alnum|alnum boundary %": 100 * bt.get("mid", 0) / len(t),
             "hazard per alnum|alnum boundary (x1e4)": 1e4 * bt.get("mid", 0) / max(1, n_mid),
             "hazard per _/symbol boundary (x1e4)": 1e4 * bt.get("sym", 0) / max(1, n_sym),
             "hazard per first token (x1e4)": 1e4 * (bt.get("first-char", 0) + bt.get("inside-token", 0) * 0) / n_occ,
             "hazard inside-token (per occ, x1e4)": 1e4 * bt.get("inside-token", 0) / n_occ,
             "hazard after last token (per occ, x1e4)": 1e4 * bt.get("end", 0) / n_occ}
        rows.append(r)
        if m in PRUNED:
            key.setdefault("token_locus", {})[m] = r
    table(pd.DataFrame(rows), "{:.1f}")

    P("### Token-level event (canonical tokens of the intended name in context; pruned models pooled and per model)")
    from textutil import token_event
    rows = []
    for m in PRUNED + ["DEV-E128-t1", "DEV-E128-minp05", "DEV-E128-topp95", "API-full", "TRAIN"]:
        t = T.get(m)
        if t is None or len(t) == 0:
            continue
        ev = Counter()
        exp_first = []
        for r in t.itertuples():
            e, j = token_event(r.typo.split(".")[-1], r.intended.split(".")[-1], list(r.i_toks))
            ev[e] += 1
            # uniform-character null: chance the first differing char falls in the first token
            b = [x for x in r.i_bounds if 0 < x < len(r.intended)]
            exp_first.append((b[0] if b else len(r.intended)) / max(1, len(r.intended)))
        row = {"model": m, "typos": len(t)}
        for k in ["continuation-lost-leading-chars", "continuation-replaced", "continuation-replaced(same 1st char)",
                  "restart(repeat-emitted-chars)", "dropped-1-token(s)", "dropped-2-token(s)", "truncated-at-boundary",
                  "continuation-truncated", "continuation-extended", "extended-after-last-token", "first-token-wrong",
                  "unaligned"]:
            row[k] = 100 * ev.get(k, 0) / len(t)
        row["null: P(first token) if chars uniform %"] = 100 * float(np.mean(exp_first))
        rows.append(row)
        key.setdefault("token_event", {})[m] = row
    table(pd.DataFrame(rows), "{:.0f}")

    P("### Divergence token index (pruned pooled)")
    vc = pool.div_tok.value_counts().sort_index()
    P(", ".join(f"token {int(k)}: {v}" for k, v in vc.items()))
    P()

    # hazard by copy status / lead (pooled pruned, per model)
    P("## T5. Copy status of the intended identifier: typo hazard per 10k occurrences (multi-token identifiers only)")
    rows = []
    for m in ALL:
        t = T.get(m)
        occ = load(m, "occ")
        if t is None or len(t) == 0 or occ is None:
            continue
        o = pd.DataFrame(occ, columns=["task_id", "pair", "attempt"] + OCC_COLS)
        o = o[o.n_tok >= 2]
        tt = t[t.n_tok_i >= 2]

        def grp_o(x):
            return np.where(~x.has_prev, "fresh", np.where(x.tok_match_any, "copy, same tokens", "copy, re-tokenized"))

        def grp_t(x):
            return np.where(~x.i_prev.astype(bool), "fresh",
                            np.where(x.i_tok_match_any.astype(bool), "copy, same tokens", "copy, re-tokenized"))
        go = pd.Series(grp_o(o)).value_counts()
        gt = pd.Series(grp_t(tt)).value_counts()
        r = {"model": m}
        for g in ["fresh", "copy, same tokens", "copy, re-tokenized"]:
            r[f"{g}: typos"] = int(gt.get(g, 0))
            r[f"{g}: occ"] = int(go.get(g, 0))
            r[f"{g}: /10k"] = 1e4 * gt.get(g, 0) / max(1, go.get(g, 0))
        rows.append(r)
    table(pd.DataFrame(rows), "{:.1f}")

    P("### Where was the intended name before? (typos, pruned models)")
    rows = []
    for m in PRUNED:
        t = T[m]
        r = {"model": m, "typos": len(t),
             "seen before %": 100 * t.i_prev.mean(),
             "in earlier code %": 100 * t.i_prev_code.mean(),
             "only in reasoning %": 100 * (t.i_prev_reason & ~t.i_prev_code).mean(),
             "never in attempt (API/builtin) %": 100 * (~t.i_prev.astype(bool)).mean(),
             "fresh & API/builtin name %": 100 * ((~t.i_prev.astype(bool)) & t.i_preloaded).mean(),
             "last seen same turn %": 100 * t.i_prev_same_turn.mean(),
             "median tokens since last occurrence": float(t[t.i_prev_dist >= 0].i_prev_dist.median()),
             "p90 tokens since": float(t[t.i_prev_dist >= 0].i_prev_dist.quantile(0.9)),
             "last seen >= 1 turn back %": 100 * (t.i_prev_turns_back >= 1).mean(),
             "last seen >= 3 turns back %": 100 * (t.i_prev_turns_back >= 3).mean(),
             "same tokens as an earlier occurrence %": 100 * t[t.i_prev.astype(bool)].i_tok_match_any.mean(),
             "typo form also in reasoning (same/earlier turn) %": 100 * t.typo_in_reason_any.mean(),
             "typo form already in earlier code %": 100 * t.typo_in_prev_code.mean()}
        rows.append(r)
    table(pd.DataFrame(rows), "{:.1f}")

    # lead char class hazard
    P("### Typo hazard by the character before the identifier (pruned pooled, multi-token identifiers, per 10k occ)")
    o_all = []
    for m in PRUNED:
        o = pd.DataFrame(load(m, "occ"), columns=["task_id", "pair", "attempt"] + OCC_COLS)
        o_all.append(o[o.n_tok >= 2])
    o_all = pd.concat(o_all)
    tt = pool[pool.n_tok_i >= 2]
    go = o_all.lead.value_counts()
    gt = tt.lead_cls.value_counts()
    o_all_any = []
    for m in PRUNED:
        o = pd.DataFrame(load(m, "occ"), columns=["task_id", "pair", "attempt"] + OCC_COLS)
        o_all_any.append(o)
    o_all_any = pd.concat(o_all_any)
    mid_by_lead = o_all_any.groupby("lead").n_mid.sum()
    occ_by_lead = o_all_any.lead.value_counts()
    multi_by_lead = o_all_any.groupby("lead").n_tok.apply(lambda s: (s >= 2).mean())
    gmid = pool[pool.btype == "mid"].lead_cls.value_counts()
    gall = pool.lead_cls.value_counts()
    rows = [{"lead": g, "typos (multi-token)": int(gt.get(g, 0)), "occurrences (multi-token)": int(go.get(g, 0)),
             "per 10k multi-token occ": 1e4 * gt.get(g, 0) / max(1, go.get(g, 0)),
             "all typos per 10k occ (any length)": 1e4 * gall.get(g, 0) / max(1, occ_by_lead.get(g, 0)),
             "share of occ multi-token %": 100 * multi_by_lead.get(g, 0),
             "alnum|alnum boundaries per occ": mid_by_lead.get(g, 0) / max(1, occ_by_lead.get(g, 0)),
             "mid-boundary typos per 10k mid boundaries": 1e4 * gmid.get(g, 0) / max(1, mid_by_lead.get(g, 0))}
            for g in go.index]
    table(pd.DataFrame(rows).sort_values("per 10k multi-token occ", ascending=False), "{:.1f}")

    # ---------------- context / turn dependence ----------------
    P("## T6. Context length and turn index: identifier-typo errors per 100 run_python calls")
    rows = []
    bins = [0, 15000, 25000, 40000, 60000, 1e9]
    labels = ["<15k", "15-25k", "25-40k", "40-60k", ">60k"]
    for m in PRUNED + ["API-full"]:
        A = load(m, "attempts")
        t = T[m]
        tkeys = Counter(zip(t.task_id, t.pair, t.attempt, t.turn))
        recs = []
        for a in A:
            for ti, tr in enumerate(a["turns"]):
                for c in tr["calls"]:
                    if c["name"] == "run_python":
                        recs.append((tr["prompt_tokens"] or 0, ti, tkeys.get((a["task_id"], a["pair"], a["attempt"], ti), 0)))
        df = pd.DataFrame(recs, columns=["ctx", "turn", "typo"])
        df["cb"] = pd.cut(df.ctx, bins, labels=labels, right=False)
        df["tb"] = pd.cut(df.turn, [0, 3, 6, 10, 20, 100], labels=["0-2", "3-5", "6-9", "10-19", "20+"], right=False)
        r = {"model": m}
        for lab, g in df.groupby("cb", observed=False):
            r[f"ctx {lab}"] = 100 * g.typo.sum() / max(1, len(g))
        for lab, g in df.groupby("tb", observed=False):
            r[f"turn {lab}"] = 100 * g.typo.sum() / max(1, len(g))
        rows.append(r)
    table(pd.DataFrame(rows), "{:.1f}")

    # per-occurrence hazard by context (controls for code volume)
    P("### Per-boundary hazard (alnum|alnum typos per 10k such boundaries) by context length, pruned pooled")
    o_all = []
    for m in PRUNED:
        o = pd.DataFrame(load(m, "occ"), columns=["task_id", "pair", "attempt"] + OCC_COLS)
        o["model"] = m
        o_all.append(o)
    o_all = pd.concat(o_all)
    o_all["cb"] = pd.cut(o_all.prompt_tokens, bins, labels=labels, right=False)
    pool2 = pool.copy()
    pool2["cb"] = pd.cut(pool2.prompt_tokens.fillna(0), bins, labels=labels, right=False)
    rows = []
    for lab in labels:
        nm = o_all[o_all.cb == lab].n_mid.sum()
        k = ((pool2.cb == lab) & (pool2.btype == "mid")).sum()
        rows.append({"context": lab, "mid-boundary typos": int(k), "mid boundaries": int(nm), "per 10k": 1e4 * k / max(1, nm)})
    table(pd.DataFrame(rows), "{:.2f}")

    # within-attempt: early vs late calls
    P("### Within attempt: typo rate in the first vs second half of an attempt's calls (attempts with >= 6 calls)")
    rows = []
    for m in PRUNED:
        A = load(m, "attempts")
        t = T[m]
        tk = Counter(zip(t.task_id, t.pair, t.attempt, t.call_seq))
        e1 = n1 = e2 = n2 = 0
        for a in A:
            seq = [(qi, c) for qi, c in enumerate([c for tr in a["turns"] for c in tr["calls"]])]
            rp = [(qi, c) for qi, c in seq if c["name"] == "run_python"]
            if len(rp) < 6:
                continue
            h = len(rp) // 2
            for j, (qi, c) in enumerate(rp):
                hit = tk.get((a["task_id"], a["pair"], a["attempt"], qi), 0)
                if j < h:
                    e1 += hit
                    n1 += 1
                else:
                    e2 += hit
                    n2 += 1
        rows.append({"model": m, "first half /100 calls": 100 * e1 / max(1, n1), "second half /100 calls": 100 * e2 / max(1, n2),
                     "calls": n1 + n2})
    table(pd.DataFrame(rows), "{:.2f}")

    # ---------------- cost ----------------
    P("## T7. Recovery: next call after an identifier typo")
    rows = []
    for m in ALL:
        t = T.get(m)
        if t is None or len(t) == 0 or "next_status" not in t:
            continue
        tn = t[t.next_status.notna()]
        r = {"model": m, "typos": len(t), "with a next call": len(tn),
             "typo gone in next call %": 100 * (~tn.next_has_typo.astype(bool)).mean(),
             "next call completes %": 100 * (tn.next_status == "completed").mean(),
             "next call errors again %": 100 * (tn.next_status == "error").mean(),
             "same error repeated %": 100 * tn.next_same_err.astype(bool).mean(),
             "next call is a near-copy (sim>=0.8) %": 100 * (tn.next_sim >= 0.8).mean(),
             "repair-turn tokens (median)": float(tn.next_turn_tokens.median()),
             "repair-turn tokens (mean)": float(tn.next_turn_tokens.mean()),
             "'Did you mean' in output": int(t.did_you_mean.sum())}
        rows.append(r)
    table(pd.DataFrame(rows), "{:.1f}")

    P("### Turns and tokens attributable to error recovery (all error calls; per attempt)")
    rows = []
    for m in EVAL + DEV:
        A = load(m, "attempts")
        tot_tok = err_tok = typo_tok = 0
        tot_turns = err_turns = typo_turns = 0
        t = T.get(m)
        tset = set(zip(t.task_id, t.pair, t.attempt, t.turn)) if t is not None and len(t) else set()
        capped = capped_fix = 0
        for a in A:
            turns = a["turns"]
            n_err_here = 0
            for ti, tr in enumerate(turns):
                ct = tr["completion_tokens"] or 0
                tot_tok += ct
                tot_turns += 1
                prev = turns[ti - 1] if ti > 0 else None
                if prev is not None and any(c["status"] == "error" for c in prev["calls"]):
                    err_tok += ct
                    err_turns += 1
                    n_err_here += 1
                    if (a["task_id"], a["pair"], a["attempt"], ti - 1) in tset:
                        typo_tok += ct
                        typo_turns += 1
            if a["stop_reason"] == "max_turns":
                capped += 1
        rows.append({"model": m, "attempts": len(A), "turns/attempt": tot_turns / len(A),
                     "turns after an error %": 100 * err_turns / max(1, tot_turns),
                     "turns after an identifier typo %": 100 * typo_turns / max(1, tot_turns),
                     "tokens in turns after an error %": 100 * err_tok / max(1, tot_tok),
                     "tokens after identifier typo %": 100 * typo_tok / max(1, tot_tok),
                     "40-turn stops": capped})
    table(pd.DataFrame(rows), "{:.1f}")

    # ---------------- paired outcome ----------------
    P("## T8. Do attempts with more typos fail more often on the same pair?")
    rows = []
    allatt = []
    for m in EVAL:
        a = ATT[m].copy()
        t = T.get(m)
        cnt = Counter(zip(t.task_id, t.pair, t.attempt)) if t is not None and len(t) else Counter()
        early = Counter()
        if t is not None and len(t):
            for r in t.itertuples():
                if r.call_seq < 6:
                    early[(r.task_id, r.pair, r.attempt)] += 1
        a["typos"] = [cnt.get(k, 0) for k in zip(a.task_id, a.pair, a.attempt)]
        a["typos_early"] = [early.get(k, 0) for k in zip(a.task_id, a.pair, a.attempt)]
        a["typo_rate"] = a.typos / a.calls.clip(lower=1)
        allatt.append(a)
        # within model, attempt 0 vs 1 on the same pair
        g = a.groupby(["task_id", "pair"])
        more_fail = more_win = 0
        e_more_fail = e_more_win = 0
        for _, x in g:
            if len(x) != 2 or x.correct.nunique() != 2:
                continue
            w = x[x.correct].iloc[0]
            l = x[~x.correct].iloc[0]
            if l.typo_rate > w.typo_rate:
                more_fail += 1
            elif w.typo_rate > l.typo_rate:
                more_win += 1
            if l.typos_early > w.typos_early:
                e_more_fail += 1
            elif w.typos_early > l.typos_early:
                e_more_win += 1
        rows.append({"comparison": f"{m}: attempt vs attempt, same pair",
                     "discordant pairs where the failing attempt had the higher typo rate": more_fail,
                     "... the winning attempt had the higher rate": more_win,
                     "p (sign)": binom_two_sided(min(more_fail, more_win), more_fail + more_win),
                     "early (first 6 calls) typos: fail > win": e_more_fail, "win > fail": e_more_win,
                     "p early": binom_two_sided(min(e_more_fail, e_more_win), e_more_fail + e_more_win)})
    allatt = pd.concat(allatt)
    pr = allatt[allatt.model.isin(PRUNED)]
    for lab, sub in [("pruned models pooled, all attempts of a pair", pr), ("all eval models pooled", allatt)]:
        more_fail = more_win = e_f = e_w = 0
        for _, x in sub.groupby(["task_id", "pair"]):
            win = x[x.correct]
            los = x[~x.correct]
            for _, w in win.iterrows():
                for _, l in los.iterrows():
                    if l.typo_rate > w.typo_rate:
                        more_fail += 1
                    elif w.typo_rate > l.typo_rate:
                        more_win += 1
                    if l.typos_early > w.typos_early:
                        e_f += 1
                    elif w.typos_early > l.typos_early:
                        e_w += 1
        rows.append({"comparison": lab + " (all win x loss attempt pairs)",
                     "discordant pairs where the failing attempt had the higher typo rate": more_fail,
                     "... the winning attempt had the higher rate": more_win, "p (sign)": float("nan"),
                     "early (first 6 calls) typos: fail > win": e_f, "win > fail": e_w, "p early": float("nan")})
    table(pd.DataFrame(rows), "{:.3g}")

    # fixed-effects LPM: correct ~ any early typo, demeaned within pair (pruned pooled)
    P("### Linear probability model, pruned models pooled, pair and model fixed effects (two-way demeaned)")

    def twoway(x, col):
        v = x[col].astype(float).copy()
        for _ in range(50):
            v = v - v.groupby([x.task_id, x.pair]).transform("mean")
            v = v - v.groupby(x.model).transform("mean")
        return v
    for var in ["typos_early", "typo_rate"]:
        x = pr.copy()
        x["xv"] = (x[var] > 0).astype(float) if var == "typos_early" else x[var]
        x["yd"] = twoway(x.assign(c=x.correct.astype(float)), "c")
        x["xd"] = twoway(x, "xv")
        beta = (x.xd * x.yd).sum() / max(1e-9, (x.xd ** 2).sum())
        # cluster bootstrap over pairs
        keys = list(x.groupby(["task_id", "pair"]).groups.keys())
        grp = {k: v for k, v in x.groupby(["task_id", "pair"])}
        rng = random.Random(0)
        bs = []
        for _ in range(500):
            s = pd.concat([grp[keys[rng.randrange(len(keys))]] for _ in range(len(keys))])
            bs.append((s.xd * s.yd).sum() / max(1e-9, (s.xd ** 2).sum()))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        lab = "any identifier typo in the first 6 calls (0/1)" if var == "typos_early" else "typo errors per call"
        P(f"- {lab}: beta = {beta:+.3f} (95% cluster-bootstrap CI {lo:+.3f} to {hi:+.3f}); "
          f"{len(x)} attempts, {len(keys)} pairs.")
        key.setdefault("lpm", {})[var] = (beta, lo, hi)
    P()

    # ---------------- auto-repair ----------------
    P("## T9. Auto-repair simulation")
    P("### (A) Static pre-execution rewrite (bare names, module attributes, method names, imports)")
    rows = []
    for m in ALL:
        R = load(m, "repair")
        if R is None:
            continue
        t = T.get(m)
        intended = {}
        if t is not None and len(t):
            for r in t.itertuples():
                intended[(r.task_id, r.pair, r.attempt, r.call_seq)] = (r.typo, r.intended, r.intended_src)
        n = len(R)
        trig = [r for r in R if any(p[2] for p in r["props"])]
        trig_ok = [r for r in trig if r["status"] == "completed"]
        trig_name_ok = [r for r in trig_ok if any(p[2] and p[0] == "name" for p in r["props"])]
        typo_calls = [r for r in R if (*r["key"], r["call_seq"]) in intended]
        hit = right = wrong = conf = conf_right = 0
        for r in typo_calls:
            ty, iv, src = intended[(*r["key"], r["call_seq"])]
            last = ty.split(".")[-1]
            p = [x for x in r["props"] if x[1] == last]
            if p and p[0][2]:
                hit += 1
                if src in ("fix=vocab", "fix", "fix-span", "fix-far"):
                    conf += 1
                    if p[0][2] == iv.split(".")[-1] or (src == "fix-span" and p[0][2] in iv):
                        conf_right += 1
        A_ = load(m, "attempts")
        codes = {}
        for a in A_:
            for qi, c in enumerate([c for tr in a["turns"] for c in tr["calls"]]):
                codes[(a["task_id"], a["pair"], a["attempt"], qi)] = c["code"] or ""
        dead = 0
        for r in trig_name_ok:
            code = codes.get((*r["key"], r["call_seq"]), "")
            names = [p[1] for p in r["props"] if p[2] and p[0] == "name"]
            if all(any(nm in line and "if False" in line for line in code.splitlines()) for nm in names):
                dead += 1
        rows.append({"model": m, "calls": n, "calls with a rewrite": len(trig),
                     "bare-name rewrites on fine calls in an 'if False' line": dead,
                     "... on calls that ran fine": len(trig_ok),
                     "... of which a bare-name rewrite": len(trig_name_ok),
                     "typo calls": len(typo_calls), "typo name rewritten %": 100 * hit / max(1, len(typo_calls)),
                     "rewrite = model's own fix (where known)": f"{conf_right}/{conf}"})
    table(pd.DataFrame(rows), "{:.1f}")

    P("### (B) Retry once after the error, replayed in a local Python session")
    rows = []
    for m in ALL:
        R = load(m, "replay")
        if R is None:
            continue
        ok = [r for r in R if "fail" not in r]
        rep = [r for r in ok if r["replay_matches"]]
        rew = [r for r in rep if r["rewrites"]]
        comp = [r for r in rew if r["final"] and r["final"][0] == "completed"]
        other = Counter((r["final"][1] if r["final"] else None) for r in rew if r["final"] and r["final"][0] != "completed")
        rows.append({"model": m, "identifier-error calls replayed": len(ok),
                     "replay reproduces the recorded exception": len(rep),
                     "retry proposed": len(rew), "retry proposed %": 100 * len(rew) / max(1, len(rep)),
                     "cell completes after retry": len(comp),
                     "completes % of reproduced": 100 * len(comp) / max(1, len(rep)),
                     "2+ rewrites needed": sum(1 for r in rew if len(r["rewrites"]) > 1),
                     "still failing: top exceptions": ", ".join(f"{k}:{v}" for k, v in other.most_common(4))})
        # agreement of the retry rewrite with the model's own fix
        t = T.get(m)
        if t is not None and len(t):
            fx = {(r.task_id, r.pair, r.attempt, r.call_seq): (r.typo, r.intended, r.intended_src) for r in t.itertuples()}
            agree = disagree = 0
            ex_dis = []
            for r in rew:
                k = (*r["key"], r["call_seq"])
                if k in fx and fx[k][2] in ("fix=vocab", "fix", "fix-far"):
                    ty, iv, _ = fx[k]
                    w = r["rewrites"][0]
                    if w[0] == ty.split(".")[-1]:
                        if w[1] == iv.split(".")[-1]:
                            agree += 1
                        else:
                            disagree += 1
                            ex_dis.append((ty, w[1], iv))
            rows[-1]["first rewrite = model's own fix"] = f"{agree}/{agree + disagree}"
            rows[-1]["disagreements (typo->rewrite vs model fix)"] = "; ".join(f"{a}->{b} vs {c}" for a, b, c in ex_dis[:4])
    table(pd.DataFrame(rows), "{:.1f}")

    # ---------------- token classes ----------------
    P("## T10. Candidate token classes (as prediction targets) in generated text")
    rows = []
    for m in ["TRAIN"] + EVAL + DEV:
        f = f"{OUT}/tokclass_{m}.json"
        if not os.path.exists(f):
            continue
        c = json.load(open(f))
        for seg in ["code", "reason"]:
            allc = c.get(f"{seg}|all", 0)
            if not allc:
                continue
            r = {"model": m, "segment": seg, "tokens": allc}
            for k in ["word", "word-first", "WIC", "WIC-alnum", "WIC-copied", "WIC-alnum-copied", "WIC-retok-copy",
                      "WIC-alnum-retok-copy", "WIC-fresh", "copied-word-token", "retok-copy-token", "def-site",
                      "def-site-WIC"]:
                r[k + " %"] = 100 * c.get(f"{seg}|{k}", 0) / allc
            rows.append(r)
    if rows:
        table(pd.DataFrame(rows), "{:.2f}")

    P("## T11. Coverage of typo loci by candidate classes vs class share of tool-block tokens (TRAIN rollouts)")
    c = json.load(open(f"{OUT}/tokclass_TRAIN.json"))
    allc = c["code|all"]
    share = {"WIC-alnum (target = word-internal continuation at alnum|alnum)": c["code|WIC-alnum"] / allc,
             "WIC any (incl. _ boundaries)": c["code|WIC"] / allc,
             "WIC-alnum in a re-tokenized copy": c["code|WIC-alnum-retok-copy"] / allc,
             "WIC in a fresh word (not seen before)": c["code|WIC-fresh"] / allc,
             "all tokens of copied words": c["code|copied-word-token"] / allc,
             "all tokens of re-tokenized copies": c["code|retok-copy-token"] / allc,
             "def/import-site tokens": c["code|def-site"] / allc}
    rows = []
    for m in PRUNED:
        t = T[m]
        n = len(t)
        prev = t.i_prev.astype(bool)
        retok = prev & ~t.i_tok_match_any.astype(bool)
        mid = t.btype == "mid"
        wic = t.btype.isin(["mid", "sym"])
        line_def = t.line.fillna("").str.match(r"\s*(import|from|def|class)\s") | t.kind.isin(["module", "import"])
        cov = {"WIC-alnum (target = word-internal continuation at alnum|alnum)": mid.mean(),
               "WIC any (incl. _ boundaries)": wic.mean(),
               "WIC-alnum in a re-tokenized copy": (mid & retok).mean(),
               "WIC in a fresh word (not seen before)": (wic & ~prev).mean(),
               "all tokens of copied words": prev.mean(),
               "all tokens of re-tokenized copies": retok.mean(),
               "def/import-site tokens": line_def.mean()}
        for k in share:
            rows.append({"model": m, "class": k, "share of typos %": 100 * cov[k],
                         "share of tool-block tokens %": 100 * share[k], "enrichment": cov[k] / share[k]})
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="class", columns="model", values=["share of typos %", "enrichment"]).round(1)
    piv.columns = [f"{a} {b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv.insert(1, "share of TRAIN tool-block tokens %", [round(100 * share[k], 2) for k in piv["class"]])
    table(piv, "{:.1f}")

    open(f"{OUT}/tables.md", "w").write("\n".join(md))
    json.dump(key, open(f"{OUT}/key_numbers.json", "w"), indent=1, default=str)
    print("\n".join(md))


if __name__ == "__main__":
    main()
