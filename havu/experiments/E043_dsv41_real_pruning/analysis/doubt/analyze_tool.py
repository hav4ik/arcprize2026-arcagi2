#!/usr/bin/env python3
"""E043 doubt analysis, step 2: tool-harness questions 1-4 from the extract.py caches.

  analyze_tool.py --cache DIR --out results/tool_stats.json

Pairing: per (task, pair), each model's attempts are averaged first, then pairs are averaged (all 167 pairs have
2 attempts in full_api / E128 / E112 / E96). The self-hosted full model (172 finished attempts, shorter ones) is
compared on exactly the same (task, pair, attempt) keys.
"""
import argparse
import json
import math
import os
import pickle
import random
import re
import statistics as st
from collections import Counter, defaultdict

MODELS = ["full_api", "full_sh", "E128", "E112", "E96"]
PRUNED = ["E128", "E112", "E96"]
CORE = ("wait", "hmm", "actually", "recheck", "hold_on", "unsure", "reconsider")


def load(cache, m):
    return {tuple(a["key"]): a for a in pickle.load(open(os.path.join(cache, f"{m}.pkl"), "rb"))}


def first_pass(a):
    for i, t in enumerate(a["turns"]):
        if t["verdict"] == "pass":
            return i
    return None


def context(a, i, fp):
    t = a["turns"]
    if i == 0:
        return "first"
    if t[i - 1]["err"]:
        return "after_error"
    if fp is not None and i == fp + 1:
        return "post_pass_first"
    if fp is not None and i > fp + 1:
        return "post_pass_later"
    if t[i - 1]["verdict"] == "fail":
        return "after_fail_check"
    return "explore"


CTXS = ["first", "explore", "after_fail_check", "after_error", "post_pass_first", "post_pass_later"]


def attempt_summary(a):
    T = a["turns"]
    fp = first_pass(a)
    comp = sum(t["comp"] for t in T)
    think = sum(t["think_tok"] for t in T)
    code = sum(t["code_tok"] for t in T)
    ans = sum(t["sub_tok"] + t["content_tok"] for t in T)
    rec = sum(T[i]["comp"] for i in range(1, len(T)) if T[i - 1]["err"])
    rec_wide = rec + sum(t["code_tok"] for t in T if t["err"])
    post = sum(t["comp"] for i, t in enumerate(T) if fp is not None and i > fp)
    post_nonrec = sum(T[i]["comp"] for i in range(len(T)) if fp is not None and i > fp and not (i and T[i - 1]["err"]))
    pre_nonrec = sum(T[i]["comp"] for i in range(len(T)) if (fp is None or i <= fp) and not (i and T[i - 1]["err"]))
    errs = sum(t["err"] for t in T)
    calls = sum(t["has_code"] for t in T)
    return {
        "comp": comp, "usage": a["comp"], "think": think, "code": code, "ans": ans, "over": comp - think - code - ans,
        "turns": len(T), "think_per_turn": think / max(1, len(T)), "fp": fp, "reached": fp is not None,
        "pre": comp - post, "post": post, "turns_post": (len(T) - fp - 1) if fp is not None else 0,
        "a_rec": rec, "a_rec_wide": rec_wide, "b_post": post_nonrec,
        "c_pre_pass": pre_nonrec if fp is not None else 0, "c_never": pre_nonrec if fp is None else 0,
        "d_resid": a["comp"] - comp, "errs": errs, "calls": calls,
        "cap": a["stop"] == "max_turns", "budget": a["stop"] == "output_budget", "correct": a["correct"],
        "words": sum(t["words"] for t in T), "core": sum(t["core"] for t in T),
    }


def pair_means(runs, keys_by_pair, f):
    out = {}
    for pr, ks in keys_by_pair.items():
        out[pr] = st.mean(f(runs[k]) for k in ks)
    return out


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


def sign_test(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = {}
    data = {m: load(a.cache, m) for m in MODELS + ["dev_E128_t1", "dev_E128_minp05", "dev_E128_topp95"]}
    S = {m: {k: attempt_summary(x) for k, x in d.items()} for m, d in data.items()}

    # ------------------------------------------------------------------ sanity: turn comps sum to usage
    R["sanity"] = {m: {"attempts": len(S[m]), "resid_tokens_total": sum(s["d_resid"] for s in S[m].values()),
                       "comp_total": sum(s["comp"] for s in S[m].values())} for m in S}

    # ------------------------------------------------------------------ Q1 per-model means (all attempts)
    def means(ss):
        n = len(ss)
        g = lambda f: sum(f(s) for s in ss) / n  # noqa: E731
        comp = g(lambda s: s["comp"])
        reached = [s for s in ss if s["reached"]]
        return {
            "n": n, "acc": g(lambda s: s["correct"]), "comp": comp, "think": g(lambda s: s["think"]),
            "code": g(lambda s: s["code"]), "ans": g(lambda s: s["ans"]), "over": g(lambda s: s["over"]),
            "think_share": g(lambda s: s["think"]) / comp, "code_share": g(lambda s: s["code"]) / comp,
            "turns": g(lambda s: s["turns"]),
            "think_per_turn": sum(s["think"] for s in ss) / sum(s["turns"] for s in ss),
            "comp_per_turn": sum(s["comp"] for s in ss) / sum(s["turns"] for s in ss),
            "reached": len(reached) / n, "pre": g(lambda s: s["pre"]), "post": g(lambda s: s["post"]),
            "post_share": g(lambda s: s["post"]) / comp,
            "post_share_reached": (sum(s["post"] for s in reached) / max(1, sum(s["comp"] for s in reached))),
            "turns_post_reached": (sum(s["turns_post"] for s in reached) / max(1, len(reached))),
            "submit_next_turn": sum(s["turns_post"] == 1 for s in reached) / max(1, len(reached)),
            "cap": sum(s["cap"] for s in ss), "budget": sum(s["budget"] for s in ss),
            "py_err_share": sum(s["errs"] for s in ss) / max(1, sum(s["calls"] for s in ss)),
            "core_per_1k": 1000 * sum(s["core"] for s in ss) / max(1, sum(s["words"] for s in ss)),
            "a_rec": g(lambda s: s["a_rec"]), "a_rec_wide": g(lambda s: s["a_rec_wide"]),
            "b_post": g(lambda s: s["b_post"]), "c_pre_pass": g(lambda s: s["c_pre_pass"]),
            "c_never": g(lambda s: s["c_never"]), "d_resid": g(lambda s: s["d_resid"]),
        }

    R["means_all"] = {m: means(list(S[m].values())) for m in S}

    # ------------------------------------------------------------------ Q1/Q3 paired vs full_api (pair-level)
    def by_pair(m):
        d = defaultdict(list)
        for k in S[m]:
            d[k[:2]].append(k)
        return d

    paired = {}
    for ref in ["full_api"]:
        for m in ["full_sh"] + PRUNED:
            if m == "full_sh":
                continue
            bp_ref, bp_m = by_pair(ref), by_pair(m)
            prs = sorted(set(bp_ref) & set(bp_m))
            row = {"pairs": len(prs)}
            for f in ["comp", "think", "code", "ans", "over", "turns", "pre", "post", "a_rec", "a_rec_wide", "b_post",
                      "c_pre_pass", "c_never", "d_resid", "correct", "core", "words"]:
                mr = st.mean(st.mean(S[ref][k][f] for k in bp_ref[p]) for p in prs)
                mm = st.mean(st.mean(S[m][k][f] for k in bp_m[p]) for p in prs)
                row[f] = {"ref": mr, "model": mm, "delta": mm - mr}
            paired[f"{m}_vs_{ref}"] = row
    # attempt-level pairing vs the self-hosted full model (172 finished attempts)
    for m in ["full_api"] + PRUNED:
        ks = sorted(set(S["full_sh"]) & set(S[m]))
        row = {"attempts": len(ks)}
        for f in ["comp", "think", "code", "ans", "turns", "pre", "post", "a_rec", "a_rec_wide", "b_post",
                  "c_pre_pass", "c_never", "correct", "core", "words"]:
            mr = st.mean(S["full_sh"][k][f] for k in ks)
            mm = st.mean(S[m][k][f] for k in ks)
            row[f] = {"ref": mr, "model": mm, "delta": mm - mr}
        paired[f"{m}_vs_full_sh"] = row
    R["paired"] = paired

    # Decomposition of the extra length, stratified by whether the pair is solved by the reference
    decomp = {}
    for m in PRUNED:
        bp_ref, bp_m = by_pair("full_api"), by_pair(m)
        prs = sorted(set(bp_ref) & set(bp_m))
        strata = {"all": prs,
                  "ref_reaches_pass_both": [p for p in prs if all(S["full_api"][k]["reached"] for k in bp_ref[p])],
                  "ref_correct_both": [p for p in prs if all(S["full_api"][k]["correct"] for k in bp_ref[p])],
                  "model_and_ref_correct_all": [p for p in prs if all(S["full_api"][k]["correct"] for k in bp_ref[p])
                                                and all(S[m][k]["correct"] for k in bp_m[p])]}
        dd = {}
        for sname, ps in strata.items():
            row = {"pairs": len(ps)}
            for f in ["comp", "a_rec", "b_post", "c_pre_pass", "c_never", "d_resid", "turns"]:
                mr = st.mean(st.mean(S["full_api"][k][f] for k in bp_ref[p]) for p in ps)
                mm = st.mean(st.mean(S[m][k][f] for k in bp_m[p]) for p in ps)
                row[f] = {"ref": mr, "model": mm, "delta": mm - mr}
            dd[sname] = row
        decomp[m] = dd
    R["decomp"] = decomp

    # Within the pre-pass exploration of attempts that reach a pass: more turns or longer turns?
    expl = {}
    for m in MODELS:
        rows = []
        for k, a_ in data[m].items():
            s = S[m][k]
            if not s["reached"]:
                continue
            T = a_["turns"]
            pre_turns = [T[i] for i in range(s["fp"] + 1) if not (i and T[i - 1]["err"])]
            rows.append((len(pre_turns), sum(t["comp"] for t in pre_turns),
                         sum(1 for i in range(s["fp"] + 1) if i and T[i - 1]["err"]),
                         sum(t["check"] and t["verdict"] == "fail" for t in T[:s["fp"] + 1])))
        expl[m] = {"attempts": len(rows), "pre_nonrec_turns": st.mean(r[0] for r in rows),
                   "pre_nonrec_tokens": st.mean(r[1] for r in rows),
                   "pre_nonrec_tok_per_turn": sum(r[1] for r in rows) / max(1, sum(r[0] for r in rows)),
                   "pre_err_recovery_turns": st.mean(r[2] for r in rows),
                   "pre_failed_checks": st.mean(r[3] for r in rows)}
    R["exploration"] = expl

    # ------------------------------------------------------------------ Q2 markers
    mk_rates = {}
    for m in MODELS + ["dev_E128_t1"]:
        tot = Counter()
        words = 0
        for a_ in data[m].values():
            for t in a_["turns"]:
                tot.update(t["mk"])
                words += t["words"]
        mk_rates[m] = {"words": words, **{k: 1000 * v / words for k, v in tot.items()},
                       "core": 1000 * sum(tot[k] for k in CORE if k != "wait") / words + 1000 * tot["wait"] / words}
    R["marker_rates"] = mk_rates

    ctx_stats = {}
    for m in MODELS:
        c_words, c_core, c_turns, c_tok, c_turn_with = Counter(), Counter(), Counter(), Counter(), Counter()
        for a_ in data[m].values():
            fp = first_pass(a_)
            for i, t in enumerate(a_["turns"]):
                c = context(a_, i, fp)
                c_words[c] += t["words"]
                c_core[c] += t["core"]
                c_turns[c] += 1
                c_tok[c] += t["comp"]
                c_turn_with[c] += t["core"] > 0
        tw, tc = sum(c_words.values()), sum(c_core.values())
        ctx_stats[m] = {c: {"turns": c_turns[c], "tokens": c_tok[c], "word_share": c_words[c] / tw,
                            "marker_share": c_core[c] / max(1, tc),
                            "core_per_1k": 1000 * c_core[c] / max(1, c_words[c]),
                            "turns_with_marker": c_turn_with[c] / max(1, c_turns[c]),
                            "tok_per_turn": c_tok[c] / max(1, c_turns[c])} for c in CTXS}
    R["contexts"] = ctx_stats

    # marker position within the turn, and tokens after the marker until the turn ends
    pos = {}
    for m in MODELS:
        occ = [o for a_ in data[m].values() for o in a_["occ"]]
        ta = [o["tok_after"] for o in occ if o["tok_after"] is not None]
        pos[m] = {"n": len(occ), "first_10pct": sum(o["pos"] < 0.1 for o in occ) / max(1, len(occ)),
                  "last_20pct": sum(o["pos"] > 0.8 for o in occ) / max(1, len(occ)),
                  "tok_after_median": st.median(ta) if ta else None,
                  "kinds": dict(Counter(o["kind"] for o in occ))}
    R["marker_position"] = pos

    # re-verification after success
    rever = {}
    for m in MODELS:
        n_att = n_post_turns = n_recheck_pass = n_same_code = n_post_fail = n_post_err = 0
        tok_recheck = tok_post = 0
        post_with_marker = 0
        for a_ in data[m].values():
            fp = first_pass(a_)
            if fp is None:
                continue
            n_att += 1
            for i, t in enumerate(a_["turns"][fp + 1:], start=fp + 1):
                n_post_turns += 1
                tok_post += t["comp"]
                post_with_marker += t["core"] > 0
                if t["check"] and t["verdict"] == "pass":
                    n_recheck_pass += 1
                    tok_recheck += t["comp"]
                if t["sim_pass"] is not None and t["sim_pass"] >= 0.6:
                    n_same_code += 1
                if t["verdict"] == "fail":
                    n_post_fail += 1
                if t["err"]:
                    n_post_err += 1
        rever[m] = {"attempts_reaching_pass": n_att, "post_turns": n_post_turns,
                    "post_turns_per_attempt": n_post_turns / max(1, n_att),
                    "post_tokens_per_attempt": tok_post / max(1, n_att),
                    "recheck_pass_turns": n_recheck_pass, "recheck_pass_tokens_per_attempt": tok_recheck / max(1, n_att),
                    "same_code_rerun_turns": n_same_code, "post_fail_checks": n_post_fail, "post_err": n_post_err,
                    "post_turns_with_marker": post_with_marker / max(1, n_post_turns)}
    R["reverification"] = rever

    # repetition
    rep = {}
    for m in MODELS:
        r6 = [t["rep6"] for a_ in data[m].values() for t in a_["turns"] if t["rep6"] is not None]
        xw = sum(t["xrep"] * t["words"] for a_ in data[m].values() for t in a_["turns"])
        w = sum(t["words"] for a_ in data[m].values() for t in a_["turns"])
        rep[m] = {"rep6_mean": st.mean(r6), "rep6_p90": sorted(r6)[int(0.9 * len(r6))],
                  "rep6_gt_0.3": sum(x > 0.3 for x in r6) / len(r6), "turns_ge50w": len(r6),
                  "xturn_repeated_sentence_word_share": xw / w}
    R["repetition"] = rep

    # ------------------------------------------------------------------ Q4 is doubt useful?
    # turn-level: what does a turn with a core marker do next, vs a turn without
    useful = {}
    for m in MODELS:
        cnt = defaultdict(Counter)
        for a_ in data[m].values():
            T = a_["turns"]
            for i, t in enumerate(T):
                if i == 0:
                    continue
                g = "marker" if t["core"] > 0 else "none"
                if "submit_answer" in t["names"]:
                    act = "submit"
                elif t["sim_prev"] is None:
                    act = "no_code"
                elif t["sim_prev"] >= 0.8:
                    act = "rerun_same"
                elif t["sim_prev"] >= 0.3:
                    act = "edit"
                else:
                    act = "new_code"
                cnt[g][act] += 1
                # did the check outcome improve (fail -> pass) in this turn?
                prev_v = next((T[j]["verdict"] for j in range(i - 1, -1, -1) if T[j]["verdict"]), None)
                if prev_v == "fail" and t["verdict"] == "pass":
                    cnt[g]["fail_to_pass"] += 1
        useful[m] = {g: dict(c) for g, c in cnt.items()}
    R["turn_outcomes"] = useful

    # post-success: does the answer change after the first passing check?  (final answer grid printed at/before
    # the first pass output -> post-pass turns did not change it)
    ans_change = {}
    for m in MODELS:
        c = Counter()
        for a_ in data[m].values():
            fp = first_pass(a_)
            if fp is None or a_["pred_len"] < 12:
                continue
            pf = next((i for i, t in enumerate(a_["turns"]) if t["pred_in_out"]), None)
            post_marker = sum(t["core"] for t in a_["turns"][fp + 1:])
            later_pass_new_code = any(t["verdict"] == "pass" and t["sim_pass"] is not None and t["sim_pass"] < 0.5
                                      for t in a_["turns"][fp + 1:])
            if pf is None:
                st_ = "pred_never_printed"
            elif pf <= fp:
                st_ = "answer_fixed_at_pass"
            else:
                st_ = "pred_first_printed_after_pass"
            c[(st_, a_["correct"])] += 1
            c[("post_marker" if post_marker else "no_post_marker", a_["correct"])] += 1
            c[("new_passing_rule_after_pass" if later_pass_new_code else "same_rule", a_["correct"])] += 1
        ans_change[m] = {f"{k[0]}|{'ok' if k[1] else 'wrong'}": v for k, v in sorted(c.items())}
    R["answer_change"] = ans_change

    # same-pair, same-model: attempts with different correctness -> is the wrong one more doubtful?
    within = {}
    for m in ["full_api", "E128", "E112", "E96"]:
        bp = defaultdict(list)
        for k in S[m]:
            bp[k[:2]].append(k)
        hi = lo = 0
        diffs = []
        for p, ks in bp.items():
            if len(ks) != 2:
                continue
            s0, s1 = S[m][ks[0]], S[m][ks[1]]
            if s0["correct"] == s1["correct"]:
                continue
            w, r_ = (s0, s1) if not s0["correct"] else (s1, s0)
            dw = 1000 * w["core"] / max(1, w["words"])
            dr = 1000 * r_["core"] / max(1, r_["words"])
            diffs.append(dw - dr)
            hi += dw > dr
            lo += dw < dr
        within[m] = {"discordant_pairs": len(diffs), "wrong_more_doubt": hi, "right_more_doubt": lo,
                     "sign_p": sign_test(hi, lo), "mean_density_diff": st.mean(diffs) if diffs else None}
    R["within_pair_doubt_vs_correct"] = within

    # cross-model same pair: E128 density vs E128 failing where full succeeded
    cross = {}
    for m in PRUNED:
        bp_ref, bp_m = by_pair("full_api"), by_pair(m)
        rows = []
        for p in sorted(set(bp_ref) & set(bp_m)):
            ref_ok = st.mean(S["full_api"][k]["correct"] for k in bp_ref[p])
            m_ok = st.mean(S[m][k]["correct"] for k in bp_m[p])
            dens_m = 1000 * sum(S[m][k]["core"] for k in bp_m[p]) / max(1, sum(S[m][k]["words"] for k in bp_m[p]))
            dens_r = 1000 * sum(S["full_api"][k]["core"] for k in bp_ref[p]) / max(1, sum(S["full_api"][k]["words"]
                                                                                        for k in bp_ref[p]))
            rows.append((ref_ok - m_ok, dens_m - dens_r))
        xs, ys = [r[0] for r in rows], [r[1] for r in rows]
        mx, my = st.mean(xs), st.mean(ys)
        cov = sum((x - mx) * (y - my) for x, y in rows)
        corr = cov / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        worse = [y for x, y in rows if x > 0]
        same = [y for x, y in rows if x == 0]
        cross[m] = {"pairs": len(rows), "corr_accloss_vs_extra_density": corr,
                    "extra_density_when_model_worse": st.mean(worse) if worse else None, "n_worse": len(worse),
                    "extra_density_when_equal": st.mean(same) if same else None, "n_equal": len(same)}
    R["cross_model_doubt_vs_loss"] = cross

    # doubt density by attempt outcome and by quartile of attempt length
    outc = {}
    for m in MODELS:
        g = defaultdict(lambda: [0, 0, 0])
        for s in S[m].values():
            key = "correct" if s["correct"] else "wrong"
            g[key][0] += s["core"]
            g[key][1] += s["words"]
            g[key][2] += 1
        outc[m] = {k: {"attempts": v[2], "core_per_1k": 1000 * v[0] / max(1, v[1])} for k, v in g.items()}
    R["doubt_by_outcome"] = outc

    # dev split (non-eval tasks): E128 sampler arms
    R["dev"] = {m: means(list(S[m].values())) for m in ["dev_E128_t1", "dev_E128_minp05", "dev_E128_topp95"]}

    json.dump(R, open(a.out, "w"), indent=1, default=float)
    print(json.dumps({"means_all": {m: {k: round(v, 3) if isinstance(v, float) else v for k, v in R["means_all"][m].items()}
                                    for m in MODELS}}, indent=0)[:6000])


if __name__ == "__main__":
    main()
