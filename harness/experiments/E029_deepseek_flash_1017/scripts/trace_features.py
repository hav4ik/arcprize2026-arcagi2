"""E029 trace analysis ($0, offline): per-attempt and per-turn features of DeepSeek-Flash tool traces,
old model (V4.1-Flash, E009/E010) vs new (V4.1-Flash-1017, E029). Reuses E026's text features (ds_features.py).

Output: experiments/E029_deepseek_flash_1017/trace_features.jsonl (one row per attempt, with per-turn lists).
Run: .venv/bin/python -I experiments/E029_deepseek_flash_1017/scripts/trace_features.py
"""
import gzip, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/E026_trace_strategy/scripts"))
from ds_features import text_feats, is_grid_line, SOLVE_DEF, COORD, RC  # noqa: E402

RES = ROOT / "evals/results"
OUT = ROOT / "experiments/E029_deepseek_flash_1017/trace_features.jsonl"
P = "tools_persistent_grid_lean2sub_deepseek-flash_low_"
RUNS = {
    ("old", "sub80s40"): RES / f"{P}e009_full_sub80s40.jsonl",
    ("new", "sub80s40"): RES / f"{P}e029_flash1017_sub80s40.jsonl",
    ("old", "s20x60"): RES / f"{P}e010_full_s20x60.jsonl",
    ("new", "s20x60"): RES / f"{P}e029_flash1017_s20x60.jsonl",
}
SOL = json.load(open(ROOT / "data/arc-agi_evaluation_solutions.json"))
CHALL = json.load(open(ROOT / "data/arc-agi_evaluation_challenges.json"))

# ---------------- grid extraction from tool output ----------------
BRACKET = re.compile(r"\[\s*(?:\[[\d,\s]*\][\s,]*)+\]")
ROW = re.compile(r"\[([\d,\s]*)\]")
FLATROW = re.compile(r"^\s*(?:\w+\s*\d*\s*[:=]?\s*)?\[\s*(\d(?:[\s,]+\d)+)\s*\]\s*$")
PLAIN = re.compile(r"^\s*(?:\d{1,2}\s*[:|]\s*)?((?:\d[ ,]?){2,30})\s*$")


def _ok(g):
    return g and all(len(r) == len(g[0]) and len(r) > 0 for r in g) and len(g) <= 30 and len(g[0]) <= 30


def extract_grids(text):
    """All rectangular digit grids printed in a tool output: nested lists, numpy prints, and plain digit-row blocks."""
    grids = []
    for m in BRACKET.finditer(text):
        rows = []
        for r in ROW.findall(m.group(0)):
            vals = [v for v in re.split(r"[,\s]+", r.strip()) if v]
            if not all(v.isdigit() for v in vals):
                rows = None
                break
            rows.append(tuple(int(v) for v in vals))
        if rows and _ok(rows):
            grids.append(tuple(rows))
    # plain blocks: consecutive lines of digits (e.g. ''.join rows, or space-separated)
    block = []
    for line in text.splitlines() + [""]:
        m = PLAIN.match(line) if "[" not in line else None
        ml = FLATROW.match(line)
        if m or ml:
            s = m.group(1) if m else ml.group(1)
            vals = [int(c) for c in re.findall(r"\d", s)] if (" " not in s and "," not in s) else \
                [int(v) for v in re.split(r"[ ,]+", s.strip()) if v]
            if block and len(vals) != len(block[-1]):
                if len(block) >= 2:
                    grids.append(tuple(block))
                block = []
            block.append(tuple(vals))
        else:
            if len(block) >= 2 and _ok(block):
                grids.append(tuple(block))
            block = []
    return grids


def as_grid(x):
    if x is None:
        return None
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            return None
    try:
        return tuple(tuple(int(v) for v in r) for r in x)
    except Exception:
        return None


# ---------------- text markers ----------------
WORD = re.compile(r"\S+")
HEDGE = {k: re.compile(rf"\b{k}\b", re.I) for k in ["wait", "hmm", "actually", "maybe"]}
REVERIFY = re.compile(r"double[- ]check|\bre-?check|\bre-?verify|verify (?:again|once more)|check (?:again|once more)|"
                      r"let(?:'s| me| us) (?:verify|confirm|re-?examine|recheck|double)|sanity[- ]check|to be (?:safe|sure)|"
                      r"make sure", re.I)
# explicit intent to use the tool: first-person/should + a code/python/tool mention within 40 chars (validated on samples;
# a looser version incl. bare "let's compute" mostly matched mental computation)
CODE_INTENT = re.compile(r"\b(?:let me|let's|let us|i'll|i will|we'll|we can|better(?: to)?|easier(?: to)?|should|need to)\b"
                         r"[^.\n]{0,40}?\b(?:code|python|script|programmatically|the tool|run_python)\b", re.I)
MENTAL = re.compile(r"\blet(?:'s| me| us) (?:compute|calculate|count|check|verify|list|enumerate|trace|simulate)\b"
                    r"(?![^.\n]{0,40}?\b(?:code|python|script|programmatically)\b)", re.I)
GIVEUP = re.compile(r"given (?:the )?(?:time|budget|limited)|time is limited|let'?s just (?:go|guess|submit|output)|"
                    r"best guess|can'?t figure|I'll guess|go with (?:this|that|it)", re.I)
DEBUG_PRINT = re.compile(r"print\([^)\n]*(?:mismatch|diff|wrong|expected|got|!=|where|argwhere|np\.where)", re.I)
TRAIN_LOOP = re.compile(r"(?:for\s+[\w, ()]+\s+in\s+(?:enumerate\()?\s*train|train\[\s*\w+\s*\])")
CMP = re.compile(r"array_equal|==|\.all\(\)|tolist\(\)\s*==|np\.all|\bsum\(.*!=")
SOLVE_CALL = re.compile(r"\b(?:solve|transform|apply|predict|f|rule|process|run)\w*\s*\(")
TEST_CALL = re.compile(r"\b(?!np\b|print\b|array\b|len\b|list\b)(\w+)\s*\(\s*(?:np\.array\()?\s*test_input\b")
DEFN = re.compile(r"^(?:def\s+(\w+)|(\w+)\s*=(?!=))", re.M)


def norm_row(l):
    d = re.findall(r"\d", is_grid_line_body(l))
    return "".join(d)


def is_grid_line_body(l):
    from ds_features import LABEL
    return LABEL.sub("", l)


def seen_rows(text):
    """Digit-row signatures (>=5 digits) of grid rows contained in a text block (tool output or prompt)."""
    out = set()
    for g in extract_grids(text):
        for r in g:
            if len(r) >= 5:
                out.add("".join(str(v) for v in r))
    return out


def attempt(run_key, r):
    try:
        with gzip.open(r["trace_path"], "rt") as fh:
            d = json.load(fh)
    except Exception:
        return None
    msgs, events = d["messages"], d.get("events", [])
    usage = [e for e in events if "usage" in e]
    status = {e["tool_call_id"]: e["status"] for e in events if "status" in e}
    toolout = {m["tool_call_id"]: m.get("content") or "" for m in msgs if m["role"] == "tool"}
    turns = [m for m in msgs if m["role"] == "assistant"]
    n = min(len(turns), len(usage))
    turns, usage = turns[:n], usage[:n]
    task, pair = r["task_id"], int(r["pair"])
    sol = as_grid(SOL[task][pair])
    test_in = as_grid(CHALL[task]["test"][pair]["input"])
    pred = as_grid(r.get("pred"))
    prompt_rows = seen_rows(msgs[0]["content"])

    T = []  # per-turn
    seen = set(prompt_rows)
    seen_tool = set()
    for i, m in enumerate(turns):
        u = usage[i]["usage"]
        tok = u.get("completion_tokens", 0) or 0
        rtok = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        rc = m.get("reasoning_content") or ""
        ct = m.get("content") or ""
        calls = []
        for c in m.get("tool_calls") or []:
            try:
                a = json.loads(c["function"]["arguments"])
                code = a.get("code") or ""
            except Exception:
                code = ""
            out = toolout.get(c["id"], "")
            calls.append({"name": c["function"]["name"], "code": code, "out": out, "status": status.get(c["id"])})
        # reasoning markers
        lines = rc.splitlines()
        glines = [l for l in lines if is_grid_line(l)]
        g_reread_tool = g_reread_prompt = 0
        for l in glines:
            sig = norm_row(l)
            if len(sig) >= 5:
                if sig in seen_tool:
                    g_reread_tool += 1
                elif sig in prompt_rows:
                    g_reread_prompt += 1
        cpt = len(rc) / rtok if rtok else None
        ci = CODE_INTENT.search(rc)
        tail_chars = len(rc) - ci.end() if ci else 0
        tf = text_feats(rc)
        t = {
            "tok": tok, "rtok": rtok, "vtok": tok - rtok, "rchars": len(rc), "cchars": len(ct),
            "words": len(WORD.findall(rc)), "finish": usage[i].get("finish_reason"),
            "grid_lines": tf["grid_lines"], "grid_chars": tf["grid_chars"], "coord_tuples": tf["coord_tuples"],
            "coord_own_chars": tf["coord_own_chars"], "coord_own_lines": tf["coord_own_lines"],
            "reread_tool": g_reread_tool, "reread_prompt": g_reread_prompt,
            "hedge": {k: len(rx.findall(rc)) for k, rx in HEDGE.items()},
            "reverify": len(REVERIFY.findall(rc)), "giveup": len(GIVEUP.findall(rc)),
            "code_intent": len(CODE_INTENT.findall(rc)), "mental": len(MENTAL.findall(rc)),
            "ci_tail_tok": (tail_chars / cpt if cpt else tail_chars / 3.5) if ci else None,
            "n_calls": len(calls), "names": [c["name"] for c in calls],
            "has_def": any(c["name"] == "run_python" and SOLVE_DEF.search(c["code"]) for c in calls),
            "uses_test": any(c["name"] == "run_python" and "test_input" in c["code"] for c in calls),
            "err": any(c["status"] in ("error", "timeout") or "Traceback" in c["out"][:4000] for c in calls),
            "out_chars": sum(len(re.sub(r"\n\[budget:[^\n]*", "", c["out"])) for c in calls),
            "code_chars": sum(len(c["code"]) for c in calls),
            "debug_print": sum(len(DEBUG_PRINT.findall(c["code"])) for c in calls),
            "n_print": sum(c["code"].count("print(") for c in calls),
            "submit": any(c["name"] == "submit_answer" for c in calls),
            "submit_code": next((c["code"] for c in calls if c["name"] == "submit_answer"), None),
            "defs": sorted({n for c in calls if c["name"] == "run_python" for a, b in DEFN.findall(c["code"]) for n in [a or b]}),
        }
        # grids produced in tool output
        has_pred = has_sol = False
        test_grids = []
        train_check = None
        for c in calls:
            if c["name"] != "run_python":
                continue
            gs = extract_grids(c["out"])
            if pred is not None and pred in gs:
                has_pred = True
            if sol in gs:
                has_sol = True
            if TEST_CALL.search(c["code"]):
                test_grids += [g for g in gs if g != test_in]
            if TRAIN_LOOP.search(c["code"]) and CMP.search(c["code"]) and SOLVE_CALL.search(c["code"]):
                o = c["out"]
                nt, nf = len(re.findall(r"\bTrue\b", o)), len(re.findall(r"\bFalse\b", o))
                bad = bool(re.search(r"Traceback|mismatch|MISMATCH|wrong|FAIL|Error", o))
                train_check = {"true": nt, "false": nf, "allpass": nt > 0 and nf == 0 and not bad}
            for g in gs:
                for row in g:
                    if len(row) >= 5:
                        seen_tool.add("".join(str(v) for v in row))
        t["has_pred"], t["has_sol"] = has_pred, has_sol
        t["test_grids"] = [[g == sol, g == pred] for g in test_grids]
        t["train_check"] = train_check
        T.append(t)

    f = {"run": run_key[0], "cfg": run_key[1], "task_id": task, "pair": pair, "attempt": int(r["attempt"]),
         "correct": bool(r.get("correct")), "has_pred": pred is not None, "stop": r.get("stop_reason"),
         "source": r.get("answer_source"), "tokens": sum(t["tok"] for t in T), "turns": T,
         "n_train": len(CHALL[task]["train"])}
    return f


def main():
    rows = []
    for key, path in RUNS.items():
        latest = {}
        for line in open(path):
            r = json.loads(line)
            if "error" in r:
                continue
            latest[(r["task_id"], int(r["pair"]), int(r["attempt"]))] = r
        k = 0
        for _, r in sorted(latest.items()):
            f = attempt(key, r)
            if f is not None:
                rows.append(f)
                k += 1
        print(key, k, "attempts")
    with open(OUT, "w") as fh:
        for f in rows:
            fh.write(json.dumps(f) + "\n")
    print(len(rows), "rows ->", OUT)


if __name__ == "__main__":
    main()
