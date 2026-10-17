"""E026 (DeepSeek side): per-attempt perception / strategy features from saved DeepSeek-Flash traces. No API calls.

Conditions:
  NT   no-tool, ARC Prize prompt, max effort (E001d attempt 2, reasoning saved for all 167 pairs)
  S80  tool harness sub80s40 (lean2sub, no summary), 3 samples
  S96  tool harness sub96n (lean2sub, no summary), 2 samples
  V1   E017 V1: lean2sub + --task-summary
  V3   E017 V3: lean2roles + --task-summary + --helpers
Output: experiments/E026_trace_strategy/ds_features.jsonl (one row per attempt).
Run: .venv/bin/python -I experiments/E026_trace_strategy/scripts/ds_features.py
"""
import gzip, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "evals/results"
OUT = ROOT / "experiments/E026_trace_strategy/ds_features.jsonl"

RUNS = {
    "NT": RES / "arcprize_deepseek-flash_max.jsonl",
    "S80": RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub80s40.jsonl",
    "S96": RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e009_full_sub96n.jsonl",
    "V1": RES / "tools_persistent_grid_lean2sub_deepseek-flash_low_e017_full_v1sum.jsonl",
    "V3": RES / "tools_persistent_grid_lean2roles_deepseek-flash_low_e017_full_v3all.jsonl",
}

# ---------- hand perception in reasoning text ----------
DIGIT_LINE = re.compile(r"^[\s\[\],0-9.|-]*\d[\s\[\],0-9.|-]*$")       # E017 definition
LABEL = re.compile(r"^\s*(?:[-*]\s*)?(?:(?:row|r|line|col|c|in|out|input|output|test)\s*\d{0,2}\s*(?:in|out|input|output)?"
                   r"\s*(?:\(\w+\))?\s*[:=]\s*)", re.I)
COORD = re.compile(r"\(\s*\d{1,2}\s*,\s*\d{1,2}\s*\)")
RC = re.compile(r"\b(?:r\d{1,2}\s*c\d{1,2}|c\d{1,2}\s*=\s*\d|row\s*\d{1,2}\s*col\s*\d{1,2})\b", re.I)
DIM = re.compile(r"\b\d{1,2}\s*[x×]\s*\d{1,2}\b|\b\d{1,2}\s+(?:rows|columns|cols)\b", re.I)
COUNTW = re.compile(r"\b(?:count(?:s|ed|ing)?|how many|number of|total of)\b|\b\d{1,3}\s+(?:cells|pixels|squares)\b", re.I)
OBJW = re.compile(r"\b(?:objects?|components?|blobs?|clusters?|bounding box|bbox|shapes?)\b", re.I)
HOLE = re.compile(r"\b(?:holes?|enclosed|interior|hollow)\b", re.I)
BG = re.compile(r"\bbackground\b", re.I)
DIFFW = re.compile(r"\b(?:changed?|unchanged|differ\w*|diff|same as (?:the )?input|stays? the same|preserved|kept)\b", re.I)
SYMW = re.compile(r"\b(?:symmetr\w*|mirror\w*|reflect\w*|rotat\w*|flip\w*|transpos\w*)\b", re.I)
CSTW = re.compile(r"\b(?:crop\w*|scal(?:e|ed|ing)|upscal\w*|downscal\w*|tile[sd]?|tiling|subgrid|sub-grid|zoom\w*)\b", re.I)
AUDIT = re.compile(r"(test[_ ]?input|the test|test grid|test case)[^\n]{0,120}?(differ|new |unseen|not seen|outside|"
                   r"novel|beyond|unlike|clipped|larger|bigger|more )|(differ|difference)[^\n]{0,60}?(test[_ ]?input|the test)",
                   re.I)
SUMREF = re.compile(r"\bsummary\b|\b(?:shape|object) [A-Z]\b(?![a-z])|#\d{1,2}\b", re.I)
SQUOTE = re.compile(r"bbox\s*=|#\d{1,2}\s*\(?c\d|\bsize\s*=\s*\d|shape\s*=\s*[A-Z]\b|\bshape [A-Z]\b|"
                    r"\b(?:moved|recoloured|unchanged) #\d|Outside train range")
NOISE = re.compile(r"\b(noise|irrelevant|distractor)", re.I)
UNCERT = re.compile(r"\b(?:maybe|perhaps|possibly|not sure|unclear|hmm|wait)\b", re.I)
GIVEUP = re.compile(r"given (?:the )?(?:time|budget|limited)|time is limited|let'?s just (?:go|guess|submit|output)|"
                    r"best guess|can'?t figure|I'll guess|go with (?:this|that|it)", re.I)


def is_grid_line(l):
    s = LABEL.sub("", l)
    return len(s.strip()) >= 7 and bool(DIGIT_LINE.match(s)) and sum(ch.isdigit() for ch in s) >= 5


def text_feats(t):
    lines = t.splitlines()
    grid = [l for l in lines if is_grid_line(l)]
    coord = [l for l in lines if not is_grid_line(l) and (len(COORD.findall(l)) >= 2 or len(RC.findall(l)) >= 2)]
    squote = [l for l in lines if SQUOTE.search(l)]
    coord_own = [l for l in coord if not SQUOTE.search(l)]
    f = {
        "chars": len(t),
        "grid_lines": len(grid), "grid_chars": sum(len(l) + 1 for l in grid),
        "coord_lines": len(coord), "coord_chars": sum(len(l) + 1 for l in coord),
        "coord_tuples": len(COORD.findall(t)),
        "coord_own_lines": len(coord_own), "coord_own_chars": sum(len(l) + 1 for l in coord_own),
        "squote_lines": len(squote), "squote_chars": sum(len(l) + 1 for l in squote),
    }
    for k, rx in [("dim", DIM), ("count", COUNTW), ("obj", OBJW), ("hole", HOLE), ("bg", BG), ("diff", DIFFW),
                  ("sym", SYMW), ("cst", CSTW), ("audit", AUDIT), ("noise", NOISE), ("uncert", UNCERT),
                  ("giveup", GIVEUP)]:
        f["n_" + k] = len(rx.findall(t))
    f["n_summary_word"] = len(re.findall(r"\bsummary\b", t, re.I))
    f["n_summary_ref"] = len(SUMREF.findall(t))
    return f


# ---------- perception in Python code ----------
CODE = {
    "c_shape": re.compile(r"\.shape\b|len\(\s*(?:grid|g|inp|out|ex|train|test_input)"),
    "c_colours": re.compile(r"np\.unique|Counter\(|bincount|\.count\(|value_counts"),
    "c_components": re.compile(r"ndimage\.label|\blabel\(|connected|flood|bfs|deque|\.pop\(\)|neighbo|def comps|"
                               r"components?\(|objects\("),
    "c_bbox": re.compile(r"bbox|argwhere|np\.where\([^)]*\)\s*\n?.*\.min|\.min\(\)|min\(r|max\(r|nonzero"),
    "c_diff": re.compile(r"!=\s*(?:out|o\b|np\.array\(ex\[.output|outp)|(?:inp|i\b|in_)\s*!=|diff"),
    "c_d8": re.compile(r"rot90|fliplr|flipud|np\.flip|\.T\b|transpose|match_d8|d8\("),
    "c_cst": re.compile(r"np\.kron|np\.tile|\.repeat\(|np\.repeat|crop|panels\("),
    "c_print_grid": re.compile(r"print\(\s*['\"].join|\bshow\(|for row in|print\(\s*(?:g|grid|inp|out|o|a)\s*\)"),
    "c_test": re.compile(r"test_input"),
}
SOLVE_DEF = re.compile(r"def\s+(?:solve|transform|apply|predict)\w*\s*\(")


def code_feats(cells):
    """cells: list of (turn_index, code). Perception cells = cells before the first solve/transform definition."""
    first_def = next((i for i, (_, c) in enumerate(cells) if SOLVE_DEF.search(c)), None)
    pre = cells[:first_def] if first_def is not None else cells
    f = {"n_cells": len(cells), "cells_before_solve": len(pre), "first_def_cell": first_def,
         "code_chars": sum(len(c) for _, c in cells), "pre_code_chars": sum(len(c) for _, c in pre)}
    for k, rx in CODE.items():
        f[k + "_any"] = any(rx.search(c) for _, c in cells)
        f[k + "_pre"] = any(rx.search(c) for _, c in pre)
    perc_keys = ["c_shape", "c_colours", "c_components", "c_bbox", "c_diff", "c_d8", "c_cst", "c_print_grid"]
    f["perc_cells_pre"] = sum(1 for _, c in pre if any(CODE[k].search(c) for k in perc_keys))
    f["perc_cells_all"] = sum(1 for _, c in cells if any(CODE[k].search(c) for k in perc_keys)
                              and not SOLVE_DEF.search(c))
    f["perc_code_chars"] = sum(len(c) for _, c in cells if any(CODE[k].search(c) for k in perc_keys)
                               and not SOLVE_DEF.search(c))
    return f


def tool_attempt(r):
    try:
        with gzip.open(r["trace_path"], "rt") as fh:
            d = json.load(fh)
    except Exception:
        return None
    msgs, events = d["messages"], d.get("events", [])
    turn_usage = [e["usage"] for e in events if "usage" in e]
    turns = [m for m in msgs if m["role"] == "assistant"]
    texts, cells, tturns = [], [], []
    for i, m in enumerate(turns):
        rc = (m.get("reasoning_content") or "") + "\n" + (m.get("content") or "")
        texts.append(rc)
        cs = []
        for c in m.get("tool_calls") or []:
            try:
                a = json.loads(c["function"]["arguments"])
            except Exception:
                a = {}
            code = a.get("code") or a.get("expr") or ""
            if c["function"]["name"] == "run_python":
                cells.append((i, code)); cs.append(code)
        tturns.append(cs)
    tok = [u.get("completion_tokens", 0) for u in turn_usage]
    rtok = [(u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0 for u in turn_usage]
    f = text_feats("\n".join(texts))
    f.update(code_feats(cells))
    f["turn_tok"] = tok
    f["tokens"] = sum(tok)
    f["turn1_tok"] = tok[0] if tok else 0
    f["turn1_chars"] = len(texts[0]) if texts else 0
    f["turn1_grid_lines"] = text_feats(texts[0])["grid_lines"] if texts else 0
    # tokens before the first solve() definition (sum of turns up to and including the turn that writes it)
    fd = f["first_def_cell"]
    if fd is not None:
        ti = cells[fd][0]
        f["tok_before_solve"] = sum(tok[:ti + 1])
    else:
        f["tok_before_solve"] = None
    # chars/token from reasoning tokens
    rchars = sum(len(m.get("reasoning_content") or "") for m in turns)
    f["chars_per_tok"] = rchars / max(1, sum(rtok)) if sum(rtok) else None
    # per-turn text features for phase plots
    f["turn_grid_lines"] = [text_feats(t)["grid_lines"] for t in texts]
    f["turn_chars"] = [len(t) for t in texts]
    f["turn_has_def"] = [any(SOLVE_DEF.search(c) for c in cs) for cs in tturns]
    f["turn_perc_code"] = [any(any(CODE[k].search(c) for k in CODE if k != "c_test") and not SOLVE_DEF.search(c)
                               for c in cs) for cs in tturns]
    return f


def nt_attempt(r):
    p = ROOT / r["reasoning_path"]
    with gzip.open(p, "rt") as fh:
        t = fh.read()
    f = text_feats(t)
    u = r["usage"] if isinstance(r["usage"], dict) else {}
    ctok = u.get("completion_tokens", 0)
    rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or ctok
    f["tokens"] = ctok
    f["chars_per_tok"] = len(t) / max(1, rt)
    # position-resolved: deciles of grid/coord chars
    n = len(t)
    pos, acc = [0] * 10, 0
    for l in t.splitlines(keepends=True):
        if is_grid_line(l) or len(COORD.findall(l)) >= 2 or len(RC.findall(l)) >= 2:
            pos[min(9, acc * 10 // max(1, n))] += len(l)
        acc += len(l)
    f["perc_decile_chars"] = pos
    # first test-application marker
    m = re.search(r"(?:now|let'?s|next)[^\n]{0,40}?(?:apply|solve|compute|construct|do)[^\n]{0,40}?test", t, re.I)
    f["test_apply_pos"] = m.start() / n if m else None
    return f


def main():
    rows = []
    for run, path in RUNS.items():
        latest = {}
        for line in open(path):
            r = json.loads(line)
            if "error" in r:
                continue
            latest[(r["task_id"], int(r["pair"]), int(r["attempt"]))] = r
        for (t, p, a), r in sorted(latest.items()):
            if run == "NT":
                if "reasoning_path" not in r or a != 2:
                    continue
                f = nt_attempt(r)
            else:
                f = tool_attempt(r)
                if f is None:
                    continue
            f.update(run=run, task_id=t, pair=p, attempt=a, correct=bool(r.get("correct")),
                     train_pass=r.get("train_pass"))
            rows.append(f)
    with open(OUT, "w") as fh:
        for f in rows:
            fh.write(json.dumps(f) + "\n")
    print(len(rows), "rows ->", OUT)


if __name__ == "__main__":
    main()
