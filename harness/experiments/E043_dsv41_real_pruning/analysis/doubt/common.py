"""E043 doubt analysis: shared parsing for tool-harness traces (read-only).

Per attempt -> per assistant turn:
  thinking / content / code / submit-expression text, exact completion tokens (trace events),
  the tool output that follows, Python-error flag, train-check flag and an all-train-pass verdict.

The all-pass detector is code-aware: a tool output counts as "every training pair passes" only when the code that
produced it compares results with the training outputs (`['output']` + a comparison) and the output shows positive
verdicts for at least n_train pairs (or an `all(...)`-style line) and no negative verdict at all. It is validated
against the harness's own `train_pass` in report.md.
"""
import ast
import gzip
import json
import os
import re

ROOT = "/home/vu/Workspace/ARC-AGI-2"
RES = f"{ROOT}/evals/results"
TOKENIZER = f"{ROOT}/experiments/E042_dsv41_l4_synthetic/metadata/shared_metadata/tokenizer.json"
P = "tools_persistent_grid_lean2sub_"

# Tool runs. E112 = attempt 0 of the main run + attempt 0 of r2 (renamed attempt 1); main attempt 1 and r3 excluded.
TOOL_RUNS = {
    "full_api": [(P + "deepseek-flash_low_e030_hdrmin", None, None)],
    "full_sh": [(P + "dsv41_low_e043_full_ref_sub80", None, None)],
    "E128": [(P + "dsv41_low_e043_e128t_w10_u2_gptq_eng64_sub80", None, None)],
    "E112": [(P + "dsv41_low_e043_e112t_w10_u2_gptq_eng64_sub80", 0, 0),
             (P + "dsv41_low_e043_e112t_w10_u2_gptq_eng64_sub80_r2", 0, 1)],
    "E96": [(P + "dsv41_low_e043_e96t_w10_u2u3_gptq_eng64_sub80", None, None)],
    "dev_E128_t1": [(P + "dsv41_low_e043_devs_e128t_t1", None, None)],
    "dev_E128_minp05": [(P + "dsv41_low_e043_devs_e128t_minp05", None, None)],
    "dev_E128_topp95": [(P + "dsv41_low_e043_devs_e128t_topp95", None, None)],
}
TRAIN_RUNS = {
    "e031_arc2_train": P + "deepseek-flash_low_e031_arc2_train",
    "e031_nvarc_b0": P + "deepseek-flash_low_e031_nvarc_b0",
    "e036_nvarc_b1": P + "deepseek-flash_low_e036_nvarc_b1",
    "e036_nvarc_b1x": P + "deepseek-flash_low_e036_nvarc_b1x",
}
NOTOOL_RUNS = {"full": "e043nt_full_a1", "E128": "e043nt_e128_u2_gptq_eng56",
               "E96": "e043nt_e96_u2u3_gptq_eng56", "E72": "e043nt_e72_w10_eng48"}


def load_records(model):
    """Finished records of a tool model, keyed (task, pair, attempt)."""
    out = {}
    for path, only_att, rename in TOOL_RUNS[model]:
        for line in open(f"{RES}/{path}.jsonl"):
            r = json.loads(line)
            if "task_id" not in r or "error" in r:
                continue
            if only_att is not None and r["attempt"] != only_att:
                continue
            if rename is not None:
                r = dict(r, attempt=rename)
            out[(r["task_id"], r["pair"], r["attempt"])] = r
    return out


def load_train_records(name):
    out = []
    for line in open(f"{RES}/{TRAIN_RUNS[name]}.jsonl"):
        r = json.loads(line)
        if "task_id" in r and "error" not in r and r.get("trace_path"):
            out.append(r)
    return out


# ----------------------------------------------------------------------------------------------- doubt markers
MARKERS = {
    "wait": r"\b[Ww]ait\b",
    "but_wait": r"\b[Bb]ut wait\b",
    "hmm": r"\b[Hh]m{2,}\b",
    "actually": r"\b[Aa]ctually\b",
    "recheck": r"double[- ]?check|\bre-?check|\bre-?verify|verify (?:it |this |that )?again|check (?:it |this |that )?again"
               r"|[Ll]et me (?:also )?(?:re-?)?verify|[Ll]et me confirm|sanity[- ]check|[Cc]onfirm again",
    "however": r"\b[Hh]owever\b",
    "alternatively": r"\b[Aa]lternatively\b",
    "unsure": r"\bnot (?:entirely |fully |so |100% )?sure\b|\bunsure\b|\buncertain\b|\bnot certain\b|\bnot confident\b",
    "maybe": r"\b(?:[Mm]aybe|[Pp]erhaps|[Pp]ossibly)\b",
    "reconsider": r"\breconsider|\bre-?think|\bre-?examine|\bre-?read|\blook again|\bre-?evaluate|\bre-?check the",
    "hold_on": r"\b[Hh]old on\b|\b[Oo]ops\b|\b[Hh]ang on\b",
    "mistake": r"\b(?:my )?mistake\b|\bI was wrong\b|\bthat'?s wrong\b|\bwrong\b|\bincorrect\b",
}
MARKER_RE = {k: re.compile(v) for k, v in MARKERS.items()}
# "core doubt" = interjections of second thoughts / re-checking (hedges and "however" reported separately)
CORE = ("wait", "hmm", "actually", "recheck", "hold_on", "unsure", "reconsider")
CORE_RE = re.compile("|".join(f"(?:{MARKERS[k]})" for k in CORE))
WORD_RE = re.compile(r"\S+")


def marker_counts(text):
    return {k: len(rx.findall(text)) for k, rx in MARKER_RE.items()}


def core_spans(text):
    return [(m.start(), m.end(), m.group(0)) for m in CORE_RE.finditer(text)]


# ----------------------------------------------------------------------------------------------- tool outputs
PYERR = re.compile(r"Traceback|\w+Error:|\w+Exception:")
OUTPUT_REF = re.compile(r"""\[\s*['"]output['"]\s*\]|\.get\(\s*['"]output['"]|\bexp(?:ected)?\b|\btarget\b|\bgt\b|\bwant\b""")
COMPARE = re.compile(r"array_equal|==|!=|\.all\(\)|\ball\(|np\.sum\(|\.sum\(\)|\bcompare|\bcheck|\bverify|\bmatch")
NEG = re.compile(r"\bFalse\b|\bFAIL|\bfail(?:ed|s)?\b|\bwrong\b|\bincorrect\b|\bmismatch(?:es)?\b(?!\D{0,4}\b0\b)"
                 r"|✗|\bnot match|\bno match\b|\bdiffer(?:s|ent)?\b|\bMISMATCH")
NUM_BAD = re.compile(r"\b(?:diff|diffs|ndiff|mismatch(?:es)?|errors?|wrong|bad|nwrong|ndiffs|diff_count|n_diff)\b[^\d\n]{0,12}?(\d+)", re.I)
POS = re.compile(r"\bTrue\b|\bOK\b|\bPASS(?:ED)?\b|\bpassed\b|\bcorrect\b|✓")
ALL_LINE = re.compile(r"\ball\b[^\n]{0,40}\b(?:True|match|pass|correct|OK|ok|equal)", re.I)


def is_train_check(code):
    return bool(OUTPUT_REF.search(code) and COMPARE.search(code))


def verdict(code, out, n_train):
    """None (not a train check / error), 'pass' (every training pair passes) or 'fail'."""
    if not code or PYERR.search(out) or not is_train_check(code):
        return None
    body = out.split("[budget:")[0].replace("[completed]", "")
    # verdicts about the test input ("test diff count 17", "test ... False") say nothing about the training pairs
    lines = [ln for ln in body.split("\n") if not re.search(r"test", ln, re.I)]
    body = "\n".join(lines)
    if NEG.search(body):
        return "fail"
    for m in NUM_BAD.finditer(body):
        if int(m.group(1)) > 0:
            return "fail"
    pos_lines = sum(bool(POS.search(ln) or NUM_BAD.search(ln)) for ln in lines)
    if pos_lines >= max(1, n_train) or ALL_LINE.search(body) or body.strip() == "True":
        return "pass"
    if "all(" in code and any(ln.strip() == "True" for ln in lines):
        return "pass"
    return None


# ----------------------------------------------------------------------------------------------- traces
def parse_calls(tc):
    if not tc:
        return []
    if isinstance(tc, str):
        try:
            tc = ast.literal_eval(tc)
        except Exception:
            try:
                tc = json.loads(tc)
            except Exception:
                return []
    out = []
    for c in tc:
        f = c.get("function", {})
        args = f.get("arguments") or "{}"
        try:
            a = json.loads(args) if isinstance(args, str) else args
        except Exception:
            a = {"_raw": args}
        if not isinstance(a, dict):
            a = {"_raw": str(a)}
        out.append((f.get("name"), a))
    return out


def n_train_of(user_text):
    return len(re.findall(r"--Example \d+--", user_text))


def load_trace(path):
    return json.load(gzip.open(path))


def attempt_turns(trace):
    """List of turn dicts (one per assistant message), with the tool output that answered it."""
    msgs = trace["messages"]
    events = [e for e in trace.get("events", []) if "usage" in e]
    user = next((m["content"] for m in msgs if m["role"] == "user"), "")
    ntr = n_train_of(user)
    turns = []
    ai = 0
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        calls = parse_calls(m.get("tool_calls"))
        code = "\n".join(str(a.get("code", a.get("_raw", ""))) for n, a in calls if n != "submit_answer")
        sub = "\n".join(str(a.get("code", a.get("expr", a.get("_raw", json.dumps(a))))) for n, a in calls
                        if n == "submit_answer")
        outs = []
        j = i + 1
        while j < len(msgs) and msgs[j]["role"] == "tool":
            outs.append(msgs[j].get("content") or "")
            j += 1
        out = "\n".join(outs)
        ev = events[ai] if ai < len(events) else {}
        ai += 1
        turns.append({
            "think": m.get("reasoning_content") or "",
            "content": m.get("content") or "",
            "code": code,
            "submit": sub,
            "names": [n for n, _ in calls],
            "out": out,
            "comp": (ev.get("usage") or {}).get("completion_tokens", 0),
            "finish": ev.get("finish_reason"),
        })
    return turns, ntr
