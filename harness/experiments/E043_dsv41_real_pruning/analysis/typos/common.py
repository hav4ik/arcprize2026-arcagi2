"""Shared loaders for the E043 typo analysis (read-only over evals/results).

Each attempt becomes a compact dict:
  {model, tag, task_id, pair, attempt, correct, turns_n, completion, stop_reason, rescues,
   turns: [{reasoning, content, prompt_tokens, completion_tokens,
            calls: [{name, code, output, status, exc_type, exc_msg}]}]}
"""
import ast
import glob
import gzip
import json
import os
import re

ROOT = "/home/vu/Workspace/ARC-AGI-2"
RES = f"{ROOT}/evals/results"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/out"
TOKENIZER = f"{ROOT}/experiments/E042_dsv41_l4_synthetic/metadata/shared_metadata/tokenizer.json"
P = "tools_persistent_grid_lean2sub_"

# model label -> [(tag, attempt filter or None)]
RUNS = {
    "API-full": [("deepseek-flash_low_e030_hdrmin", None)],
    "SH-full": [("dsv41_low_e043_full_ref_sub80", None)],          # partial: errored records dropped
    "E128": [("dsv41_low_e043_e128t_w10_u2_gptq_eng64_sub80", None)],
    "E112": [("dsv41_low_e043_e112t_w10_u2_gptq_eng64_sub80", {0}),  # attempt 1 + r3 excluded (anomalous)
             ("dsv41_low_e043_e112t_w10_u2_gptq_eng64_sub80_r2", None)],
    "E96": [("dsv41_low_e043_e96t_w10_u2u3_gptq_eng64_sub80", None)],
    "E72": [("dsv41_low_e043_e72_w30_eng64_rescue2", None)],
    "DEV-E128-t1": [("dsv41_low_e043_devs_e128t_t1", None)],
    "DEV-E128-minp05": [("dsv41_low_e043_devs_e128t_minp05", None)],
    "DEV-E128-topp95": [("dsv41_low_e043_devs_e128t_topp95", None)],
}
EVAL_MODELS = ["API-full", "SH-full", "E128", "E112", "E96", "E72"]
TRAIN_TAGS = ["deepseek-flash_low_e031_arc2_train", "deepseek-flash_low_e031_nvarc_b0",
              "deepseek-flash_low_e036_nvarc_b1", "deepseek-flash_low_e036_nvarc_b1x"]

EXC_LINE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Iteration|Warning))(?::\s?(.*))?$")


def parse_tool_calls(tc):
    if tc is None:
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
    for c in tc or []:
        f = c.get("function", {}) if isinstance(c, dict) else {}
        args = f.get("arguments", "")
        code = None
        if isinstance(args, dict):
            code = args.get("code")
        else:
            try:
                code = json.loads(args).get("code")
            except Exception:
                try:
                    code = ast.literal_eval(args).get("code")
                except Exception:
                    m = re.search(r'"code"\s*:\s*"(.*)"\s*}?\s*$', args or "", re.S)
                    code = m.group(1) if m else args
        out.append({"id": c.get("id"), "name": f.get("name"), "code": code if isinstance(code, str) else ""})
    return out


def classify_output(text):
    """(status, exc_type, exc_msg) from a tool message."""
    text = text or ""
    head = text.split("\n", 1)[0]
    if head.startswith("[completed]"):
        status = "completed"
    elif head.startswith("[error]"):
        status = "error"
    elif head.startswith("[timeout]"):
        status = "timeout"
    elif head.startswith("[tool error"):
        status = "tool_error"
    elif head.startswith("[answer accepted"):
        status = "answer_ok"
    elif head.startswith("[answer"):
        status = "answer_other"
    else:
        status = "other"
    exc_type = exc_msg = None
    if status in ("error", "timeout", "answer_other", "other"):
        lines = [l for l in text.split("\n") if l.strip() and not l.startswith("[budget") and not l.startswith("[")]
        for l in reversed(lines):
            m = EXC_LINE.match(l.strip())
            if m:
                exc_type, exc_msg = m.group(1), (m.group(2) or "")
                break
    return status, exc_type, exc_msg


def load_trace(path):
    t = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))
    return t.get("messages", t) if isinstance(t, dict) else t, (t.get("events", []) if isinstance(t, dict) else [])


def compact_attempt(rec, model, tag):
    p = rec.get("trace_path")
    if not p or not os.path.exists(p):
        return None
    msgs, events = load_trace(p)
    usage = {}
    for ev in events:
        if "usage" in ev and ev.get("turn") is not None and ev["usage"]:
            usage[ev["turn"]] = ev["usage"]
    outputs = {m.get("tool_call_id"): m.get("content") or "" for m in msgs if m.get("role") == "tool"}
    turns = []
    user = ""
    for m in msgs:
        if m.get("role") == "user" and not user:
            user = m.get("content") or ""
        if m.get("role") != "assistant":
            continue
        k = len(turns) + 1
        u = usage.get(k, {})
        calls = []
        for c in parse_tool_calls(m.get("tool_calls")):
            o = outputs.get(c["id"], None)
            st, et, em = classify_output(o) if o is not None else ("no_output", None, None)
            calls.append({"name": c["name"], "code": c["code"], "output": o or "", "status": st,
                          "exc_type": et, "exc_msg": em})
        turns.append({"reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
                      "content": m.get("content") or "", "calls": calls,
                      "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens")})
    return {"model": model, "tag": tag, "task_id": rec["task_id"], "pair": rec["pair"], "attempt": rec["attempt"],
            "correct": bool(rec.get("correct")), "turns_n": rec.get("turns"),
            "completion": (rec.get("usage") or {}).get("completion"), "stop_reason": rec.get("stop_reason"),
            "rescues": rec.get("rescues", 0), "train_pass": rec.get("train_pass"), "user_chars": len(user),
            "turns": turns}


def iter_records(tag, attempts=None):
    f = f"{RES}/{P}{tag}.jsonl"
    rows = {}
    for line in open(f):
        r = json.loads(line)
        if "task_id" not in r:
            continue
        rows[(r["task_id"], r["pair"], r["attempt"])] = r  # last record wins (resumed runs)
    for k, r in sorted(rows.items()):
        if "error" in r:
            continue
        if attempts is not None and r["attempt"] not in attempts:
            continue
        yield r
