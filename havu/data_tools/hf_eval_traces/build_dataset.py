"""Build a local Hugging Face dataset of every saved model trace on the 120 ARC-AGI-2 public evaluation tasks.

Read-only on every input. Writes Parquet shards + runs table under --out (no upload, no API calls).

Sources
- evals/results/*.jsonl (+ failed/e024_bolt_tools_*.jsonl) with their per-attempt transcripts
  (*.messages.json.gz, *_traces/*.txt.gz, *.reasoning.txt.gz)
- experiments/E016_cutoff_continuation/results/continuations.jsonl (continuation turns)
- solver arc_logs (attempts_*.jsonl + trace_<task>_<pair>_<attempt>.txt): experiments/E027_l4_throughput/**,
  experiments/E021_kaggle_v3/runs/*/working/arc_logs, kaggle/runs/*/arc_logs (local-mode eval runs only)

Run metadata (run_id, model, harness, set, description) comes from reports/build_report.py's run specs, plus a few
manual specs for files the report does not cover.

    .venv/bin/python data_tools/hf_eval_traces/build_dataset.py --out <dir>
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "reports"))
import build_report as B  # noqa: E402  (module-level code only reads files)

RES = ROOT / "evals/results"
ABS_PREFIX = str(ROOT) + "/"

# ------------------------------------------------------------------ ground truth / scope
GH = B.TASKS  # GitHub public eval (120 tasks, 167 test pairs)
EVAL_IDS = set(GH)
KAG_CH = json.loads((ROOT / "data/arc-agi_evaluation_challenges.json").read_text())
KAG_SOL = json.loads((ROOT / "data/arc-agi_evaluation_solutions.json").read_text())
TEST_IDS = set(json.loads((ROOT / "data/arc-agi_test_challenges.json").read_text()))
assert set(KAG_CH) == EVAL_IDS and not (TEST_IDS & EVAL_IDS)


def in_gh(t, p):
    return t in GH and p is not None and 0 <= p < len(GH[t]["test"])


def truth(t, p):
    """(GitHub answer or None, Kaggle answer or None)."""
    g = GH[t]["test"][p]["output"] if in_gh(t, p) else None
    k = KAG_SOL[t][p] if (t in KAG_SOL and p is not None and 0 <= p < len(KAG_SOL[t])) else None
    return g, k


# ------------------------------------------------------------------ redaction
RED_COUNTS = Counter()
_P = r"[^\s'\"\\,;)\]}<>`|]*"
RX_STRICT = [
    ("url_local", re.compile(r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\d{1,3}(?:\.\d{1,3}){3}|[\w.-]*vast\.ai)(?::\d+)?" + _P), "<host>"),
    ("ssh", re.compile(r"ssh\s+(?:-\w\s*\S+\s+)*-p\s*\d+\s+\S+"), "<host>"),
    ("vast_host", re.compile(r"\bssh\d*\.vast\.ai\b"), "<host>"),
    ("ip_port", re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}:\d{2,5}\b"), "<host>"),
    ("localhost", re.compile(r"\blocalhost:\d{2,5}\b"), "<host>"),
    ("private_ip", re.compile(r"(?<![\d.])(?:172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}(?![\d.])|(?<![\d.])127\.0\.0\.1(?![\d.])"), "<host>"),
    ("home_path", re.compile(r"/home/vu" + _P), "<path>"),
    ("path", re.compile(r"(?:(?<![\w.~/-])|(?<=\\n)|(?<=\\t))(?:/root|/workspace|/kaggle)(?=[/\s'\"\\,;)\]}<>`|]|$)" + _P), "<path>"),
    ("email", re.compile(r"[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|org|net|edu|io|ai|gov|co|uk|de|ru|cn|me|dev)\b"), "<email>"),
]
# metadata only (configs, record extras): any IPv4 literal
RX_META = RX_STRICT + [("ip", re.compile(r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])"), "<host>")]


def redact(s, meta=False):
    if s is None:
        return None
    for name, rx, rep in (RX_META if meta else RX_STRICT):
        s, n = rx.subn(rep, s)
        if n:
            RED_COUNTS[("meta" if meta else "text", name)] += n
    return s


def jd(x):
    return None if x is None else json.dumps(x, ensure_ascii=False)


# ------------------------------------------------------------------ run specs
def short_model(m):
    if "SFT" in m:
        return "Qwen3.8-27B + E028 LoRA SFT"
    base = m.split(",")[0].replace(" (DeepSeek API)", "").strip()
    return re.sub(r"\bfp8\b", "FP8", base)


SPECS_BY_FILE = defaultdict(list)
for s in B.T1_SPECS + B.T2_SPECS:
    for f in s.get("files", []):
        SPECS_BY_FILE[f].append(s)

FAMS = {B.DS: "DeepSeek", B.QW: "Qwen", B.NM: "Nemotron", B.OT: "Other"}
QWEN_AWQ_LOCAL = "Qwen3.8-27B AWQ-INT4, local SGLang server (2×RTX 3090)"
SFT_L4 = "Qwen3.8-27B + E028 LoRA SFT, ModelOpt FP8, vLLM on vast.ai 4×L4 (Kaggle notebook solver in local mode)"
QWEN_POOL = B.QWEN_POOL
E021_DESC = "Kaggle v3/v4 notebook solver, local functional run on 6 eval tasks (E021 README, 'Functional runs')"

MANUAL = {
    # evals/results files the report does not list
    "e027_sft_l4_kaggle.jsonl": dict(id="E027 SFT L4 DeepSeek-harness (stopped)", fam=B.QW, harness="tools-grid", set="non-dev-93 (partial)",
                                     model="Qwen3.8-27B + E028 LoRA SFT, self-hosted on vast.ai 4×L4 (undocumented short run)", status="stopped",
                                     desc="evals/run_deepseek_tools.py with lean2roles, submit_answer, 80k / 64k per turn on the E028 SFT model; stopped after 3 of 90 attempts"),
    "qwen_tools_baseline_gw_dev56_s1.jsonl": dict(id="E019 dev A submit_answer s1 (partial)", fam=B.QW, model=QWEN_POOL, harness="tools-grid", set="dev-56",
                                                  status="stopped", desc="extra sample of E019 dev A (qwen-tools-4 baseline, medium 48k / 24k)"),
    "qwen_tools_baseline_gw_dev56_s2.jsonl": dict(id="E019 dev A submit_answer s2 (partial)", fam=B.QW, model=QWEN_POOL, harness="tools-grid", set="dev-56",
                                                  status="stopped", desc="extra sample of E019 dev A"),
    "qwen_tools_program_gw_dev56_s1.jsonl": dict(id="E019 dev B program-only s1 (partial)", fam=B.QW, model=QWEN_POOL, harness="tools-program", set="dev-56",
                                                 status="stopped", desc="extra sample of E019 dev B (submit_program only)"),
    "qwen_tools_program_gw_dev56_s2.jsonl": dict(id="E019 dev B program-only s2 (partial)", fam=B.QW, model=QWEN_POOL, harness="tools-program", set="dev-56",
                                                 status="stopped", desc="extra sample of E019 dev B"),
    "qwen_tools_program_gw_dev56_s3.jsonl": dict(id="E019 dev B program-only s3 (partial)", fam=B.QW, model=QWEN_POOL, harness="tools-program", set="dev-56",
                                                 status="stopped", desc="extra sample of E019 dev B"),
    "qwen_tools_baseline_smoke_local.jsonl": dict(id="E019 smoke A (local)", fam=B.QW, model=QWEN_AWQ_LOCAL, harness="tools-grid", set="smoke",
                                                  status="probe", desc="E019 harness smoke test, submit mode"),
    "qwen_tools_program_smoke_local.jsonl": dict(id="E019 smoke B program (local)", fam=B.QW, model=QWEN_AWQ_LOCAL, harness="tools-program", set="smoke",
                                                 status="probe", desc="E019 harness smoke test, program mode"),
    "qwen_tools_program_smoke2_local.jsonl": dict(id="E019 smoke B2 program (local)", fam=B.QW, model=QWEN_AWQ_LOCAL, harness="tools-program", set="smoke",
                                                  status="probe", desc="E019 harness smoke test 2, program mode"),
    "qwen_tools_e020_smoke_local.jsonl": dict(id="E020 smoke (local)", fam=B.QW, model=QWEN_AWQ_LOCAL, harness="tools-program", set="smoke",
                                              status="probe", desc="E020 options smoke test (xhigh 32k, all options)"),
    "qwen_tools_e020_slots_smoke_local.jsonl": dict(id="E020 slots smoke (local)", fam=B.QW, model=QWEN_AWQ_LOCAL, harness="tools-program", set="smoke",
                                                    status="probe", desc="worker-slot scheduling smoke test (3k cap)"),
    "tools_persistent_grid_lean2roles_deepseek-flash_low_e017_smoke_v3.jsonl": dict(id="E017 V3 smoke", fam=B.DS, model=B.FLASH, harness="tools-grid",
                                                                                     set="smoke", status="probe", desc="E017 V3 options smoke test"),
    "failed/e024_bolt_tools_illegaladdr.jsonl": dict(id="E024 Nemotron-3.5-Bolt tools (crashed, illegal address)", fam=B.NM, model=B.BOLT, harness="tools-grid",
                                                     set="full-167 (partial)", status="crashed",
                                                     desc=B.E024_T + "; every attempt ended in a server error (b12x kernel crash), transcripts are partial"),
    "failed/e024_bolt_tools_segfault.jsonl": dict(id="E024 Nemotron-3.5-Bolt tools (crashed, segfault)", fam=B.NM, model=B.BOLT, harness="tools-grid",
                                                  set="full-167 (partial)", status="crashed",
                                                  desc=B.E024_T + "; every attempt ended in a server error (segfault), transcripts are partial"),
}
SKIP_FILES = {
    "cache_gw_dev56.jsonl": "SGLang cache-monitor log, not model output",
    "cache_local_radix.jsonl": "SGLang cache-monitor log, not model output",
    "tools_persistent_grid_lean2roles_deepseek-flash_low_e025_arc2_train.jsonl": "E025 training-task rollouts (out of scope)",
    "tools_persistent_grid_lean2roles_deepseek-flash_low_e025_nvarc_b0.jsonl": "E025 NVARC synthetic rollouts (out of scope)",
    "arc2_deepseek-flash_max.jsonl": "E001a: NVIDIA val subset; no reasoning saved (final boxed answer only); 50/120 test instances differ from the official test pairs",
    "arc2_deepseek-flash_high.jsonl": "E001b: as E001a",
    "arc2_deepseek-v4-pro_max.jsonl": "E001c: as E001a",
    "qwen_nvfp4_gen_xhigh_240k_t1_pilot.jsonl": "E008 Qwen xhigh pilot: its 52 records (and traces) were copied into qwen_nvfp4_gen_xhigh_240k_t1.jsonl; skipped to avoid duplicates",
    "failed/e024_bolt_notools_illegaladdr.jsonl": "E024 Bolt no-tools crashed tries: error records only, nothing saved",
    "failed/e024_bolt_notools_segfault.jsonl": "E024 Bolt no-tools crashed tries: error records only, nothing saved",
}

# arc_logs solver runs: dir -> spec
REPORT_AT = {f[1:]: s for s in B.T1_SPECS + B.T2_SPECS for f in s.get("files", []) if f.startswith("@")}
COMMIT = {"v2-v2": ("Kaggle v2 tools (commit run)", B.QWEN_KAGGLE, "tools-grid", "E013/E021"),
          "plain-v1": ("Kaggle plain v1 (commit run)", B.QWEN_KAGGLE, "no tools", "E013"),
          "v3-v3": ("Kaggle v3 tools (commit run)", B.QWEN_KAGGLE, "tools-grid", "E021"),
          "v4c-p2p": ("Kaggle v4c P2P (commit run)", B.QWEN_KAGGLE, "tools-program", "E027"),
          "v5-sft-commit": ("Kaggle v5 SFT, ModelOpt FP8 (commit run)", "Qwen3.8-27B E028 SFT, ModelOpt FP8, vLLM on Kaggle 4×L4", "tools-grid", "E028"),
          "v5-sft-commit-v2": ("Kaggle v5 SFT, block FP8 (commit run)", "Qwen3.8-27B E028 SFT, block FP8, vLLM on Kaggle 4×L4", "tools-grid", "E028")}


def arc_log_runs():
    out = []
    for d in sorted(glob.glob(str(ROOT / "experiments/E027_l4_throughput/e6/*/arc_logs"))):
        rd = Path(d).parent.relative_to(ROOT).as_posix()
        if rd in REPORT_AT:
            s = dict(REPORT_AT[rd])
        else:  # e6_tq
            s = dict(id="E027 vLLM 0.31 TurboQuant k8v4 (stopped)", fam=B.QW, model=B.QWEN_E027, harness="tools-program", set="full-172 (partial)",
                     status="stopped", desc=B.E027_V4B + "; TurboQuant k8v4 KV cache; stopped at 129/172 pairs without a summary file")
        s.setdefault("exp", "E027")
        out.append((d, s))
    out.append((str(ROOT / "experiments/E027_l4_throughput/sft_nb_kaggle/arc_logs"),
                dict(id="E027/E028 SFT Kaggle-paced (L4 box)", exp="E027", fam=B.QW, model=SFT_L4, harness="tools-grid", set="non-dev-93",
                     desc="exact Kaggle v5 SFT notebook solver (cad74873, kaggle_v5_sft_settings.json) in local mode at Kaggle's 11 h / 259-pair pace; "
                          "71 non-dev tasks / 93 pairs, 1 attempt each (E027 README 'Kaggle-paced run')")))
    out.append((str(ROOT / "experiments/E027_l4_throughput/sft_nb_kaggle_cut45/arc_logs"),
                dict(id="E027/E028 SFT Kaggle-paced, 45-min window (stopped)", exp="E027", fam=B.QW, model=SFT_L4, harness="tools-grid",
                     set="non-dev-93 (partial)", status="stopped", desc="first try of the Kaggle-paced run; read a 45-min commit window and stopped after 4 attempts")))
    for d in sorted(glob.glob(str(ROOT / "kaggle/runs/*/arc_logs"))):
        rd = Path(d).parent.relative_to(ROOT).as_posix()
        name = Path(rd).name
        if rd in REPORT_AT:
            s = dict(REPORT_AT[rd]); s.setdefault("exp", "E021")
        elif name in COMMIT:
            lab, model, harness, exp = COMMIT[name]
            s = dict(id=lab, exp=exp, fam=B.QW, model=model, harness=harness, set="Kaggle commit-run local test",
                     desc=f"Kaggle notebook commit run (local mode on eval tasks), kaggle/runs/{name}")
        elif name.startswith("bench-mtp"):
            mtp = name.split("-")[1]
            s = dict(id=f"E013 Kaggle bench {mtp.upper()} e2e", exp="E013", fam=B.QW, model=B.QWEN_KAGGLE.replace("vLLM 0.23", "vLLM 0.19") + f" ({mtp.upper()})",
                     harness="no tools", set="bench (15 tasks)", status="probe",
                     desc="E013 throughput bench on Kaggle 4×L4: end-to-end scheduler phase with real ARC prompts (plain agent, 16k budget)")
        else:
            s = None
        if s is not None:
            out.append((d, s))
    for d in sorted(glob.glob(str(ROOT / "experiments/E021_kaggle_v3/runs/*/working/arc_logs"))):
        name = Path(d).parents[1].name
        s = dict(id=f"E021 local {name.replace('local_', '')}", exp="E021", fam=B.QW, model=QWEN_AWQ_LOCAL, harness="tools-program" if "program" in name or "K3" in name else "tools-grid",
                 set="local functional (6 tasks)", status="probe", desc=E021_DESC + f"; run {name}")
        out.append((d, s))
    return out


def exp_of(s):
    if s.get("exp"):
        return s["exp"]
    m = re.match(r"(E\d{3}[a-d]?)", s["id"])
    return m.group(1) if m else ""


def set_label(s):
    lab = s.get("setlabel") or s.get("set")
    return {"full": "full-167", "pilot": "pilot-26", "dev": "dev-56", "e004": "E004a-20", "probe48": "probe-48", "covered": "partial"}.get(lab, lab)


def pick_spec(fname, attempt):
    if fname in MANUAL:
        return MANUAL[fname]
    cands = SPECS_BY_FILE.get(fname, [])
    if not cands:
        return None
    ok = [s for s in cands if s.get("attempts") is None or attempt in s["attempts"]] or cands
    ok.sort(key=lambda s: len(s.get("files", [])))
    return ok[0]


# ------------------------------------------------------------------ schema
SCHEMA = pa.schema([
    ("run_id", pa.string()), ("experiment", pa.string()), ("family", pa.string()), ("model", pa.string()), ("model_detail", pa.string()),
    ("harness", pa.string()), ("harness_version", pa.string()), ("eval_set", pa.string()), ("run_status", pa.string()),
    ("source_file", pa.string()), ("record_index", pa.int32()), ("is_latest", pa.bool_()),
    ("task_id", pa.string()), ("pair", pa.int32()), ("attempt", pa.int32()), ("in_github_167", pa.bool_()),
    ("correct", pa.bool_()), ("correct_kaggle", pa.bool_()), ("stored_correct", pa.bool_()),
    ("predicted_output", pa.string()), ("final_content", pa.string()),
    ("answer_source", pa.string()), ("stop_reason", pa.string()), ("finish_reason", pa.string()), ("error", pa.string()),
    ("output_tokens", pa.int64()), ("prompt_tokens", pa.int64()), ("turns", pa.int32()), ("tool_calls", pa.int32()), ("tool_errors", pa.int32()),
    ("train_pass", pa.bool_()), ("seconds", pa.float64()),
    ("has_trace", pa.bool_()), ("trace_format", pa.string()), ("trace_file", pa.string()),
    ("messages", pa.string()), ("turn_events", pa.string()), ("trace_text", pa.string()),
    ("run_config", pa.string()), ("record_extra", pa.string()),
])
CONSUMED = {"task_id", "pair", "attempt", "sample", "pred", "content", "correct", "answer_source", "stop_reason", "finish_reason", "error",
            "turns", "tool_calls", "tool_errors", "train_pass", "seconds", "elapsed_s", "config", "trace", "trace_path", "reasoning_path",
            "harness_version", "final_content", "new_messages", "source"}


def as_int(x):
    try:
        return None if x is None else int(x)
    except (TypeError, ValueError):
        return None


def as_float(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def as_bool(x):
    if x is None:
        return None
    if isinstance(x, str):
        return {"true": True, "false": False}.get(x.lower())
    return bool(x)


def prompt_tokens(r):
    u = r.get("usage") or {}
    if not isinstance(u, dict):
        return None
    for k in ("prompt_total", "prompt", "prompt_tokens"):
        if k in u:
            return as_int(u[k])
    return as_int(r.get("prompt_tokens"))


def output_tokens(r):
    if "tokens" in r and "usage" not in r:
        return as_int(r["tokens"])
    try:
        return as_int(B.tokens(r))
    except Exception:
        return None


def rel(p):
    if not p:
        return None
    p = str(p)
    if p.startswith(ABS_PREFIX):
        p = p[len(ABS_PREFIX):]
    return p


def load_trace(relpath):
    """-> (format, messages_json, events_json, text) or None if missing."""
    p = ROOT / relpath
    if not p.is_file():
        return None
    if relpath.endswith(".messages.json.gz"):
        obj = json.loads(gzip.decompress(p.read_bytes()))
        msgs = obj.get("messages") if isinstance(obj, dict) else obj
        ev = obj.get("events") if isinstance(obj, dict) else None
        return "messages", jd(msgs), jd(ev), None
    raw = gzip.decompress(p.read_bytes()) if relpath.endswith(".gz") else p.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    fmt = "reasoning_text" if ".reasoning." in relpath else "chat_template_text"
    return fmt, None, None, text


STATS = {"rows": 0, "skipped_out_of_scope": 0, "skipped_empty_error": 0, "trace_missing": 0, "shared_trace_detached": 0, "orphan_traces": {}}


def make_row(spec, source_file, idx, latest, r, trace, trace_file, run_config, harness_version):
    t = r["task_id"]
    p = as_int(r.get("pair"))
    a = as_int(r.get("attempt", r.get("sample", 0)))
    pred = r.get("pred")
    g, k = truth(t, p)
    correct = (pred == g) if g is not None else ((pred == k) if k is not None else None)
    if r.get("error") and not pred:
        correct = False if correct is not None else None
    extra = {kk: v for kk, v in r.items() if kk not in CONSUMED}
    fmt, msgs, ev, text = trace if trace else (None, None, None, None)
    content = r.get("content", r.get("final_content"))
    return {
        "run_id": spec["id"], "experiment": exp_of(spec), "family": FAMS.get(spec["fam"], spec["fam"]), "model": short_model(spec["model"]),
        "model_detail": redact(spec["model"], meta=True), "harness": spec["harness"], "harness_version": harness_version,
        "eval_set": set_label(spec), "run_status": spec.get("status", "done"),
        "source_file": source_file, "record_index": idx, "is_latest": latest,
        "task_id": t, "pair": p, "attempt": a, "in_github_167": in_gh(t, p),
        "correct": correct, "correct_kaggle": (pred == k) if k is not None else None, "stored_correct": as_bool(r.get("correct")),
        "predicted_output": jd(pred), "final_content": redact(content if isinstance(content, str) else jd(content)),
        "answer_source": r.get("answer_source", r.get("source")), "stop_reason": r.get("stop_reason"),
        "finish_reason": r.get("finish_reason"), "error": redact(str(r["error"]), meta=True) if r.get("error") else None,
        "output_tokens": output_tokens(r), "prompt_tokens": prompt_tokens(r), "turns": as_int(r.get("turns")),
        "tool_calls": as_int(r.get("tool_calls")) if not isinstance(r.get("tool_calls"), list) else len(r["tool_calls"]),
        "tool_errors": as_int(r.get("tool_errors")), "train_pass": as_bool(r.get("train_pass")),
        "seconds": as_float(r.get("seconds", r.get("elapsed_s"))),
        "has_trace": trace is not None, "trace_format": fmt or ("final_content_only" if content else "none"), "trace_file": trace_file,
        "messages": redact(msgs), "turn_events": redact(ev, meta=True), "trace_text": redact(text),
        "run_config": redact(run_config, meta=True), "record_extra": redact(jd(extra), meta=True) if extra else None,
    }


def read_jsonl(path):
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def record_config(r):
    c = r.get("config")
    if c is None:
        c = {k: r[k] for k in ("model", "effort", "max_tokens", "extra_body", "base_url", "tier", "budget", "provider", "endpoint", "served_model") if k in r}
    return jd(c) if c else None


def rows_from_results(fname):
    recs = read_jsonl(RES / fname)
    keyed = {}
    for i, r in enumerate(recs):
        keyed[(r.get("task_id"), str(r.get("pair")), str(r.get("attempt", r.get("sample"))))] = i
    path_last = {}
    for i, r in enumerate(recs):
        tp = rel(r.get("trace_path") or r.get("trace") or r.get("reasoning_path"))
        if tp:
            path_last[tp] = i
    groups = defaultdict(list)
    for i, r in enumerate(recs):
        t = r.get("task_id")
        if t not in EVAL_IDS:
            STATS["skipped_out_of_scope"] += 1
            continue
        latest = keyed[(t, str(r.get("pair")), str(r.get("attempt", r.get("sample"))))] == i
        tp = rel(r.get("trace_path") or r.get("trace") or r.get("reasoning_path"))
        trace = None
        if tp:
            if path_last[tp] != i:
                STATS["shared_trace_detached"] += 1  # trace file was overwritten by a later record for the same key
                tp_use = None
            else:
                tp_use = tp
            if tp_use:
                trace = load_trace(tp_use)
                if trace is None:
                    STATS["trace_missing"] += 1
        if trace is None and r.get("error") and not r.get("pred") and not r.get("content"):
            STATS["skipped_empty_error"] += 1
            continue
        spec = pick_spec(fname, as_int(r.get("attempt", 0)))
        hv = r.get("harness_version") or ((r.get("config") or {}).get("harness_version") if isinstance(r.get("config"), dict) else None)
        row = make_row(spec, f"evals/results/{fname}", i, latest, r, trace, tp if trace else None, record_config(r), hv)
        groups[spec["id"]].append(row)
    # orphan trace files (in the run folder, not referenced by any record)
    stem = fname[:-6]
    refd = set(path_last)
    for d in (RES / stem, RES / (stem + "_traces")):
        if d.is_dir() and not (d.name != stem and (RES / (d.name + ".jsonl")).exists()):
            orph = [f for f in os.listdir(d) if f"evals/results/{d.name}/{f}" not in refd]
            if orph:
                STATS["orphan_traces"][f"evals/results/{d.name}"] = len(orph)
    return groups


def rows_from_e016():
    path = ROOT / "experiments/E016_cutoff_continuation/results/continuations.jsonl"
    groups = defaultdict(list)
    for i, r in enumerate(read_jsonl(path)):
        if r["task_id"] not in EVAL_IDS:
            STATS["skipped_out_of_scope"] += 1
            continue
        if r["variant"] == "baseline":  # host-side probe of the replayed session, no model call
            STATS["skipped_e016_baseline_probe"] = STATS.get("skipped_e016_baseline_probe", 0) + 1
            continue
        spec = dict(id=f"E016 continuation {r['variant']}", exp="E016", fam=B.DS, model=B.FLASH, harness="tools-grid (continuation)",
                    set="s20x60 cut-off attempts", status="done")
        msgs = r.get("new_messages")
        trace = ("messages", jd(msgs), None, None) if msgs else None
        cfg = jd({"variant": r["variant"], "continues_run": "E010 s20x60", "continues_file": "evals/results/" + B.tool_file("lean2sub", "low", "e010_full_s20x60"),
                  "note": "messages holds only the new continuation turns; the original transcript is the E010 s20x60 row with the same task/pair/attempt"})
        groups[spec["id"]].append(make_row(spec, "experiments/E016_cutoff_continuation/results/continuations.jsonl", i, True, r, trace,
                                           None, cfg, "e016-continuation"))
    return groups


def rows_from_arc_logs(d, spec):
    d = Path(d)
    att = sorted(d.glob("attempts_*.jsonl"))
    recs = []
    for f in att:
        recs += [(f, r) for r in read_jsonl(f)]
    run_dir = d.parent
    summ = None
    for cand in (run_dir / "summary_local.json", run_dir / "summary_submit.json"):
        if cand.exists():
            summ = json.loads(cand.read_text())
            break
    cfg = None
    if summ:
        ac = (summ.get("phases_s") or {}).get("agent_config") if isinstance(summ.get("phases_s"), dict) else None
        cfg = {"summary": {k: v for k, v in summ.items() if k not in ("per_pair", "cache_stats", "phases_s")}}
        if ac:
            cfg["agent_config"] = ac
    cfg = jd(cfg)
    keyed, groups, used = {}, defaultdict(list), set()
    for i, (f, r) in enumerate(recs):
        keyed[(r["task_id"], r.get("pair"), r.get("attempt"))] = i
    for i, (f, r) in enumerate(recs):
        t = r["task_id"]
        if t not in EVAL_IDS:
            STATS["skipped_out_of_scope"] += 1
            continue
        latest = keyed[(t, r.get("pair"), r.get("attempt"))] == i
        tf = d / f"trace_{t}_{r.get('pair')}_{r.get('attempt')}.txt"
        trace = None
        if latest and tf.exists():
            trace = load_trace(tf.relative_to(ROOT).as_posix())
            used.add(tf.name)
        groups[spec["id"]].append(make_row(spec, f.relative_to(ROOT).as_posix(), i, latest, r, trace,
                                           tf.relative_to(ROOT).as_posix() if trace else None, cfg, r.get("harness_version") or r.get("agent")))
    # traces without an attempt record (e.g. bench runs, attempts cut by a deadline)
    for tf in sorted(d.glob("trace_*.txt")):
        if tf.name in used:
            continue
        m = re.match(r"trace_(.+)_(\d+)_(\d+)\.txt$", tf.name)
        t, p, a = m.group(1), int(m.group(2)), int(m.group(3))
        if t not in EVAL_IDS:
            STATS["skipped_out_of_scope"] += 1
            continue
        r = {"task_id": t, "pair": p, "attempt": a, "note": "trace without an attempt record (unfinished or not logged)"}
        row = make_row(spec, tf.relative_to(ROOT).as_posix(), -1, True, r, load_trace(tf.relative_to(ROOT).as_posix()),
                       tf.relative_to(ROOT).as_posix(), cfg, None)
        row["correct"] = None
        row["correct_kaggle"] = None
        groups[spec["id"]].append(row)
    return groups


def config_of(row):
    fam = row["family"]
    tools = row["harness"].startswith("tools")
    if fam == "DeepSeek":
        return "deepseek_tools" if tools else "deepseek_no_tools"
    if fam == "Qwen":
        return "qwen_tools" if tools else "qwen_no_tools"
    if fam == "Nemotron":
        return "nemotron"
    return "other_models"


class ShardWriter:
    def __init__(self, out, name, max_bytes):
        self.out, self.name, self.max = out / "data" / name, name, max_bytes
        self.out.mkdir(parents=True, exist_ok=True)
        self.idx, self.w, self.files = 0, None, []

    def _open(self):
        p = self.out / f"train-{self.idx:05d}.parquet"
        self.files.append(p)
        self.w = pq.ParquetWriter(p, SCHEMA, compression="zstd", compression_level=9)

    def write(self, rows):
        for i in range(0, len(rows), 32):
            if self.w is None:
                self._open()
            self.w.write_table(pa.Table.from_pylist(rows[i:i + 32], schema=SCHEMA))
            if self.files[-1].stat().st_size > self.max:
                self.w.close()
                self.w = None
                self.idx += 1

    def close(self):
        if self.w is not None:
            self.w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-shard-mb", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)

    sources = []  # (label, loader)
    skipped = {}
    files = sorted(f for f in os.listdir(RES) if f.endswith(".jsonl")) + ["failed/" + f for f in sorted(os.listdir(RES / "failed")) if f.endswith(".jsonl")]
    for f in files:
        if f in SKIP_FILES:
            skipped[f"evals/results/{f}"] = SKIP_FILES[f]
            continue
        if f not in MANUAL and f not in SPECS_BY_FILE:
            skipped[f"evals/results/{f}"] = "no run spec (unknown file)"
            continue
        sources.append((f"evals/results/{f}", lambda f=f: rows_from_results(f)))
    sources.append(("experiments/E016_cutoff_continuation/results/continuations.jsonl", rows_from_e016))
    for d, spec in arc_log_runs():
        sources.append((Path(d).relative_to(ROOT).as_posix(), lambda d=d, spec=spec: rows_from_arc_logs(d, spec)))

    writers = {}
    runs = {}
    for label, loader in sources:
        groups = loader()
        if not groups:
            skipped[label] = "no in-scope records"
            continue
        for rid, rows in groups.items():
            rows.sort(key=lambda x: (x["task_id"], x["pair"] if x["pair"] is not None else -1, x["attempt"] or 0, x["record_index"]))
            cfg = config_of(rows[0])
            if cfg not in writers:
                writers[cfg] = ShardWriter(out, cfg, args.max_shard_mb * 1024 * 1024)
            writers[cfg].write(rows)
            STATS["rows"] += len(rows)
            R = runs.setdefault(rid, {"run_id": rid, "experiment": rows[0]["experiment"], "family": rows[0]["family"], "model": rows[0]["model"],
                                      "model_detail": rows[0]["model_detail"], "harness": rows[0]["harness"], "eval_set": rows[0]["eval_set"],
                                      "run_status": rows[0]["run_status"], "config": cfg, "files": set(), "rows": 0, "latest_rows": 0,
                                      "rows_with_trace": 0, "scored_latest": 0, "correct_latest": 0, "tasks": set(), "pairs": set(), "out_tokens": 0})
            for x in rows:
                R["files"].add(x["source_file"] if not x["source_file"].endswith(".txt") else str(Path(x["source_file"]).parent))
                R["rows"] += 1
                R["rows_with_trace"] += x["has_trace"]
                R["out_tokens"] += x["output_tokens"] or 0
                R["tasks"].add(x["task_id"])
                R["pairs"].add((x["task_id"], x["pair"]))
                if x["is_latest"]:
                    R["latest_rows"] += 1
                    if x["correct"] is not None and x["in_github_167"]:
                        R["scored_latest"] += 1
                        R["correct_latest"] += bool(x["correct"])
        print(f"{label}: {sum(len(v) for v in groups.values())} rows", flush=True)
    for w in writers.values():
        w.close()

    spec_desc = {s["id"]: s.get("desc", "") for s in B.T1_SPECS + B.T2_SPECS}
    for m in MANUAL.values():
        spec_desc[m["id"]] = m.get("desc", "")
    for _, s in arc_log_runs():
        spec_desc.setdefault(s["id"], s.get("desc", ""))
    run_rows = []
    for rid, R in runs.items():
        run_rows.append({
            "run_id": rid, "experiment": R["experiment"], "family": R["family"], "model": R["model"], "model_detail": R["model_detail"],
            "harness": R["harness"], "eval_set": R["eval_set"], "run_status": R["run_status"], "config": R["config"],
            "description": redact(spec_desc.get(rid, "") or ("E016 continuation turn for E010 s20x60 cut-off attempts" if rid.startswith("E016") else ""), meta=True),
            "source_files": ";".join(sorted(R["files"])), "rows": R["rows"], "latest_rows": R["latest_rows"], "rows_with_trace": R["rows_with_trace"],
            "tasks": len(R["tasks"]), "pairs": len(R["pairs"]), "scored_attempts_gh167": R["scored_latest"], "correct_attempts_gh167": R["correct_latest"],
            "attempt_accuracy_gh167": round(R["correct_latest"] / R["scored_latest"], 4) if R["scored_latest"] else None,
            "mean_output_tokens": round(R["out_tokens"] / R["rows"]) if R["rows"] else None})
    run_rows.sort(key=lambda x: (x["config"], x["experiment"], x["run_id"]))
    pq.write_table(pa.Table.from_pylist(run_rows), out / "runs.parquet")
    import csv
    with open(out / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(run_rows[0]))
        w.writeheader()
        w.writerows(run_rows)
    meta = {"stats": STATS, "skipped": skipped, "redactions": {f"{a}:{b}": n for (a, b), n in sorted(RED_COUNTS.items())},
            "shards": {k: [p.relative_to(out).as_posix() for p in w.files] for k, w in writers.items()}}
    (out / "build_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
