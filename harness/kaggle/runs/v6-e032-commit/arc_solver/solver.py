"""ARC Prize 2026 (ARC-AGI-2) offline solver: Qwen3.8-27B-FP8 on vLLM (TP=4, 4xL4), no tools.

One self-contained file; kaggle/build_notebook.py embeds it in the Kaggle notebook. Modes (env ARC_MODE):

- submit:   full run on the competition test file (used automatically on the competition rerun).
- fallback: write a valid fallback submission.json and exit (used for the commit run before submitting).
- bench:    test run: provenance hashes, server start, long-context decode benchmark, short end-to-end run on
            evaluation tasks with a near deadline, scoring against the evaluation solutions.
            ARC_HASH=0 / ARC_BENCH_DECODE=0 skip the first two (validation run: ARC_E2E_TASKS, ARC_E2E_MIN).

Robustness rules: a complete, valid submission.json is written before anything else and rewritten atomically
after every finished attempt; every failure path keeps the last valid file.

Prompt, parser and Qwen raw-template rendering follow evals/arc_harness/arcprize.py and evals/run_qwen_generate.py
(ARC Prize official prompt, tier xhigh, temp 1.0, top_p 0.95, top_k 20, forced final answer at the budget).
"""
import asyncio
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

T0 = float(os.environ.get("ARC_T0", time.time()))  # notebook start (set by the first cell)
MODE = os.environ.get("ARC_MODE", "submit")
WORK = "/kaggle/working"
WORK = os.environ.get("ARC_WORK", WORK)
OUT_SUB = os.path.join(WORK, "submission.json")
LOG_DIR = os.path.join(WORK, "arc_logs")
PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"
SERVED = os.environ.get("ARC_MODEL_ID", "qwen3.8-27b")
# Off-Kaggle validation: use an already running OpenAI-compatible server and skip install / launch.
EXTERNAL = os.environ.get("ARC_EXTERNAL_SERVER")  # e.g. http://localhost:30000/v1
if EXTERNAL:
    BASE = EXTERNAL.rstrip("/").removesuffix("/v1")
INPUT = os.environ.get("ARC_INPUT", "/kaggle/input")

# ---------------------------------------------------------------- settings (env overridable)

def envf(name, default):
    return type(default)(os.environ.get(name, default))

FORCE_AT_S = envf("ARC_FORCE_AT_S", 11.0 * 3600)    # force final answers for all in-flight attempts
HARD_STOP_S = envf("ARC_HARD_STOP_S", 11.5 * 3600)  # stop everything, final write (notebook guard: 11.75 h)
B_MAX1 = envf("ARC_B_MAX1", 96000)                  # attempt-1 budget cap (thinking + answer)
B_MIN1 = envf("ARC_B_MIN1", 16000)
B_MAX2 = envf("ARC_B_MAX2", 64000)
B_MIN2 = envf("ARC_B_MIN2", 32000)                  # attempt 2 is launched only if it can get this much
FINAL_TOKENS = envf("ARC_FINAL_TOKENS", 3000)
RATE_PRIOR = envf("ARC_RATE_PRIOR", 350.0)          # tok/s before the run measures its own rate
MAX_MODEL_LEN = envf("ARC_MAX_MODEL_LEN", 131072)
MAX_SEQS = envf("ARC_MAX_SEQS", 64)
CONCURRENCY = envf("ARC_CONCURRENCY", 0)            # 0 = from KV capacity
# KV tokens per sequence when sizing concurrency = p90 context of a live tool attempt (E014/E019 rule: KV / p90,
# not the mean): 1.40M (vLLM 0.23) / 48k = 29. v2 used 40000 (-> 35).
TOKENS_PER_SEQ_KV = envf("ARC_KV_PER_SEQ", 48000)
MTP = envf("ARC_MTP", 0)                            # speculative tokens (0 = off)
TIER = os.environ.get("ARC_TIER", "xhigh")
TEMPERATURE = envf("ARC_TEMPERATURE", 1.0)
GPU_UTIL = envf("ARC_GPU_UTIL", 0.95)
TP = int(envf("ARC_TP", 4))                          # tensor parallel size; main() sets it from the GPUs present
SERVER = os.environ.get("ARC_SERVER", "vllm")        # vllm (user's vllm-0.23.0 cu130 source) | sglang (ours)
AGENT = os.environ.get("ARC_AGENT", "tools")         # tools (E012 agent_qwen) | plain (v1 no-tool prompt)
TOOLS_CAP = envf("ARC_TOOLS_CAP", 48000)             # E012: hard cap per attempt (min with the adaptive share)
TOOLS_SOFT = envf("ARC_TOOLS_SOFT", 24000)
TOOLS_TIER = os.environ.get("ARC_TOOLS_TIER", "medium")
TOOL_TIMEOUT = envf("ARC_TOOL_TIMEOUT", 30.0)
TOOLS_ANSWER_MODE = os.environ.get("ARC_TOOLS_ANSWER_MODE", "submit")  # E019: "program" = submit_program only
# ---- v3 (E021) toggles; every default reproduces v2 except the robustness items (memory cap, KV/p90 concurrency)
TOOLS_PROMPT = os.environ.get("ARC_TOOLS_PROMPT", "")  # "" = lean2sub (submit) / lean2prog (program); lean2progroles
TASK_SUMMARY = envf("ARC_TASK_SUMMARY", 0)          # E017/E020: host-computed perception pack in the first message
HELPERS = envf("ARC_HELPERS", 0)                    # E017/E020: arc_helpers preloaded in session + programs
FEEDBACK = os.environ.get("ARC_FEEDBACK", "basic")  # program mode: basic | rich (E020)
TURN_THINK_CAP = envf("ARC_TURN_THINK_CAP", 0)      # E020: thinking tokens per turn before the "run code" nudge
TURN_THINK_GRACE = envf("ARC_TURN_THINK_GRACE", 2048)  # then close the thinking into a run_python call
THINK_CALL_RESCUE = envf("ARC_THINK_CALL_RESCUE", 1)  # E020/E021: after a nudge, run a <tool_call> written inside
COMPACT_AT = envf("ARC_COMPACT_AT", 0)            # E027: compact transcripts past this many tokens (0 = off)
COMPACT_KEEP = envf("ARC_COMPACT_KEEP", 2)        # E027: recent turns left untouched by a compaction
COMPACT_TOOL_CHARS = envf("ARC_COMPACT_TOOL_CHARS", 1500)  # E027: older tool outputs cut to this length
                                                    # still-open thinking (only acts with the think cap on)
# ---- v5_sft (E028) toggles, all opt-in; with every one unset the solver behaves exactly as v4
MODEL_SOURCE = os.environ.get("ARC_MODEL_SOURCE", "")  # owner/slug of the Kaggle model to serve, e.g.
                                                    # chankhavu/qwen3-8-27b-e028-sft ("" = the official FP8 mirror)
TOKENIZER = os.environ.get("ARC_TOKENIZER", "")     # vLLM --tokenizer: "" = the model's own; "official" = the
                                                    # official FP8 mirror's dir (E028: vLLM 0.23 cannot load the SFT
                                                    # checkpoint's Qwen3_5Tokenizer config); else an owner/slug or path
TOOLS_DESC = os.environ.get("ARC_TOOLS_DESC", "")   # AgentConfig.tool_desc ("e025": the E025 training tool texts)
TOOLS_FALLBACK_SOLVE = int(os.environ.get("ARC_TOOLS_SOLVE_FALLBACK", os.environ.get("ARC_TOOLS_FALLBACK_SOLVE", 0)))
                                                    # AgentConfig.solve_fallback: no answer -> solve(test_input)
TOOLS_MAX_TURNS = envf("ARC_TOOLS_MAX_TURNS", 60)   # AgentConfig.max_turns (E025 training rollouts: 40)
MODEL_REQUIRE = os.environ.get("ARC_MODEL_REQUIRE", "")  # file that must exist in the served model dir, else the run
                                                    # fails loudly (E028: e028_quant_info.json, only in our checkpoint)
EXEC_STYLE = os.environ.get("ARC_EXEC_STYLE", "plain")  # LocalExecutor output format; "ipython" = the E025 sandbox's
                                                    # text format (use with ARC_TOOLS_DESC=e025)
LOCAL_WALL_MIN = envf("ARC_LOCAL_WALL_MIN", 0.0)    # > 0: local (commit) runs only: emergency exit at T0 + this
TOOLS_SOFT_FRAC = envf("ARC_TOOLS_SOFT_FRAC", 0.0)  # > 0: budget_soft = this x the attempt's hard cap (E028 trained on
                                                    # 40k/80k = 0.5); 0 = min(ARC_TOOLS_SOFT, cap) as before
# ---- E032 (e031 mode) toggles, all opt-in: the E031 interface the E032 SFT model was trained on, with
# ARC_TOOLS_DESC=e031, ARC_TOOLS_PROMPT=lean2sub, ARC_EXEC_STYLE=ipython, ARC_TOOLS_SOFT=20000, ARC_TOOLS_MAX_TURNS=40,
# ARC_TOOLS_SOFT_WORDING=strong (experiments/E027_l4_throughput/e032_kaggle_settings.json)
EFFORT_HEADER = os.environ.get("ARC_TOOLS_EFFORT_HEADER", "")  # AgentConfig.effort_header ("minimal": E031's system
                                                    # message, rendered after the tools block)
TURNS_NOTE = envf("ARC_TOOLS_TURNS_NOTE", 0)        # AgentConfig.turns_note (E031: 5)
SOFT_WORDING = os.environ.get("ARC_TOOLS_SOFT_WORDING", "gentle")  # AgentConfig.soft_wording past the soft target
                                                    # (E031: strong = "submit now, unless one more experiment ...")
FORCE_SUBMIT = envf("ARC_TOOLS_FORCE_SUBMIT", 256)  # tokens for the forced submit_answer at the cap (0 = off;
                                                    # E031/E032: 0, as E031 never forced a submit)
FORCE_SUBMIT_LAST = envf("ARC_TOOLS_FORCE_SUBMIT_LAST", 0)  # 1: force the submit only when the session answer,
                                                    # solve() and salvage fallbacks all fail (E032 submission)
NOTE_CAP = envf("ARC_TOOLS_NOTE_CAP", 0)            # AgentConfig.budget_note_cap: the hard limit the budget notes state
                                                    # (E031: 60000) whatever the attempt's real cap; with it set,
                                                    # budget_soft = ARC_TOOLS_SOFT unchanged (training wording)
AGENT_EXTRA = os.environ.get("ARC_AGENT_EXTRA", "")  # JSON {AgentConfig field: value}, applied last (unknown fields
                                                    # are logged and skipped); hook for new harness options
STRICT_FEATURES = envf("ARC_STRICT_FEATURES", 0)    # 1: a requested agent option the self-test switches off fails
                                                    # the commit gate (summary v3.dropped)
FREE_ATTEMPT2 = envf("ARC_FREE_ATTEMPT2", 0)        # 1: attempt_2 from the attempt's own alternatives (no tokens)
ATTEMPT2 = os.environ.get("ARC_ATTEMPT2", "legacy")  # legacy (v2/v1 rule) | off | gated (E021)
ATTEMPT2_UNKNOWN = envf("ARC_ATTEMPT2_UNKNOWN", 1)  # gated: also retry submitted answers without a train check (E012)
ATTEMPT2_SAFETY = envf("ARC_ATTEMPT2_SAFETY", 0.85)  # gated: budget <= safety x per-attempt speed x time to force
SESSION_MEM_MB = envf("ARC_SESSION_MEM_MB", 2048)   # session kernel RLIMIT_AS = VmSize + this; programs min(3072, .)
                                                    # 0 = v2 (only the executor's absolute 4 GB RLIMIT_AS)
LOCAL_FORCE_MIN = envf("ARC_LOCAL_FORCE_MIN", 40.0)  # local validation: force at max(T0 + this, ready + 20 min)
LOCAL_FORCE_CAP_MIN = envf("ARC_LOCAL_FORCE_CAP_MIN", 60.0)  # ... but never later than T0 + this
LOCAL_READY_MIN = envf("ARC_LOCAL_READY_MIN", 0.0)  # > 0: local force = server ready + this (overrides the above)
LOCAL_HARD_MIN = envf("ARC_LOCAL_HARD_MIN", 8.0)    # local hard stop = force + this
LOCAL_TASKS = os.environ.get("ARC_LOCAL_TASKS", "")  # comma-separated subset of the local test set
LOCAL_SOURCE = os.environ.get("ARC_LOCAL_SOURCE", "local")  # local: arc2-local-testcases; eval: competition evaluation set
                                                    # (load probe: real scheduler on public-eval pairs, self-scored)
PUBLIC_TEST_IDS_SHA256 = "4cab35dea7bde6ee39bbfcfdcfc101153e19f333c14b68c68b84a7a3278c265e"  # public test file
PHASES = {}  # phase timings (s since T0), written to the summary
KNOWN_SOURCES = ("submit", "forced", "session", "salvage", "parse", "lenient", "program_pass", "program_best_fail",
                 "session_solve")
DROPPED = []  # requested agent options the self-test switched off (summary v3.dropped; ARC_STRICT_FEATURES gate)
PKG = os.path.join(WORK, "arc_solver")  # evals/arc_harness + notes/prompts + solver.py, written by the notebook

PROMPT_TEMPLATE = """You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.

Respond in the format of the training output examples

--Training Examples--
{training_examples}
--End of Training Examples--

--Test Input--
{test_input}
--End of Test Input--

Your response:"""  # noqa: F821  (ARC Prize official prompt, injected by build_notebook.py)

CLOSE = "</think>"
FORCE_TEXT = "\n\nI have to give my final answer now.\n</think>\n\n"
LOW_YIELD = 20
TIER_SYSTEM = {
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
             "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final "
             "answer.",
    "medium": None,
}


def log(*a):
    el = time.time() - T0
    print(f"[{el / 3600:5.2f}h {time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- prompt / parse (verbatim from arc_harness)

def build_prompt(train, test_input):
    ex = ""
    for i, pair in enumerate(train):
        ex += f"--Example {i}-- \n\n INPUT: \n\n" + json.dumps(pair["input"]) + "\n\n"
        ex += "OUTPUT: \n\n" + json.dumps(pair["output"]) + "\n\n"
    return PROMPT_TEMPLATE.format(training_examples=ex, test_input=json.dumps(test_input))


def render_prompt(user_content, tier=TIER):
    s = ""
    if TIER_SYSTEM.get(tier):
        s += f"<|im_start|>system\n{TIER_SYSTEM[tier]}<|im_end|>\n"
    return s + f"<|im_start|>user\n{user_content.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def answer_section(gen):
    i = gen.rfind(CLOSE)
    return None if i < 0 else gen[i + len(CLOSE):]


def is_grid(x):
    return isinstance(x, list) and len(x) > 0 and all(isinstance(r, list) for r in x)


def extract_from_boxed(s):
    m = re.search(r"\\boxed\{(.*?)\}", s, re.DOTALL)
    if m:
        try:
            g = json.loads(m.group(1).strip())
            return g if is_grid(g) else None
        except json.JSONDecodeError:
            return None
    return None


def backscan_json_parser(s):
    end = max(s.rfind("]"), s.rfind("}"))
    if end < 0:
        return None
    close = s[end]
    open_ = "[" if close == "]" else "{"
    depth = 1
    for i in range(end - 1, -1, -1):
        depth += (s[i] == close) - (s[i] == open_)
        if depth == 0:
            try:
                g = json.loads(s[i:end + 1])
                return g if is_grid(g) else None
            except json.JSONDecodeError:
                return None
    return None


def parse(s):
    return extract_from_boxed(s or "") or backscan_json_parser(s or "")


def valid_grid(grid):
    return (isinstance(grid, list) and 1 <= len(grid) <= 30
            and all(isinstance(row, list) and 1 <= len(row) <= 30 for row in grid)
            and len({len(row) for row in grid}) == 1
            and all(type(v) is int and 0 <= v <= 9 for row in grid for v in row))


# ---------------------------------------------------------------- paths

def find_one(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None


def comp_file(name):
    return (find_one(f"{INPUT}/competitions/arc-prize-2026-arc-agi-2/{name}")
            or find_one(f"{INPUT}/arc-prize-2026-arc-agi-2/{name}")
            or find_one(f"{INPUT}/**/{name}"))


def official_model_dir():
    """The official Qwen3.8-27B-FP8 mirror (mikedan7/qwen3-8-27b-fp8-official), mounted by Kaggle at
    /kaggle/input/models/mikedan7/qwen3-8-27b-fp8-official/pytorch/hf-fp8/1 (framework lower-cased)."""
    for pat in [f"{INPUT}/models/mikedan7/qwen3-8-27b-fp8-official/**/config.json",
                f"{INPUT}/qwen3-8-27b-fp8-official/**/config.json",
                f"{INPUT}/**/qwen3-8-27b-fp8*/**/config.json"]:
        p = find_one(pat)
        if p:
            return os.path.dirname(p)
    raise FileNotFoundError("model dir")


def source_dir(source, need="config.json"):
    """Directory of an attached Kaggle model given as owner/slug (mounted at
    /kaggle/input/models/<owner>/<slug>/<framework>/<variation>/<version>/) or as an existing path."""
    if os.path.isdir(source):
        return source
    owner, _, slug = source.strip("/").partition("/")
    for pat in [f"{INPUT}/models/{owner}/{slug}/**/{need}", f"{INPUT}/{slug}/**/{need}"]:
        p = find_one(pat)
        if p:
            return os.path.dirname(p)
    raise FileNotFoundError(f"model {source!r} not attached (no {need} under {INPUT})")


def model_dir():
    """Model to serve: ARC_MODEL_SOURCE if set (never silently another model), else the official FP8 mirror."""
    return source_dir(MODEL_SOURCE) if MODEL_SOURCE else official_model_dir()


def model_identity(mdir):
    """What is about to be served (logged as MODEL IDENTITY before the server starts). Raises if ARC_MODEL_REQUIRE
    is set and that file is missing, or if ARC_MODEL_SOURCE's slug is not in the resolved path."""
    info = {"dir": mdir, "tokenizer": tokenizer_dir()}
    if MODEL_SOURCE and "/" + MODEL_SOURCE.split("/")[-1] + "/" not in mdir + "/":
        raise RuntimeError(f"WRONG MODEL: {mdir} is not {MODEL_SOURCE}")
    if MODEL_REQUIRE:
        req = os.path.join(mdir, MODEL_REQUIRE)
        if not os.path.isfile(req):
            raise RuntimeError(f"WRONG MODEL: {MODEL_REQUIRE} missing in {mdir}")
        info[MODEL_REQUIRE] = open(req).read()[:2000]
    raw = open(os.path.join(mdir, "config.json"), "rb").read()
    info["config_md5"] = hashlib.md5(raw).hexdigest()
    cfg = json.loads(raw)
    qc = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
    info["quant_method"] = qc.get("quant_method")
    info["shards"] = {os.path.basename(p): os.path.getsize(p) for p in sorted(glob.glob(f"{mdir}/*.safetensors"))}
    return info


def server_identity(slog):
    """After startup: the server's quantization / KV lines and /v1/models (served id -> model path)."""
    import urllib.request
    out = {}
    try:
        lines = open(slog, errors="ignore").read().splitlines()
        out["quant_lines"] = [ln[-300:] for ln in lines if re.search(r"quantiz|modelopt", ln, re.I)][:6]
        out["kv_lines"] = [ln[-200:] for ln in lines if re.search(r"KV cache size|GPU KV cache|Maximum concurrency",
                                                                  ln)][:4]
    except Exception as exc:
        out["log_error"] = repr(exc)
    try:
        with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as r:
            out["models"] = [(m.get("id"), m.get("root")) for m in json.load(r).get("data", [])]
    except Exception as exc:
        out["models_error"] = repr(exc)
    return out


def tokenizer_dir():
    """vLLM --tokenizer override (None = the model's own tokenizer)."""
    if not TOKENIZER:
        return None
    return official_model_dir() if TOKENIZER == "official" else source_dir(TOKENIZER, "tokenizer.json")


def wheel_dir():
    """Our own wheel sources (attached as kernel sources): SGLang 0.5.20 cu130 or the user's vLLM 0.23.0 cu128."""
    pat = "sglang-0.5.20-*.whl" if SERVER == "sglang" else "vllm-0.23.0*.whl"
    p = find_one(f"{INPUT}/notebooks/chankhavu/**/{pat}") or find_one(f"{INPUT}/**/{pat}")
    if not p:
        raise FileNotFoundError(f"wheel source for {SERVER}")
    return os.path.dirname(p)


# ---------------------------------------------------------------- submission

class Submission:
    def __init__(self, tasks):
        self.tasks = tasks
        self.answers = {}  # (tid, pair) -> {attempt_index: grid}
        self.meta = {}     # (tid, pair) -> {attempt_index: {train_pass, train_frac, answer_source, alternatives}}
        self.sources = {}  # (tid, pair) -> (source of attempt_1, source of attempt_2), set by build()

    def fallback(self, tid, p):
        g = self.tasks[tid]["test"][p]["input"]
        return g if valid_grid(g) else [[0]]

    def _pick(self, tid, p):
        """Top-2 distinct grids with their sources (v3 ranking, arc_harness.selection); v2's rule if that fails."""
        answers = self.answers.get((tid, p), {})
        try:
            from arc_harness import selection as SEL
            meta = self.meta.get((tid, p), {})
            recs = {a: {**meta.get(a, {}), "pred": answers.get(a)} for a in set(answers) | set(meta)}
            top = SEL.select_two(SEL.candidates(recs))
            return [c["grid"] for c in top], [f"a{c['attempt']}:{c['src']}" for c in top]
        except Exception:
            got = [g for _, g in sorted(answers.items()) if valid_grid(g)]
            return got, ["v2"] * len(got)

    def build(self):
        sub = {}
        for tid, task in self.tasks.items():
            lst = []
            for p in range(len(task["test"])):
                got, src = self._pick(tid, p)
                fb = self.fallback(tid, p)
                a1 = got[0] if got else fb
                a2 = next((g for g in got[1:] if g != a1), None)
                s2 = src[1] if a2 is not None else None
                if a2 is None:
                    a2, s2 = (fb, "fallback") if a1 != fb else (a1, "dup")
                self.sources[(tid, p)] = (src[0] if got else "fallback", s2)
                lst.append({"attempt_1": a1, "attempt_2": a2})
            sub[tid] = lst
        return sub

    def write(self):
        sub = self.build()
        tmp = OUT_SUB + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sub, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, OUT_SUB)


def check_submission(path, tasks):
    sub = json.load(open(path))
    assert set(sub) == set(tasks), "task ids differ"
    for tid, task in tasks.items():
        assert len(sub[tid]) == len(task["test"]), tid
        for slot in sub[tid]:
            assert valid_grid(slot["attempt_1"]) and valid_grid(slot["attempt_2"]), tid
    return True


# ---------------------------------------------------------------- environment

def sh(cmd, timeout=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def driver_env():
    """Host driver libcuda first (images also carry an older compat libcuda); set before any CUDA process starts."""
    real = "/usr/local/nvidia/lib64"
    if os.path.exists(real + "/libcuda.so"):
        os.environ["LD_LIBRARY_PATH"] = real + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_PRELOAD"] = real + "/libcuda.so"


def cuda_toolkit_env(env=None, shim=None):
    """CUDA 13 toolkit from the pip wheels for JIT builds (SGLang's sglang-kernel JIT, FlashInfer, tvm-ffi).

    The image has no /usr/local/cuda/bin/nvcc; our SGLang wheel source installs nvidia-cuda-nvcc/crt/nvvm/cccl/runtime
    into site-packages/nvidia/cu13. JIT builders expect $CUDA_HOME/{bin/nvcc,include,nvvm} and link with
    `-L$CUDA_HOME/lib64 -lcudart`, while the wheels ship lib/ with only libcudart.so.13, so a shim directory of
    symlinks gives the expected layout. Updates `env` (default os.environ); returns the shim path or None."""
    env = os.environ if env is None else env
    shim = shim or os.environ.get("ARC_CUDA_SHIM", "/tmp/cuda13")
    import site
    roots = list(dict.fromkeys(site.getsitepackages() + glob.glob("/usr/local/lib/python3*/dist-packages")
                               + glob.glob("/usr/lib/python3/dist-packages")))
    cu = next((os.path.join(r, "nvidia/cu13") for r in roots if os.path.exists(os.path.join(r, "nvidia/cu13/bin/nvcc"))),
              None)
    if cu is None:
        log("cuda toolkit: no nvidia/cu13/bin/nvcc in", roots)
        return None
    os.makedirs(os.path.join(shim, "lib64"), exist_ok=True)
    for d in ("bin", "include", "nvvm", "lib", "share"):
        src, dst = os.path.join(cu, d), os.path.join(shim, d)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)
    for f in os.listdir(os.path.join(cu, "lib")):
        dst = os.path.join(shim, "lib64", f)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(cu, "lib", f), dst)
    for name in ("cudart", "nvrtc"):
        real = sorted(glob.glob(os.path.join(cu, "lib", f"lib{name}.so.*")))
        dst = os.path.join(shim, "lib64", f"lib{name}.so")
        if real and not os.path.lexists(dst):
            os.symlink(real[0], dst)
    env.update(CUDA_HOME=shim, CUDA_PATH=shim, CUDA_ROOT=shim,
               PATH=os.path.join(shim, "bin") + ":" + env.get("PATH", ""),
               LD_LIBRARY_PATH=(env.get("LD_LIBRARY_PATH", "") + ":" + os.path.join(shim, "lib64")).lstrip(":"),
               TORCH_CUDA_ARCH_LIST=os.environ.get("ARC_CUDA_ARCH", "8.9"))
    return shim


def sglang_jit_env(env):
    """SGLang JIT kernel cache: a writable dir, seeded from a pre-built cache attached as input if there is one."""
    import shutil
    jit = os.environ.get("ARC_SGL_JIT_DIR", "/tmp/sglang_jit")
    seed = find_one(f"{INPUT}/**/sglang_jit_cache/sm89") if os.path.isdir(INPUT) else None
    if seed and not os.path.exists(os.path.join(jit, "sm89")):
        shutil.copytree(seed, os.path.join(jit, "sm89"), symlinks=True)
        log("sglang JIT cache seeded from", seed)
    os.makedirs(jit, exist_ok=True)
    env["SGLANG_JIT_CACHE_DIR"] = jit
    return jit


def gpu_inventory():
    """[(name, memory MiB, compute capability)] of the GPUs actually present (nvidia-smi)."""
    _, out = sh("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits", timeout=60)
    gpus = []
    for ln in out.splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) == 3:
            try:
                gpus.append((parts[0], int(float(parts[1])), float(parts[2])))
            except ValueError:
                pass
    return gpus


def choose_tp(gpus, weights_gb=29.0, reserve_gb=6.0):
    """Tensor-parallel size for Qwen3.8-27B-FP8 on the GPUs present, or 0 if they cannot run it.

    Kaggle once gave a `machine_shape: NvidiaL4` session 2x T4 (E014). FP8 needs SM89+; TP must divide the 4 KV
    heads; each GPU needs its weight shard plus `reserve_gb` for activations, graphs, KV and Mamba state.
    ARC_TP overrides (0 = auto)."""
    forced = int(envf("ARC_TP", 0))
    if forced:
        return forced
    good = [g for g in gpus if g[2] >= 8.9]
    for tp in (4, 2, 1):
        if len(good) >= tp and min(g[1] for g in good[:tp]) / 1024 >= weights_gb / tp + reserve_gb:
            return tp
    return 0


def install_server():
    wd = wheel_dir()
    t = time.time()
    if SERVER == "sglang":
        cmd = f"{sys.executable} -m pip install --no-index --find-links {wd} -r {wd}/requirements.lock"
    else:
        cmd = (f"{sys.executable} -m pip install --no-index --find-links {wd} vllm==0.23.0 flashinfer-python "
               f"flashinfer-jit-cache")
    code, out = sh("uv " + cmd.replace(f"{sys.executable} -m pip install", "pip install --system --python "
                                       + sys.executable), timeout=2400)
    log(f"uv offline install of {SERVER} from {wd}: rc={code} in {time.time() - t:.0f}s")
    if code != 0:
        print(out[-1500:])
        t = time.time()
        code, out = sh(cmd, timeout=2400)
        log(f"pip offline install of {SERVER} from {wd}: rc={code} in {time.time() - t:.0f}s")
    if code != 0:
        print(out[-4000:])
        raise RuntimeError("pip install failed")
    mod = "sglang" if SERVER == "sglang" else "vllm"
    code, out = sh(f"{sys.executable} -c 'import torch; print(torch.version.cuda)'", timeout=600)
    if out.strip().splitlines()[-1:] == ["None"]:
        # A CPU torch of the same version in the image satisfies the pin and is kept: force the CUDA build.
        tw = sorted(glob.glob(f"{wd}/torch-*.whl")) + sorted(glob.glob(f"{wd}/torchvision-*.whl")) + \
            sorted(glob.glob(f"{wd}/torchaudio-*.whl"))
        code, out2 = sh(f"{sys.executable} -m pip install --no-index --force-reinstall --no-deps " + " ".join(tw),
                        timeout=1800)
        log(f"CPU torch found; reinstalled CUDA torch from the wheel source: rc={code}")
    code, out = sh(f"{sys.executable} -c 'import torch, {mod}; print(torch.__version__, torch.version.cuda, "
                   f"torch.cuda.is_available(), torch.cuda.device_count(), {mod}.__version__)'", timeout=600)
    log("torch/server:", out.strip().splitlines()[-1] if out.strip() else out)
    if SERVER == "sglang":
        shim = cuda_toolkit_env()
        code, out = sh(f"{shim}/bin/nvcc --version | tail -2", timeout=120) if shim else (1, "no shim")
        log(f"cuda toolkit shim {shim}: nvcc rc={code} {out.strip().splitlines()[-1:] if out.strip() else ''}")
    if SERVER == "vllm":
        _, ldd = sh(f"ldd $({sys.executable} -c 'import glob, os, vllm; "
                    f"print(glob.glob(os.path.dirname(vllm.__file__) + \"/_C*.so\")[0])') | grep -i 'not found'",
                    timeout=120)
        log("vllm._C missing libraries:", ldd.strip() or "none")
    _, mem = sh("free -g | head -2", timeout=30)
    log("host memory:", " | ".join(mem.strip().splitlines()))
    _, freeze = sh(f"{sys.executable} -m pip freeze", timeout=300)
    open(os.path.join(LOG_DIR, "pip_freeze.txt"), "w").write(freeze)
    log("installed:", " ".join(l for l in freeze.splitlines() if l.split("==")[0].lower() in
                               ("vllm", "torch", "transformers", "flashinfer-python", "triton", "sglang", "numpy")))


def install_fp8_configs():
    """E027: copy tuned W8A8 block-FP8 Triton configs (ARC_FP8_CONFIGS=<dir> of vLLM-named JSON files) into the installed
    vLLM's configs folder, which is where vLLM 0.23 looks them up (no env override). Never fails the run."""
    src = os.environ.get("ARC_FP8_CONFIGS", "")
    if not src or SERVER != "vllm":
        return
    try:
        files = sorted(glob.glob(os.path.join(src, "*.json")))
        code, out = sh(f"{sys.executable} -c 'import os, vllm; print(os.path.join(os.path.dirname(vllm.__file__), "
                       f"\"model_executor/layers/quantization/utils/configs\"))'", timeout=600)
        dst = out.strip().splitlines()[-1] if code == 0 and out.strip() else ""
        if not files or not os.path.isdir(dst):
            log(f"fp8 configs: nothing installed (files={len(files)}, configs dir {dst!r})")
            return
        for f in files:
            shutil.copy(f, os.path.join(dst, os.path.basename(f)))
        log(f"fp8 configs: installed {len(files)} tuned files from {src} into {dst}")
    except Exception as exc:
        log(f"fp8 configs: install failed ({type(exc).__name__}: {exc}); vLLM uses its default configs")


def file_hashes(paths, workers=16):
    def h(p):
        s256, s1 = hashlib.sha256(), hashlib.sha1()
        size = os.path.getsize(p)
        s1.update(f"blob {size}\0".encode())
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 24)
                if not b:
                    break
                s256.update(b)
                s1.update(b)
        return p, {"size": size, "sha256": s256.hexdigest(), "git_sha1": s1.hexdigest()}
    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(h, paths))


def preload(mdir, workers=16):
    files = [os.path.join(r, f) for r, _, fs in os.walk(mdir) for f in fs]

    def rd(p):
        with open(p, "rb") as f:
            while f.read(1 << 26):
                pass
    t = time.time()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(rd, files))
    log(f"preloaded {len(files)} model files in {time.time() - t:.0f}s")


def launch_server(mdir):
    env = os.environ.copy()
    env.update(PYTORCH_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
    if SERVER == "sglang":
        cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", mdir, "--served-model-name", SERVED,
               "--host", "127.0.0.1", "--port", str(PORT), "--tp-size", str(TP), "--kv-cache-dtype", "fp8_e4m3",
               "--context-length", str(MAX_MODEL_LEN), "--mem-fraction-static", os.environ.get("ARC_SGL_MEM", "0.85"),
               "--max-running-requests", str(MAX_SEQS), "--trust-remote-code",
               "--attention-backend", os.environ.get("ARC_SGL_ATTN", "triton"), "--log-level", "info",
               "--enable-cache-report"]
        # Radix (prefix) cache is mandatory for the tool agent (every turn re-sends the whole transcript); the
        # hybrid model's Mamba state slots bound the running batch: max_running = slots / slots-per-request
        # (5 with extra_buffer + overlap; 3 with extra_buffer_lazy + SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1).
        if os.environ.get("ARC_RADIX", "1") != "1":
            cmd.append("--disable-radix-cache")
        if os.environ.get("ARC_SGL_MAMBA_SLOTS"):
            cmd += ["--max-mamba-cache-size", os.environ["ARC_SGL_MAMBA_SLOTS"]]
        if os.environ.get("ARC_SGL_MAMBA_STRATEGY"):
            cmd += ["--mamba-radix-cache-strategy", os.environ["ARC_SGL_MAMBA_STRATEGY"]]
        if os.environ.get("ARC_SGL_SKIP_LOCK", "0") == "1":
            env["SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK"] = "1"
        if envf("ARC_SGL_HICACHE", 0) > 0:  # host-memory KV offload (hierarchical cache), ratio = host / device
            cmd += ["--enable-hierarchical-cache", "--hicache-ratio", os.environ["ARC_SGL_HICACHE"],
                    "--hicache-write-policy", os.environ.get("ARC_SGL_HICACHE_POLICY", "write_through")]
        cuda_toolkit_env(env)
        sglang_jit_env(env)
    else:
        attn = "TRITON_ATTN"  # FlashInfer fp8-KV kernels need SM90+; L4 is SM89
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", mdir,
               "--served-model-name", SERVED, "--host", "127.0.0.1", "--port", str(PORT),
               "--tensor-parallel-size", str(TP), "--gpu-memory-utilization", str(GPU_UTIL),
               "--max-model-len", str(MAX_MODEL_LEN), "--max-num-seqs", str(MAX_SEQS),
               "--kv-cache-dtype", "fp8", "--attention-backend", attn, "--enable-prefix-caching",
               "--trust-remote-code", "--no-async-scheduling"]
        if MTP > 0:
            cmd += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": MTP})]
        env.update(VLLM_FORCE_ATTN_BACKEND=attn, VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="1")
    tok = tokenizer_dir()
    if tok:  # E028: the SFT checkpoint's tokenizer_config names Qwen3_5Tokenizer (transformers 4.57 can't load it)
        cmd += ["--tokenizer-path" if SERVER == "sglang" else "--tokenizer", tok]
    cmd += json.loads(os.environ.get("ARC_SERVER_EXTRA", "[]"))  # extra server flags (bench / experiments)
    log("server cmd:", " ".join(cmd))
    path = os.path.join(LOG_DIR, f"{SERVER}_server.log")
    fh = open(path, "a")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True, env=env)
    return proc, path


def kill_server(proc=None):
    """Kill the server's whole process tree (vLLM renames its workers VLLM::Worker_TP*), then wait for GPU memory."""
    if proc is not None:
        try:
            os.killpg(proc.pid, 9)
        except Exception:
            pass
    if proc is not None:
        for pid in descendants(proc.pid):
            try:
                os.kill(pid, 9)
            except OSError:
                pass
    if ON_KAGGLE:  # name patterns only on Kaggle (a shared host has other users' vllm/sglang processes)
        sh("pkill -9 -f '[v]llm.entrypoints' ; pkill -9 -f '[V]LLM::' ; pkill -9 -f '[s]glang.launch_server' ; "
           "pkill -9 -f '[s]glang::'", timeout=60)
    time.sleep(5)
    for _ in range(24):
        _, out = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", timeout=30)
        used = [int(x) for x in out.split() if x.strip().isdigit()]
        if used and max(used) < 2000:
            break
        time.sleep(5)


def server_log_tail(path, n=60):
    try:
        return "".join(open(path, errors="ignore").readlines()[-n:])
    except Exception:
        return ""


def wait_ready(proc, path, timeout):
    import urllib.request
    t = time.time()
    while time.time() - t < timeout:
        if proc.poll() is not None:
            log(f"server exited rc={proc.returncode}\n" + server_log_tail(path))
            return False
        try:
            with urllib.request.urlopen(BASE + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    log(f"server ready in {time.time() - t:.0f}s")
                    return True
        except Exception:
            pass
        time.sleep(5)
    log("server not ready in time\n" + server_log_tail(path))
    return False


def server_capacity(path):
    """(kv_tokens, max_running) from the server log."""
    txt = open(path, errors="ignore").read()
    if SERVER == "sglang":
        kv, mr = None, None
        try:  # runtime values (after the Mamba cap); the log is the fallback
            import urllib.request
            with urllib.request.urlopen(BASE + "/get_server_info", timeout=20) as r:
                st = json.loads(r.read())["internal_states"][0]
            kv = st.get("memory_usage", {}).get("token_capacity")
            mr = st.get("effective_max_running_requests_per_dp")
        except Exception as exc:
            log("get_server_info failed:", exc)
        kvl = re.findall(r"KV Cache is allocated\. dtype: \S+ #tokens: (\d+)", txt)
        mrl = re.findall(r"max_running_requests is capped to (\d+)", txt)
        out = (kv or (int(kvl[-1]) if kvl else None), mr or (int(mrl[-1]) if mrl else None))
    else:
        m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", txt)
        out = (int(m[-1].replace(",", "")) if m else None, None)
    log("server capacity (kv tokens, max running):", out)
    return out


_VLLM_METRICS = {"vllm:prefix_cache_queries_total": "pc_queries", "vllm:prefix_cache_hits_total": "pc_hits",
                 "vllm:num_preemptions_total": "preemptions", "vllm:kv_cache_usage_perc": "kv_usage",
                 "vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
                 "vllm:prompt_tokens_total": "prompt_tokens", "vllm:generation_tokens_total": "gen_tokens"}


def server_stats(timeout=5.0):
    """Cache / load counters from the server (None if unavailable). Cumulative counters are summed over engines.

    vLLM: /metrics (prefix-cache queries/hits in tokens, preemptions, KV usage fraction, running/waiting).
    SGLang: /v1/loads (running/waiting, token_usage, cumulative uncached prefill tokens, retracted queue)."""
    import urllib.request
    try:
        if SERVER == "sglang" and _SGL_PROM.get("use"):
            return _sglang_prometheus(timeout)
        if SERVER == "sglang":
            try:
                r = urllib.request.urlopen(BASE + "/v1/loads", timeout=timeout)
            except Exception:  # older SGLang (e.g. 0.5.14 locally): /v1/loads fails; use /metrics from now on
                _SGL_PROM["use"] = True
                return _sglang_prometheus(timeout)
            with r:
                ld = json.loads(r.read())["loads"]
            return {"running": sum(x.get("num_running_reqs", 0) for x in ld),
                    "waiting": sum(x.get("num_waiting_reqs", 0) for x in ld),
                    "kv_usage": max(x.get("token_usage", 0.0) for x in ld),
                    "kv_used_tokens": sum(x.get("num_used_tokens", 0) for x in ld),
                    "uncached_prefill": sum(x.get("total_prefill_uncached_tokens", 0) for x in ld),
                    "retracted": sum((x.get("queues") or {}).get("retracted", 0) for x in ld)}
        with urllib.request.urlopen(BASE + "/metrics", timeout=timeout) as r:
            txt = r.read().decode(errors="ignore")
        out = {}
        for line in txt.splitlines():
            if line.startswith("#"):
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name in _VLLM_METRICS:
                try:
                    v = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    continue
                k = _VLLM_METRICS[name]
                out[k] = max(out.get(k, 0.0), v) if k == "kv_usage" else out.get(k, 0.0) + v
        return out or None
    except Exception:
        return None


_SGL_PROM = {}
_SGL_METRICS = {"sglang:num_running_reqs": "running", "sglang:num_queue_reqs": "waiting",
                "sglang:token_usage": "kv_usage", "sglang:prompt_tokens_total": "pc_queries",
                "sglang:cached_tokens_total": "pc_hits", "sglang:num_retracted_reqs": "preemptions"}


def _sglang_prometheus(timeout):
    """SGLang /metrics (needs --enable-metrics) in the vLLM field names: cached / prompt tokens, retractions."""
    import urllib.request
    with urllib.request.urlopen(BASE + "/metrics", timeout=timeout) as r:
        txt = r.read().decode(errors="ignore")
    out = {"_prom": True}
    for line in txt.splitlines():
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if line.startswith("#") or name not in _SGL_METRICS:
            continue
        try:
            v = float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
        k = _SGL_METRICS[name]
        out[k] = max(out.get(k, 0.0), v) if k == "kv_usage" else out.get(k, 0.0) + v
    return out


def cache_line(cur, prev):
    """One progress-line fragment: prefix-cache hit rate over the window, KV usage, preemptions/retractions."""
    if not cur:
        return "server stats n/a"
    prev = prev or {}
    if SERVER == "sglang" and not cur.get("_prom"):
        new = cur["uncached_prefill"] - prev.get("uncached_prefill", 0)
        sent = cur.get("_client_prompt", 0) - prev.get("_client_prompt", 0)
        hit = f"{100 * (1 - new / sent):.1f}%" if sent > 0 else "n/a"
        return (f"cache hit {hit} (uncached prefill {new} of {sent}), running {cur['running']}, "
                f"waiting {cur['waiting']}, "
                f"KV usage {100 * cur['kv_usage']:.1f}%, retracted queue {cur['retracted']}")
    dq = cur.get("pc_queries", 0) - prev.get("pc_queries", 0)
    dh = cur.get("pc_hits", 0) - prev.get("pc_hits", 0)
    hit = f"{100 * dh / dq:.1f}%" if dq > 0 else "n/a"
    return (f"cache hit {hit} (of {dq:.0f} prompt tokens), running {cur.get('running', 0):.0f}, "
            f"waiting {cur.get('waiting', 0):.0f}, KV usage {100 * cur.get('kv_usage', 0):.1f}%, "
            f"preemptions {cur.get('preemptions', 0) - prev.get('preemptions', 0):.0f} "
            f"(total {cur.get('preemptions', 0):.0f})")


# ---------------------------------------------------------------- client

class Client:
    def __init__(self):
        import httpx
        self.last_prompt_tokens = None
        self.prompt_tokens = 0  # summed over finished requests (final usage), for the cache hit rate
        self.cached_tokens = 0  # usage.prompt_tokens_details.cached_tokens when the server reports it
        self.http = httpx.AsyncClient(base_url=BASE, timeout=httpx.Timeout(connect=30.0, read=900.0, write=120.0, pool=None),
                                      limits=httpx.Limits(max_connections=512, max_keepalive_connections=512))

    async def count(self, prompt):
        try:
            r = await self.http.post("/tokenize", json={"model": SERVED, "prompt": prompt})
            return int(r.json()["count"])
        except Exception:
            return int(len(prompt) / 2.5)

    async def stream(self, prompt, max_tokens, deadline, on_tokens, stop=None, info=None):
        """Streamed completion bounded by `deadline` even when no data arrives (finish 'deadline')."""
        parts = {"text": [], "ntok": 0}
        try:
            return await asyncio.wait_for(self._stream(prompt, max_tokens, deadline, on_tokens, stop, info, parts),
                                          timeout=max(1.0, deadline - time.time() + 5))
        except asyncio.TimeoutError:
            return "".join(parts["text"]), parts["ntok"], "deadline"

    async def _stream(self, prompt, max_tokens, deadline, on_tokens, stop, info, parts):
        """Streamed completion. Returns (text, completion_tokens, finish_reason); finish 'deadline' if cut off.

        Breaking out of the stream closes the connection, which makes vLLM abort the request.
        """
        body = {"model": SERVED, "prompt": prompt, "max_tokens": max_tokens, "temperature": TEMPERATURE,
                "top_p": 0.95, "top_k": 20, "stream": True, "skip_special_tokens": False,
                "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
        if stop:
            body["stop"] = stop
        text, ntok, finish = parts["text"], 0, None
        last_usage = None
        async with self.http.stream("POST", "/v1/completions", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {(await r.aread())[:300]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                d = json.loads(data)
                if d.get("error"):
                    raise RuntimeError(f"server error event: {str(d['error'])[:300]}")
                u = d.get("usage")
                got_text = False
                for c in d.get("choices") or []:
                    text.append(c.get("text") or "")
                    got_text = got_text or bool(c.get("text"))
                    if c.get("finish_reason"):
                        finish = c["finish_reason"]
                if u:
                    last_usage = u
                if u and u.get("completion_tokens") is not None:
                    self.last_prompt_tokens = u.get("prompt_tokens") or self.last_prompt_tokens
                    if info is not None and u.get("prompt_tokens"):
                        info["prompt_tokens"] = u["prompt_tokens"]
                    on_tokens(u["completion_tokens"] - ntok)
                    ntok = u["completion_tokens"]
                elif got_text:  # server without continuous usage stats: ~1 token per chunk until usage arrives
                    on_tokens(1)
                    ntok += 1
                parts["ntok"] = ntok
                if time.time() > deadline:
                    finish = "deadline"
                    break
        if last_usage and last_usage.get("prompt_tokens"):
            self.prompt_tokens += last_usage["prompt_tokens"]
            self.cached_tokens += (last_usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        return "".join(text), ntok, finish or "stop"


# ---------------------------------------------------------------- tool executor (Kaggle hardening)

def session_prelude(mem_mb=None, helpers=None):
    """Extra session prelude (re-run after every restart): the E020 --session-mem-mb cap first, then the helpers."""
    from arc_harness import agent_qwen as AQ
    from arc_harness import executor as EX
    mem_mb = SESSION_MEM_MB if mem_mb is None else mem_mb
    helpers = HELPERS if helpers is None else helpers
    extra = EX.SESSION_MEM_CAP.format(mb=int(mem_mb)) if mem_mb and hasattr(EX, "SESSION_MEM_CAP") else ""
    return extra + (AQ.HELPERS_SRC if helpers else "")


def robust_executor_factory(timeout, prelude_extra=None):
    """E012 LocalExecutor (qwen-tools-3 executor: RLIMIT_AS/CPU/FSIZE, nice, oom_score_adj=1000, SIGINT to the
    process group, retried start with a longer timeout). nice 15 so tool code never starves the model server.
    v3: the session prelude adds the per-session memory cap and (with ARC_HELPERS) the helper functions."""
    from arc_harness import executor as EX
    extra = session_prelude() if prelude_extra is None else prelude_extra
    kw = {"prelude_extra": extra} if extra else {}
    if EXEC_STYLE != "plain":
        kw["style"] = EXEC_STYLE
    try:
        return EX.local_factory(timeout=timeout, nice=15, **kw)
    except TypeError:  # older executor without the hardening knobs
        return EX.local_factory(timeout=timeout)


def build_agent_config(AQ, cap):
    """The AgentConfig every tool attempt runs with (hard cap `cap` = min(ARC_TOOLS_CAP, the attempt's budget))."""
    fields = AQ.AgentConfig.__dataclass_fields__
    soft = int(TOOLS_SOFT_FRAC * cap) if TOOLS_SOFT_FRAC > 0 else int(TOOLS_SOFT) if NOTE_CAP else min(TOOLS_SOFT, cap)
    kw = dict(tier=TOOLS_TIER, max_total_output=cap, budget_soft=soft,
              budget_notes=0.01, budget_warn=0.75, verify_submit=False, max_turns=int(TOOLS_MAX_TURNS),
              chunk_tokens=16384, force_submit_tokens=int(FORCE_SUBMIT), tool_timeout=TOOL_TIMEOUT,
              temperature=TEMPERATURE, top_p=0.95, top_k=20)
    if "context" in fields:
        kw["context"] = MAX_MODEL_LEN
    if FORCE_SUBMIT_LAST:
        if "force_submit_last" in fields:
            kw["force_submit_last"] = True
        else:
            log("SELF-TEST: bundled agent has no force_submit_last; ARC_TOOLS_FORCE_SUBMIT_LAST ignored")
    kw.update(agent_options(fields))
    return AQ.AgentConfig(**kw)


def agent_extra(fields):
    """ARC_AGENT_EXTRA (JSON object) restricted to the bundled agent's fields; ({}, []) when unset."""
    if not AGENT_EXTRA:
        return {}, []
    try:
        extra = json.loads(AGENT_EXTRA)
        assert isinstance(extra, dict)
    except Exception:
        return {}, ["ARC_AGENT_EXTRA (not a JSON object)"]
    return {k: v for k, v in extra.items() if k in fields}, [k for k in extra if k not in fields]


def agent_options(fields):
    """v3 AgentConfig options from the settings (only fields the bundled agent knows). The defaults give v2's
    config exactly. record_train_check / record_alternatives are bookkeeping: no model tokens."""
    opt = {}
    if TOOLS_ANSWER_MODE == "program":
        opt.update(answer_mode="program", prompt="lean2prog")  # E019 (qwen-tools-4)
    if TOOLS_PROMPT:
        opt["prompt"] = TOOLS_PROMPT
    if TASK_SUMMARY:
        opt["task_summary"] = True
    if HELPERS:
        opt["helpers"] = True
    if FEEDBACK != "basic" and TOOLS_ANSWER_MODE == "program":
        opt["feedback"] = FEEDBACK
    if TURN_THINK_CAP:
        opt.update(turn_think_cap=int(TURN_THINK_CAP), turn_think_grace=int(TURN_THINK_GRACE))
        if THINK_CALL_RESCUE:
            opt["think_call_rescue"] = True
    if FREE_ATTEMPT2:
        opt["record_alternatives"] = True
    if COMPACT_AT:  # E027 (opt-in)
        opt.update(compact_at=int(COMPACT_AT), compact_keep=int(COMPACT_KEEP), compact_tool_chars=int(COMPACT_TOOL_CHARS))
    if (FREE_ATTEMPT2 or ATTEMPT2 == "gated") and TOOLS_ANSWER_MODE != "program":
        opt["record_train_check"] = True  # train check of the submitted expression (ranking, retry gating)
    if TOOLS_DESC:  # E028: tool descriptions as in the E025 training rollouts
        opt["tool_desc"] = TOOLS_DESC
    if TOOLS_FALLBACK_SOLVE:  # E028: no answer at the end -> the session's solve(test_input)
        opt["solve_fallback"] = opt["fallback_solve"] = True  # the harness field is solve_fallback
    if EFFORT_HEADER:  # E032: E031's system message after the tools block
        opt["effort_header"] = EFFORT_HEADER
    if SOFT_WORDING != "gentle":  # E032: E031's strong wording past the soft target
        opt["soft_wording"] = SOFT_WORDING
    if TURNS_NOTE:  # E032: "[N turns left: ...]" once N or fewer turns remain
        opt["turns_note"] = int(TURNS_NOTE)
    if NOTE_CAP:  # E032: budget notes keep the training scale (e.g. 20k / 60k) under a lower real cap
        opt["budget_note_cap"] = int(NOTE_CAP)
    opt.update(agent_extra(fields)[0])
    return {k: v for k, v in opt.items() if k in fields}


# ---------------------------------------------------------------- scheduler

class Scheduler:
    def __init__(self, tasks, sub, client, force_at, hard_stop, concurrency, attempts=2, tag="run"):
        self.tasks, self.sub, self.client = tasks, sub, client
        self.force_at, self.hard_stop = force_at, hard_stop
        self.concurrency, self.attempts = concurrency, attempts
        self.pending1 = []
        self.pending2 = []
        self.inflight = {}  # job -> [budget, used]
        self.tokens = 0
        self.rate_samples = []  # (time, total tokens)
        self.gen_start = None
        self.records = []
        self.rec_path = os.path.join(LOG_DIR, f"attempts_{tag}.jsonl")
        self.launch_margin = 120  # no new attempts in the last 2 min before forcing
        self.retries = {}
        self.server_down = lambda: False  # set by main(): True while the vLLM server is dead or restarting
        self.cache_stats = []  # periodic server cache/load counters (summary)
        self.priority = {}  # gated attempt 2: job -> selection.retry_priority key
        self.gated = {"eligible": 0, "started": 0, "dropped_time": 0}

    def may_get_work(self):
        """Gated mode: a worker with nothing to do now waits while first attempts can still queue retries."""
        if ATTEMPT2 != "gated" or time.time() >= self.force_at - self.launch_margin:
            return False
        return bool(self.pending1 or self.pending2 or any(j[2] == 0 for j in self.inflight))

    def on_tokens(self, n):
        self.tokens += n
        if "first_token" not in PHASES and n > 0:
            PHASES["first_token"] = round(time.time() - T0)

    def rate(self):
        now = time.time()
        self.rate_samples.append((now, self.tokens))
        self.rate_samples = [s for s in self.rate_samples if now - s[0] < 1800] or [(now, self.tokens)]
        t_old, k_old = self.rate_samples[0]
        if self.gen_start is None or now - self.gen_start < 900 or now - t_old < 300:
            return RATE_PRIOR
        return max(50.0, (self.tokens - k_old) / (now - t_old))

    def capacity_left(self):
        cap = self.rate() * max(0.0, self.force_at - time.time())
        return cap - sum(max(0, b - u) for b, u in self.inflight.values())

    def attempt_speed(self):
        """Generated tokens per second of one attempt (tool time included): median over finished first attempts
        that ran >= 60 s, else the total rate / concurrency."""
        sp = sorted(r["tokens"] / r["elapsed_s"] for r in self.records
                    if r.get("attempt") == 0 and (r.get("elapsed_s") or 0) >= 60 and r.get("tokens"))
        if len(sp) >= 5:
            return sp[len(sp) // 2]
        return self.rate() / max(1, self.concurrency)

    def gated_budget(self):
        """(budget, reason) for a gated second attempt now; budget 0 if it cannot start now."""
        left_s = self.force_at - self.launch_margin - time.time()
        by_time = ATTEMPT2_SAFETY * self.attempt_speed() * left_s
        cap = min(B_MAX2, TOOLS_CAP) if AGENT == "tools" else B_MAX2
        budget = min(cap, self.capacity_left(), by_time)
        if by_time < B_MIN2:
            return 0, "time"
        return (int(budget), "ok") if budget >= B_MIN2 else (0, "capacity")

    def next_job(self):
        """(job, budget) or None. Attempt-1 jobs first; attempt 2 only with spare capacity."""
        if self.pending1:
            job = self.pending1.pop(0)
            share = self.capacity_left() / (len(self.pending1) + 1)
            return job, int(min(B_MAX1, max(B_MIN1, share)))
        if ATTEMPT2 == "off":
            return None
        if ATTEMPT2 == "gated":
            if not self.pending2:
                return None
            budget, why = self.gated_budget()
            if why == "time":  # time only shrinks: no gated attempt can start any more
                log(f"gated attempt 2: no time left for {len(self.pending2)} queued pairs; dropping them")
                self.gated["dropped_time"] += len(self.pending2)
                self.pending2.clear()
                return None
            if not budget:
                return None  # capacity frees up as running attempts finish under budget
            self.pending2.sort(key=lambda j: self.priority.get(j, (9,)))
            self.gated["started"] += 1
            return self.pending2.pop(0), budget
        while self.pending2:
            job = self.pending2.pop(0)
            share = self.capacity_left() / (len(self.pending2) + 1)
            if share < B_MIN2:
                continue
            return job, int(min(B_MAX2, share))
        return None

    async def attempt(self, job, budget):
        if AGENT == "tools":
            return await self.attempt_tools(job, budget)
        return await self.attempt_plain(job, budget)

    async def attempt_tools(self, job, budget):
        """One E012 tool-agent attempt (agent_qwen.solve + LocalExecutor) with deadline forcing.

        DeadlineLLM streams each completion; past force_at it cuts the stream and reports the attempt's budget as
        used, so the agent goes straight to its forced submit_answer (then the session `answer`). The forced call
        (stop=["</parameter>"]) may run until the hard stop."""
        import dataclasses
        from arc_harness import agent_qwen as AQ
        tid, p, a = job
        task = self.tasks[tid]
        cap = int(min(TOOLS_CAP, budget))
        cfg = build_agent_config(AQ, cap)
        sch, client = self, self.client
        stats = {"tokens": 0, "requests": 0, "deadline_cuts": 0, "stream_errors": 0}

        class DeadlineLLM:
            async def complete(self, prompt, max_tokens, stop=None):
                info = {}
                stats["requests"] += 1
                if stop is None and time.time() >= sch.force_at:
                    stats["deadline_cuts"] += 1
                    return AQ.Completion("", cap, 0, "length")
                # forced submit expression: must end well before the hard-stop cancel so that submit_answer and
                # the session-answer fallback still run (review S2)
                dl = (max(sch.hard_stop - 180, min(time.time() + 30, sch.hard_stop - 10)) if stop
                      else sch.force_at)

                def cb(n):
                    stats["tokens"] += n
                    sch.on_tokens(n)
                    if job in sch.inflight:
                        sch.inflight[job][1] = stats["tokens"]
                try:
                    text, ntok, finish = await client.stream(prompt, max_tokens, dl, cb, stop=stop, info=info)
                except Exception as exc:
                    stats["stream_errors"] += 1
                    stats["last_stream_error"] = f"{type(exc).__name__}: {exc}"[:300]
                    if sch.server_down():
                        raise  # attempt is requeued once the server is back
                    # otherwise: end the attempt through the forced submit / session answer path
                    return AQ.Completion("", cap, 0, "length")
                if finish == "deadline":
                    stats["deadline_cuts"] += 1
                    return AQ.Completion(text, ntok + cap, info.get("prompt_tokens", 0), "length")
                return AQ.Completion(text, ntok, info.get("prompt_tokens", 0),
                                     "length" if finish == "length" else "stop")

        self.inflight[job] = [cap, 0]
        rec = {"task_id": tid, "pair": p, "attempt": a, "budget": cap, "agent": "tools"}
        t0 = time.time()
        try:
            transcript, res = await AQ.solve(DeadlineLLM(), robust_executor_factory(TOOL_TIMEOUT), task["train"],
                                             task["test"][p]["input"], cfg)
        except Exception as exc:
            transcript, res = "", {"pred": None, "error": f"{type(exc).__name__}: {exc}"[:400]}
        finally:
            self.inflight.pop(job, None)
        pred = res.get("pred")
        if hasattr(pred, "tolist"):
            pred = pred.tolist()
        rec.update(tokens=stats["tokens"], requests=stats["requests"], deadline_cuts=stats["deadline_cuts"],
                   stream_errors=stats["stream_errors"], last_stream_error=stats.get("last_stream_error"),
                   answered=pred is not None and valid_grid(pred), pred=pred,
                   forced=res.get("answer_source") in ("forced", "session") or stats["deadline_cuts"] > 0,
                   force_reason=("deadline" if stats["deadline_cuts"] else res.get("stop_reason")),
                   elapsed_s=round(time.time() - t0),
                   **{k: res.get(k) for k in ("answer_source", "stop_reason", "turns", "tool_calls", "tool_errors",
                                              "parse_failures", "eos_continues", "chunks", "seconds", "turn_log",
                                              "train_pass", "train_frac", "alternatives", "think_nudges",
                                              "think_forced", "think_call_rescued", "n_submissions",
                                              "harness_version")})
        if res.get("error"):
            rec["error"] = res["error"]
        try:
            with open(os.path.join(LOG_DIR, f"trace_{tid}_{p}_{a}.txt"), "w") as f:
                f.write(transcript)
        except Exception:
            pass
        return rec

    async def attempt_plain(self, job, budget):
        tid, p, a = job
        task = self.tasks[tid]
        raw = render_prompt(build_prompt(task["train"], task["test"][p]["input"]))
        ptok = await self.client.count(raw)
        rec = {"task_id": tid, "pair": p, "attempt": a, "budget": budget, "prompt_tokens": ptok}
        self.inflight[job] = [budget, 0]
        gen, used, conts, streak, forced, force_reason, after_eos, chunks = "", 0, 0, 0, False, None, False, []
        t0 = time.time()
        try:
            while True:
                room = MAX_MODEL_LEN - ptok - used - 64
                if forced:
                    mt, dl = min(FINAL_TOKENS, room), self.hard_stop
                else:
                    mt, dl = min(budget - used, room - FINAL_TOKENS), self.force_at
                if mt <= 0 and not forced:
                    forced, force_reason = True, "budget"
                    gen += FORCE_TEXT
                    continue
                if mt <= 0:
                    break
                text, ntok, finish = await self.client.stream(raw + gen, mt, dl, self.on_tokens)
                gen += text
                used += ntok
                self.inflight[job][1] = used
                chunks.append([ntok, finish])
                if after_eos:
                    streak = streak + 1 if ntok < LOW_YIELD else 0
                if forced:
                    break
                if answer_section(gen) is not None:
                    if finish == "length" and used < budget and room - ntok > 64:
                        continue
                    break
                if finish == "deadline" or used >= budget or finish == "length":
                    forced, force_reason = True, ("deadline" if finish == "deadline" else "budget")
                    gen += FORCE_TEXT
                    continue
                # EOS mid-thought: continue, force after two low-yield continuations in a row
                if streak >= 2:
                    forced, force_reason = True, "low_yield_eos"
                    gen += FORCE_TEXT
                    continue
                conts += 1
                after_eos = True
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            self.inflight.pop(job, None)
        content = answer_section(gen)
        pred = parse(content) if content is not None else None
        rec.update(tokens=used, continuations=conts, forced=forced, force_reason=force_reason, chunks=chunks,
                   elapsed_s=round(time.time() - t0), answered=pred is not None and valid_grid(pred), pred=pred,
                   tail=(content or gen[-300:])[-300:])
        with open(os.path.join(LOG_DIR, f"trace_{tid}_{p}_{a}.txt"), "w") as f:
            f.write(gen)
        return rec

    async def worker(self, wid):
        while time.time() < self.force_at - self.launch_margin:
            nj = self.next_job()
            if nj is None:
                if self.may_get_work():
                    await asyncio.sleep(5)
                    continue
                return
            job, budget = nj
            rec = await self.attempt(job, budget)
            if "error" in rec and not rec["answered"] and self.server_down():
                # server crashed: requeue the job (at most twice) and wait for the watchdog to restart it
                self.retries[job] = self.retries.get(job, 0) + 1
                if self.retries[job] <= 2:
                    (self.pending1 if job[2] == 0 else self.pending2).insert(0, job)
                    log(f"server down; requeued {job}, waiting")
                    while self.server_down() and time.time() < self.force_at - self.launch_margin:
                        await asyncio.sleep(15)
                    continue
            self.records.append(rec)
            with open(self.rec_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            tid, p, a = job
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
                    self.gated["eligible"] += 1
            log(f"done {tid}.{p}#{a} answered={rec['answered']} tok={rec['tokens']} budget={budget} "
                f"forced={rec['forced']}/{rec['force_reason']} "
                f"{rec['elapsed_s']}s {rec.get('error', '')}")

    async def monitor(self):
        last_t, last_k = time.time(), 0
        prev = None
        while True:
            await asyncio.sleep(120)
            now = time.time()
            r = (self.tokens - last_k) / (now - last_t)
            last_t, last_k = now, self.tokens
            cur = await asyncio.to_thread(server_stats)
            if cur is not None:
                cur["_client_prompt"] = getattr(self.client, "prompt_tokens", 0)
            log(f"progress: {len(self.records)} attempts done, in flight {len(self.inflight)}, pending1 "
                f"{len(self.pending1)}, pending2 {len(self.pending2)}"
                + (f" (gated {self.gated}, attempt speed {self.attempt_speed():.1f} tok/s)" if ATTEMPT2 == "gated"
                   else "") + f", tokens {self.tokens}, window {r:.0f} tok/s, "
                f"rate est {self.rate():.0f} tok/s, avg {self.tokens / max(1, now - self.gen_start):.0f} tok/s | "
                + cache_line(cur, prev))
            if cur is not None:
                self.cache_stats.append({"t": round(now - T0), **cur})
                prev = cur

    async def run(self, jobs1):
        self.pending1 = list(jobs1)
        self.gen_start = time.time()
        mon = asyncio.create_task(self.monitor())
        try:
            await asyncio.wait_for(asyncio.gather(*(self.worker(i) for i in range(self.concurrency))),
                                   timeout=max(60, self.hard_stop - time.time()))
        except asyncio.TimeoutError:
            log("hard stop reached; cancelling in-flight attempts")
        mon.cancel()
        el = time.time() - self.gen_start
        log(f"scheduler finished: {len(self.records)} attempts, {self.tokens} tokens in {el / 60:.1f} min "
            f"({self.tokens / max(1, el):.0f} tok/s)")


def order_jobs(tasks):
    """Attempt-1 jobs, shortest prompts first (approximated by character length)."""
    jobs = [(tid, p, 0) for tid, t in tasks.items() for p in range(len(t["test"]))]
    size = {tid: len(json.dumps(t["train"])) + len(json.dumps(t["test"])) for tid, t in tasks.items()}
    return sorted(jobs, key=lambda j: (size[j[0]], j[0], j[1]))


# ---------------------------------------------------------------- benchmark (test runs only)

async def bench_decode(client, prefixes, n_seqs, gen_tokens, tasks, max_chars=None, n_unique=None):
    """Decode throughput at long context: real ARC prompts + real Qwen trace prefixes, n_seqs in parallel.

    n_unique: number of distinct prompts (sequence i uses prompt i % n_unique), so repeats can hit a prefix cache.
    """
    total = {"tok": 0}
    prompts = []
    for i in range(n_seqs):
        pr = prefixes[i % (n_unique or len(prefixes))]
        t = tasks[pr["task_id"]]
        prompts.append(render_prompt(build_prompt(t["train"], t["test"][pr["pair"]]["input"]))
                       + pr["text"][:max_chars])
    ctx = [await client.count(p) for p in prompts]
    log(f"bench: {n_seqs} seqs, context tokens mean {sum(ctx) / len(ctx):.0f} min {min(ctx)} max {max(ctx)}")
    first = {}
    t_start = time.time()

    def cb(n):
        total["tok"] += n

    async def one(i):
        def cbi(n):
            if i not in first:
                first[i] = time.time()
            cb(n)
        return await client.stream(prompts[i], gen_tokens, time.time() + 3600, cbi)

    samples = []

    async def sampler():
        while True:
            await asyncio.sleep(10)
            samples.append((time.time(), total["tok"]))
    s = asyncio.create_task(sampler())
    res = await asyncio.gather(*(one(i) for i in range(n_seqs)), return_exceptions=True)
    s.cancel()
    t_end = time.time()
    t_all_first = max(first.values()) if first else t_end
    dec = [x for x in samples if x[0] >= t_all_first]
    steady = (dec[-1][1] - dec[0][1]) / (dec[-1][0] - dec[0][0]) if len(dec) >= 2 else None
    errs = [repr(r)[:200] for r in res if isinstance(r, Exception)]
    out = {"n_seqs": n_seqs, "ctx_mean": sum(ctx) / len(ctx), "gen_tokens": gen_tokens, "total_tokens": total["tok"],
           "wall_s": t_end - t_start, "prefill_phase_s": t_all_first - t_start, "steady_decode_tok_s": steady,
           "overall_tok_s": total["tok"] / (t_end - t_start), "errors": errs}
    log("bench result:", json.dumps(out))
    return out


def score(sub, solutions, tasks):
    tot, pairs = 0.0, 0
    for tid, t in tasks.items():
        n = len(t["test"])
        ok = 0
        for p in range(n):
            s = sub[tid][p]
            ok += solutions[tid][p] in (s["attempt_1"], s["attempt_2"])
        tot += ok / n
        pairs += n
    return tot, len(tasks)


# ---------------------------------------------------------------- main

def setup_harness():
    """Make the bundled E012 harness importable and self-test the executor; False -> use the plain agent."""
    try:
        sys.path.insert(0, os.path.join(PKG, "evals"))
        from arc_harness import agent_qwen
        from arc_harness.executor import LocalExecutor, submit_answer
        agent_qwen.CONTEXT = MAX_MODEL_LEN  # the server's context, not the model's 262k
        t = time.time()
        with LocalExecutor([{"input": [[1, 2]], "output": [[2, 1]]}], [[3, 4]], timeout=3) as ex:
            r1 = ex.execute("import numpy as np\nx = np.array(test_input)[:, ::-1]\nprint(x.tolist())")
            r2 = ex.execute("print(x.sum())")
            r3 = ex.execute("1/0")
            r4 = ex.execute("while True:\n    pass")
            r5 = ex.execute("print(x.shape)")
            out, grid = submit_answer(ex, "x")
        ok = (r1["stdout"].strip() == "[[4, 3]]" and r2["stdout"].strip() == "7" and "ZeroDivisionError" in r3["stderr"]
              and r4["process_status"] == "timeout" and r5["stdout"].strip() == "(1, 2)" and grid == [[4, 3]])
        log(f"harness {agent_qwen.HARNESS_VERSION}; executor self-test {'OK' if ok else 'FAILED'} in "
            f"{time.time() - t:.1f}s")
        if not ok:
            log("self-test results:", json.dumps([r1, r2, r3, r4, r5, out])[:1500])
            DROPPED.append("tools_agent")
        if ok:
            v3_selftest()
        return ok
    except Exception as exc:
        log(f"harness/executor self-test error: {type(exc).__name__}: {exc}")
        return False


def v3_selftest():
    """Check each enabled v3 feature on CPU before any GPU work; a broken one is switched off (v2 behaviour for
    that feature) and logged loudly. Sets agent_qwen.PROGRAM_MEMORY_MB from the session memory cap."""
    global SESSION_MEM_MB, HELPERS, TASK_SUMMARY, TOOLS_ANSWER_MODE, TOOLS_PROMPT, FREE_ATTEMPT2, TURN_THINK_CAP, \
        THINK_CALL_RESCUE, TOOLS_DESC, TOOLS_FALLBACK_SOLVE, AGENT_EXTRA, EXEC_STYLE, EFFORT_HEADER, TURNS_NOTE, \
        NOTE_CAP
    from arc_harness import agent_qwen as AQ
    from arc_harness import executor as EX
    fields = AQ.AgentConfig.__dataclass_fields__
    train = [{"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]}, {"input": [[2, 3]], "output": [[3, 2]]}]
    test_input = [[4, 5], [6, 7]]
    status = {}

    def run(prelude, cells):
        with EX.LocalExecutor(train, test_input, timeout=10, prelude_extra=prelude) as ex:
            return [ex.execute(c) for c in cells]

    if SESSION_MEM_MB:
        try:
            big = int(SESSION_MEM_MB) + 512
            r = run(session_prelude(SESSION_MEM_MB, False),
                    [f"x = bytearray({big} * 2 ** 20)", "print('alive', len(train))"])
            ok = "MemoryError" in r[0]["stderr"] and r[1]["stdout"].strip() == "alive 2" and not r[1].get("state_reset")
            status["session_mem"] = ok
            if ok:
                AQ.PROGRAM_MEMORY_MB = min(AQ.PROGRAM_MEMORY_MB, int(SESSION_MEM_MB))
            else:
                log("SELF-TEST: session memory cap FAILED; using the executor's 4 GB limit only:", json.dumps(r)[:600])
                SESSION_MEM_MB = 0
        except Exception as exc:
            log(f"SELF-TEST: session memory cap error {exc!r}; off")
            SESSION_MEM_MB, status["session_mem"] = 0, False
    if HELPERS:
        try:
            r = run(session_prelude(SESSION_MEM_MB, True), ["print(d8([[1, 2], [3, 4]])['fliplr'].tolist(), len(objects(train[0]['input'])))"])
            ok = "helpers" in fields and bool(AQ.HELPERS_SRC) and r[0]["stdout"].strip() == "[[2, 1], [4, 3]] 1"
            status["helpers"] = ok
            if not ok:
                log("SELF-TEST: helpers FAILED; off:", json.dumps(r)[:400])
                HELPERS = 0
                DROPPED.append("helpers")
        except Exception as exc:
            log(f"SELF-TEST: helpers error {exc!r}; off")
            HELPERS, status["helpers"] = 0, False
            DROPPED.append("helpers")
    if TOOLS_ANSWER_MODE == "program":
        try:
            code = ("def solve(grid):\n    return [row[::-1] for row in grid]" if not HELPERS else
                    "def solve(grid):\n    return d8(grid)['fliplr'].tolist()")
            with EX.LocalExecutor(train, test_input, timeout=10, prelude_extra=session_prelude()) as ex:
                outs = [EX.run_program(ex, code, g, 10, AQ.PROGRAM_MEMORY_MB, AQ.HELPERS_SRC if HELPERS else "")
                        for g in [pair["input"] for pair in train] + [test_input]]
                hog = EX.run_program(ex, "def solve(grid):\n    x = bytearray(%d * 2 ** 20)\n    return grid"
                                     % (AQ.PROGRAM_MEMORY_MB + 512), test_input, 10, AQ.PROGRAM_MEMORY_MB)
            ok = ("answer_mode" in fields and [o.get("out") for o in outs] == [p["output"] for p in train]
                  + [[[5, 4], [7, 6]]] and hog.get("kind") in ("MemoryError", "crash"))
            status["program"] = ok
            if not ok:
                log("SELF-TEST: program mode FAILED; back to submit_answer:", json.dumps([outs, hog])[:600])
                TOOLS_ANSWER_MODE, TOOLS_PROMPT = "submit", ""
        except Exception as exc:
            log(f"SELF-TEST: program mode error {exc!r}; back to submit_answer")
            TOOLS_ANSWER_MODE, TOOLS_PROMPT, status["program"] = "submit", "", False
    if TOOLS_PROMPT and TOOLS_PROMPT not in getattr(AQ, "PROMPTS", {}):
        log(f"SELF-TEST: unknown prompt {TOOLS_PROMPT!r}; using the mode default")
        DROPPED.append(f"prompt:{TOOLS_PROMPT}")
        TOOLS_PROMPT = ""
    if TOOLS_PROMPT and ("prog" in TOOLS_PROMPT) != (TOOLS_ANSWER_MODE == "program"):
        log(f"SELF-TEST: prompt {TOOLS_PROMPT!r} does not match answer mode {TOOLS_ANSWER_MODE}; mode default")
        DROPPED.append(f"prompt:{TOOLS_PROMPT}")
        TOOLS_PROMPT = ""
    if TASK_SUMMARY:
        try:
            from task_summary import task_summary
            txt = task_summary(train, test_input)
            ok = "task_summary" in fields and isinstance(txt, str) and len(txt) > 50
            status["task_summary"] = ok
            if not ok:
                log("SELF-TEST: task summary FAILED; off")
                TASK_SUMMARY = 0
                DROPPED.append("task_summary")
        except Exception as exc:
            log(f"SELF-TEST: task summary error {exc!r}; off")
            TASK_SUMMARY, status["task_summary"] = 0, False
            DROPPED.append("task_summary")
    if TURN_THINK_CAP and "turn_think_cap" not in fields:
        log("SELF-TEST: bundled agent has no turn_think_cap; off")
        TURN_THINK_CAP, status["think_cap"] = 0, False
    if TURN_THINK_CAP and THINK_CALL_RESCUE and "think_call_rescue" not in fields:
        log("SELF-TEST: bundled agent has no think_call_rescue; off")
        THINK_CALL_RESCUE, status["think_call_rescue"] = 0, False
    if FREE_ATTEMPT2 and "record_alternatives" not in fields:
        log("SELF-TEST: bundled agent has no record_alternatives; free attempt_2 off")
        FREE_ATTEMPT2, status["free_attempt2"] = 0, False
    # E028 options: the field must exist in the bundled agent (agent_options drops unknown fields silently)
    if TOOLS_DESC and "tool_desc" not in fields:
        log(f"SELF-TEST: bundled agent has no tool_desc; ARC_TOOLS_DESC={TOOLS_DESC!r} ignored")
        status["tool_desc"] = False
        DROPPED.append("tool_desc")
    if TOOLS_DESC == "e031" and not hasattr(AQ, "E031_CONFIG"):  # an older agent would take it as the default
        log("SELF-TEST: bundled agent has no e031 mode; ARC_TOOLS_DESC=e031 ignored")
        status["tool_desc"] = False
        DROPPED.append("tool_desc")
        TOOLS_DESC = ""
    # E032 options (e031 mode)
    if EFFORT_HEADER and ("effort_header" not in fields or EFFORT_HEADER not in getattr(AQ, "EFFORT_HEADERS", {})):
        log(f"SELF-TEST: bundled agent has no effort header {EFFORT_HEADER!r}; ARC_TOOLS_EFFORT_HEADER ignored")
        status["effort_header"], EFFORT_HEADER = False, ""
        DROPPED.append("effort_header")
    if TURNS_NOTE and "turns_note" not in fields:
        log("SELF-TEST: bundled agent has no turns_note; ARC_TOOLS_TURNS_NOTE ignored")
        status["turns_note"], TURNS_NOTE = False, 0
        DROPPED.append("turns_note")
    if NOTE_CAP and "budget_note_cap" not in fields:
        log("SELF-TEST: bundled agent has no budget_note_cap; ARC_TOOLS_NOTE_CAP ignored")
        status["budget_note_cap"], NOTE_CAP = False, 0
        DROPPED.append("budget_note_cap")
    if TOOLS_FALLBACK_SOLVE and "solve_fallback" not in fields and "fallback_solve" not in fields:
        log("SELF-TEST: bundled agent has no solve_fallback; ARC_TOOLS_FALLBACK_SOLVE ignored")
        status["fallback_solve"] = False
        DROPPED.append("solve_fallback")
    if EXEC_STYLE != "plain":  # E028: executor text format of the training rollouts
        try:
            with EX.LocalExecutor(train, test_input, timeout=10, prelude_extra=session_prelude(),
                                  style=EXEC_STYLE) as ex:
                r = [ex.execute(c) for c in ("print(len(train))", "1/0", "x = 3\nx + 1")]
            ok = (r[0]["stdout"].strip() == "2" and "ZeroDivisionError" in r[1]["stdout"] + r[1]["stderr"]
                  and "4" in r[2]["stdout"])
            status["exec_style"] = ok
            if not ok:
                log(f"SELF-TEST: executor style {EXEC_STYLE!r} FAILED; plain:", json.dumps(r)[:600])
        except Exception as exc:
            log(f"SELF-TEST: executor style {EXEC_STYLE!r} error {exc!r}; plain")
            ok = status["exec_style"] = False
        if not ok:
            DROPPED.append(f"exec_style:{EXEC_STYLE}")
            EXEC_STYLE = "plain"
    unknown = agent_extra(fields)[1]
    if unknown:
        log(f"SELF-TEST: ARC_AGENT_EXTRA fields unknown to the bundled agent, ignored: {unknown}")
        DROPPED.extend(f"extra:{k}" for k in unknown)
    try:  # the agent config the attempts will use (also checks the field names)
        cfg = build_agent_config(AQ, int(TOOLS_CAP))
        AQ.initial_prompt(train, test_input, cfg)
        PHASES["agent_config"] = AQ.config_dict(cfg, SERVED)  # what a full-budget attempt runs with
    except Exception as exc:
        log(f"SELF-TEST: agent config/prompt error {exc!r}; dropping all v3 agent options")
        DROPPED.append("all_agent_options")
        HELPERS = TASK_SUMMARY = TURN_THINK_CAP = FREE_ATTEMPT2 = TOOLS_FALLBACK_SOLVE = TURNS_NOTE = NOTE_CAP = 0
        EFFORT_HEADER = ""
        TOOLS_ANSWER_MODE, TOOLS_PROMPT, TOOLS_DESC, AGENT_EXTRA = "submit", "", "", ""
    PHASES["v3_selftest"] = status
    log(f"v3 self-test {status}; effective: answer_mode={TOOLS_ANSWER_MODE} prompt={TOOLS_PROMPT or 'default'} "
        f"summary={TASK_SUMMARY} helpers={HELPERS} think_cap={TURN_THINK_CAP}/{TURN_THINK_GRACE} "
        f"rescue={THINK_CALL_RESCUE} tier={TOOLS_TIER} feedback={FEEDBACK} "
        f"free_attempt2={FREE_ATTEMPT2} attempt2={ATTEMPT2} session_mem={SESSION_MEM_MB} MB "
        f"program_mem={AQ.PROGRAM_MEMORY_MB} MB tool_desc={TOOLS_DESC or 'default'} "
        f"fallback_solve={TOOLS_FALLBACK_SOLVE} exec_style={EXEC_STYLE} effort_header={EFFORT_HEADER or '-'} "
        f"turns_note={TURNS_NOTE} note_cap={NOTE_CAP} soft_wording={SOFT_WORDING} extra={AGENT_EXTRA or '-'} dropped={DROPPED}")


def is_rerun(test_path):
    """Belt and braces: the env var, or a test file whose task ids differ from the public one."""
    import hashlib
    env = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))
    try:
        ids = sorted(json.load(open(test_path)))
        differs = hashlib.sha256(",".join(ids).encode()).hexdigest() != PUBLIC_TEST_IDS_SHA256
    except Exception:
        differs = False
    return env or differs, env, differs


def descendants(pid=None):
    """PIDs of all descendants of `pid` (default: this process), from /proc parent links."""
    pid = os.getpid() if pid is None else pid
    children = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
                children.setdefault(ppid, []).append(int(d))
            except (OSError, ValueError, IndexError):
                pass
    out, todo = [], [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


ON_KAGGLE = os.path.isdir("/kaggle/input") and not EXTERNAL


def kill_children():
    """Kill everything this solver started (server, its workers, tool executors), by descendant PIDs. The
    name-based pkill of the server is added only on Kaggle: on a shared host it would hit other users'
    processes whose command line mentions vllm/sglang (E021 incident)."""
    for pid in descendants():
        try:
            os.kill(pid, 9)
        except OSError:
            pass
    if ON_KAGGLE:
        subprocess.run("pkill -9 -f '[V]LLM::'; pkill -9 -f '[v]llm.entrypoints'; pkill -9 -f '[s]glang'",
                       shell=True)


def emergency_timer(at_s=None):
    """Last-resort timer (review S1): at T0 + ARC_EMERGENCY_S (or `at_s`), make sure a valid submission.json exists,
    kill the server and all children, and exit, whatever the asyncio side is doing."""
    import threading

    def fire():
        log("EMERGENCY timer fired: validating submission.json, killing children, exiting")
        try:
            path = os.path.join(WORK, "submission.json")
            tasks = json.load(open(comp_file("arc-agi_test_challenges.json")))
            try:
                check_submission(path, tasks)
            except Exception:
                global OUT_SUB
                OUT_SUB = path
                Submission(tasks).write()
        finally:
            try:
                kill_children()
            finally:
                os._exit(0)
    at_s = envf("ARC_EMERGENCY_S", 11.7 * 3600) if at_s is None else at_s
    t = threading.Timer(max(1.0, T0 + at_s - time.time()), fire)
    t.daemon = True
    t.start()


def main():
    global OUT_SUB
    emergency_timer()
    os.makedirs(LOG_DIR, exist_ok=True)
    driver_env()
    global AGENT
    rerun, env_rerun, ids_differ = is_rerun(comp_file("arc-agi_test_challenges.json"))
    mode = "submit" if rerun else MODE
    log(f"rerun={rerun} (env {env_rerun}, test ids differ from public {ids_differ})")
    if os.environ.get("ARC_INTERNET"):  # set by the notebook's connectivity cell
        PHASES["internet"] = os.environ["ARC_INTERNET"]
        log(f"INTERNET: {os.environ['ARC_INTERNET']}")
    log(f"mode={mode} server={SERVER} agent={AGENT} rerun={rerun} "
        f"T0={time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(T0))} UTC")
    test_tasks = json.load(open(comp_file("arc-agi_test_challenges.json")))
    Submission(test_tasks).write()  # valid fallback for the competition test file, always
    log(f"fallback submission written ({len(test_tasks)} tasks)")
    if mode == "fallback":
        return
    if mode == "local" and LOCAL_WALL_MIN > 0:  # commit-run wall-clock bound (never on the rerun: mode is submit)
        emergency_timer(LOCAL_WALL_MIN * 60)
        log(f"local wall-clock limit: emergency exit at {LOCAL_WALL_MIN:.0f} min")
    if mode == "local":
        # Non-rerun commit/interactive run: small local test set with known solutions, self-scored, time-boxed.
        if LOCAL_SOURCE == "eval":
            tasks = json.load(open(comp_file("arc-agi_evaluation_challenges.json")))
            solutions = json.load(open(comp_file("arc-agi_evaluation_solutions.json")))
        else:
            tasks = json.load(open(find_one(f"{INPUT}/**/arc-agi_local_challenges.json")))
            solutions = json.load(open(find_one(f"{INPUT}/**/arc-agi_local_solutions.json")))
        if LOCAL_TASKS:
            tasks = {t: tasks[t] for t in LOCAL_TASKS.split(",") if t in tasks}
        force_at = hard_stop = None  # set once the server is ready
        OUT_SUB = os.path.join(WORK, "local_submission.json")
    else:
        tasks, solutions = test_tasks, None
        force_at, hard_stop = T0 + FORCE_AT_S, T0 + HARD_STOP_S
    sub = Submission(tasks)
    sub.write()
    log(f"{mode}: {len(tasks)} tasks, {sum(len(t['test']) for t in tasks.values())} test inputs"
        + (f"; force at {(force_at - T0) / 3600:.2f}h, hard stop {(hard_stop - T0) / 3600:.2f}h" if force_at else ""))
    if AGENT == "tools" and not setup_harness():
        AGENT = "plain"
        log("falling back to the plain (no-tool) agent")
    PHASES["harness_ready"] = round(time.time() - T0)

    if EXTERNAL:
        # off-Kaggle validation against an already running server: no install, no launch, no watchdog restarts
        class _Alive:
            returncode = None

            def poll(self):
                return None
        mdir, proc, slog = None, _Alive(), os.path.join(LOG_DIR, "external_server.log")
        open(slog, "a").close()
        log(f"external server {BASE} (model {SERVED}); skipping install and launch")
    else:
        global TP
        gpus = gpu_inventory()
        TP = choose_tp(gpus)
        PHASES["gpus"] = [f"{n} {m}MiB sm{c}" for n, m, c in gpus]
        log(f"GPUs present: {gpus}; tensor parallel {TP}")
        if TP == 0:
            log("NO USABLE GPUS for Qwen3.8-27B-FP8 (need SM89+ and enough memory): keeping the fallback submission")
            return
        if TP != 4:
            log(f"WARNING: running with TP={TP} instead of 4; KV capacity and concurrency shrink accordingly")
        mdir = model_dir()
        log("model dir:", mdir, "tokenizer:", tokenizer_dir() or "(model's own)")
        PHASES["model"] = model_identity(mdir)  # raises (run fails, fallback kept) on the wrong model
        log("MODEL IDENTITY", json.dumps(PHASES["model"]))
        install_server()
        install_fp8_configs()
        PHASES["installed"] = round(time.time() - T0)
        preload(mdir)
        PHASES["preloaded"] = round(time.time() - T0)
        proc, slog = launch_server(mdir)
    if not wait_ready(proc, slog, timeout=1800):
        log("server start failed; killing the process tree and retrying once")
        kill_server(proc)
        proc, slog = launch_server(mdir)
        if not wait_ready(proc, slog, timeout=1800):
            raise RuntimeError("server failed to start (twice)")
    PHASES["server_ready"] = round(time.time() - T0)
    PHASES["server_identity"] = server_identity(slog)
    log("SERVER IDENTITY", json.dumps(PHASES["server_identity"]))
    if mode == "local":
        if LOCAL_READY_MIN > 0:
            force_at = time.time() + LOCAL_READY_MIN * 60
        else:
            force_at = min(T0 + LOCAL_FORCE_CAP_MIN * 60, max(T0 + LOCAL_FORCE_MIN * 60, time.time() + 20 * 60))
        hard_stop = force_at + LOCAL_HARD_MIN * 60
        log(f"local time box: force at {(force_at - T0) / 60:.0f} min, hard stop {(hard_stop - T0) / 60:.0f} min")
    kv, max_running = server_capacity(slog)
    conc = CONCURRENCY or min(MAX_SEQS, max_running or MAX_SEQS,
                              max(8, int(kv / TOKENS_PER_SEQ_KV)) if kv else 16)
    log(f"concurrency {conc} (max_seqs {MAX_SEQS}, server max running {max_running}, kv {kv})")
    server = {"proc": proc, "down": False, "restarts": 0}

    async def watchdog(force_at):
        """Restart the server (at most twice) if it dies while there is time left to use it."""
        while True:
            await asyncio.sleep(15)
            if server["proc"].poll() is None:
                continue
            server["down"] = True
            log(f"server died (rc={server['proc'].returncode}):\n" + server_log_tail(slog, 40))
            if server["restarts"] >= 2 or force_at - time.time() < 45 * 60:
                log("not restarting the server")
                return
            server["restarts"] += 1
            kill_server(server["proc"])
            server["proc"], _ = launch_server(mdir)
            ok = await asyncio.to_thread(wait_ready, server["proc"], slog, 1800)
            log(f"server restart {'ok' if ok else 'FAILED'}")
            if not ok:
                return
            server["down"] = False

    async def cpu_monitor():
        while True:
            await asyncio.sleep(120)
            try:
                la = os.getloadavg()
                log(f"cpu load avg {la[0]:.1f}/{la[1]:.1f}/{la[2]:.1f} on {os.cpu_count()} cpus")
            except Exception:
                pass

    async def go():
        from concurrent.futures import ThreadPoolExecutor
        # agent_qwen runs executor calls via asyncio.to_thread: one thread per concurrent attempt (E012)
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=conc + 8))
        client = Client()
        sch = Scheduler(tasks, sub, client, force_at, hard_stop, conc, attempts=2, tag=mode)
        sch.server_down = lambda: server["down"] or server["proc"].poll() is not None
        wd = asyncio.create_task(watchdog(sch.force_at))
        cm = asyncio.create_task(cpu_monitor())
        await sch.run(order_jobs(tasks))
        wd.cancel()
        cm.cancel()
        sub.write()
        check_submission(OUT_SUB, tasks)
        recs = sch.records
        summary = {"mode": mode, "server": SERVER, "agent": AGENT, "concurrency": conc, "kv_tokens": kv,
                   "phases_s": PHASES, "server_restarts": server["restarts"],
                   "attempts": len(recs), "tokens": sch.tokens,
                   "gen_minutes": round((time.time() - sch.gen_start) / 60, 1),
                   "answered": sum(r["answered"] for r in recs), "errors": sum("error" in r for r in recs),
                   "stream_errors": sum(r.get("stream_errors") or 0 for r in recs),
                   "server_preemption_lines": sum("preempt" in ln.lower() for ln in open(slog, errors="ignore")),
                   "cache_stats": sch.cache_stats[-1] if sch.cache_stats else None,
                   "cache_hit_overall": cache_line(sch.cache_stats[-1], None) if sch.cache_stats else None,
                   "kv_usage_max_pct": max([float(x) for x in re.findall(r"KV cache usage: ([\d.]+)%",
                                                                          open(slog, errors="ignore").read())] or [0]),
                   "tool_calls": sum(r.get("tool_calls") or 0 for r in recs),
                   "tool_errors": sum(r.get("tool_errors") or 0 for r in recs),
                   "parse_failures": sum(r.get("parse_failures") or 0 for r in recs),
                   "answer_sources": {k: sum(r.get("answer_source") == k for r in recs)
                                      for k in ["submit", "forced", "session", "salvage", "parse", "lenient",
                                                "program_pass", "program_best_fail", "session_solve", None]
                                      + sorted({r["answer_source"] for r in recs if isinstance(  # new harness
                                          r.get("answer_source"), str)} - set(KNOWN_SOURCES))},  # sources too
                   "v3": {"answer_mode": TOOLS_ANSWER_MODE, "prompt": TOOLS_PROMPT or "default",
                          "task_summary": TASK_SUMMARY, "helpers": HELPERS, "turn_think_cap": TURN_THINK_CAP,
                          "think_call_rescue": THINK_CALL_RESCUE, "tier": TOOLS_TIER, "feedback": FEEDBACK,
                          "think_rescued": sum(r.get("think_call_rescued") or 0 for r in recs),
                          "free_attempt2": FREE_ATTEMPT2, "attempt2": ATTEMPT2, "session_mem_mb": SESSION_MEM_MB,
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
                                            for k in ("a0:alt", "a1:alt", "a0:", "a1:", "fallback", "dup")}}}
        if solutions:
            final = json.load(open(OUT_SUB))
            sc, n = score(final, solutions, tasks)
            summary["score"] = f"{sc:.2f}/{n} tasks"
            only1 = {t: [{"attempt_1": s["attempt_1"], "attempt_2": s["attempt_1"]} for s in v] for t, v in final.items()}
            summary["score_attempt1_only"] = f"{score(only1, solutions, tasks)[0]:.2f}/{n} tasks"
            summary["slots"] = [{"task": t, "pair": p, "src": sub.sources.get((t, p)),
                                 "correct1": final[t][p]["attempt_1"] == solutions[t][p],
                                 "correct2": final[t][p]["attempt_2"] == solutions[t][p]}
                                for t in tasks for p in range(len(tasks[t]["test"]))]
            summary["per_pair"] = [{"task": r["task_id"], "pair": r["pair"], "attempt": r["attempt"],
                                    "correct": r["pred"] == solutions[r["task_id"]][r["pair"]],
                                    "tokens": r["tokens"], "turns": r.get("turns"), "tool_calls": r.get("tool_calls"),
                                    "tool_errors": r.get("tool_errors"), "source": r.get("answer_source"),
                                    "stop": r.get("stop_reason"), "s": r["elapsed_s"], "error": r.get("error"),
                                    "train_pass": r.get("train_pass"), "n_alt": len(r.get("alternatives") or []),
                                    "alt_correct": any(a.get("pred") == solutions[r["task_id"]][r["pair"]]
                                                       for a in r.get("alternatives") or [] if isinstance(a, dict))}
                                   for r in recs]
        json.dump(summary, open(os.path.join(WORK, f"summary_{mode}.json"), "w"), indent=1)
        log("SUMMARY", json.dumps(summary))
    asyncio.run(go())
    log("server log tail:\n" + server_log_tail(slog, 10))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:  # keep whatever submission.json exists
        import traceback
        traceback.print_exc()
        log(f"FATAL {type(exc).__name__}: {exc}")
    finally:
        try:
            tasks_path = comp_file("arc-agi_test_challenges.json")
            check_submission(os.path.join(WORK, "submission.json"), json.load(open(tasks_path)))
            log("submission.json valid")
        except Exception as exc:
            log(f"submission check failed ({exc}); rewriting fallback")
            OUT_SUB = os.path.join(WORK, "submission.json")
            Submission(json.load(open(comp_file("arc-agi_test_challenges.json")))).write()
        try:
            kill_children()  # executors / server still alive after a hard-stop cancel (descendants only off Kaggle)
        finally:
            os._exit(0)
