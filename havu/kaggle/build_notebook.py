"""Build the Kaggle notebook(s) from kaggle/src/solver.py.

    .venv/bin/python kaggle/build_notebook.py --variant submit   # -> kaggle/nb_submit/  (commit run: fallback only)
    .venv/bin/python kaggle/build_notebook.py --variant bench --mtp 3   # -> kaggle/nb_bench_mtp3/

The notebook writes solver.py to /kaggle/working and runs it in a subprocess. On the competition rerun
(KAGGLE_IS_COMPETITION_RERUN set) it always runs mode "submit"; otherwise it runs the variant's mode
("fallback" for the submit notebook, so its commit run finishes in seconds; "bench" for test notebooks).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAG = ROOT / "kaggle"
USER = "chankhavu"

MODEL = "mikedan7/qwen3-8-27b-fp8-official/PyTorch/hf-fp8/1"
WHEELS = "mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"
BENCH_DATA = f"{USER}/arc-qwen-bench-prefixes"
# Python 3.12 Kaggle GPU image (the wheelhouse is cp312; the current default image is Python 3.13).
# Same image as the public notebook delongyang/arc-agi-2-inference-qwen3-8-27b-fp8-within-4-hours.
DOCKER = "gcr.io/kaggle-images/python@sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9"


def solver_source():
    src = (KAG / "src/solver.py").read_text()
    tmpl = (ROOT / "notes/prompts/arcprize_official_system_prompt.txt").read_text()
    assert "__TEMPLATE__" in src
    assert '"""' not in tmpl and "\\" not in tmpl
    return src.replace("__TEMPLATE__", '"""' + tmpl + '"""', 1)  # readable triple-quoted literal


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


SGL_SOURCE = f"{USER}/sglang-0-5-20-py312-cu130"
SGL_DOCKER = "gcr.io/kaggle-images/python@sha256:e5452ce6268c2e8345cfe5141f31ca7ff47032aca46a7ea532bbb87481281d0c"


def build_sglang_bench():
    """Offline SGLang throughput notebook: our wheel source + verified model + bench prefixes, internet off."""
    solver = solver_source()
    bench = (KAG / "src/bench_sglang.py").read_text()
    compile(solver, "solver.py", "exec")
    compile(bench, "bench_sglang.py", "exec")
    cells = [
        md("# SGLang 0.5.20 throughput test on 4xL4 (offline)\n\nSource: `kaggle/src/bench_sglang.py`."),
        code("import os, time\nos.environ['ARC_T0'] = str(time.time())\nos.environ['ARC_MODE'] = 'bench'"),
        code("open('/kaggle/working/arc_solver/solver.py', 'w').write(" + json.dumps(solver) + ")\n"
             "open('/kaggle/working/bench_sglang.py', 'w').write(" + json.dumps(bench) + ")"),
        code("import subprocess, sys\n"
             "p = subprocess.Popen([sys.executable, '-u', '/kaggle/working/bench_sglang.py'], stdout=subprocess.PIPE,\n"
             "                     stderr=subprocess.STDOUT, text=True, cwd='/kaggle/working')\n"
             "for line in p.stdout:\n    print(line, end='', flush=True)\n"
             "print('exit code', p.wait())"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    slug = "arc2-sglang-bench"
    out = KAG / "nb_sglang_bench"
    out.mkdir(exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "machine_shape": "NvidiaL4",
            "docker_image": SGL_DOCKER, "docker_image_pinning_type": "original",
            "dataset_sources": [BENCH_DATA], "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "kernel_sources": [SGL_SOURCE], "model_sources": [MODEL]}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(out)


VLLM_SOURCE = f"{USER}/vllm-0-23-0-py312-cu130"  # cu128 source has a CUDA-13 vllm wheel
LOCAL_DATA = f"{USER}/arc2-local-testcases"
HARNESS_FILES = ["evals/arc_harness/__init__.py", "evals/arc_harness/agent_qwen.py", "evals/arc_harness/executor.py",
                 "evals/arc_harness/arcprize.py", "evals/arc_harness/grids.py", "evals/arc_harness/prompts.py",
                 "notes/prompts/arcprize_official_system_prompt.txt"]


def harness_bundle():
    """E012 tool harness files, written by the notebook under /kaggle/working/pkg/ (same relative layout, so
    arcprize.py finds the prompt file at ROOT/notes/prompts)."""
    return {f: (ROOT / f).read_text() for f in HARNESS_FILES}


# Solver in its own process group; a guard thread kills the group (server + executors) at T0 + 11.75 h.
GUARD_CELL = """import os, signal, subprocess, sys, threading, time
T0 = float(os.environ['ARC_T0'])
GUARD_AT = T0 + float(os.environ.get('ARC_GUARD_S', 11.75 * 3600))
p = subprocess.Popen([sys.executable, '-u', '/kaggle/working/arc_solver/solver.py'], stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, cwd='/kaggle/working', start_new_session=True)
def descendants(root):
    kids = {}
    for d in os.listdir('/proc'):
        if d.isdigit():
            try:
                kids.setdefault(int(open(f'/proc/{d}/stat').read().rsplit(')', 1)[1].split()[1]), []).append(int(d))
            except Exception:
                pass
    out, todo = [], [root]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out
def guard():
    while p.poll() is None:
        if time.time() > GUARD_AT:
            print('GUARD: wall-clock limit reached; killing the solver and everything it started', flush=True)
            tree = descendants(p.pid)  # server and tool executors run in their own sessions
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception as exc:
                print('GUARD kill failed', exc, flush=True)
            for pid in tree:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
            if os.path.isdir('/kaggle/input'):  # name patterns only on Kaggle, never on a shared host
                subprocess.run("pkill -9 -f 'VLLM::'; pkill -9 -f vllm.entrypoints; pkill -9 -f sglang", shell=True)
            return
        time.sleep(10)
threading.Thread(target=guard, daemon=True).start()
for line in p.stdout:
    print(line, end='', flush=True)
print('solver exit code', p.wait(), 'at', round((time.time() - T0) / 3600, 2), 'h')
"""

# Pure-python final check: valid submission.json for the competition test file (fallback rewrite if needed), then
# on non-rerun runs a gate that fails the commit if the local validation did not work.
FINAL_CELL = """import glob, json, os, shutil
def _valid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and all(isinstance(r, list) and 1 <= len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1 and all(type(v) is int and 0 <= v <= 9 for r in g for v in r))
test_path = (glob.glob('/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json')
             or sorted(p for d in ('', '*/', '*/*/', 'datasets/*/*/')  # fixed depths, never a recursive scan
                       for p in glob.glob(f'/kaggle/input/{d}arc-agi_test_challenges.json')))[0]
tasks = json.load(open(test_path))
try:
    sub = json.load(open('/kaggle/working/submission.json'))
except Exception as exc:
    print('submission.json unreadable:', exc)
    sub = {}
out, repaired = {}, 0
for tid, t in tasks.items():
    slots = sub.get(tid) if isinstance(sub.get(tid), list) else []
    lst = []
    for i, pair in enumerate(t['test']):
        s = slots[i] if i < len(slots) and isinstance(slots[i], dict) else {}
        fb = pair['input'] if _valid(pair['input']) else [[0]]
        slot = {}
        for k in ('attempt_1', 'attempt_2'):
            ok = _valid(s.get(k))
            repaired += not ok
            slot[k] = s[k] if ok else fb
        lst.append(slot)
    out[tid] = lst
json.dump(out, open('/kaggle/working/submission.json', 'w'))
print(len(out), 'tasks;', sum(len(v) for v in out.values()), 'test inputs; slots repaired:', repaired)
shutil.rmtree('/kaggle/working/pkg', ignore_errors=True)
for f in glob.glob('/kaggle/working/*.tmp'):
    os.remove(f)
# Commit-run gate (never on the rerun; the solver also detects the rerun by the test ids and then writes
# summary_submit.json): the local validation must have worked, else this cell fails the commit.
if os.environ.get('ARC_MODE') != 'submit' and not os.path.exists('/kaggle/working/summary_submit.json'):
    problems = []
    try:
        s = json.load(open('/kaggle/working/summary_local.json'))
    except Exception:
        s = None
        problems.append('summary_local.json missing')
    if s is not None:
        if not s.get('answered'):
            problems.append(f"no answered attempt ({s.get('answered')}/{s.get('attempts')})")
        if s.get('agent') == 'tools' and not s.get('tool_calls'):
            problems.append('no tool calls')
        if s.get('agent') != os.environ.get('ARC_AGENT', 'tools'):
            problems.append('tool agent self-test failed (plain fallback in use)')
        if os.environ.get('ARC_STRICT_FEATURES') == '1' and (s.get('v3') or {}).get('dropped'):
            problems.append(f"requested agent options switched off: {s['v3']['dropped']}")
    print('COMMIT GATE:', 'PASS' if not problems else 'FAIL: ' + '; '.join(problems))
    if problems:
        raise SystemExit('commit gate failed: ' + '; '.join(problems))
"""


V2_MODULES = [  # (repo path, path under /kaggle/working/arc_solver)
    ("evals/arc_harness/__init__.py", "evals/arc_harness/__init__.py"),
    ("evals/arc_harness/grids.py", "evals/arc_harness/grids.py"),
    ("evals/arc_harness/arcprize.py", "evals/arc_harness/arcprize.py"),
    ("evals/arc_harness/prompts.py", "evals/arc_harness/prompts.py"),
    ("evals/arc_harness/executor.py", "evals/arc_harness/executor.py"),
    ("evals/arc_harness/agent_qwen.py", "evals/arc_harness/agent_qwen.py"),
]
DEST = "/kaggle/working/arc_solver"

V2_INTRO = """# ARC-AGI-2: Qwen3.8-27B-FP8 + Python tools, offline on 4xL4

Generated by `kaggle/build_notebook.py --variant v2` from the team repo (the repo is the source of truth; do not
edit this notebook by hand).

**What it does**
- **Mode switch.** On the competition rerun (`KAGGLE_IS_COMPETITION_RERUN`, or a test file whose task ids differ
  from the public one) it solves the hidden test set. Otherwise (commit or interactive run) it solves a small local
  test set with known solutions (`chankhavu/arc2-local-testcases`), scores itself, and the last cell fails the run
  if that validation did not work. Both modes write a valid `submission.json` for the competition test file first.
- **Model and server.** Qwen3.8-27B-FP8 (hash-verified mirror) on vLLM 0.23.0 from the user's own offline wheel
  source (`chankhavu/vllm-0-23-0-py312-cu130`): TP=4, FP8 KV cache with the Triton attention backend, prefix caching.
- **Agent.** E012 tool agent (`agent_qwen.solve`): raw `/v1/completions` with the Qwen template, `run_python` and
  `submit_answer` tools in a local child-process executor, budget notes, forced `submit_answer` at the cap.
- **Scheduler.** One attempt per test pair, budget = min(48k, remaining capacity / remaining pairs); a second attempt
  only with spare capacity. Streams are cut at the deadline, which forces the answer.
- **Time limits (from the first cell).** Force answers at 11h00m, hard stop 11h30m, in-solver emergency writer at
  11h42m, notebook guard kills the solver process group at 11h45m; the last cell validates `submission.json` and
  repairs any slot. If installation or server start is slow, per-pair budgets shrink accordingly.
"""


def code_cell(src):
    import nbformat
    return nbformat.v4.new_code_cell(src)


def md_cell(src):
    import nbformat
    return nbformat.v4.new_markdown_cell(src)


V3_MODULES = V2_MODULES[:4] + [
    ("evals/arc_harness/executor.py", "evals/arc_harness/executor.py"),
    ("evals/arc_harness/audit.py", "evals/arc_harness/audit.py"),
    ("evals/arc_harness/selection.py", "evals/arc_harness/selection.py"),
    ("evals/arc_harness/worked_example.txt", "evals/arc_harness/worked_example.txt"),
    ("evals/arc_helpers.py", "evals/arc_helpers.py"),
    ("evals/task_summary.py", "evals/task_summary.py"),
    ("evals/arc_harness/agent_qwen.py", "evals/arc_harness/agent_qwen.py"),
]

V3_INTRO = """# ARC-AGI-2: Qwen3.8-27B-FP8 + Python tools, offline on 4xL4 (v3)

Generated by `kaggle/build_notebook.py --variant v3` from the team repo (the repo is the source of truth; do not
edit this notebook by hand). Experiment notes: `experiments/E021_kaggle_v3/README.md`.

**Same machinery as v2**: mode switch (competition rerun -> hidden test set; otherwise a self-scored local test set
and a commit gate), Qwen3.8-27B-FP8 (hash-verified mirror) on vLLM 0.23.0 from our own offline wheel source, TP=4,
FP8 KV, prefix caching; the tool agent (`agent_qwen.solve`, the same modules as the evals in `evals/arc_harness/`)
with a local child-process executor; deadline forcing; force 11h00m, hard stop 11h30m, emergency writer 11h42m,
guard 11h45m; the last cell validates and repairs `submission.json`.

**v3 options** (section 1, `SETTINGS`; documented in `kaggle/README.md`):
- `ARC_FREE_ATTEMPT2`: attempt_2 from other test grids of the same attempt (failing programs, the session's
  `solve` / `answer`, salvage), ranked by train check (`arc_harness/selection.py`). No model tokens.
- `ARC_ATTEMPT2=gated`: with leftover capacity before the force deadline, a second attempt only for pairs whose
  first answer fails the train check or that have no answer, highest expected value first; never before every
  first attempt has started.
- `ARC_TURN_THINK_CAP` / `ARC_TURN_THINK_GRACE`: per-turn thinking cap with a "test it with code now" nudge.
- `ARC_TOOLS_ANSWER_MODE=program`, `ARC_TASK_SUMMARY`, `ARC_HELPERS`, `ARC_TOOLS_PROMPT`, `ARC_FEEDBACK`.
- Robustness: per-session memory cap (`ARC_SESSION_MEM_MB`, RLIMIT_AS = VmSize + 2 GB; programs 2 GB) and
  concurrency = KV tokens / p90 context (`ARC_KV_PER_SEQ` 48000 -> 29 on vLLM 0.23's 1.40M KV).
"""


def build_v3(env):
    env = dict(env)
    env.setdefault("_slug", "arc2-qwen38-tools-submit")
    env.setdefault("_outdir", "nb_v3")
    return build_v2(env, modules=V3_MODULES, intro=V3_INTRO)


# Connectivity check (v5_sft): the notebook must run offline; logs INTERNET: OFF, or flags ON and continues.
NET_CELL = """import os, socket, threading
def _reach(host, port, out):
    try:
        socket.create_connection((host, port), timeout=3).close()
        out[f'{host}:{port}'] = True
    except Exception as exc:
        out[f'{host}:{port}'] = type(exc).__name__
_net = {}
_ts = [threading.Thread(target=_reach, args=(h, p, _net), daemon=True) for h, p in [('8.8.8.8', 53), ('huggingface.co', 443)]]
for _t in _ts:
    _t.start()
for _t in _ts:
    _t.join(5)  # DNS may not honour the socket timeout
_on = any(v is True for v in _net.values())
os.environ['ARC_INTERNET'] = 'ON (UNEXPECTED)' if _on else 'OFF'
print('INTERNET: ON (UNEXPECTED)' if _on else 'INTERNET: OFF', _net)
"""


def build_v2(env, modules=None, intro=None, net_check=False):
    """Readable submission notebook: %%writefile cells with the real module sources, then run + validate cells."""
    import nbformat
    env = dict(env)
    modules = modules or V2_MODULES
    slug = env.pop("_slug", "arc2-qwen38-tools-submit")
    outdir = env.pop("_outdir", "nb_v2")
    models = env.pop("_models", [MODEL])
    solver = solver_source()
    compile(solver, "solver.py", "exec")
    tmpl = (ROOT / "notes/prompts/arcprize_official_system_prompt.txt").read_text()
    cells = [
        md_cell(intro or V2_INTRO),
        md_cell("## 1. Configuration\n\nThe wall clock for every deadline starts here (`ARC_T0`)."),
        code_cell("import os, time\n"
                  "os.environ['ARC_T0'] = str(time.time())  # deadline clock starts at the first cell\n"
                  "os.environ['ARC_MODE'] = 'submit' if os.environ.get('KAGGLE_IS_COMPETITION_RERUN') else 'local'\n"
                  "SETTINGS = " + json.dumps(env, indent=1) + "\n"
                  "for k, v in SETTINGS.items():\n    os.environ.setdefault(k, str(v))\n"
                  f"os.makedirs('{DEST}/evals/arc_harness', exist_ok=True)\n"
                  f"os.makedirs('{DEST}/notes/prompts', exist_ok=True)\n"
                  "print('mode', os.environ['ARC_MODE'], {k: v for k, v in os.environ.items() if k.startswith('ARC_')})"),
        md_cell("## 2. Modules\n\nThe E012 tool harness (`evals/arc_harness/`), the ARC Prize prompt, and the solver "
                "(server install/launch, scheduler, deadlines, submission writer) are written to "
                f"`{DEST}/` with their repo layout."),
    ]
    if net_check:  # right after the configuration cell
        cells.insert(3, code_cell(NET_CELL))
    for repo_path, dest in modules:
        cells.append(code_cell(f"%%writefile {DEST}/{dest}\n" + (ROOT / repo_path).read_text()))
    cells.append(md_cell("The ARC Prize official prompt (exact bytes, no trailing newline, so it is "
                         "written from Python rather than with `%%writefile`)."))
    cells.append(code_cell("ARC_PROMPT = " + '"""' + tmpl + '"""' + "\n"
                           f"open('{DEST}/notes/prompts/arcprize_official_system_prompt.txt', 'w').write(ARC_PROMPT)"))
    cells.append(md_cell("### Solver\n\nPhases: fallback `submission.json` -> executor self-test -> offline install "
                         "of vLLM from the attached wheel source (uv, pip fallback; versions logged) -> model preload -> "
                         "vLLM server (retried once) -> scheduler with the tool agent -> summary. A watchdog restarts "
                         "a dead server; an emergency timer guarantees a valid file."))
    cells.append(code_cell(f"%%writefile {DEST}/solver.py\n" + solver))
    cells.append(md_cell("## 3. Import check\n\nEvery written module must import before any GPU work starts."))
    cells.append(code_cell("import importlib, sys\n"
                           f"sys.path[:0] = ['{DEST}', '{DEST}/evals']\n"
                           "for m in " + repr([m[1][len("evals/"):-3].replace("/", ".") for m in modules
                                                if m[1].endswith(".py") and not m[1].endswith("__init__.py")]
                                               + ["solver"]) + ":\n"
                           "    mod = importlib.import_module(m)\n"
                           "    print('ok', m, getattr(mod, 'HARNESS_VERSION', ''))"))
    cells.append(md_cell("## 4. Run\n\nThe solver runs in its own process group. A guard thread kills that group "
                         "(server and tool children included) at T0 + 11h45m."))
    cells.append(code_cell(GUARD_CELL))
    cells.append(md_cell("## 5. Validate `submission.json` (and gate the commit run)\n\nRepairs any invalid slot "
                         "with the fallback. On non-rerun runs it fails the run if the local validation did not work."))
    cells.append(code_cell(FINAL_CELL))
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}})
    nbformat.validate(nb)
    out = KAG / outdir
    out.mkdir(exist_ok=True)
    nbformat.write(nb, str(out / f"{slug}.ipynb"))
    server = env.get("ARC_SERVER", "vllm")
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "machine_shape": "NvidiaL4",
            "docker_image": SGL_DOCKER, "docker_image_pinning_type": "original",
            "dataset_sources": [LOCAL_DATA], "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "kernel_sources": [SGL_SOURCE if server == "sglang" else VLLM_SOURCE], "model_sources": models}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(out)


# E028: LoRA-SFT Qwen3.8-27B (ModelOpt static FP8), private Kaggle model; the official FP8 mirror stays attached for
# its tokenizer (vLLM 0.23 / transformers 4.57 cannot load the SFT checkpoint's Qwen3_5Tokenizer config).
SFT_MODEL = f"{USER}/qwen3-8-27b-e028-sft/Transformers/fp8/1"  # framework casing as in the model URL
V5_SFT_MODULES = V3_MODULES
V5_SFT_INTRO = """# ARC-AGI-2: Qwen3.8-27B E028 LoRA-SFT (FP8) + Python tools, offline on 4xL4 (v5_sft)

Generated by `kaggle/build_notebook.py --variant v5_sft` from the team repo (the repo is the source of truth; do not
edit this notebook by hand). Settings: `experiments/E028_qwen38_sft/kaggle_v5_sft_settings.json`; notes:
`experiments/E028_qwen38_sft/KAGGLE_V5_SFT.md`.

**The v3/v4 machinery with a different model and the E028 training interface**:
- Model: `chankhavu/qwen3-8-27b-e028-sft` (E028 LoRA-SFT merged, ModelOpt static FP8) on vLLM 0.23.0 from our own
  offline wheel source; TP=4, FP8 KV, prefix caching; tokenizer from the official Qwen3.8-27B-FP8 mirror (identical
  chat template and token ids).
- Agent: the eval modules in `evals/arc_harness/` with the interface the model was trained on (E025 rollouts):
  `lean2roles` prompt, E025 tool descriptions, xhigh, task summary, helpers, `submit_answer`, budget notes.
- Same deadlines (force 11h00m, hard stop 11h30m, emergency writer 11h42m, guard 11h45m), mode switch (rerun ->
  hidden test set; otherwise a self-scored local test set and a commit gate that also fails if a requested agent
  option was switched off by the self-test), and `submission.json` repair cell.
"""


def build_v5_sft(env):
    env = dict(env)
    env.setdefault("_slug", "arc2-qwen38-e028-sft-submit")
    env.setdefault("_outdir", "nb_v5_sft")
    env.setdefault("_models", [SFT_MODEL, MODEL])
    return build_v2(env, modules=V5_SFT_MODULES, intro=V5_SFT_INTRO, net_check=True)


# E032: LoRA-SFT Qwen3.8-27B trained on E031 rollouts (block FP8, the official FP8 layout), run with the e031 mode
E032_MODEL = f"{USER}/qwen3-8-27b-e032-sft/transformers/fp8block/1"
V5_E032_INTRO = """# ARC-AGI-2: Qwen3.8-27B E032 LoRA-SFT (block FP8) + Python tools, offline on 4xL4 (v5_e032)

Generated by `kaggle/build_notebook.py --variant v5_e032` from the team repo (the repo is the source of truth; do not
edit this notebook by hand). Settings: `experiments/E027_l4_throughput/e032_kaggle_settings.json` (its local window
is for the 4-hour L4 box run; the notebook build sets the commit window).

**The v5_sft machinery with the E032 model and the E031 training interface** (`agent_qwen.E031_CONFIG`):
- Model: `chankhavu/qwen3-8-27b-e032-sft` (E032 LoRA-SFT merged, block FP8) on vLLM 0.23.0 from our own offline
  wheel source; TP=4, FP8 KV, prefix caching; tokenizer from the official Qwen3.8-27B-FP8 mirror.
- Agent: the interface the model was trained on (E031 DeepSeek rollouts, token-identical replay): xhigh, E031 tool
  texts, the "Reasoning Effort: Minimal" system message after the tools, `lean2sub` prompt (no helpers, no task
  summary), `submit_answer`, budget notes at the training scale (20k target, 60k limit, strong wording), the turns
  note 5 turns before the 40-turn limit, IPython-style tool output.
- Same deadlines, mode switch, commit gate and `submission.json` repair cell as v5_sft.
"""


def build_v5_e032(env):
    env = dict(env)
    env.setdefault("_slug", "arc2-qwen38-e032-sft-submit")
    env.setdefault("_outdir", "nb_v5_e032")
    env.setdefault("_models", [E032_MODEL, MODEL])
    return build_v2(env, modules=V5_SFT_MODULES, intro=V5_E032_INTRO, net_check=True)


REPLAY_DATA = f"{USER}/arc2-agentic-replay"


def build_replay(env):
    """E014 agentic replay benchmark notebook (offline; our wheel source + verified model + replay dataset).

    env: ARC_SERVER, ARC_REPLAY_PLAN (JSON list), ARC_BENCH_STOP_MIN, optional _slug / _outdir."""
    import nbformat
    env = dict(env)
    server = env.get("ARC_SERVER", "sglang")
    slug = env.pop("_slug", f"arc2-e014-replay-{server}")
    outdir = env.pop("_outdir", f"nb_replay_{server}")
    solver = solver_source()
    bench = (KAG / "src/replay_bench.py").read_text()
    compile(solver, "solver.py", "exec")
    compile(bench, "replay_bench.py", "exec")
    stop_min = float(env.get("ARC_BENCH_STOP_MIN", 60))
    cells = [
        md_cell(f"# E014 agentic replay benchmark: {server} on 4xL4 (offline)\n\nGenerated by "
                "`kaggle/build_notebook.py --variant replay` from the team repo (`kaggle/src/replay_bench.py`, "
                "`kaggle/src/solver.py`). Replays recorded E012 tool-use transcripts turn by turn and measures "
                "throughput, prefix-cache hit rate and evictions at several concurrency levels. Internet off; "
                f"time-boxed to {stop_min:.0f} min from the first cell."),
        code_cell("import json, os, time\nos.environ['ARC_T0'] = str(time.time())\nos.environ['ARC_MODE'] = 'bench'\n"
                  "SETTINGS = " + json.dumps(env, indent=1) + "\n"
                  "for k, v in SETTINGS.items():\n    os.environ[k] = v if isinstance(v, str) else json.dumps(v)\n"
                  f"os.makedirs('{DEST}', exist_ok=True)\n"
                  "print({k: v for k, v in os.environ.items() if k.startswith('ARC_')})"),
        code_cell(f"%%writefile {DEST}/solver.py\n" + solver),
        code_cell(f"%%writefile {DEST}/replay_bench.py\n" + bench),
        code_cell("import os, signal, subprocess, sys, threading, time\n"
                  f"KILL_AT = float(os.environ['ARC_T0']) + {stop_min * 60 + 240:.0f}  # outer guard: time box + 4 min\n"
                  f"p = subprocess.Popen([sys.executable, '-u', '{DEST}/replay_bench.py'], stdout=subprocess.PIPE,\n"
                  "                     stderr=subprocess.STDOUT, text=True, cwd='/kaggle/working', start_new_session=True)\n"
                  "def guard():\n"
                  "    while p.poll() is None and time.time() < KILL_AT:\n        time.sleep(10)\n"
                  "    if p.poll() is None:\n"
                  "        print('OUTER GUARD: killing the bench process group', flush=True)\n"
                  "        try:\n            os.killpg(p.pid, signal.SIGKILL)\n        except Exception:\n            pass\n"
                  "        os.system(\"pkill -9 -f '[s]glang' ; pkill -9 -f '[v]llm' ; pkill -9 -f '[V]LLM::'\")\n"
                  "threading.Thread(target=guard, daemon=True).start()\n"
                  "for line in p.stdout:\n    print(line, end='', flush=True)\n"
                  "print('bench exit code', p.wait())"),
        code_cell("import json\nr = json.load(open('/kaggle/working/replay_bench.json'))\n"
                  "print(json.dumps({k: {'starts': v.get('starts'), 'capacity': v.get('capacity'), 'levels': v.get('levels')}\n"
                  "                  for k, v in r['configs'].items()}, indent=1)[:20000])\nprint(r.get('phases'), r.get('fatal'))"),
    ]
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}})
    nbformat.validate(nb)
    out = KAG / outdir
    out.mkdir(exist_ok=True)
    nbformat.write(nb, str(out / f"{slug}.ipynb"))
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "machine_shape": "NvidiaL4",
            "docker_image": SGL_DOCKER, "docker_image_pinning_type": "original",
            "dataset_sources": [REPLAY_DATA], "competition_sources": [],
            "kernel_sources": [SGL_SOURCE if server == "sglang" else VLLM_SOURCE], "model_sources": [MODEL]}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(out)


def build(variant, env):
    if variant == "replay":
        return build_replay(env)
    if variant == "sglang_bench":
        return build_sglang_bench()
    if variant == "v2":
        return build_v2(env)
    if variant == "v3":
        return build_v3(env)
    if variant == "v5_sft":
        return build_v5_sft(env)
    if variant == "v5_e032":
        return build_v5_e032(env)
    solver = solver_source()
    compile(solver, "solver.py", "exec")
    default_mode = "fallback" if variant == "submit" else "bench"
    cells = [
        md("# ARC-AGI-2: Qwen3.8-27B-FP8 on vLLM (4xL4), no tools\n\n"
           "Source: `kaggle/src/solver.py` in the team repo. On the competition rerun it solves the hidden test "
           "set with a deadline-aware scheduler; interactive/commit runs only write a fallback submission "
           "(or run the benchmark in test variants)."),
        code("import os, time\nos.environ['ARC_T0'] = str(time.time())\n"
             f"DEFAULT_MODE = {default_mode!r}\n"
             "os.environ['ARC_MODE'] = 'submit' if os.environ.get('KAGGLE_IS_COMPETITION_RERUN') else DEFAULT_MODE\n"
             f"for k, v in {env!r}.items():\n    os.environ.setdefault(k, str(v))\n"
             "print('mode', os.environ['ARC_MODE'], {k: v for k, v in os.environ.items() if k.startswith('ARC_')})"),
        code("SOLVER = " + json.dumps(solver) + "\n"
             "open('/kaggle/working/arc_solver/solver.py', 'w').write(SOLVER)\nprint(len(SOLVER), 'chars written')"),
        code("import subprocess, sys\n"
             "p = subprocess.Popen([sys.executable, '-u', '/kaggle/working/arc_solver/solver.py'], stdout=subprocess.PIPE,\n"
             "                     stderr=subprocess.STDOUT, text=True, cwd='/kaggle/working')\n"
             "for line in p.stdout:\n    print(line, end='', flush=True)\n"
             "print('solver exit code', p.wait())"),
        code("import json, os\nsub = json.load(open('/kaggle/working/submission.json'))\n"
             "print(len(sub), 'tasks;', sum(len(v) for v in sub.values()), 'test inputs')\n"
             "for f in ['solver.py', 'submission.json.tmp']:\n"
             "    if os.path.exists('/kaggle/working/' + f) and f.endswith('.tmp'):\n"
             "        os.remove('/kaggle/working/' + f)"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    name = f"nb_{variant}" + (f"_{env.get('ARC_TAG')}" if env.get("ARC_TAG") else "")
    out = KAG / name
    out.mkdir(exist_ok=True)
    slug = {"submit": "arc2-qwen38-27b-submit", "bench": "arc2-qwen38-27b-bench"}[variant]
    if env.get("ARC_TAG"):
        slug += "-" + env["ARC_TAG"]
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"{USER}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "machine_shape": "NvidiaL4",
            "docker_image": DOCKER, "docker_image_pinning_type": "original",
            "dataset_sources": [WHEELS] + ([BENCH_DATA] if variant == "bench" else []),
            "competition_sources": ["arc-prize-2026-arc-agi-2"], "kernel_sources": [],
            "model_sources": [MODEL]}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["submit", "bench", "sglang_bench", "v2", "v3", "v5_sft", "v5_e032", "replay"],
                    required=True)
    ap.add_argument("--mtp", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--env", default="{}", help="extra ARC_* settings as JSON")
    a = ap.parse_args()
    env = {**({"ARC_MTP": a.mtp} if a.variant not in ("v2", "v3", "v5_sft", "v5_e032", "replay")
                                     else {}), **json.loads(a.env)}
    if a.tag:
        env["ARC_TAG"] = a.tag
    build(a.variant, env)
