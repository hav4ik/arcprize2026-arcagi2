"""Shared fakes for the deep audit of the pinned mtf V6 code. Runtime = Kaggle's: the v3 kit's evals (envB) with the
pinned harness a8a5b665 copied over it, the pinned solver 4b344011 loaded from the repo's pinned_mtf_v6 copy."""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

AUDIT = Path(__file__).resolve().parents[1]
ENVB = AUDIT / "envB"
EVALS = ENVB / "evals"
PINNED = Path("/home/vu/Workspace/ARC-AGI-2/kaggle/dsv41_tools/pinned_mtf_v6")
V3 = AUDIT / "v3"
sys.path.insert(0, str(EVALS))

assert hashlib.sha256((EVALS / "run_deepseek_tools.py").read_bytes()).hexdigest().startswith("a8a5b665")
assert hashlib.sha256((PINNED / "solver_dsv41.py").read_bytes()).hexdigest().startswith("4b344011")
assert hashlib.sha256((EVALS / "train_check.py").read_bytes()).hexdigest().startswith("44c4d210")

import run_deepseek_tools as H  # noqa: E402  (pinned)
from arc_harness.executor import LocalExecutor  # noqa: E402  (v3 kit)


def load_v3_harness():
    spec = importlib.util.spec_from_file_location("rdt_v3", V3 / "repo/evals/run_deepseek_tools.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_solver(env, name=None):
    """Import the pinned solver as a fresh module with `env` (dict; None values = unset) applied at import."""
    base = {"ARC_OUT": tempfile.mkdtemp(prefix="solver_out_"), "ARC_KIT": str(AUDIT / "kitB")}
    saved = {}
    keys = set(base) | set(env) | {k for k in os.environ if k.startswith("ARC_")}
    for k in keys:
        saved[k] = os.environ.get(k)
    try:
        for k in list(os.environ):
            if k.startswith("ARC_"):
                del os.environ[k]
        for k, v in {**base, **env}.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        spec = importlib.util.spec_from_file_location(name or f"solver_{abs(hash(json.dumps(env, sort_keys=True)))}",
                                                      PINNED / "solver_dsv41.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        (Path(base["ARC_OUT"]) / "dsv41_logs" / "traces").mkdir(parents=True, exist_ok=True)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def local_factory(url, train, test_input, timeout=30, max_output=16000, prelude_extra=""):
    """The solver's session_factory (LocalExecutor, ipython style), with a short timeout for tests."""
    return LocalExecutor(train, test_input, min(timeout, 20), max_output, memory_mb=2048, python=sys.executable,
                         prelude_extra=prelude_extra, style="ipython", cpu_seconds=600)


def open_session(train, test_input, test_inputs=None, max_output=16000):
    extra = ""
    if test_inputs is not None:  # exactly the harness's multi prelude
        extra = f"\nimport json as _arc_json\ntest_inputs = _arc_json.loads({json.dumps(test_inputs)!r})\n"
    s = local_factory("local", train, test_input, 20, max_output, prelude_extra=extra)
    s.__enter__()
    return s


# ---------------------------------------------------------------- fake token-exact vLLM server

DS_EOS, DSML = "<｜end▁of▁sentence｜>", "｜DSML｜"


def ds_call_text(code, name="run_python", reasoning="thinking", content=""):
    """A canonical V4.1 thinking-mode completion with one tool call (parse_ds_completion -> clean)."""
    d = DSML
    return (f"{reasoning}</think>{content}\n\n<{d} calls>\n<{d} invoke name=\"{name}\">\n"
            f"<{d} parameter name=\"code\" string=\"true\">{code}</{d} parameter>\n</{d} invoke>\n"
            f"</{d} calls>{DS_EOS}")


def ds_text_answer(content, reasoning="thinking"):
    return f"{reasoning}</think>{content}{DS_EOS}"


class FakeTokenServer:
    """/tokenize: one token per character of a canonical rendering (prefix-stable); /v1/completions: scripted turns.
    `pad`: extra prompt tokens added to every render (to push prompts towards the context limit)."""

    def __init__(self, turns, pad=0, completion_tokens=None):
        self.turns, self.pad = list(turns), pad
        self.requests = []  # (len(prompt ids), max_tokens)
        self.completion_tokens = completion_tokens
        self.base_url = "http://fake:1/v1"

    @staticmethod
    def _render(messages, gen, pad):
        s = "P" * pad
        for m in messages:
            s += f"<{m['role']}>" + (m.get("content") or "") + (m.get("reasoning_content") or "")
            for c in m.get("tool_calls") or []:
                s += c["function"]["name"] + c["function"]["arguments"]
        if gen:
            s += "<assistant><think>"
        return [ord(ch) % 50000 for ch in s]

    def post(self, url, json=None, timeout=None):  # httpx.post replacement
        assert url.endswith("/tokenize"), url
        toks = self._render(json["messages"], json["add_generation_prompt"], self.pad)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"tokens": toks})

    @property
    def completions(self):
        server = self

        class _C:
            @staticmethod
            def create(model, prompt, max_tokens, stream, stream_options, extra_body):
                server.requests.append((len(prompt), max_tokens))
                assert max_tokens >= 1, max_tokens
                if not server.turns:
                    raise AssertionError("no scripted turn left")
                text = server.turns.pop(0)
                if callable(text):
                    text = text(len(prompt), max_tokens)
                finish = "stop"
                ids = [ord(ch) % 50000 for ch in text]
                if len(ids) > max_tokens:  # the server cuts at max_tokens
                    ids, text, finish = ids[:max_tokens], text[:max_tokens], "length"
                n = server.completion_tokens or len(ids)
                ch = SimpleNamespace(text=text, finish_reason=finish, model_extra={"token_ids": ids}, index=0)
                last = SimpleNamespace(choices=[], usage=SimpleNamespace(model_dump=lambda: {
                    "prompt_tokens": len(prompt), "completion_tokens": n, "total_tokens": len(prompt) + n}))
                return _Stream([SimpleNamespace(choices=[ch], usage=None), last])
        return _C()


class _Stream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        pass


def harness_args(extra=(), max_turns=40):
    solver = load_solver({}, name="solver_for_argv")
    argv = solver.HARNESS + list(extra) + ["--max-turns", str(max_turns), "--base-url", "http://fake:1/v1",
                                           "--extra-body", json.dumps(solver.EXTRA_BODY), "--sandbox-url", "local",
                                           "--attempts", "2"]
    return H.parser().parse_args(argv)
