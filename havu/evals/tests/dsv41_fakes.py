"""Fakes for the dsv41 Kaggle-path tests of the live harness (ported from the 2026-10-26 deep audit,
kaggle/dsv41_tools/tests_mtf_v6/common.py): a fake token-exact vLLM server, canonical DeepSeek-V4.1 DSML completions,
LocalExecutor sessions as the Kaggle solver opens them, and the solver's harness argv."""
import importlib.util
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

EVALS = Path(__file__).resolve().parents[1]
ROOT = EVALS.parent
sys.path.insert(0, str(EVALS))
SOLVER = ROOT / "kaggle/dsv41_tools/solver_dsv41.py"
KIT_V5 = ROOT / "kaggle/dsv41_tools/kaggle_kit_v5/dsv41_kit.tgz"  # what V5 runs and V6's rerun fallback restores

import run_deepseek_tools as H  # noqa: E402  (live)
from arc_harness.executor import LocalExecutor  # noqa: E402


def load_solver_module(env=None, name="solver_for_tests"):
    """The live solver as a fresh module, with `env` (dict; None = unset) applied while it imports."""
    env = {"ARC_OUT": tempfile.mkdtemp(prefix="solver_out_"), **(env or {})}
    with patch.dict(os.environ, {}, clear=False):
        for k in [k for k in os.environ if k.startswith("ARC_")]:
            del os.environ[k]
        for k, v in env.items():
            if v is not None:
                os.environ[k] = v
        spec = importlib.util.spec_from_file_location(name, SOLVER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


_V3 = {}


def load_v3_harness():
    """The Kaggle kit's harness (1ad5ee24, what V5 runs), extracted from kaggle_kit_v5/dsv41_kit.tgz."""
    if "mod" not in _V3:
        d = Path(tempfile.mkdtemp(prefix="kit_v5_"))
        with tarfile.open(KIT_V5) as tf:
            tf.extractall(d, filter="data")
        spec = importlib.util.spec_from_file_location("rdt_v3", d / "repo/evals/run_deepseek_tools.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _V3["mod"] = mod
    return _V3["mod"]


def local_factory(url, train, test_input, timeout=30, max_output=16000, prelude_extra=""):
    """The solver's session_factory (LocalExecutor, ipython style), with a short timeout for tests."""
    return LocalExecutor(train, test_input, min(timeout, 20), max_output, memory_mb=2048, python=sys.executable,
                         prelude_extra=prelude_extra, style="ipython", cpu_seconds=600)


def open_session(train, test_input, test_inputs=None, max_output=16000):
    extra = (f"\nimport json as _arc_json\ntest_inputs = _arc_json.loads({json.dumps(test_inputs)!r})\n"
             if test_inputs is not None else "")  # the harness's multi prelude (inputs_prelude)
    s = local_factory("local", train, test_input, 20, max_output, prelude_extra=extra)
    s.__enter__()
    return s


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
    `pad`: extra prompt tokens added to every render (to push prompts towards the context limit). A turn may be a
    callable (n_prompt, max_tokens) -> text, or an exception instance (raised by create)."""

    def __init__(self, turns, pad=0, completion_tokens=None, tokenize_fail_after=None):
        self.turns, self.pad = list(turns), pad
        self.requests = []  # (len(prompt ids), max_tokens)
        self.completion_tokens = completion_tokens
        self.tokenize_fail_after, self.tokenizes = tokenize_fail_after, 0
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
        self.tokenizes += 1
        if self.tokenize_fail_after is not None and self.tokenizes > self.tokenize_fail_after:
            import httpx
            raise httpx.HTTPStatusError("Server error '500 Internal Server Error' for url 'http://fake:1/tokenize'",
                                        request=None, response=None)
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
                if isinstance(text, BaseException):
                    raise text
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


_SOLVER = {}


def harness_args(extra=(), max_turns=40):
    """The Kaggle solver's harness argv (live solver_dsv41.HARNESS + EXTRA_BODY) plus `extra`."""
    if "mod" not in _SOLVER:
        _SOLVER["mod"] = load_solver_module({}, name="solver_for_argv")
    solver = _SOLVER["mod"]
    argv = solver.HARNESS + list(extra) + ["--max-turns", str(max_turns), "--base-url", "http://fake:1/v1",
                                           "--extra-body", json.dumps(solver.EXTRA_BODY), "--sandbox-url", "local",
                                           "--attempts", "2"]
    return H.parser().parse_args(argv)


def run_solve(turns, args, task, test_inputs=None, pad=0, completion_tokens=None, tokenize_fail_after=None,
              prior=None):
    srv = FakeTokenServer(turns, pad=pad, completion_tokens=completion_tokens, tokenize_fail_after=tokenize_fail_after)
    with patch("httpx.post", srv.post):
        trace, result = H.solve(srv, args, task, task["test"][0]["input"], session_factory=local_factory,
                                **({"test_inputs": test_inputs} if test_inputs else {}),
                                **({"prior": prior} if prior else {}))
    return srv, trace, result
