"""Synchronous client for the AIMO/NeMo-Skills persistent IPython sandbox.

Each attempt owns a session and HTTP client. Cells are never retried: a lost
response may represent code that already executed. No generated code runs here.
"""
import json
import warnings
from uuid import uuid4

import httpx


class SandboxSession:
    def __init__(self, url, train, test_input, timeout=30, max_output=16000, transport=None):
        self.session_id = uuid4().hex
        self.timeout, self.max_output = timeout, max_output
        # Host-side copies for submission checks: model code may rebind train/test_input in the session.
        self.train, self.test_input = train, test_input
        self.url, self.transport = url, transport
        self.client = httpx.Client(base_url=url.rstrip("/"), trust_env=False, transport=transport)
        # Never serialize a whole task containing test answers.
        data = json.dumps({"train": train, "test_input": test_input})
        self.prelude = (
            "import json, numpy as np\n"
            f"_arc_data = json.loads({data!r})\n"
            "train, test_input = _arc_data['train'], _arc_data['test_input']\n"
            "del _arc_data\n"
        )
        self.initialized = self.closed = False

    def _execute(self, code):
        response = self.client.post(
            "/execute", headers={"X-Session-ID": self.session_id},
            json={"generated_code": code, "language": "ipython", "timeout": self.timeout,
                  "max_output_characters": self.max_output, "traceback_verbosity": "Plain"},
            # Sessions share 24 single-process workers, so a cell may queue behind another session's cell.
            timeout=2 * self.timeout + 60,
        )
        response.raise_for_status()
        result = response.json()
        if result.get("process_status") not in {"completed", "error", "timeout"}:
            raise RuntimeError("Invalid sandbox execution response")
        return result

    def initialize(self):
        result = self._execute(self.prelude)
        if result["process_status"] != "completed":
            raise RuntimeError(f"Sandbox initialization failed: {result.get('stderr', '')[:500]}")
        self.initialized = True

    def execute(self, code):
        if self.closed:
            raise RuntimeError("Sandbox session is closed")
        if not self.initialized:
            self.initialize()
        result = self._execute(code)
        # The patched server interrupts timed-out cells (keeping state) and reports them as errors.
        if (result["process_status"] == "error"
                and "KeyboardInterrupt" in result.get("stdout", "") + result.get("stderr", "")):
            result["process_status"] = "timeout"
        if result.get("new_session_created"):
            # Restore only task data. Replaying generated cells can repeat side effects.
            self.initialize()
            result["state_reset"] = True
        return result

    def checker(self):
        """A fresh session holding only the task inputs, for running submitted code away from the answers."""
        return SandboxSession(self.url, [{"input": p["input"]} for p in self.train], self.test_input,
                              self.timeout, self.max_output, self.transport)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            response = self.client.delete(f"/sessions/{self.session_id}", timeout=10)
            if response.status_code != 404:
                response.raise_for_status()
        except Exception as exc:
            warnings.warn(f"Could not clean up sandbox session {self.session_id}: {type(exc).__name__}")
        finally:
            self.client.close()

    def __enter__(self):
        try:
            self.initialize()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc):
        self.close()


def tool_output(result, limit=16000):
    text = result.get("stdout", "")
    if result.get("stderr"):
        text += "\n[stderr]\n" + result["stderr"]
    if result.get("state_reset"):
        text = "[Python state lost. train/test_input/np restored; redefine other variables.]\n" + text
    text = f"[{result['process_status']}]\n" + (text or "[no output; use print()]")
    return text if len(text) <= limit else text[:limit] + "\n[output truncated]"


def check_health(url):
    with httpx.Client(trust_env=False) as client:
        response = client.get(url.rstrip("/") + "/health", timeout=10)
        response.raise_for_status()
        if response.json().get("status") != "healthy":
            raise RuntimeError("Sandbox health check failed")
