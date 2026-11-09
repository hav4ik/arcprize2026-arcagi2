"""Tests for the E032 e031 mode of the Qwen tool agent (agent_qwen.E031_CONFIG: the E031 DeepSeek interface the E032
SFT model was trained on). No network, no sandbox, no model.

The replay tests run the real agent loop (aq.solve) on recorded E031 rollouts: the recorded DeepSeek turns are the
completions (with DeepSeek's completion-token counts, so the harness writes its own budget and turns notes), and the
recorded tool outputs with their notes stripped are the executor results. The transcript must equal the E032
training sequence (data/sft/e032/train.npz, decoded) character for character, and token for token when the official
tokenizer files are in the HF cache. Fixtures: evals/tests/fixtures/e031_replay.json.gz (4 training rollouts).
The full replay over all 1,900 training rollouts is experiments/E027_l4_throughput/e031mode_replay.py.

    .venv/bin/python -m unittest evals/tests/test_qwen_e031.py
"""
import argparse
import asyncio
import gzip
import hashlib
import json
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arc_harness import agent_qwen as aq  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures/e031_replay.json.gz"
E031_TOOLS = ROOT / "experiments/E032_qwen38_sft_e031/e031_tools.json"
_TURNS_NOTE = re.compile(r"\n\[\d+ turns? left: call submit_answer with your best answer now, or at least set `answer`\.\]$")
_BUDGET_NOTE = re.compile(r"\n\[budget: \d+k output tokens used; aim to finish within \d+k \(hard limit \d+k\)\][^\n]*$")
# Tool messages the harness writes itself (not executor results)
_HARNESS_MADE = ("[skipped: solution already accepted]", "[tool error: Unknown tool:",
                 "[tool error: Expected an object with a string code field]")


def e031_cfg(**kw):
    return aq.AgentConfig(**{**aq.E031_CONFIG, **kw})


def strip_notes(text):
    """A recorded last-tool-output of a turn without the budget / turns notes run_deepseek_tools appended."""
    return _BUDGET_NOTE.sub("", _TURNS_NOTE.sub("", text))


class _ReplayLLM:
    """The recorded assistant turns as raw completions, each costing its recorded DeepSeek completion tokens."""

    def __init__(self, turns):
        self.turns = list(turns)

    async def complete(self, prompt, max_tokens, stop=None):
        if stop or not self.turns:  # a forced submit or a turn past the recording: nothing
            return aq.Completion("", 0, len(prompt) // 3, "stop")
        text, n = self.turns.pop(0)
        return aq.Completion(text, n, len(prompt) // 3, "stop")


class _ReplayExecutor:
    """Executor whose results are the recorded tool outputs, in order (used with patched tool_output/submit_answer)."""

    def __init__(self, outputs, pred):
        self.outputs, self.pred = list(outputs), pred

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        pass

    def _next(self):
        return self.outputs.pop(0) if self.outputs else "[answer not accepted] no value returned"

    def execute(self, code):
        return {"process_status": "completed", "_text": self._next()}

    def submit(self, code):
        text = self._next()
        return text, (self.pred if text.startswith("[answer accepted") else None)


def replay(messages, completion_tokens, train, test_input, pred, cfg=None):
    """Run aq.solve over a recorded E031 conversation (DeepSeek API messages, system message first or not).

    Returns (harness transcript + the recorded trailing tool messages, result). The harness ends an accepted
    attempt at the assistant turn; the training render also holds the trailing "[answer accepted]" message."""
    cfg = cfg or e031_cfg(salvage=False)
    msgs = [m for m in messages if m["role"] != "system"]
    turns, outputs, k = [], [], 0
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in m.get("tool_calls") or []]
            text = aq.render_assistant(m.get("reasoning_content") or "", m.get("content") or "", calls)
            turns.append((text[len(aq.GEN_PROMPT):-len("<|im_end|>\n")], completion_tokens[k]))
            k += 1
        elif m["role"] == "tool":
            last = i + 1 == len(msgs) or msgs[i + 1]["role"] != "tool"
            text = strip_notes(m["content"]) if last else m["content"]
            if not text.startswith(_HARNESS_MADE):
                outputs.append(text)
    trailing = []
    for m in reversed(msgs):
        if m["role"] != "tool":
            break
        trailing.insert(0, m["content"])
    if msgs[-1]["role"] == "tool" and not any(t.startswith("[answer accepted") for t in trailing):
        trailing = []  # an attempt that did not end on an accepted answer: the harness rendered these itself
    with mock.patch.object(aq, "tool_output", lambda result, limit=16000: result["_text"]), \
            mock.patch.object(aq, "submit_answer", lambda session, code: session.submit(code)):
        transcript, res = asyncio.run(aq.solve(_ReplayLLM(turns), lambda tr, ti: _ReplayExecutor(outputs, pred),
                                               train, test_input, cfg))
    return transcript + (aq.render_tool_responses(trailing) if trailing else ""), res


def official_tokenizer():
    """tokenizers.Tokenizer of Qwen/Qwen3.8-27B-FP8 from the local HF cache, or None (no download)."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None
    for name in ("models--Qwen--Qwen3.8-27B-FP8", "models--Qwen--Qwen3.8-27B"):
        for path in sorted((Path.home() / ".cache/huggingface/hub" / name).glob("snapshots/*/tokenizer.json")):
            return Tokenizer.from_file(str(path))
    return None


class E031InterfaceTest(unittest.TestCase):
    def test_tool_specs_match_training(self):
        import run_deepseek_tools as ds
        ref = ds.tools_for(argparse.Namespace(tool_timeout=30, helpers=False, answer_tool=True, mode="grid"))
        self.assertEqual(json.dumps(aq.tool_specs(e031_cfg())), json.dumps(ref))
        if E031_TOOLS.exists():
            self.assertEqual(json.dumps(aq.tool_specs(e031_cfg())), json.dumps(json.loads(E031_TOOLS.read_text())))

    def test_effort_header_and_prompt_match_deepseek(self):
        import run_deepseek_tools as ds
        self.assertEqual(aq.EFFORT_HEADERS, ds.EFFORT_HEADERS)
        tasks = json.loads((ROOT / "data/arc-agi_training_challenges.json").read_text())
        for task_id in list(tasks)[:4]:
            t = tasks[task_id]
            ref = ds.build_prompt(t["train"], t["test"][0]["input"]) + ds.PROMPT_SUFFIXES["lean2sub"]
            self.assertEqual(aq.user_content(t["train"], t["test"][0]["input"], "lean2sub"), ref)
            p = aq.initial_prompt(t["train"], t["test"][0]["input"], e031_cfg())
            self.assertTrue(p.startswith("<|im_start|>system\n" + aq.TIER_SYSTEM["xhigh"] + "\n\n# Tools\n"))
            self.assertIn("</IMPORTANT>\n\n" + ds.EFFORT_HEADERS["minimal"] + "<|im_end|>\n<|im_start|>user\n" + ref
                          + "<|im_end|>\n" + aq.GEN_PROMPT, p)

    def test_notes_match_deepseek(self):
        import run_deepseek_tools as ds
        args = argparse.Namespace(budget_notes=0.01, max_total_output=60000, budget_soft=20000, budget_warn=0.75,
                                  soft_wording="strong", answer_tool=True, turns_note=5, max_turns=40)
        for cap in (60000, 48000, 32000):  # the stated limit stays 60k whatever the real cap
            cfg = e031_cfg(max_total_output=cap)
            for used in range(0, 62000, 37):
                self.assertEqual(aq.budget_note(cfg, used), ds.budget_note(args, used))
        for turn in range(1, 41):
            self.assertEqual(aq.turns_note(e031_cfg(), turn), ds.turns_note(args, turn - 1))
        self.assertIn("[2 turns left:", aq.turns_note(e031_cfg(), 38))

    def test_defaults_unchanged(self):
        d = aq.config_dict(aq.AgentConfig(), "m")
        for key in ("effort_header", "turns_note", "budget_note_cap", "tool_desc"):
            self.assertNotIn(key, d)
        self.assertEqual(d["harness_version"], "qwen-tools-4")
        self.assertEqual(aq.config_dict(e031_cfg(), "m")["harness_version"], "qwen-tools-5")
        tools = aq.tool_specs(aq.AgentConfig())
        self.assertEqual(aq.render_system("xhigh", tools), aq.render_system("xhigh", tools, None))
        self.assertEqual(aq.turns_note(aq.AgentConfig(), 59), "")


class E031LoopTest(unittest.TestCase):
    def test_turns_note_in_loop(self):
        """The last turns get the turns note after the budget note; the attempt ends at max_turns."""
        from test_qwen_tools import TEST, TRAIN, FakeLLM, call
        from arc_harness.executor import local_factory
        llm = FakeLLM([(call("run_python", "print(1)"), "stop")] * 7, tokens=1000)
        _, res = asyncio.run(aq.solve(llm, local_factory(5), TRAIN, TEST, e031_cfg(max_turns=7, salvage=False)))
        self.assertEqual(res["stop_reason"], "max_turns")
        last = [p for p, _, _ in llm.prompts]
        self.assertNotIn("turns left", last[1])
        self.assertIn("(hard limit 60k)]\n[5 turns left: call submit_answer with your best answer now, or at least "
                      "set `answer`.]\n</tool_response>", last[2])
        self.assertIn("[1 turn left:", last[6])


@unittest.skipUnless(FIXTURES.exists(), "no e031 replay fixtures")
class E031ReplayTest(unittest.TestCase):
    """Recorded E031 training rollouts through the agent loop == the E032 training sequences."""

    @classmethod
    def setUpClass(cls):
        cls.fixtures = json.load(gzip.open(FIXTURES))
        cls.tok = official_tokenizer()

    def test_replay_matches_training(self):
        for fx in self.fixtures:
            with self.subTest(trace=fx["trace"]):
                text, res = replay(fx["messages"], fx["completion_tokens"], fx["train"], fx["test_input"], fx["pred"])
                self.assertEqual(res["stop_reason"], "submitted")
                self.assertEqual(text, fx["expected_text"] + "\n")  # training ends at <|im_end|>, no separator
                if self.tok is not None:
                    ids = self.tok.encode(fx["expected_text"], add_special_tokens=False).ids
                    self.assertEqual(hashlib.sha256(json.dumps(ids).encode()).hexdigest(), fx["ids_sha256"])
                    ours = self.tok.encode(text, add_special_tokens=False).ids
                    self.assertEqual(ours[:-1], ids)

    def test_lower_cap_keeps_training_notes(self):
        """A 48k Kaggle cap with the 60k note scale: the transcript is unchanged for attempts under 48k."""
        for fx in self.fixtures:
            if sum(fx["completion_tokens"]) < 47000:
                text, _ = replay(fx["messages"], fx["completion_tokens"], fx["train"], fx["test_input"], fx["pred"],
                                 e031_cfg(salvage=False, max_total_output=48000))
                self.assertEqual(text, fx["expected_text"] + "\n", fx["trace"])


if __name__ == "__main__":
    unittest.main()
