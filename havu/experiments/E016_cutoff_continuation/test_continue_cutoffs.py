"""Offline tests of the trace -> continuation-messages reconstruction (no API, no sandbox)."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continue_cutoffs as cc  # noqa: E402


class Reconstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = cc.cutoff_records()
        cfg = cls.recs[0]["config"]
        cls.tasks = cc.load_tasks(Path(cfg["challenges"]))
        cls.cfg = cfg

    def test_count(self):
        self.assertEqual(len(self.recs), 24)

    def test_prompt_and_structure(self):
        for rec in self.recs:
            msgs = cc.load_trace(rec)["messages"]
            task = self.tasks[rec["task_id"]]
            prompt = cc.build_prompt(task["train"], task["test"][rec["pair"]]["input"]) + cc.PROMPT_SUFFIXES["lean2sub"]
            self.assertEqual(msgs[0], {"role": "user", "content": prompt})
            n_calls = sum(len(m.get("tool_calls", [])) for m in msgs if m["role"] == "assistant")
            self.assertEqual(n_calls, rec["tool_calls"])
            self.assertEqual(sum(m["role"] == "assistant" for m in msgs), rec["turns"])
            for variant in ("toolnote", "usermsg"):
                out = cc.build_continuation(msgs, variant)
                cc.validate_messages(out)
                if variant == "usermsg":
                    self.assertEqual(out[:-1], msgs)
                    self.assertEqual(out[-1]["role"], "user")
                else:
                    self.assertTrue(cc.TIME_UP in out[-1]["content"])
                    # everything before the modified message is the exact trace prefix
                    k = len(out) - 1
                    self.assertEqual(out[:k], msgs[:k])
                    self.assertTrue(out[-1]["content"].startswith(msgs[k]["content"]) or out[-1]["role"] == "user")
            # assistant turns keep their reasoning_content for the tool loop
            self.assertTrue(all("reasoning_content" in m for m in msgs if m["role"] == "assistant"))

    def test_executed_calls_skip(self):
        rec = next(r for r in self.recs if r["task_id"] == "a47bf94d")
        msgs = cc.load_trace(rec)["messages"]
        calls = cc.executed_calls(msgs)
        self.assertEqual(len(calls), rec["tool_calls"] - 1)  # the truncated turn's call was skipped

    def test_strip_note(self):
        self.assertEqual(cc.strip_note("[completed]\nx\n[budget: 5k output tokens used]"), "[completed]\nx")


if __name__ == "__main__":
    unittest.main()
