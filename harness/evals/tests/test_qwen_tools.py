"""Tests for the Qwen tool agent (evals/arc_harness/agent_qwen.py) and LocalExecutor (no network, no sandbox).

    .venv/bin/python -m unittest evals/tests/test_qwen_tools.py
"""
import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arc_harness import agent_qwen as aq  # noqa: E402
from arc_harness.executor import LocalExecutor, check_answer_on_train, local_factory, submit_answer  # noqa: E402

TRAIN = [{"input": [[1, 2]], "output": [[2, 1]]}, {"input": [[3, 4, 5]], "output": [[5, 4, 3]]}]
TEST = [[7, 8, 9]]
ANSWER = [[9, 8, 7]]


def call(name, code, pre="I will run code."):
    return (f"{pre}\n</think>\n\n<tool_call>\n<function={name}>\n<parameter=code>\n{code}\n</parameter>\n"
            "</function>\n</tool_call>")


class FakeLLM:
    """Scripted completions: each item is (text, finish) or a callable(prompt, max_tokens, stop) -> that."""

    def __init__(self, script, tokens=None):
        self.script, self.prompts, self.tokens = list(script), [], tokens

    async def complete(self, prompt, max_tokens, stop=None):
        self.prompts.append((prompt, max_tokens, stop))
        item = self.script.pop(0)
        text, finish = item(prompt, max_tokens, stop) if callable(item) else item
        n = self.tokens if self.tokens is not None else max(1, len(text) // 4)
        return aq.Completion(text, min(n, max_tokens), len(prompt) // 4, finish)


def run(llm, cfg=None):
    cfg = cfg or aq.AgentConfig(tool_timeout=5)
    return asyncio.run(aq.solve(llm, local_factory(5), TRAIN, TEST, cfg))


class ParseTest(unittest.TestCase):
    def test_tool_call(self):
        _, content = aq.split_turn(call("run_python", "x = 1\nprint(x)"))
        calls, errors = aq.parse_tool_calls(content)
        self.assertEqual(calls, [("run_python", {"code": "x = 1\nprint(x)"})])
        self.assertEqual(errors, [])

    def test_two_calls_and_malformed(self):
        content = ("<tool_call>\n<function=run_python>\n<parameter=code>\na\n</parameter>\n</function>\n</tool_call>\n"
                   "<tool_call>\n<function=submit_answer>\n<parameter=code>\nb\n</parameter>\n</function>\n</tool_call>")
        self.assertEqual(aq.parse_tool_calls(content)[0], [("run_python", {"code": "a"}), ("submit_answer", {"code": "b"})])
        self.assertEqual(len(aq.parse_tool_calls("<tool_call>\n<function=run_python>\n<parameter=code>\nx")[1]), 1)
        self.assertEqual(len(aq.parse_tool_calls("<tool_call>\nrun_python(x)\n</tool_call>")[1]), 1)
        self.assertEqual(aq.parse_tool_calls("The answer is [[1]]."), ([], []))

    def test_split_turn(self):
        self.assertEqual(aq.split_turn("thinking"), ("thinking", None))
        self.assertEqual(aq.split_turn("a</think>b"), ("a", "b"))

    def test_lenient(self):
        self.assertEqual(aq.lenient_parse("x [[1, 2],\n [3, 4],\n] y"), [[1, 2], [3, 4]])
        self.assertEqual(aq.lenient_parse("[1, 2]\n[3, 4]\n"), [[1, 2], [3, 4]])
        self.assertIsNone(aq.lenient_parse("none"))


class RenderTest(unittest.TestCase):
    def test_initial_prompt(self):
        p = aq.initial_prompt(TRAIN, TEST, aq.AgentConfig(tier="medium"))
        self.assertTrue(p.startswith("<|im_start|>system\n# Tools\n\nYou have access"))
        self.assertTrue(p.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n"))
        self.assertIn('"name": "submit_answer"', p)
        x = aq.initial_prompt(TRAIN, TEST, aq.AgentConfig(tier="xhigh"))
        self.assertTrue(x.startswith("<|im_start|>system\nReasoning effort is set to xhigh."))

    def test_tool_responses(self):
        self.assertEqual(aq.render_tool_responses(["a", " b \n"]),
                         "<|im_start|>user\n<tool_response>\na\n</tool_response>\n<tool_response>\nb\n</tool_response>"
                         "<|im_end|>\n")

    def test_verbatim_turn_matches_template(self):
        """A generated turn appended verbatim equals the template's rendering of the parsed turn."""
        gen = call("run_python", "print(1)", pre="Inspect.")
        reasoning, content = aq.split_turn(gen)
        calls, _ = aq.parse_tool_calls(content)
        self.assertEqual("<|im_start|>assistant\n<think>\n" + gen + "<|im_end|>\n",
                         aq.render_assistant(reasoning, content.split("<tool_call>")[0], calls))

    def test_budget_note(self):
        cfg = aq.AgentConfig(max_total_output=80000, budget_soft=40000)
        self.assertIn("aim to finish within 40k (hard limit 80k)]", aq.budget_note(cfg, 10000))
        self.assertNotIn("Past the target", aq.budget_note(cfg, 10000))
        self.assertIn("Past the target", aq.budget_note(cfg, 45000))
        self.assertIn("Budget nearly spent", aq.budget_note(cfg, 61000))
        self.assertEqual(aq.budget_note(aq.AgentConfig(budget_notes=0), 50000), "")
        strong = aq.AgentConfig(max_total_output=80000, budget_soft=40000, soft_wording="strong")
        self.assertIn("submit your best answer now", aq.budget_note(strong, 45000))
        self.assertNotIn("soft_wording", aq.config_dict(cfg, "m"))


class ExecutorTest(unittest.TestCase):
    def test_state_errors_timeout(self):
        with LocalExecutor(TRAIN, TEST, timeout=2) as ex:
            self.assertEqual(ex.execute("x = len(train)\nx + 1")["stdout"], "3\n")
            r = ex.execute("1/0")
            self.assertEqual(r["process_status"], "error")
            self.assertIn("ZeroDivisionError", r["stderr"])
            self.assertEqual(ex.execute("while True: pass")["process_status"], "timeout")
            self.assertEqual(ex.execute("print(x, test_input)")["stdout"], "2 [[7, 8, 9]]\n")
            r = ex.execute("import os; os._exit(3)")
            self.assertTrue(r.get("state_reset"))
            self.assertEqual(ex.execute("print(train[0]['output'])")["stdout"], "[[2, 1]]\n")

    def test_answer_helpers(self):
        with LocalExecutor(TRAIN, TEST) as ex:
            ex.execute("def solve(g): return [r[::-1] for r in g]\ndef bad(g): return g")
            self.assertEqual(submit_answer(ex, "solve(test_input)")[1], ANSWER)
            self.assertEqual(check_answer_on_train(ex, "solve(test_input)"), ([], ""))
            self.assertEqual(check_answer_on_train(ex, "bad(test_input)")[0], [0, 1])
            self.assertIsNone(submit_answer(ex, "[[1, 2], [3]]")[1])


class AgentTest(unittest.TestCase):
    def test_run_then_submit(self):
        llm = FakeLLM([(call("run_python", "def solve(g): return [r[::-1] for r in g]\nprint(solve(train[0]['input']))"), "stop"),
                       (call("submit_answer", "solve(test_input)"), "stop")], tokens=1000)
        transcript, r = run(llm)
        self.assertEqual((r["pred"], r["answer_source"], r["stop_reason"]), (ANSWER, "submit", "submitted"))
        self.assertEqual((r["turns"], r["tool_calls"], r["parse_failures"]), (2, 2, 0))
        self.assertIn("<tool_response>\n[completed]\n[[2, 1]]\n\n[budget:", transcript)
        # Append-only: each request extends the previous one.
        self.assertTrue(llm.prompts[1][0].startswith(llm.prompts[0][0]))

    def test_eos_mid_thought_and_length_continue(self):
        llm = FakeLLM([("thinking part one", "stop"), (" more thinking", "length"),
                       (call("submit_answer", "[[9, 8, 7]]", pre=" done"), "stop")])
        _, r = run(llm)
        self.assertEqual((r["pred"], r["eos_continues"], r["chunks"], r["turns"]), (ANSWER, 1, 3, 1))
        self.assertEqual(llm.prompts[1][0], llm.prompts[0][0] + "thinking part one")

    def test_final_text_answer(self):
        _, r = run(FakeLLM([("think\n</think>\n\nThe output is [[9, 8, 7]]", "stop")]))
        self.assertEqual((r["pred"], r["answer_source"], r["stop_reason"]), (ANSWER, "parse", "final"))

    def test_malformed_call_gets_feedback(self):
        llm = FakeLLM([("x\n</think>\n\n<tool_call>\nrun_python: print(1)\n</tool_call>", "stop"),
                       (call("submit_answer", "[[9, 8, 7]]"), "stop")])
        _, r = run(llm)
        self.assertEqual((r["pred"], r["parse_failures"]), (ANSWER, 1))
        self.assertIn("[tool error: malformed", llm.prompts[1][0])

    def test_budget_exhausted_forces_submit(self):
        def forced(prompt, max_tokens, stop):
            self.assertTrue(prompt.endswith(aq.FORCE_THINK + aq.FORCE_CALL))
            self.assertEqual(stop, ["</parameter>"])
            return "solve(test_input)\n", "stop"
        llm = FakeLLM([(call("run_python", "def solve(g): return [r[::-1] for r in g]"), "stop"),
                       ("long thinking", "length"), forced], tokens=600)
        _, r = run(llm, aq.AgentConfig(max_total_output=1000, budget_soft=0))
        self.assertEqual((r["pred"], r["answer_source"], r["stop_reason"]), (ANSWER, "forced", "output_budget"))

    def test_force_submit_last_keeps_session_answer(self):
        # force_submit_last: a set `answer` wins and the forced call is never made.
        llm = FakeLLM([(call("run_python", "answer = [[9, 8, 7]]"), "stop"), ("long thinking", "length")], tokens=600)
        _, r = run(llm, aq.AgentConfig(max_total_output=1000, budget_soft=0, force_submit_last=True))
        self.assertEqual((r["pred"], r["answer_source"], r["stop_reason"]), (ANSWER, "session", "output_budget"))
        self.assertEqual(len(llm.prompts), 2)

    def test_force_submit_last_runs_when_fallbacks_fail(self):
        def forced(prompt, max_tokens, stop):
            self.assertTrue(prompt.endswith(aq.FORCE_THINK + aq.FORCE_CALL))
            return "[[9, 8, 7]]\n", "stop"
        llm = FakeLLM([(call("run_python", "x = 1"), "stop"), ("long thinking", "length"), forced], tokens=600)
        _, r = run(llm, aq.AgentConfig(max_total_output=1000, budget_soft=0, force_submit_last=True))
        self.assertEqual((r["pred"], r["answer_source"], r["stop_reason"]), (ANSWER, "forced", "output_budget"))

    def test_session_answer_fallback(self):
        llm = FakeLLM([(call("run_python", "answer = [[9, 8, 7]]"), "stop"), ("x", "length"), ("garbage", "stop")],
                      tokens=600)
        _, r = run(llm, aq.AgentConfig(max_total_output=1000, budget_soft=0))
        self.assertEqual((r["pred"], r["answer_source"]), (ANSWER, "session"))

    def test_verify_submit_rejects_once(self):
        llm = FakeLLM([(call("run_python", "def bad(g): return g"), "stop"),
                       (call("submit_answer", "bad(test_input)"), "stop"),
                       (call("submit_answer", "bad(test_input)"), "stop")])
        _, r = run(llm, aq.AgentConfig(verify_submit=True))
        self.assertEqual((r["pred"], r["submit_rejections"], r["submit_verified"]), (TEST, 1, False))
        self.assertIn("[answer not accepted yet", llm.prompts[2][0])


class FailingLLM(FakeLLM):
    """Scripted turns, then raises (an HTTP 400 / stream error)."""

    async def complete(self, prompt, max_tokens, stop=None):
        if not self.script:
            raise RuntimeError("HTTP 400: context length exceeded")
        return await super().complete(prompt, max_tokens, stop)


class HangingExecutor(LocalExecutor):
    """execute('hang') blocks far beyond any timeout, like a wedged sandbox."""
    instances = 0

    def __enter__(self):
        HangingExecutor.instances += 1
        return super().__enter__()

    def execute(self, code):
        if code == "hang":
            import time
            time.sleep(3)
        return super().execute(code)


class RobustnessTest(unittest.TestCase):
    """qwen-tools-3 fixes from kaggle/reviews (R2, R3, R7, R8/S4, S7, S8)."""

    def test_error_keeps_session_answer(self):  # R2 / S3
        _, r = run(FailingLLM([(call("run_python", "answer = [[9, 8, 7]]"), "stop")]))
        self.assertEqual((r["pred"], r["answer_source"], r["stop_reason"]), (ANSWER, "session", "error"))
        self.assertIn("HTTP 400", r["error"])

    def test_salvage_best_function(self):  # R7
        code = "def ident(g): return g\ndef flip(g): return [row[::-1] for row in g]\nprint('ok')"
        _, r = run(FakeLLM([(call("run_python", code), "stop"), ("x\n</think>\n\nI could not solve it.", "stop")]))
        self.assertEqual((r["pred"], r["answer_source"]), (ANSWER, "salvage"))
        self.assertEqual((r["salvage"]["function"], r["salvage"]["train_matched"]), ("flip", 2))
        _, r = run(FakeLLM([(call("run_python", "x = 1"), "stop"), ("x\n</think>\n\nNo idea.", "stop")]))
        self.assertIsNone(r["pred"])
        _, r = run(FakeLLM([("x\n</think>\n\nNo idea.", "stop")]), aq.AgentConfig(salvage=False))
        self.assertIsNone(r["pred"])

    def test_executor_call_cap_replaces_session(self):  # R3
        HangingExecutor.instances = 0
        llm = FakeLLM([(call("run_python", "x = 5"), "stop"), (call("run_python", "hang"), "stop"),
                       (call("run_python", "print(test_input)"), "stop"), (call("submit_answer", "[[9, 8, 7]]"), "stop")])
        cfg = aq.AgentConfig(tool_timeout=0.5, exec_margin=0.5)
        _, r = asyncio.run(aq.solve(llm, lambda tr, te: HangingExecutor(tr, te, 5), TRAIN, TEST, cfg))
        self.assertEqual((r["pred"], r["executor_resets"], HangingExecutor.instances), (ANSWER, 1, 2))
        self.assertIn("executor call exceeded 1s", llm.prompts[2][0])
        self.assertIn("[[7, 8, 9]]", llm.prompts[3][0])  # the fresh session has the task data

    def test_context_limit_forces_submit(self):  # R8 / S4
        first = (call("run_python", "print('" + "x" * 4000 + "')"), "stop")
        probe = FakeLLM([first, ("t", "length")], tokens=None)
        run(probe, aq.AgentConfig(max_total_output=2000, force_submit_tokens=0, salvage=False))
        p1, p2 = probe.prompts[0][0], probe.prompts[1][0]
        t1 = first[0]
        est2 = aq.estimate_tokens(len(p1) // 4 + len(t1) // 4, len(p1) + len(t1), p2)
        self.assertGreater(est2, len(p1) // 4 + 2000)  # the 4000-char tool output is counted before sending
        llm = FakeLLM([first, lambda prompt, mt, stop: ("[[9, 8, 7]]", "stop")])
        cfg = aq.AgentConfig(context=est2 + 64 + 200, max_total_output=10 ** 6, force_submit_tokens=64)
        _, r = run(llm, cfg)
        self.assertEqual((r["stop_reason"], r["answer_source"], r["pred"]), ("context", "forced", ANSWER))
        self.assertEqual(len(llm.prompts), 2)  # no oversized request was sent

    def test_executor_start_failure_is_recorded(self):  # S7
        ex = LocalExecutor(TRAIN, TEST, python="/bin/false", start_timeout=2)
        with self.assertRaises(RuntimeError):
            ex.__enter__()
        _, r = asyncio.run(aq.solve(FakeLLM([]), lambda tr, te: LocalExecutor(tr, te, python="/bin/false",
                                                                               start_timeout=2),
                                    TRAIN, TEST, aq.AgentConfig()))
        self.assertIn("initialization failed", r["error"])
        self.assertIsNone(r["pred"])

    def test_child_limits_and_group_kill(self):  # S8
        import os
        import time
        path = f"/tmp/e012_child_{os.getpid()}.pid"
        with LocalExecutor(TRAIN, TEST, timeout=1, nice=7, fsize_mb=1) as ex:
            out = ex.execute("import os, resource\nprint(os.nice(0), resource.getrlimit(resource.RLIMIT_FSIZE)[0],"
                             " open('/proc/self/oom_score_adj').read().strip())")["stdout"].split()
            self.assertEqual(out[1:], [str(2 ** 20), "1000"])
            self.assertGreaterEqual(int(out[0]), 7)
            self.assertEqual(ex.execute("open('big.bin', 'wb').write(b'x' * 2 ** 21)")["process_status"], "error")
            r = ex.execute(f"import subprocess\np = subprocess.Popen(['sleep', '60'])\nopen({path!r}, 'w')"
                           f".write(str(p.pid))\nwhile True: pass")
            self.assertEqual(r["process_status"], "timeout")
            with open(path) as f:
                pid = int(f.read())
        os.remove(path)
        time.sleep(0.5)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)


class CompactTest(unittest.TestCase):
    """E027 transcript compaction (AgentConfig.compact_at)."""

    def test_compact_keeps_recent_turns_and_structure(self):
        cfg = aq.AgentConfig(prompt="lean2sub")
        p = head = aq.initial_prompt(TRAIN, TEST, cfg)
        for i in range(4):
            p += (f"thinking {i} " * 300 + f"\n</think>\n\nreply {i}\n\n<tool_call>\n<function=run_python>\n"
                  f"<parameter=code>\nprint({i})\n</parameter>\n</function>\n</tool_call><|im_end|>\n")
            p += aq.render_tool_responses([f"out {i} " + "x" * 3000]) + aq.GEN_PROMPT
        new, removed = aq.compact(p, keep=2, tool_chars=1500)
        self.assertGreater(removed, 0)
        self.assertTrue(new.startswith(head.rsplit(aq.GEN_PROMPT, 1)[0]) and new.endswith(aq.GEN_PROMPT))
        for i in range(4):
            self.assertIn(f"reply {i}", new)
            self.assertIn(f"print({i})", new)
            self.assertEqual(f"thinking {i} " in new, i >= 2)
        self.assertEqual(new.count("characters cut]"), 2)
        self.assertEqual(aq.compact(new, 2, 1500)[1], 0)  # idempotent

    def test_loop_compacts_and_still_answers(self):
        long = "consider the grid " * 1500  # ~27k chars of thinking per turn (FakeLLM: chars / 4 tokens)
        script = [(call("run_python", f"x{i} = {i}\nprint(x{i})", pre=long), "stop") for i in range(5)]
        script += [(call("run_python", "def solve(g): return [r[::-1] for r in g]"), "stop"),
                   (call("submit_answer", "solve(test_input)"), "stop")]
        llm = FakeLLM(script)
        cfg = aq.AgentConfig(tool_timeout=5, compact_at=12000, compact_keep=2, max_total_output=200000)
        _, res = run(llm, cfg)
        self.assertEqual(res["pred"], ANSWER)
        self.assertGreaterEqual(res["compactions"], 1)
        last_prompt = llm.prompts[-1][0]
        self.assertLess(last_prompt.count("consider the grid"), 5 * 1500)  # old thinking removed
        self.assertIn("print(x0)", last_prompt)  # tool calls kept
        self.assertNotIn("compactions", run(FakeLLM(script[5:]))[1])  # off by default: no bookkeeping


if __name__ == "__main__":
    unittest.main()
