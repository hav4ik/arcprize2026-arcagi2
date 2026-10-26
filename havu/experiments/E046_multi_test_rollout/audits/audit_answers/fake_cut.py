"""Real harness solve() + solver GuardedClient with a fake chat stream: a multi-test (3 inputs) attempt sets state with
run_python, then its next turn is cut by the force deadline; what reaches split_multi / build_submission?"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

SCR = Path(__file__).resolve().parent
os.environ.update(ARC_OUT=str(SCR / "fake_cut_out"), ARC_MULTI_TEST="1", ARC_MULTI_BUDGET="0.5",
                  ARC_CONTEXT_LIMIT="163840")
sys.path[:0] = [str(SCR / "kit"), str(SCR / "kit" / "repo" / "evals")]
import solver_dsv41 as S  # noqa: E402

S.LOG.mkdir(parents=True, exist_ok=True)
(S.LOG / "traces").mkdir(exist_ok=True)
args = S.make_args()
import run_deepseek_tools as h  # noqa: E402  (the module make_args patched)
args.token_exact = False  # chat path (the token-exact path needs a /tokenize server); post-turn logic is shared
TASK = {"train": [{"input": [[1, 0]], "output": [[0, 1]]}], "test": [{"input": [[1, 2]]}, {"input": [[3, 4]]}, {"input": [[5, 6]]}]}


class Usage:
    def __init__(self, c):
        self.c = c

    def model_dump(self):
        return {"prompt_tokens": 10, "completion_tokens": self.c, "total_tokens": 10 + self.c}


def chunk(content="", reasoning="", calls=None, finish=None, usage=None):
    d = NS(content=content, reasoning_content=reasoning, tool_calls=calls, model_extra={})
    return NS(usage=usage, choices=[NS(index=0, delta=d, finish_reason=finish, model_extra={})], model_extra={})


class FakeReal:
    def __init__(self, code):
        self.code, self.n = code, 0
        self.base_url = "http://fake/v1"
        outer = self

        class _C:
            @staticmethod
            def create(**kw):
                outer.n += 1
                if outer.n == 1:
                    tc = NS(index=0, id="call_1", function=NS(name="run_python", arguments=json.dumps({"code": outer.code})))
                    return iter([chunk(reasoning="think"), chunk(calls=[tc]), chunk(finish="tool_calls"),
                                 NS(usage=Usage(50), choices=[], model_extra={})])
                S.STATE["forced"] = True  # the force deadline passes while this turn streams

                def gen():
                    yield chunk(reasoning="still thinking")
                    yield chunk(reasoning="more")
                return gen()
        self.chat = NS(completions=_C())
        self.completions = NS(create=lambda **kw: None)


SCEN = {
    "answer = list of 3": "answer = [np.array(t)[:, ::-1].tolist() for t in test_inputs]",
    "answer 2 of 3 valid, no solve": "answer = [test_inputs[0], None, test_inputs[2]]",
    "answer one grid, solve defined": "def solve(g):\n    return np.array(g)[:, ::-1]\nanswer = solve(test_input)",
    "answer 2 of 3, solve raises on #1": ("def solve(g):\n    if g[0][0] == 3: raise ValueError('x')\n    return np.array(g)[:, ::-1]\n"
                                          "answer = [solve(t) if t[0][0] != 3 else None for t in test_inputs]"),
}
for name, code in SCEN.items():
    S.STATE["forced"] = False
    S.RECORDS.clear()
    client = S.GuardedClient(FakeReal(code))
    trace, rec = h.solve(client, args, TASK, TASK["test"][0]["input"], session_factory=S.session_factory,
                           test_inputs=[x["input"] for x in TASK["test"]])
    rec.update(task_id="T", pair=0, attempt=0, cuts=client.stats)
    for r in S.split_multi(rec, TASK):
        S.RECORDS.setdefault(("T", r["pair"]), []).append(r)
    sub = S.build_submission({"T": TASK})
    print(f"{name:36s} stop={rec['stop_reason']} src={rec.get('answer_source')} cuts={rec['cuts']} "
          f"train={rec.get('train_trace')} fallbacks={[(e['fallback'], e['accepted']) for e in trace['events'] if 'fallback' in e]}")
    print("    rows:", [row["attempt_1"] for row in sub["T"]])
