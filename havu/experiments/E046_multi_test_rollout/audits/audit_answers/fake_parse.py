import json, os, sys
from pathlib import Path
from types import SimpleNamespace as NS
SCR = Path(__file__).resolve().parent
os.environ.update(ARC_OUT=str(SCR / "fake_cut_out"))
sys.path[:0] = [str(SCR / "kit"), str(SCR / "kit" / "repo" / "evals")]
import solver_dsv41 as S
S.LOG.mkdir(parents=True, exist_ok=True)
args = S.make_args()
import run_deepseek_tools as h
args.token_exact = False
TASK = {"train": [{"input": [[1, 0]], "output": [[0, 1]]}], "test": [{"input": [[1, 2]]}]}
class U:
    def __init__(s, c): s.c = c
    def model_dump(s): return {"prompt_tokens": 10, "completion_tokens": s.c, "total_tokens": 10 + s.c}
def ch(content="", reasoning="", calls=None, finish=None, usage=None):
    return NS(usage=usage, choices=[NS(index=0, delta=NS(content=content, reasoning_content=reasoning, tool_calls=calls, model_extra={}), finish_reason=finish, model_extra={})], model_extra={})
class Real:
    def __init__(s): s.n = 0; s.base_url = "http://fake/v1"; outer = s
    def create(s, **kw):
        s.n += 1
        if s.n == 1:
            tc = NS(index=0, id="c1", function=NS(name="run_python", arguments=json.dumps({"code": "answer = [[2, 1]]"})))
            return iter([ch(reasoning="r"), ch(calls=[tc]), ch(finish="tool_calls"), NS(usage=U(50), choices=[], model_extra={})])
        def g():
            yield ch(content="Let me compare with the earlier draft [[9, 9]] and then submit")
            S.STATE["forced"] = True
            yield ch(content=" ...")
        return g()
r = Real(); r.chat = NS(completions=NS(create=r.create)); r.completions = NS(create=lambda **k: None)
S.STATE["forced"] = False
tr, rec = h.solve(S.GuardedClient(r), args, TASK, TASK["test"][0]["input"], session_factory=S.session_factory)
print("pred", rec["pred"], "source", rec.get("answer_source"), "stop", rec["stop_reason"],
      "fallback events", [e for e in tr["events"] if "fallback" in e])
