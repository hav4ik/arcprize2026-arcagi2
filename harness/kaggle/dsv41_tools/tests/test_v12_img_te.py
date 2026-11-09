"""V12 + images + token-exact images (build_v12_img_te.py; PR #3 of hav4ik/arcprize2026-arcagi2). No GPU, network or
Kaggle. Run: .venv/bin/python -m unittest discover -s kaggle/dsv41_tools/tests -p 'test_v12_img_te.py'

- the new notebook differs from nb_v12_img_ab only in the markdown, cell 00's header and cell 0b (overlay, SHAs, feature check);
- the overlay's solver / harness differ from the AB ones by the approved hunks only;
- the solver's text-mode argv and Namespace are V12's exactly; image mode = V12's argv + --grid-format text+image
  --image-cell 0, token-exact kept;
- the harness's text path makes the same calls as the AB harness (V12's), the image path uses /inference/v1/generate
  (a fake vision server), the placeholders / images / echo checks hold, and the PR's own tests pass on the pinned file.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_v12_img as T  # noqa: E402  (V12 helpers: probe, kit_with, run_text, ...)

D = T.D
ROOT = T.ROOT
TE = D / "pinned_v12img_te"
AB_NB = D / "out/nb_v12_img_final/dsv41-tools-v12-img.ipynb"
TE_NB = D / "out/nb_v12_img_te/dsv41-tools-v12-img.ipynb"
PR_TESTS = os.environ.get("PR3_TESTS", "")  # path of the PR's evals/tests/test_deepseek_tools.py


def cells(path):
    return [T.src(c) for c in json.loads(Path(path).read_text())["cells"]]


AB_CELLS, TE_CELLS = cells(AB_NB), cells(TE_NB)
AB_FILES = T.overlay(T.cell(json.loads(AB_NB.read_text())["cells"], "# Cell 0b"))
TE_FILES = T.overlay(T.cell(json.loads(TE_NB.read_text())["cells"], "# Cell 0b"))
H_AB = T.load_harness(AB_FILES["repo/evals/run_deepseek_tools.py"], "rdt_ab_ed66")
H_TE = T.load_harness(TE_FILES["repo/evals/run_deepseek_tools.py"], "rdt_te")


class Built(unittest.TestCase):
    def test_only_these_cells_change(self):
        self.assertEqual(len(AB_CELLS), len(TE_CELLS))
        self.assertEqual([i for i, (a, b) in enumerate(zip(AB_CELLS, TE_CELLS)) if a != b], [0, 1, 4])  # (2m is shared with final)

    def test_overlay(self):
        self.assertEqual(set(AB_FILES), set(TE_FILES))
        changed = {k for k in AB_FILES if AB_FILES[k] != TE_FILES[k]}
        self.assertEqual(changed, {"solver_dsv41.py", "repo/evals/run_deepseek_tools.py"})
        c = T.cell(json.loads(TE_NB.read_text())["cells"], "# Cell 0b")
        pins = json.loads(c.split("_MT_SHA = ", 1)[1].split("\n", 1)[0])
        self.assertEqual(pins, {k: T.sha(v) for k, v in TE_FILES.items()})
        self.assertEqual(TE_FILES["solver_dsv41.py"], (TE / "solver_dsv41.py").read_bytes())
        self.assertEqual(TE_FILES["repo/evals/run_deepseek_tools.py"], (TE / "run_deepseek_tools.py").read_bytes())
        need = json.loads(c.split("_MT_NEED = ", 1)[1].split("\n", 1)[0])
        self.assertIn("def generate_with_images", need["repo/evals/run_deepseek_tools.py"])
        for f, needs in need.items():
            for n in needs:
                self.assertIn(n, TE_FILES[f].decode(), (f, n))
        self.assertEqual(c.split("_MT_EXPECT = ", 1)[1].split("\n", 1)[0], T.cell(
            json.loads(AB_NB.read_text())["cells"], "# Cell 0b").split("_MT_EXPECT = ", 1)[1].split("\n", 1)[0])

    def test_solver_diff(self):
        rem, add = T.added_removed(AB_FILES["solver_dsv41.py"].decode(), TE_FILES["solver_dsv41.py"].decode())
        self.assertEqual(rem, [x for x in rem if x.startswith("#") or "token-exact" in x or x.strip().endswith("if IMAGES else HARNESS)") or "rglob" in x or x.strip().startswith('""')])
        self.assertIn('    harness = HARNESS + ["--grid-format", "text+image", "--image-cell", "0"] if IMAGES else HARNESS  # token-exact stays', add)
        self.assertTrue(any("self.timeout = real.timeout" in x for x in add))  # GuardedClient.timeout (the PR's stream needs it)
        self.assertTrue(any("def guard(self, factory)" in x for x in add))
        self.assertTrue(any('glob("models/*/*/*/*/*/e043_manifest.json")' in x for x in add))

    def test_harness_diff(self):
        rem, add = T.added_removed(AB_FILES["repo/evals/run_deepseek_tools.py"].decode(), TE_FILES["repo/evals/run_deepseek_tools.py"].decode())
        # removed: the docstring's last line, the old completions-stream block (re-indented under `else`), the guard (3)
        self.assertEqual(len(rem), 8, rem)
        self.assertIn('    if args.token_exact and args.grid_format != "json":', rem)
        text = TE_FILES["repo/evals/run_deepseek_tools.py"].decode()
        for needle in ("def generate_with_images", "/inference/v1/generate", "--enable-scale-out", "DS_IMAGE ="):
            self.assertIn(needle, text)
        self.assertIn('if args.token_exact and args.grid_format == "image":', text)
        self.assertNotIn("--token-exact has no image path", text)

    def test_cell0b_runs_and_checks(self):
        nbc = json.loads(TE_NB.read_text())["cells"]
        c00, c0b = T.src(nbc[1]), T.cell(nbc, "# Cell 0b")
        for rerun in (False, True):
            kit = T.SCRATCH / f"te0b_{rerun}"
            kit.mkdir()
            import tarfile
            with tarfile.open(T.KIT_TGZ) as tf:
                tf.extractall(kit, filter="data")
            env = {k: v for k, v in os.environ.items() if not k.startswith(("ARC_", "MANH_", "KAGGLE_"))}
            env["ARC_SMOKE_ENV_LOG"] = str(kit / "smoke_env.jsonl")
            if rerun:
                env["KAGGLE_IS_COMPETITION_RERUN"] = "1"
            code = (f"import os\nns = {{}}\nexec(compile({c00!r}, 'c00', 'exec'), ns)\n"
                    f"ns.update(KIT={str(kit)!r}, MODE='submit', RERUN={rerun!r})\nexec(compile({c0b!r}, 'c0b', 'exec'), ns)\n")
            script = T.SCRATCH / f"te_run0b_{rerun}.py"
            script.write_text(code)
            r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-1500:])
            self.assertIn("E046 CHECK OK", r.stdout, r.stdout[-1500:])
            for arc, data in TE_FILES.items():
                self.assertEqual((kit / arc).read_bytes(), data, arc)


class SolverSwitch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v12 = T.probe(T.V12_FILES, "kit_v12", {})
        cls.text = T.probe(TE_FILES, "kit_te", {})
        cls.img = T.probe(TE_FILES, "kit_te", {"ARC_GRID_FORMAT": "text+image"})

    def test_text_mode_is_v12_exactly(self):
        self.assertEqual(self.text["argv"], self.v12["argv"])
        self.assertEqual(self.text["ns"], self.v12["ns"])
        self.assertEqual(self.text["server_cmd"], self.v12["server_cmd"])

    def test_image_mode_keeps_token_exact(self):
        v12 = self.v12["argv"]
        i = v12.index("--multi-test")
        self.assertEqual(self.img["argv"], v12[:i] + ["--grid-format", "text+image", "--image-cell", "0"] + v12[i:])
        ns = self.img["ns"]
        self.assertEqual((ns["token_exact"], ns["grid_format"], ns["image_cell"], ns["context_limit"]),
                         ("True", "text+image", "0", "163840"))
        self.assertEqual({k for k in ns if ns[k] != self.v12["ns"][k]}, {"grid_format", "image_cell"})


class Harness(unittest.TestCase):
    def test_text_path_same_as_ab(self):
        flip_all = T.FLIP + "print(solve(test_inputs[0]))"
        scripts = [
            (T.TASK1, None, [T.C.ds_call_text(T.FLIP + "print(solve(test_input))"),
                             T.C.ds_call_text("solve(test_input)", name="submit_answer")], 0),
            (T.TASK3, [T.T0, T.T1, T.T2], [T.C.ds_call_text(flip_all), T.C.ds_call_text(
                "[solve(t) for t in test_inputs]", name="submit_answer")], 0),
            (T.TASK1, None, [T.C.ds_call_text("print(undefined_nmae)"), T.C.ds_call_text(T.FLIP + "answer = solve(test_input)"),
                             T.C.ds_text_answer("done")], 0),
            (T.TASK3, [T.T0, T.T1, T.T2], [T.C.ds_call_text(T.FLIP + "answer = [solve(t) for t in test_inputs]")], 163000),
        ]
        for task, tests, turns, pad in scripts:
            a = T.run_text(H_AB, list(turns), task, tests, pad=pad)
            b = T.run_text(H_TE, list(turns), task, tests, pad=pad)
            self.assertEqual(a, b)
        self.assertEqual(b[2]["stop_reason"], "context")

    def solver_module(self):
        import importlib.util
        kit = T.kit_with(TE_FILES, "kit_te")
        env = {**{k: v for k, v in os.environ.items() if not k.startswith(("ARC_", "MANH_"))}, "ARC_KIT": str(kit),
               "ARC_OUT": str(T.SCRATCH / "out_te_inproc"), **T.V12_ENV, "ARC_GRID_FORMAT": "text+image"}
        with patch.dict(os.environ, env, clear=True):
            spec = importlib.util.spec_from_file_location("solver_te_inproc", kit / "solver_dsv41.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        return mod

    def test_forced_deadline_cuts_a_generate_stream(self):
        """The solver's force sweep ends an image turn like any other stream: no exception, the attempt falls back."""
        import httpx
        sol = self.solver_module()
        sol.STATE["forced"] = True
        client = sol.GuardedClient(type("Real", (), {"base_url": "http://fake:1/v1", "timeout": 30})())
        calls = []
        mock = httpx.Client(transport=httpx.MockTransport(lambda r: calls.append(r.url.path) or httpx.Response(
            200, json={"tokens": [1, 2, 3]} if r.url.path == "/tokenize" else {})))
        argv = T.solver_argv(True)
        i = argv.index("--multi-test")
        args = H_TE.parser().parse_args(argv[:i] + ["--token-exact"] + argv[i:])
        with mock, patch("httpx.post", mock.post), patch("httpx.stream", mock.stream):
            trace, result = H_TE.solve(client, args, T.TASK1, T.T0, session_factory=__import__('dsv41_fakes').local_factory)
        self.assertNotIn("/inference/v1/generate", calls)  # forced before the request: never sent
        self.assertIn(result["stop_reason"], ("output_budget", "length", "error", "cut", "deadline") if result.get("stop_reason") else (None,))

    def test_image_turn_uses_generate(self):
        """The solver's image argv (token-exact kept) on a fake vision server: turns go to /inference/v1/generate."""
        import httpx
        sys.path.insert(0, str(D.parent.parent / "evals/tests"))
        sys.path.insert(0, str(TE_FILES and Path(T.SCRATCH)))
        import dsv41_fakes as C
        IMG = 7

        class Srv:
            def __init__(self, turns):
                self.turns, self.reqs = list(turns), []
                self.base_url = "http://fake:1/v1"
                self.timeout = 30

            def handler(self, request):
                path, body = request.url.path, json.loads(request.content)
                self.reqs.append((path, body))
                if path == "/tokenize" and "prompt" in body:
                    return httpx.Response(200, json={"tokens": [IMG]})
                if path == "/tokenize":
                    ids = []
                    for m in body["messages"]:
                        ids.append(2)
                        parts = m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m.get("content") or ""}]
                        for p in parts:
                            ids += [IMG] * 5 if p["type"] == "image_url" else [100 + len(p["text"]) % 50]
                        for c in m.get("tool_calls") or []:
                            ids.append(300)
                        ids.append(sum(ord(ch) for ch in (m.get("reasoning_content") or "")) % 90)
                    return httpx.Response(200, json={"tokens": ids + [5] * body["add_generation_prompt"]})
                if path == "/v1/completions/render":
                    return httpx.Response(200, json=[{"token_ids": body["prompt"], "sampling_params": {
                        "temperature": 1.0, "max_tokens": body["max_tokens"]}}])
                assert path == "/inference/v1/generate", path
                ran = [x for t in body["token_ids"] for x in ([t] * 5 if t == IMG else [t])]
                text = self.turns.pop(0)
                ids = [ord(ch) % 5000 + 1000 for ch in text]
                ev = [{"choices": [{"index": 0, "text": text, "token_ids": ids, "finish_reason": "stop"}], "prompt_token_ids": ran},
                      {"choices": [], "usage": {"prompt_tokens": len(ran), "completion_tokens": len(ids), "total_tokens": len(ran) + len(ids)}}]
                sse = "".join(f"data: {json.dumps(e)}\n\n" for e in ev) + "data: [DONE]\n\n"
                return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})

        argv = T.solver_argv(True)
        i = argv.index("--multi-test")
        self.assertNotIn("--token-exact", T.solver_argv(True))  # the helper mirrors the AB solver; add it back here
        argv = argv[:i] + ["--token-exact"] + argv[i:]
        args = H_TE.parser().parse_args(argv)
        turns = [C.ds_call_text(T.FLIP + "print(solve(test_input))"), C.ds_call_text("solve(test_input)", name="submit_answer")]
        srv = Srv(turns)
        mock = httpx.Client(transport=httpx.MockTransport(srv.handler))
        sol = self.solver_module()
        client = sol.GuardedClient(type("Real", (), {"base_url": "http://fake:1/v1", "timeout": 30})())
        with mock, patch("httpx.post", mock.post), patch("httpx.stream", mock.stream):
            trace, result = H_TE.solve(client, args, T.TASK1, T.T0, session_factory=C.local_factory)
        paths = [p for p, _ in srv.reqs]
        self.assertEqual(paths.count("/inference/v1/generate"), 2)
        for p, b in srv.reqs:
            if p == "/inference/v1/generate":
                self.assertEqual(b["token_ids"].count(IMG), len(b["content_parts"]))
                self.assertEqual(len(b["content_parts"]), 2 * len(T.TASK1["train"]) + 1)
        self.assertEqual(result["token_exact"]["turns"], 2)
        self.assertNotIn("/v1/chat/completions", paths)


class PRTests(unittest.TestCase):
    @unittest.skipUnless(PR_TESTS, "set PR3_TESTS=<PR #3's evals/tests/test_deepseek_tools.py>")
    def test_pr_tests_on_the_pinned_harness(self):
        live = ROOT / "evals"
        with tempfile.TemporaryDirectory() as d:
            ev = Path(d) / "evals"
            ev.mkdir()
            for p in live.iterdir():
                if p.name not in ("run_deepseek_tools.py", "tests"):
                    (ev / p.name).symlink_to(p)
            (ev / "run_deepseek_tools.py").write_bytes(TE_FILES["repo/evals/run_deepseek_tools.py"])
            (ev / "tests").mkdir()
            for p in (live / "tests").iterdir():
                (ev / "tests" / p.name).symlink_to(p) if p.name != "test_deepseek_tools.py" else None
            (ev / "tests" / "test_deepseek_tools.py").write_text(Path(PR_TESTS).read_text())
            r = subprocess.run([sys.executable, "-m", "unittest", "-k", "TokenExact", "tests.test_deepseek_tools"], cwd=str(ev),
                               capture_output=True, text=True, timeout=300, env={**os.environ, "PYTHONPATH": str(ev)})
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])


if __name__ == "__main__":
    unittest.main()
