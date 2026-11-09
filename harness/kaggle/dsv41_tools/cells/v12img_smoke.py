# Cell 2i (V12 + images smoke, commit runs only): the image path of his server (cell 2m), checked in a background
# thread once it serves, so it costs no wall time; results go to smoke_image.json and cell 4i fails the commit run
# on a hard failure. Uses cell 0's WORK, KIT, MODE, RERUN, cell 2's PY / BASE_ENV and cell 2m's MANH_BASE_URL,
# MANH_MODEL and _manh_logs().
# Hard (fatal):
# 1. causal image attention. Upstream #57152 (in his wheels from g8cf0c0664b on, g22df163d2d included) removed the
#    fork's `is_mm_prefix_lm = vision_n_layers > 0`; E068: a bidirectional server reads grids about 2x worse. Checked
#    two ways: the installed vllm/transformers_utils/configs/deepseek_v41.py has no such line, and vLLM's own config
#    resolution gives is_mm_prefix_lm False for this checkpoint (when that check runs). A server log that says
#    "multimodal-bidirectional" is also fatal; vLLM prints it only without --disable-chunked-mm-input, so with
#    cell 2m's flag the two checks above are the ones that count.
# 2. one grid image is accepted and adds prompt tokens;
# 3. a prompt with MANH_VISION_MAX_IMAGES images (the harness sends one per grid; ARC's most is 22) is accepted.
# Recorded only: the image's prompt tokens vs the aligned geometry (grid_image.deepseek_tokens), limit + 1 images
# rejected, disable_chunked_mm_input in the server's non-default args, vllm:mm_cache_* on /metrics, and E068's
# small-grid perception probe: 15 ARC training grids of <= 36 cells, the image alone, thinking effort 50, one sample
# each (E068, full model: causal 67% exact, bidirectional 30%); fewer than 5 exact is a warning, not a failure
# (15 samples cannot separate the two reliably, and a 2-bit model may read worse than the full one).
import re, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

IMG_SMOKE_OUT = f"{WORK}/smoke_image.json"
IMG_PROBE_GRIDS = {  # E068 perception_probe.probes(60, seed 0), the <= 36-cell bin
    "90347967:0:input": [[6, 0, 0], [2, 5, 0], [2, 1, 0]],
    "539a4f51:2:input": [[2, 3, 4, 1, 6], [3, 3, 4, 1, 6], [4, 4, 4, 1, 6], [1, 1, 1, 1, 6], [6, 6, 6, 6, 6]],
    "662c240a:3:input": [[8, 9, 8], [9, 8, 8], [8, 8, 8], [2, 2, 1], [2, 2, 1], [1, 1, 2], [4, 4, 4], [4, 4, 3],
                         [3, 3, 3]],
    "e5790162:0:output": [[0, 0, 0, 3, 0, 0], [0, 0, 0, 3, 0, 0], [3, 3, 3, 3, 8, 0], [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
    "ccd554ac:4:input": [[2, 0, 2], [0, 2, 0], [2, 2, 2]],
    "88a62173:2:output": [[0, 8], [8, 0]],
    "9caf5b84:1:output": [[7, 9, 7, 9], [9, 2, 7, 2], [7, 2, 7, 7], [7, 2, 9, 2]],
    "5783df64:1:output": [[6, 4, 7], [2, 3, 9], [1, 5, 8]],
    "54d82841:2:output": [[0, 0, 0, 0, 0, 0, 0], [0, 8, 8, 8, 0, 0, 0], [0, 8, 0, 8, 6, 6, 6], [0, 0, 0, 0, 6, 0, 6],
                          [0, 0, 4, 0, 0, 4, 0]],
    "bc4146bd:3:input": [[2, 2, 2, 2], [8, 2, 2, 2], [2, 2, 8, 2], [8, 2, 8, 8]],
    "7b7f7511:2:output": [[2, 3], [3, 2], [4, 4]],
    "913fb3ed:2:input": [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
    "d4469b4b:5:input": [[0, 2, 0, 2, 0], [0, 2, 2, 2, 0], [0, 2, 2, 0, 2], [2, 2, 2, 0, 0], [0, 0, 2, 0, 2]],
    "ca8de6ea:0:output": [[6, 2, 7], [4, 3, 4], [7, 2, 6]],
    "c1990cce:1:input": [[0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]],
}
IMG_PROBE_WARN_BELOW = 5
_VI_CHECK = r'''
import json, sys
out = {}
try:
    from vllm.transformers_utils.config import get_config
    cfg = get_config(sys.argv[1], trust_remote_code=True)
    out["model_type"] = getattr(cfg, "model_type", None)
    out["hf_config_is_mm_prefix_lm"] = getattr(cfg, "is_mm_prefix_lm", None)
    from vllm.transformers_utils.model_arch_config_convertor import (MODEL_ARCH_CONFIG_CONVERTORS,
                                                                      ModelArchConfigConvertorBase)
    conv = MODEL_ARCH_CONFIG_CONVERTORS.get(cfg.model_type, ModelArchConfigConvertorBase)
    out["convertor"] = conv.__name__
    text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    out["is_mm_prefix_lm"] = bool(conv(cfg, text).is_mm_prefix_lm(True))
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"[:400]
print("VI_CHECK " + json.dumps(out))
'''
_VI_PROBE_Q = (
    "The image shows one grid from a puzzle. Every cell is a solid square of one colour; there are no grid lines. "
    "Colour key: {legend}.\n\nLook at the image and answer with a JSON object only, no other text:\n"
    '{{"rows": <number of rows>, "cols": <number of columns>, "colors": <sorted list of the colour numbers present>, '
    '"corners": [<top-left>, <top-right>, <bottom-left>, <bottom-right>], "grid": <the grid as a list of rows of '
    "colour numbers>}}")


def _vi_source_check(vroot):
    """Fatal problems in the installed vLLM's V4.1 config class (the pre-#57152 bidirectional line), or []."""
    p = os.path.join(vroot, "transformers_utils", "configs", "deepseek_v41.py")
    try:
        text = open(p).read()
    except OSError as exc:
        return [f"cannot read {p}: {type(exc).__name__}"]
    return ([f"{p} sets is_mm_prefix_lm (bidirectional image attention: the pre-#57152 fork)"]
            if re.search(r"^\s*self\.is_mm_prefix_lm\s*=", text, re.M) else [])


def _vi_parse(text):
    s = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None


def _vi_score(grid, ans):
    """E068 perception_probe.score_one: size / colours / corners / exact transcription."""
    h, w = len(grid), len(grid[0])
    a = ans if isinstance(ans, dict) else {}
    g = a.get("grid")
    shape = isinstance(g, list) and len(g) == h and all(isinstance(r, list) and len(r) == w for r in g)
    return {"parsed": ans is not None, "size": a.get("rows") == h and a.get("cols") == w,
            "colors": a.get("colors") == sorted({v for r in grid for v in r}),
            "corners": a.get("corners") == [grid[0][0], grid[0][-1], grid[-1][0], grid[-1][-1]],
            "exact": shape and g == grid}


def img_smoke_run(base, model="dsv41", wait_s=3600, out=None):
    res = out if out is not None else {}
    res.update(fail=[], warn=[], hard_done=False, done=False)
    fail, warn = res["fail"], res["warn"]

    def save():
        with open(IMG_SMOKE_OUT + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(IMG_SMOKE_OUT + ".tmp", IMG_SMOKE_OUT)

    def post(path, body, timeout=900):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    t = time.time()
    while time.time() - t < wait_s:
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
                if any(m.get("id") == model for m in json.loads(r.read())["data"]):
                    break
        except Exception:
            pass
        time.sleep(10)
    else:
        fail.append(f"server never served {model} within {wait_s}s")
        res.update(hard_done=True, done=True)
        save()
        return res
    res["server_up_at_s"] = round(time.time() - T0)
    # 1. causal image attention
    try:
        import importlib.util
        vroot = list(importlib.util.find_spec("vllm").submodule_search_locations)[0]
        res["vllm_root"] = vroot
        fail += _vi_source_check(vroot)
    except Exception as exc:
        fail.append(f"vllm source check: {type(exc).__name__}: {exc}"[:300])
    try:
        r = subprocess.run([PY, "-c", _VI_CHECK, MANH_MODEL], env=BASE_ENV, capture_output=True, text=True,
                           errors="replace", timeout=900)
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("VI_CHECK ")), None)
        res["config_check"] = json.loads(line[9:]) if line else {"error": (r.stdout + r.stderr)[-400:]}
    except Exception as exc:
        res["config_check"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    cc = res["config_check"]
    if cc.get("is_mm_prefix_lm") or cc.get("hf_config_is_mm_prefix_lm"):
        fail.append(f"vLLM resolves bidirectional image attention for this checkpoint: {cc}")
    elif "is_mm_prefix_lm" not in cc:
        warn.append(f"config check did not run ({cc.get('error')}); the source check stands")
    try:
        logs = _manh_logs()
        res["log_bidirectional"] = "multimodal-bidirectional" in logs
        res["log_disable_chunked_mm_input"] = bool(re.search(r"'disable_chunked_mm_input': True", logs))
        if res["log_bidirectional"]:
            fail.append("server log: 'multimodal-bidirectional' (bidirectional image attention)")
    except Exception as exc:
        warn.append(f"server log: {type(exc).__name__}: {exc}"[:200])
    kw = {"thinking": True, "reasoning_effort": 50}
    try:
        sys.path.insert(0, f"{KIT}/repo/evals")
        from arc_harness.grid_image import COLOUR_NAMES, aligned_cell, data_uri, deepseek_tokens
    except Exception as exc:
        fail.append(f"grid_image: {type(exc).__name__}: {exc}"[:300])
        res.update(hard_done=True, done=True)
        save()
        return res

    def img(g):
        return {"type": "image_url", "image_url": {"url": data_uri(g, aligned_cell(g, "deepseek"))}}
    # 2. one image, its prompt tokens vs the aligned geometry
    try:
        g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        text = {"type": "text", "text": "Describe this 3x3 grid in one sentence."}
        n = [post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": c}],
                                           "temperature": 0, "max_tokens": 16, "chat_template_kwargs": kw}
                  )["usage"]["prompt_tokens"] for c in ([text], [text, img(g)])]
        res["image_prompt_tokens"], res["image_tokens_expected"] = n[1] - n[0], deepseek_tokens(g, aligned_cell(g))
        if res["image_prompt_tokens"] <= 0:
            fail.append(f"an image added no prompt tokens ({n}): the server dropped it")
        elif res["image_prompt_tokens"] != res["image_tokens_expected"]:
            warn.append(f"image tokens {res['image_prompt_tokens']} != aligned geometry {res['image_tokens_expected']}")
    except Exception as exc:
        fail.append(f"image chat request: {type(exc).__name__}: {exc}"[:300])
    # 3. MANH_VISION_MAX_IMAGES images in one prompt accepted; one more rejected (recorded)
    lim = int(os.environ.get("MANH_VISION_MAX_IMAGES", "16"))
    res["image_limit"] = lim
    for k in (lim, lim + 1):
        try:
            c = [{"type": "text", "text": f"How many images are there? ({k})"}] + [img([[i % 10]]) for i in range(k)]
            u = post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": c}],
                                              "temperature": 0, "max_tokens": 1, "chat_template_kwargs": kw})
            res[f"images_{k}"] = {"accepted": True, "prompt_tokens": u["usage"]["prompt_tokens"]}
        except Exception as exc:
            res[f"images_{k}"] = {"accepted": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    if not res[f"images_{lim}"]["accepted"]:
        fail.append(f"{lim} images in one prompt rejected: {res[f'images_{lim}'].get('error')}")
    if res[f"images_{lim + 1}"]["accepted"]:
        warn.append(f"{lim + 1} images accepted: the server's limit is above MANH_VISION_MAX_IMAGES")
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
            m = r.read().decode(errors="replace")
        res["mm_cache_metrics"] = {k: [ln for ln in m.splitlines() if ln.startswith(f"vllm:{k}")][:1]
                                   for k in ("mm_cache_queries", "mm_cache_hits")}
    except Exception as exc:
        res["mm_cache_metrics"] = f"{type(exc).__name__}"
    res["hard_done"] = True
    save()
    # perception probe (recorded; warning only)
    legend = ", ".join(f"{k} = {v}" for k, v in COLOUR_NAMES.items())

    def probe(item):
        gid, g = item
        t1 = time.time()
        try:
            r = post("/v1/chat/completions", {
                "model": model, "max_tokens": 8000, "temperature": 1.0, "top_p": 1.0, "chat_template_kwargs": kw,
                "messages": [{"role": "user", "content": [img(g), {"type": "text",
                                                                   "text": _VI_PROBE_Q.format(legend=legend)}]}]},
                timeout=2400)
            msg = r["choices"][0]["message"]
            return {"gid": gid, "s": round(time.time() - t1), "out_tokens": r["usage"]["completion_tokens"],
                    **_vi_score(g, _vi_parse(msg.get("content")))}
        except Exception as exc:
            return {"gid": gid, "error": f"{type(exc).__name__}: {exc}"[:200]}
    with ThreadPoolExecutor(len(IMG_PROBE_GRIDS)) as ex:
        rows = list(ex.map(probe, IMG_PROBE_GRIDS.items()))
    ok = [r for r in rows if "error" not in r]
    res["probe"] = {"n": len(rows), "answered": len(ok), **{k: sum(bool(r[k]) for r in ok)
                                                           for k in ("parsed", "size", "colors", "corners", "exact")},
                    "rows": rows}
    if res["probe"]["exact"] < IMG_PROBE_WARN_BELOW:
        warn.append(f"perception probe: {res['probe']['exact']} of {len(rows)} small grids exact "
                    f"(E068 full model: causal ~10 of 15, bidirectional ~4.5 of 15)")
    res["done"] = True
    save()
    return res


IMG_SMOKE = {}


def img_smoke_thread():
    try:
        img_smoke_run(MANH_BASE_URL[:-3] if MANH_BASE_URL.endswith("/v1") else MANH_BASE_URL, out=IMG_SMOKE)
    except Exception as exc:
        IMG_SMOKE.setdefault("fail", []).append(f"image smoke crashed: {type(exc).__name__}: {exc}"[:300])
        IMG_SMOKE.update(hard_done=True, done=True)
        try:
            with open(IMG_SMOKE_OUT, "w") as f:
                json.dump(IMG_SMOKE, f, indent=1)
        except Exception:
            pass
    print("V12 + images smoke:", {k: v for k, v in IMG_SMOKE.items() if k != "probe"},
          "| probe:", {k: v for k, v in (IMG_SMOKE.get("probe") or {}).items() if k != "rows"}, flush=True)


if not RERUN and MODE != "fallback" and globals().get("MANH_SERVER_PID"):
    threading.Thread(target=img_smoke_thread, daemon=True).start()
    print("V12 + images smoke started (background)", flush=True)
