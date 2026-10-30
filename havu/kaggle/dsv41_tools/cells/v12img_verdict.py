# Cell 4i (V12 + images smoke verdict, commit runs only): cell 2i's smoke_image.json. RAISES on a hard failure
# (bidirectional image attention, an image or a MANH_VISION_MAX_IMAGES-image prompt rejected, or no result); the
# perception probe and the other recorded checks only print. Runs after his verdict cell, which raises first on
# anything it found.
import json, os, time
if not os.environ.get("KAGGLE_IS_COMPETITION_RERUN") and os.environ.get("ARC_GRID_FORMAT") == "text+image":
    _vi_path = f"{os.environ.get('ARC_OUT', '/kaggle/working')}/smoke_image.json"
    _vi, _vi_bad = {}, []
    for _ in range(60):  # the hard checks finish within minutes of server start; the solver cell ran ~40 min since
        try:
            _vi = json.load(open(_vi_path))
        except Exception:
            _vi = {}
        if _vi.get("hard_done"):
            break
        time.sleep(10)
    if not _vi.get("hard_done"):
        _vi_bad.append(f"no finished image smoke in {_vi_path} (cell 2i)")
    _vi_bad += _vi.get("fail", [])
    _p = _vi.get("probe") or {}
    print("V12 + images smoke:", {k: _vi.get(k) for k in ("server_up_at_s", "config_check", "log_bidirectional",
                                                         "log_disable_chunked_mm_input", "image_prompt_tokens",
                                                         "image_tokens_expected", "image_limit", "mm_cache_metrics")},
          {k: v for k, v in _vi.items() if k.startswith("images_")}, flush=True)
    print("V12 + images perception probe:", {k: v for k, v in _p.items() if k != "rows"} or "not finished", flush=True)
    for _w in _vi.get("warn", []) + ([] if _vi.get("done") else ["perception probe not finished"]):
        print("V12 + images smoke warning:", _w, flush=True)
    if _vi_bad:
        print("=" * 100, f"V12 + IMAGES SMOKE FAILED: {_vi_bad}", "=" * 100, sep="\n", flush=True)
        raise RuntimeError(f"V12 + IMAGES SMOKE FAILED: {_vi_bad}")
    print("V12 + IMAGES SMOKE OK", flush=True)
