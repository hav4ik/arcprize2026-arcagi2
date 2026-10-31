#!/usr/bin/env python3
"""V12 + images + token-exact images (2026-10-30): nb_v12_img_final (e5385b59: A+B + replay CUDA graphs) with the image turns sent as
exact token ids through the server's /inference/v1/generate (PR #3 of hav4ik/arcprize2026-arcagi2, by Manh), instead of
the chat path. Only cell 0b's overlay changes (and notes):
- harness run_deepseek_tools.py ed668f98 -> pinned_v12img_te/ = ed668f98 + the PR's generate_with_images / turn_token_exact
  change (git diff main...pr3) + the `--token-exact` guard in main() narrowed to --grid-format image (text+image is the
  PR's path);
- solver 0c2f81e8 -> pinned_v12img_te/solver_dsv41.py: ARC_GRID_FORMAT=text+image keeps `--token-exact` (it dropped it);
- cell 0b: new overlay, SHAs and the feature check (`def generate_with_images`);
- text path: byte-identical calls to V12's (tests/test_v12_img_te.py).
Usage: build_v12_img_te.py [--src DIR] [--out DIR]. Nothing is pushed.
"""
import argparse, base64, gzip, hashlib, io, json, re, tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE / "pinned_v12img_te"


def jstr(s):  # a snippet as it appears inside a JSON string
    return json.dumps(s, ensure_ascii=False)[1:-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE / "out/nb_v12_img_final"))
    ap.add_argument("--out", default=str(HERE / "out/nb_v12_img_te"))
    a = ap.parse_args()
    raw = (Path(a.src) / "dsv41-tools-v12-img.ipynb").read_text()
    nb = json.loads(raw)
    c0b = "".join(nb["cells"][4]["source"])
    old_b64 = re.search(r'_MT = "([A-Za-z0-9+/=]+)"', c0b).group(1)
    old_sha = json.loads(c0b.split("_MT_SHA = ", 1)[1].split("\n", 1)[0])
    tf = tarfile.open(fileobj=io.BytesIO(base64.b64decode(old_b64)))
    files = {m.name: (m, tf.extractfile(m).read()) for m in tf}
    assert {k: hashlib.sha256(v[1]).hexdigest() for k, v in files.items()} == old_sha, "source overlay != its _MT_SHA"
    new = {"solver_dsv41.py": (P / "solver_dsv41.py").read_bytes(),
           "repo/evals/run_deepseek_tools.py": (P / "run_deepseek_tools.py").read_bytes()}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as out:  # same member order and header fields as the source overlay
        for name, (m, data) in files.items():
            data = new.get(name, data)
            ti = tarfile.TarInfo(name)
            ti.size, ti.mode, ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = len(data), m.mode, m.mtime, m.uid, m.gid, m.uname, m.gname
            out.addfile(ti, io.BytesIO(data))
    new_b64 = base64.b64encode(buf.getvalue()).decode()
    new_sha = {n: hashlib.sha256(new.get(n, d)).hexdigest() for n, (m, d) in files.items()}
    reps = [(old_b64, new_b64),
            (json.dumps(old_sha), json.dumps(new_sha)),
            ('"repo/evals/run_deepseek_tools.py": ["--auto-repair", "def chat_context_cap", "def text_image_content"]',
             '"repo/evals/run_deepseek_tools.py": ["--auto-repair", "def chat_context_cap", "def text_image_content", "def generate_with_images"]'),
            ("# V12 + images: the pinned files are pinned_v12img/'s",
             "# V12 + images + token-exact images (build_v12_img_te.py): harness = ed668f98 + PR #3's /inference/v1/generate path,\n"
             "# solver keeps --token-exact with images. Otherwise the pinned files are pinned_v12img/'s")]
    # the repr of the cell-0b strings inside the raw JSON: replace the escaped snippets
    text = raw
    for old, new_ in reps:
        o, n = jstr(old), jstr(new_)
        assert text.count(o) == 1, f"{old[:60]!r} occurs {text.count(o)} times"
        text = text.replace(o, n)
    o = jstr("# V12 + IMAGES (build_v12_img.py, 2026-10-30)")
    assert text.count(o) == 1
    text = text.replace(o, jstr("# V12 + IMAGES + TOKEN-EXACT IMAGES (build_v12_img_te.py, 2026-10-30): image turns go to /inference/v1/generate\n"
                                "# with the exact token ids (PR #3), not the chat path. Based on nb_v12_img_final.\n# V12 + IMAGES (build_v12_img.py, 2026-10-30)"))
    o = jstr("(one\ntoken-aligned image per grid next to the unchanged JSON prompt, chat path instead of `--token-exact`)")
    if text.count(o) == 1:
        text = text.replace(o, jstr("(one\ntoken-aligned image per grid next to the unchanged JSON prompt; `--token-exact` through /inference/v1/generate)"))
    json.loads(text)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "dsv41-tools-v12-img.ipynb").write_text(text)
    (out / "kernel-metadata.json").write_text((Path(a.src) / "kernel-metadata.json").read_text())
    print("notebook sha256", hashlib.sha256(text.encode()).hexdigest()); print(json.dumps({k: v[:8] for k, v in new_sha.items()}))


if __name__ == "__main__":
    main()
