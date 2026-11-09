"""Compare hashes computed inside Kaggle (provenance_hashes.json from a bench run) against upstream.

    .venv/bin/python kaggle/provenance/check.py kaggle/runs/<run>/provenance_hashes.json

- Model files vs the HF repo Qwen/Qwen3.8-27B-FP8 at the mirrored revision (kaggle/provenance/hf_tree.json):
  LFS files by sha256, small files by git blob sha1.
- Wheels vs PyPI's published sha256 for the same filename; wheels not on PyPI are checked against the PyTorch
  index (download.pytorch.org) when they look like torch builds, else listed as unverified.
Writes kaggle/provenance/report.json and prints a summary.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode()


def check_model(hashes):
    tree = {x["path"]: x for x in json.load(open(HERE / "hf_tree.json")) if x["type"] == "file"}
    res = {"match": [], "mismatch": [], "extra_on_kaggle": [], "missing_on_kaggle": []}
    for path, h in hashes.items():
        x = tree.get(path)
        if x is None:
            res["extra_on_kaggle"].append(path)
            continue
        ok = (x["lfs"]["oid"] == h["sha256"]) if "lfs" in x else (x["oid"] == h["git_sha1"])
        res["match" if ok else "mismatch"].append(path)
    res["missing_on_kaggle"] = sorted(set(tree) - set(hashes))
    return res


def pypi_sha(fname):
    m = re.match(r"^([A-Za-z0-9_.]+)-([^-]+)-", fname)
    if not m:
        return None, "unparsed"
    name, ver = m.group(1), m.group(2)
    try:
        d = json.loads(get(f"https://pypi.org/pypi/{name}/{ver}/json"))
    except Exception as exc:
        return None, f"pypi error {type(exc).__name__}"
    for u in d.get("urls", []):
        if u["filename"] == fname:
            return u["digests"]["sha256"], "pypi"
    return None, "not on pypi"


def torch_index_sha(fname):
    name = fname.split("-")[0].lower().replace("_", "-")
    for idx in ["cu128", "cu126", "cu129", "cu130"]:
        try:
            page = get(f"https://download.pytorch.org/whl/{idx}/{name}/")
        except Exception:
            continue
        m = re.search(re.escape(fname.replace("+", "%2B")) + r"#sha256=([0-9a-f]{64})", page) or \
            re.search(re.escape(fname) + r"#sha256=([0-9a-f]{64})", page)
        if m:
            return m.group(1), f"pytorch/{idx}"
    return None, "not on pytorch index"


def check_wheels(hashes):
    res = {"match": [], "mismatch": [], "unverified": [], "non_wheel": []}
    for path, h in sorted(hashes.items()):
        fname = Path(path).name
        if not fname.endswith(".whl"):
            res["non_wheel"].append(path)
            continue
        sha, src = pypi_sha(fname)
        if sha is None:
            sha, src2 = torch_index_sha(fname)
            src = src2 if sha else f"{src}; {src2}"
        if sha is None:
            res["unverified"].append([fname, src])
        elif sha == h["sha256"]:
            res["match"].append([fname, src])
        else:
            res["mismatch"].append([fname, src])
    return res


if __name__ == "__main__":
    hashes = json.load(open(sys.argv[1]))
    rep = {"model": check_model(hashes["model"]), "wheels": check_wheels(hashes["wheels"])}
    json.dump(rep, open(HERE / "report.json", "w"), indent=1)
    for k, v in rep.items():
        print(k, {kk: len(vv) for kk, vv in v.items()})
    print("model mismatches:", rep["model"]["mismatch"], "extra:", rep["model"]["extra_on_kaggle"],
          "missing:", rep["model"]["missing_on_kaggle"])
    print("wheel mismatches:", rep["wheels"]["mismatch"])
    print("unverified wheels:", rep["wheels"]["unverified"])
