#!/usr/bin/env python3
"""Cell-by-cell diff of a built V12 + images notebook against V12 (every changed line; the cell 0b overlay and the
cell 1b INTEGRITY blobs are summarised: they are checked by sha256 / file list instead).
usage: v12img_diff.py [NEW.ipynb] [V12.ipynb]"""
import difflib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = HERE / "out/nb_v12_img/dsv41-tools-v12-img.ipynb"
V12 = HERE / "out/manh_v12_363638159/dsv41-tools-submit-b32m-external-server.ipynb"


def src(c):
    return c["source"] if isinstance(c["source"], str) else "".join(c["source"])


def short(line):
    if line.startswith('_MT = "'):
        return f'_MT = "<base64 tar.gz overlay, {len(line)} chars>"'
    m = re.match(r'(INTEGRITY = json\.loads\(r""")(.*)("""\).*)', line)
    if m:
        d = json.loads(m.group(2))["model"]
        return f"{m.group(1)}<{len(d)} files, {sum(v[0] for v in d.values()) / 1e9:.2f} GB>{m.group(3)}"
    return line


def main():
    new = json.load(open(sys.argv[1] if len(sys.argv) > 1 else NEW))["cells"]
    old = json.load(open(sys.argv[2] if len(sys.argv) > 2 else V12))["cells"]
    key = lambda c: src(c).splitlines()[0][:40] if src(c) else ""  # noqa: E731
    sm = difflib.SequenceMatcher(a=[key(c) for c in old], b=[key(c) for c in new], autojunk=False)
    same = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):  # replace: a changed first line
            for i, j in zip(range(i1, i2), range(j1, j2)):
                a, b = src(old[i]).splitlines(), src(new[j]).splitlines()
                if a == b:
                    same += 1
                    continue
                print(f"\n### V12 cell {i} -> new cell {j}: {key(old[i])}")
                for ln in difflib.unified_diff([short(x) for x in a], [short(x) for x in b], lineterm="", n=0):
                    if not ln.startswith(("---", "+++")):
                        print(ln)
        else:
            for i in range(i1, i2):
                print(f"\n### V12 cell {i} removed: {key(old[i])}")
            for j in range(j1, j2):
                print(f"\n### new cell {j} (added, {len(src(new[j]).splitlines())} lines): {key(new[j])}")
    print(f"\n{same} of {len(old)} V12 cells unchanged")


if __name__ == "__main__":
    main()
