"""E033: parse a grid written as rows of digits (the image-prompt arm answers this way, since its training outputs are
images rather than JSON). Takes the last block of consecutive lines that each hold only digits 0-9 (space-separated
or not) and that all have the same width. Falls back to nothing (None) when no such block exists.
"""
import re

_ROW = re.compile(r"^\s*(?:\d\s*)+$")


def _row(line):
    s = line.strip().strip("`").strip()
    if not s or not _ROW.match(s):
        return None
    tokens = s.split()
    if len(tokens) == 1 and len(tokens[0]) > 1:  # "0123" style
        tokens = list(tokens[0])
    if not all(len(t) == 1 for t in tokens):
        return None
    return [int(t) for t in tokens]


def parse_digit_rows(text):
    blocks, cur = [], []
    for line in (text or "").splitlines():
        r = _row(line)
        if r is not None:
            cur.append(r)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    for b in reversed(blocks):
        if len({len(r) for r in b}) == 1:
            return b
    return None
