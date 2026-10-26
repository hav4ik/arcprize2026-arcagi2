"""Text/token utilities: identifier occurrences in code, edit distance, BPE spans in context."""
import io
import keyword
import re
import tokenize as pytok
from functools import lru_cache

from tokenizers import Tokenizer

from common import TOKENIZER

KW = set(keyword.kwlist) | {"True", "False", "None"}
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TK = None


def tk():
    global _TK
    if _TK is None:
        _TK = Tokenizer.from_file(TOKENIZER)
    return _TK


def encode(text):
    e = tk().encode(text, add_special_tokens=False)
    return e.ids, e.offsets, e.tokens


def ident_occurrences(code):
    """[(start, end, name, kind)] of NAME tokens (keywords excluded). kind: 'attr' (after '.'), 'import'
    (in an import statement), 'kwarg' (name followed by '=' inside a call), else 'name'. Falls back to a regex
    over code with strings/comments blanked when tokenize fails."""
    lines = code.splitlines(keepends=True)
    starts = [0]
    for l in lines:
        starts.append(starts[-1] + len(l))
    out = []
    try:
        toks = list(pytok.generate_tokens(io.StringIO(code).readline))
        prev = None
        line_is_import = {}
        for t in toks:
            if t.type == pytok.NAME and t.string in ("import", "from"):
                line_is_import[t.start[0]] = True
        for i, t in enumerate(toks):
            if t.type == pytok.NAME and t.string not in KW and t.start[0] - 1 < len(lines):
                s = starts[t.start[0] - 1] + t.start[1]
                nxt = toks[i + 1].string if i + 1 < len(toks) else ""
                if prev is not None and prev.string == ".":
                    kind = "attr"
                elif line_is_import.get(t.start[0]):
                    kind = "import"
                elif nxt == "=" and prev is not None and prev.string in ("(", ","):
                    kind = "kwarg"
                else:
                    kind = "name"
                out.append((s, s + len(t.string), t.string, kind))
            if t.type not in (pytok.NL, pytok.NEWLINE, pytok.INDENT, pytok.DEDENT, pytok.COMMENT):
                prev = t
        return out
    except Exception:
        pass
    blank = re.sub(r"#[^\n]*", lambda m: " " * len(m.group()), code)
    blank = re.sub(r"'[^'\n]*'|\"[^\"\n]*\"", lambda m: " " * len(m.group()), blank)
    for m in IDENT.finditer(blank):
        if m.group() in KW:
            continue
        s = m.start()
        j = s - 1
        while j >= 0 and blank[j] == " ":
            j -= 1
        kind = "attr" if j >= 0 and blank[j] == "." else "name"
        out.append((s, m.end(), m.group(), kind))
    return out


def osa(a, b):
    """Optimal-string-alignment (restricted Damerau-Levenshtein) distance."""
    la, lb = len(a), len(b)
    if a == b:
        return 0
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def edit_type(t, i):
    """Character-level edit class of typo t vs intended i."""
    if t == i:
        return "same"
    if i.startswith(t):
        return "truncation" if len(i) - len(t) >= 2 else "deletion"
    dist = osa(t, i)
    if dist == 1:
        if len(t) == len(i) + 1:
            # doubled char?
            for k in range(len(t)):
                if t[:k] + t[k + 1:] == i:
                    if (k > 0 and t[k] == t[k - 1]) or (k + 1 < len(t) and t[k] == t[k + 1]):
                        return "insertion(doubled)"
            return "insertion"
        if len(t) + 1 == len(i):
            return "deletion"
        if len(t) == len(i):
            diff = [k for k in range(len(t)) if t[k] != i[k]]
            return "transposition" if len(diff) == 2 else "substitution"
    # segment repeat: t = i[:j] + i[k:] with k < j
    for j in range(1, len(i) + 1):
        for k in range(0, j):
            if i[:j] + i[k:] == t:
                return "segment-repeat"
    # chunk deletion: t = i[:j] + i[k:], k > j + 1
    for j in range(0, len(i)):
        for k in range(j + 2, len(i) + 1):
            if i[:j] + i[k:] == t:
                return "chunk-deletion"
    # common prefix, then different tail
    cp = 0
    while cp < min(len(t), len(i)) and t[cp] == i[cp]:
        cp += 1
    if cp >= 1:
        return "tail-replaced"
    return f"multi({dist})"


def first_diff(t, i):
    k = 0
    while k < min(len(t), len(i)) and t[k] == i[k]:
        k += 1
    return k


@lru_cache(maxsize=200000)
def word_tokens_cached(text):
    return tuple(tk().encode(text, add_special_tokens=False).tokens)


def span_tokens(ids, offsets, toks, s, e):
    """Indices of tokens overlapping char span [s, e)."""
    return [k for k, (a, b) in enumerate(offsets) if a < e and b > s and b > a]


def token_event(typo, intended, i_toks):
    """Token-level event class for a typo, given the intended's canonical tokens in context.
    Token strings: leading non-identifier characters of the first token are stripped ('(m' -> 'm', 'Ġcomp' -> 'comp').
    Returns (event, j) with j = index of the first token that is not reproduced."""
    pieces = [t.replace("Ġ", " ").replace("Ċ", "\n") for t in i_toks]
    # strip to the intended identifier's characters
    s = "".join(pieces)
    off = s.find(intended)
    if off < 0:
        return "unaligned", None
    cum, bounds = 0, []
    for p in pieces:
        bounds.append((max(cum, off) - off, min(cum + len(p), off + len(intended)) - off))
        cum += len(p)
    toks = [intended[a:b] for a, b in bounds if b > a]
    # j = number of whole tokens reproduced as a prefix of the typo
    j, pre = 0, ""
    while j < len(toks) and typo.startswith(pre + toks[j]):
        pre += toks[j]
        j += 1
    if j == len(toks):
        return ("extended-after-last-token" if len(typo) > len(intended) else "same"), j
    if j == 0:
        return "first-token-wrong", 0
    rest_i = "".join(toks[j:])
    x = typo[len(pre):]
    if x == "":
        return "truncated-at-boundary", j
    # dropped whole tokens (1-3)
    for k in range(1, 4):
        if j + k <= len(toks) and x == "".join(toks[j + k:]) and x != rest_i:
            return f"dropped-{k}-token(s)", j
    # restart: x = (suffix of pre) + rest_i
    for L in range(1, len(pre) + 1):
        if x == pre[-L:] + rest_i:
            return "restart(repeat-emitted-chars)", j
        if x.startswith(pre[-L:]) and x[L:] and rest_i.startswith(x[L:][:1]) and len(x) - L >= len(rest_i) - 1 \
                and pre[-L:] + rest_i[:1] == x[:L + 1]:
            pass
    if x == "":
        return "truncated-at-boundary", j
    if rest_i.startswith(x):
        return "continuation-truncated", j
    if rest_i.endswith(x):
        return "continuation-lost-leading-chars", j
    if x.startswith(rest_i):
        return "continuation-extended", j
    if x and rest_i and x[0] == rest_i[0]:
        return "continuation-replaced(same 1st char)", j
    return "continuation-replaced", j
