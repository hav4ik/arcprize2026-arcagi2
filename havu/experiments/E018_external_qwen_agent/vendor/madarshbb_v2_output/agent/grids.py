"""Grid <-> text, task prompts and failure feedback for the program-synthesis agent.

Grids are shown as rows of digits with no separators, the same layout NVARC's
formatter uses, plus explicit dimensions -- reasoning models miscount rows and
columns of long digit strings otherwise.
"""

SYSTEM = (
    "You are an expert at solving ARC-AGI puzzles by writing Python programs. "
    "Each puzzle has a few input->output example pairs that share one hidden "
    "transformation rule. Grids are 2D lists of integers 0-9 (colours); 0 is "
    "usually the background."
)

INSTRUCTIONS = """Find the transformation rule that maps EVERY example input to its output, then write it as a Python function.

Requirements:
- Define `def transform(grid: list[list[int]]) -> list[list[int]]:`.
- Use only the Python standard library and numpy (already importable as `np` if you `import numpy as np`).
- The function must work for every example input AND the test input(s); do not hard-code example outputs.
- Return a new 2D list of ints 0-9 (numpy arrays are accepted).

Reason about the rule first, then give the final program in ONE ```python code block."""


def grid_text(g):
    return "\n".join("".join(str(int(v)) for v in row) for row in g)


def shape(g):
    return (len(g), len(g[0]) if g else 0)


def show(g, name):
    r, c = shape(g)
    return f"{name} ({r}x{c}):\n{grid_text(g)}"


def task_prompt(task):
    parts = [INSTRUCTIONS, ""]
    for i, pair in enumerate(task["train"], 1):
        parts.append(f"## Example {i}")
        parts.append(show(pair["input"], "Input"))
        parts.append(show(pair["output"], "Output"))
        parts.append("")
    for j, t in enumerate(task["test"], 1):
        parts.append(f"## Test input {j}")
        parts.append(show(t["input"], "Input"))
        parts.append("")
    return "\n".join(parts).rstrip()


def diff_text(expected, got, max_cells=900):
    """Expected grid with wrong cells marked, so the model sees WHERE it failed.

    '.' marks a correct cell; a wrong cell shows `expected/got` inline is too
    wide, so two aligned grids are given: the expected output, and the
    program's output with correct cells replaced by '.'.
    """
    re_, ce = shape(expected)
    rg, cg = shape(got)
    if (re_, ce) != (rg, cg):
        return (f"Wrong output SIZE: expected {re_}x{ce}, your program returned {rg}x{cg}.\n"
                + show(expected, "Expected") + "\n" + show(got, "Your output"))
    if re_ * ce > max_cells:
        wrong = sum(expected[r][c] != got[r][c] for r in range(re_) for c in range(ce))
        return f"{wrong} of {re_ * ce} cells are wrong (grid too large to print the diff)."
    wrong = 0
    rows = []
    for r in range(re_):
        s = []
        for c in range(ce):
            if expected[r][c] == got[r][c]:
                s.append(".")
            else:
                s.append(str(int(got[r][c])))
                wrong += 1
        rows.append("".join(s))
    return (f"{wrong} of {re_ * ce} cells are wrong.\n"
            + show(expected, "Expected") + "\n"
            + f"Your output, with CORRECT cells shown as '.' ({re_}x{ce}):\n" + "\n".join(rows))


def feedback_prompt(results, train, max_examples=3):
    """Tell the model which examples failed and how; ask for a corrected program."""
    lines = []
    n_ok = sum(r["ok"] for r in results)
    lines.append(f"Your program reproduced {n_ok} of {len(results)} examples.")
    shown = 0
    for i, (r, pair) in enumerate(zip(results, train), 1):
        if r["ok"]:
            continue
        if shown >= max_examples:
            break
        shown += 1
        lines.append(f"\n## Example {i} failed")
        if r.get("error"):
            lines.append(f"Error: {r['error']}")
        else:
            lines.append(diff_text(pair["output"], r["output"]))
    lines.append("\nRe-examine the rule against ALL examples, then give the corrected "
                 "complete program in ONE ```python code block.")
    return "\n".join(lines)
