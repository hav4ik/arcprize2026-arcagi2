"""Everything the model reads: the opening message, tool descriptions and fixed instructions.

Kept byte-identical to harness persistent-v3 so results stay comparable across refactors.
"""
from .arcprize import build_prompt

# Optional suffixes appended to the ARC Prize prompt (--prompt). "plain" keeps the original prompt.
PROMPT_SUFFIXES = {
    "plain": "",
    "lean": ("\n\nYou have a Python tool. Work efficiently: inspect the grids with code instead of reading them "
             "by hand, form a hypothesis, and write code that computes the output. Never construct, copy or count "
             "grids by hand. Check your rule on every training pair with code before answering, and if a pair "
             "fails, fix the rule rather than explaining the mismatch away. Keep your reasoning brief."),
    "lean2": ("\n\nYou have a Python tool. The variables `train` (list of {'input', 'output'} grids) and "
              "`test_input` are already loaded in it: use them directly and never retype or reassign them. "
              "Work efficiently: inspect the grids with code instead of reading them by hand, form a hypothesis, "
              "and write code that computes the output. Never construct, copy or count grids by hand. Check your "
              "rule on every training pair with code before answering, and if a pair fails, fix the rule rather "
              "than explaining the mismatch away. Before answering, check that the rule's assumptions also hold "
              "on test_input. Keep your reasoning brief."),
}
CODE_MODE_SUFFIX = "\n\nUse run_python to investigate; submit self-contained transform(grid) code with submit_solution."
FINALIZE = "Finalize now. Return only your best test output as a JSON grid."
MODES = ("grid", "code")


def initial_messages(train, test_input, prompt, mode):
    content = build_prompt(train, test_input) + PROMPT_SUFFIXES[prompt]
    if mode == "code":
        content += CODE_MODE_SUFFIX
    return [{"role": "user", "content": content}]


def tool_specs(mode, cell_timeout):
    descriptions = {"run_python": (
        "Execute a Python cell in your persistent IPython session. Variables/functions persist. "
        "Preloaded: train (list of {'input': grid, 'output': grid}), test_input (grid), numpy as np. "
        f"scipy is available. No internet or host files. Timeout {cell_timeout}s. "
        "Use print() to inspect results. Check transformations on all training pairs."
    )}
    if mode == "code":
        descriptions["submit_solution"] = (
            "Submit self-contained Python code defining transform(grid), which returns a rectangular "
            "grid of ints 0-9 (or numpy array). Include imports/helpers; execution uses a fresh namespace. "
            "The harness checks every training example. Failures return feedback so you can revise. "
            "If all pass, it applies transform to test_input and ends this attempt."
        )
    return [{"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                       "required": ["code"], "additionalProperties": False},
    }} for name, description in descriptions.items()]
