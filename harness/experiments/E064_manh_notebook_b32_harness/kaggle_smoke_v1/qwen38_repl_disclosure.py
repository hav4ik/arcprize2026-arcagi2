"""Label-blind staged test-input disclosure. No test answers or model calls here."""

from copy import deepcopy


MODE = "last-first"


def initial_view(task, mode="all"):
    visible = deepcopy({"train": [{"input": p["input"], "output": p["output"]} for p in task["train"]],
                        "test": [{"input": p["input"]} for p in task["test"]]})
    indices = list(range(len(visible["test"])))
    if mode == MODE and len(indices) > 1:
        indices = indices[-1:]
        visible["test"] = [visible["test"][-1]]
    return visible, indices


def initial_note(task, mode="all"):
    if mode != MODE or len(task["test"]) <= 1:
        return ""
    return (f"\n\n[staged test inputs] Start with only the LAST original test input "
            f"({len(task['test'])-1}), available as test_inputs[0]. All training pairs are visible. "
            "Other test inputs are withheld from text, images and Python until disclosure. "
            "Find one general solve(grid), check ALL training pairs and submit it when ready. "
            "Before submitting, inspect whether its assumptions also apply to the shown test input; "
            "executing a function successfully is not evidence that its test answer is correct. "
            "Briefly state the rule and unresolved assumptions; no special note format is required. "
            "A saved training-passing solver triggers the next input, not a correctness verdict. "
            "Later inputs are appended to test_inputs without changing existing indices. "
            "Every stage shares the SAME task and review token/time limits. Test answers are unknown.")


def original_order(record):
    """Snapshot publication uses original test indices, even before all inputs are revealed."""
    if not record.get("test_disclosure"):
        return record
    result = deepcopy(record)
    total = result["test_disclosure"]["total"]
    for entry in [*result["submissions"], *([result["fallback"]] if result.get("fallback") else [])]:
        indices = entry.pop("test_indices")
        grids = [None] * total
        for index, grid in zip(indices, entry["test"], strict=True):
            grids[index] = grid
        entry["test"] = grids
        entry["revealed_test_indices"] = indices
        if "test_at_submission" in entry:
            initial = [None] * total
            for index, grid in zip(entry["submission_test_indices"], entry["test_at_submission"], strict=True):
                initial[index] = grid
            entry["test_at_submission"] = initial
    result["test_disclosure"]["prediction_order"] = "original"
    return result


class Disclosure:
    def __init__(self, task):
        self.full, _ = initial_view(task)
        self.visible, self.indices = initial_view(task, MODE)
        self.events = []

    @property
    def pending(self):
        return [i for i in range(len(self.full["test"])) if i not in self.indices]

    def state(self):
        return dict(mode=MODE, total=len(self.full["test"]), visible_indices=list(self.indices),
                    events=deepcopy(self.events), prediction_order="visible")

    def advance(self, *, all_remaining=False):
        new = self.pending if all_remaining else self.pending[:1]
        for index in new:
            self.indices.append(index)
            self.visible["test"].append(deepcopy(self.full["test"][index]))
        return new
