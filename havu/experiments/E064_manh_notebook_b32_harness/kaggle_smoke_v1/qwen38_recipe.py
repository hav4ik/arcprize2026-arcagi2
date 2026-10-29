"""Opt-in solver recipes shared by the standalone and 4B cascade entrypoints.

These are configuration presets, not a second solver or a task-specific router.
No experiment artifacts, solutions, model weights or remote services are read.
The serving/kernel profile and the 4B stage remain caller-controlled.
"""

from copy import deepcopy
import sys

from qwen38_flash_profile import ProfileParser as ServerProfileParser
import qwen38_flash_profile as flash_profile


VERSION = 5
NAMES = ("manual", "vision-repl", "vision-repl-audit", "vision-repl-execution", "vision-repl-interactions",
         "vision-repl-search", "vision-repl-staged")
COMMON = dict(
    solve_mode="tir", tir_interface="repl", input_mode="text-image",
    context=128000, concurrency=20,
    task_token_budget=70000, tir_repl_budget=70000, tir_repl_trajectories=1,
    tir_repl_turn_tokens=40000, tir_repl_think_tokens=8192,
    tir_repl_min_turn=12000, tir_repl_wrap_up=20000,
    tir_repl_keep_reasoning=-1, tir_repl_progress_notes="optional",
    tir_repl_context_policy="legacy", tir_repl_require_train_pass=True,
    tir_repl_review=True, tir_repl_visual_tools=True,
    vision_layout="pairs-plain", vision_cell_pixels=24, vision_max_images=16,
    tir_repl_finalize_reserve=12000, tir_repl_finalize_think_tokens=2048,
    tir_repl_finalize_turn_tokens=6000, tir_repl_solver_by_tokens=12000,
)
AUDIT = dict(
    tir_repl_input_audit=True, tir_repl_candidate_pool=True,
    tir_repl_review_token_budget=8000, tir_repl_review_seconds=600,
    tir_repl_finalize_reserve_seconds=720,
)
SEARCH = dict(tir_repl_input_audit=False, tir_repl_targeted_review=True,
              tir_repl_solver_max_turns=2, tir_repl_solver_checkpoint_action="resume",
              tir_repl_auto_check_solver=True)


def defaults(name, *, flash_next=False):
    if name not in NAMES:
        raise ValueError(f"Unknown Qwen3.8 recipe: {name}")
    if name == "manual":
        return {}
    return deepcopy({**COMMON,
                     **(flash_profile.CAPACITY_DEFAULTS if flash_next else {}),
                     **(AUDIT if name in NAMES[2:] else {}),
                     **({"tir_repl_execution_audit": True} if name == "vision-repl-execution" else {}),
                     **({"tir_repl_interaction_audit": True} if name == "vision-repl-interactions" else {}),
                     **(SEARCH if name in ("vision-repl-search", "vision-repl-staged") else {}),
                     **({"tir_repl_test_reveal": "last-first"} if name == "vision-repl-staged" else {})})


class RecipeParser(ServerProfileParser):
    """Explicit CLI values win in either order; parser reuse never leaks defaults."""

    def parse_args(self, args=None, namespace=None):
        argv = list(sys.argv[1:] if args is None else args)
        explicit = set(vars(namespace)) if namespace is not None else set()
        result = super().parse_args(argv, namespace)
        # Solver recipes select the prompting/tool policy. The serving profile
        # owns capacity defaults; do not replace FN's 102400/B24 with 128000/B20.
        values = defaults(getattr(result, "qwen38_recipe", "manual"),
                          flash_next=flash_profile.active(result))
        if not values:
            return result
        for token in argv:
            option = token.split("=", 1)[0]
            action = self._option_string_actions.get(option)
            if action is None and self.allow_abbrev and option.startswith("--"):
                matches = [a for name, a in self._option_string_actions.items()
                           if name.startswith(option)]
                if matches and len({a.dest for a in matches}) == 1:
                    action = matches[0]
            if action is not None:
                explicit.add(action.dest)
        for field, value in values.items():
            if field not in explicit:
                setattr(result, field, value)
        # A single task-budget override should not retain an invisible 70K
        # trajectory ceiling (or an invisible 70K task ceiling). Both may still
        # be set explicitly for multiple-trajectory experiments.
        task, trajectory = "task_token_budget", "tir_repl_budget"
        if task in explicit and trajectory not in explicit:
            result.tir_repl_budget = result.task_token_budget
        elif trajectory in explicit and task not in explicit:
            result.task_token_budget = result.tir_repl_budget
        return result


def add_arguments(parser):
    parser.add_argument("--qwen38-recipe", choices=NAMES, default="manual",
        help="Opt-in multimodal solver preset (27B, Flash-Next or dsv4). vision-repl: retained reasoning, PNGs, visual Python tools, "
             "soft solver checkpoint and token-reserved finalization. vision-repl-audit: also input "
             "audits, alternative candidates and bounded review/wall finalization (experimental). "
             "vision-repl-execution: saved-code state/effect checks and mechanism challenge instead of "
             "the required raw-input audit (experimental, no demonstrated accuracy gain). "
             "vision-repl-interactions: bounded coverage/operation-interaction review of EVERY test input "
             "instead of the required raw-input audit (experimental, regressed in the matched screen). "
             "vision-repl-search: executable-search reference with bounded targeted review and automatic "
             "training checks. vision-repl-staged: same reference, last test input first, then sequential "
             "disclosure (untested on GPU; no extra budget). "
             "Defaults: context 128000, concurrency 20, 70000 generated tokens/task; the FP4-Local serving "
             "profile keeps its context 102400/concurrency 24 defaults. Explicit flags win; "
             "manual preserves existing behavior. No task subset, deadline or serving environment is set.")


def validate(args):
    name = getattr(args, "qwen38_recipe", "manual")
    if name == "manual":
        return
    defaults(name)  # Reject unknown programmatic values too.
    # dsv4: the same REPL loop on DeepSeek-V4.1 (llama.cpp with its mmproj); dsv4-vllm on our vLLM fork.
    for field, expected in (("qwen38_variant", ("27b", "flash-next", "dsv4", "dsv4-vllm")), ("solve_mode", ("tir",)),
                            ("tir_interface", ("repl",)), ("input_mode", ("text-image",))):
        if getattr(args, field) not in expected:
            raise ValueError(f"--qwen38-recipe {name} requires {field} in {expected!r}; "
                             "use --qwen38-recipe manual for a different solver/backend")


def describe(args):
    """Record effective preset values and deviations, never labels or secrets."""
    name = getattr(args, "qwen38_recipe", "manual")
    values = defaults(name, flash_next=flash_profile.active(args))
    effective = {field: getattr(args, field) for field in values}
    return dict(name=name, version=VERSION if values else None,
                effective=effective,
                overrides={field: value for field, value in effective.items() if value != values[field]},
                validation=("experimental staged test disclosure; local correctness only, not promoted"
                            if name == "vision-repl-staged" else
                            "small-cohort executable-search reference; full/private validation still required"
                            if name == "vision-repl-search" else
                            "experimental per-input interaction audit; matched screen regressed, not promoted"
                            if name == "vision-repl-interactions" else
                            "experimental execution audit; no demonstrated accuracy gain, not promoted"
                            if name == "vision-repl-execution" else
                            "experimental bounded-audit recipe; full/private validation still required"
                            if name == "vision-repl-audit" else
                            "completed small-cohort recipe; full/private validation still required"
                            if name == "vision-repl" else "caller-configured"))
