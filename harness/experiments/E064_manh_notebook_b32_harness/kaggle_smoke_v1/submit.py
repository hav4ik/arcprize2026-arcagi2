# need run below command first
# !UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12 python starter.py --end-time {global_end_time}

import argparse
import json
import os

import numpy as np

from arc_decoder import (
    DEFAULT_SELECTION_ALGORITHM,
    SELECTION_ALGORITHMS,
    ArcDecoder,
)
from arc_loader import ArcDataset, train_prediction_key
from early_routing import early_routed_task_keys
from confidence_routing import (
    CASCADE_MERGE_POLICIES,
    merge_routed_predictions,
    print_cascade_merge_summary,
    print_routing_simulation_summary,
    route_tasks_by_confidence,
    zero_routed_submission_tasks,
)
from output_nll_report import (
    find_output_nll_entry,
    load_output_nll_reports,
)
from run_config import (
    resolve_data_challenge_path,
    resolve_evaluation_solution_path,
    resolve_train_prediction_output_dir,
    select_task_keys,
)
from run_status import (
    STATUS_SUCCESS,
    extract_run_status_path,
    run_entrypoint_with_status,
)


ARC_COLORS = (
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
)


def coordinate_ticks(length):
    """Keep coordinates readable for both tiny outputs and 30x30 inputs."""
    if length <= 12:
        step = 1
    elif length <= 20:
        step = 2
    else:
        step = 5
    return np.arange(0, length, step)


def draw_arc_grid(
    ax,
    grid,
    title,
    cmap,
    norm,
    border_color=None,
    mismatch_mask=None,
):
    grid = np.asarray(grid, dtype=np.uint8)
    if grid.ndim != 2:
        ax.axis("off")
        ax.set_title(f"{title}\ninvalid shape {grid.shape}")
        return

    height, width = grid.shape
    ax.imshow(
        grid,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        origin="upper",
    )
    ax.set_title(f"{title}\n{height}x{width}", fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.set_xticks(coordinate_ticks(width))
    ax.set_yticks(coordinate_ticks(height))
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.55, alpha=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", labelsize=8)

    if mismatch_mask is not None and 0: # ignore
        mismatch_y, mismatch_x = np.where(mismatch_mask)
        # A black underlay keeps the white X visible on every ARC color.
        ax.scatter(
            mismatch_x,
            mismatch_y,
            marker="x",
            s=58,
            c="black",
            linewidths=3.2,
            zorder=3,
        )
        ax.scatter(
            mismatch_x,
            mismatch_y,
            marker="x",
            s=58,
            c="white",
            linewidths=1.4,
            zorder=4,
        )

    if border_color is not None:
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(3)


def attempt_title(attempt_name, guess, truth):
    label = attempt_name.replace("_", " ").title()
    if truth is None:
        return label, None

    guess = np.asarray(guess)
    truth = np.asarray(truth)
    if guess.shape != truth.shape:
        return f"{label} — wrong shape", "#FF4136"
    mismatched_cells = int(np.count_nonzero(guess != truth))
    if mismatched_cells == 0:
        return f"{label} — EXACT", "#2ECC40"
    return f"{label} — {mismatched_cells} cells wrong", "#FF4136"


def output_nll_text(entry):
    """Format one canonical total/per-token NLL for logs and figure titles."""
    if entry is None:
        return "NLL unavailable"
    if entry.get("status") != "ok" or entry.get("total_nll") is None:
        return f"NLL unavailable ({entry.get('status', 'unknown')})"
    return (
        f"NLL {float(entry['total_nll']):.5f} · "
        f"{float(entry['mean_nll']):.6f}/token"
    )


def print_output_nll_summary(
    reports,
    submission,
    selected_keys,
):
    """Print canonical NLLs for ground truth and the selected two attempts."""
    if not reports:
        return 0
    print(
        "*** Canonical output NLLs "
        "(total sequence NLL · mean NLL/token):"
    )
    output_count = 0
    for task_key in selected_keys:
        report = reports.get(task_key)
        if report is None:
            continue
        for test_index, attempts in enumerate(submission[task_key]):
            base_key = f"{task_key}_{test_index}"
            report_output = report.get("outputs", {}).get(base_key, {})
            truth = report_output.get("ground_truth")
            truth_entry = (
                find_output_nll_entry(report, base_key, truth)
                if truth is not None
                else None
            )
            fields = [
                "ground_truth="
                + (
                    output_nll_text(truth_entry)
                    if truth is not None
                    else "unavailable"
                )
            ]
            for attempt_name in ("attempt_1", "attempt_2"):
                grid = attempts[attempt_name]
                fields.append(
                    f"{attempt_name}="
                    f"{output_nll_text(find_output_nll_entry(report, base_key, grid))}"
                )
            print(f"  {base_key}: " + "; ".join(fields))
            output_count += 1
    return output_count


def visualize_submission(
    data,
    submission,
    selected_keys,
    train_predictions=None,
    output_nll_reports=None,
):
    """Display demonstrations and test predictions as task worksheets."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import colors

    backend = matplotlib.get_backend()
    if "inline" not in backend.lower():
        print(
            f"*** Matplotlib backend is {backend!r}. For inline notebook "
            "figures, run `%run submit.py --visualize` instead of "
            "`!python submit.py --visualize`."
        )

    cmap = colors.ListedColormap(ARC_COLORS)
    norm = colors.BoundaryNorm(np.arange(-0.5, 10.5, 1), cmap.N)
    displayed_count = 0

    for task_key in selected_keys:
        training_examples = data.queries[task_key]["train"]
        test_cases = data.queries[task_key]["test"]
        known_outputs = data.replies.get(task_key)
        output_nll_report = (
            output_nll_reports.get(task_key)
            if output_nll_reports is not None
            else None
        )

        row_count = len(training_examples) + len(test_cases)
        figure = plt.figure(
            figsize=(18.6, 3.6 * row_count + 0.8),
            constrained_layout=True,
        )
        grid_spec = figure.add_gridspec(
            row_count,
            1,
            hspace=0.5,
        )

        for row, example in enumerate(training_examples):
            diagnostic_key = train_prediction_key(task_key, row)
            diagnostic_guesses = (
                list(train_predictions.get(diagnostic_key, []))
                if train_predictions is not None
                else []
            )
            if train_predictions is None:
                training_grid = grid_spec[row, 0].subgridspec(
                    1,
                    3,
                    width_ratios=(1, 0.08, 1),
                    wspace=0.12,
                )
                input_ax = figure.add_subplot(training_grid[0, 0])
                arrow_ax = figure.add_subplot(training_grid[0, 1])
                output_ax = figure.add_subplot(training_grid[0, 2])
            else:
                training_grid = grid_spec[row, 0].subgridspec(
                    1,
                    4,
                    wspace=0.35,
                )
                input_ax = figure.add_subplot(training_grid[0, 0])
                output_ax = figure.add_subplot(training_grid[0, 1])
            draw_arc_grid(
                input_ax,
                example["input"],
                f"Training example {row + 1} — input",
                cmap,
                norm,
            )
            draw_arc_grid(
                output_ax,
                example["output"],
                f"Training example {row + 1} — output",
                cmap,
                norm,
                border_color="#2ECC40" if train_predictions is not None else None,
            )
            if train_predictions is None:
                arrow_ax.axis("off")
                arrow_ax.text(
                    0.5,
                    0.5,
                    "→",
                    ha="center",
                    va="center",
                    fontsize=24,
                    fontweight="bold",
                )
            else:
                if diagnostic_guesses and len(diagnostic_guesses) < 2:
                    diagnostic_guesses.append(diagnostic_guesses[-1])
                while len(diagnostic_guesses) < 2:
                    diagnostic_guesses.append(np.asarray([[0]]))
                truth = np.asarray(example["output"], dtype=np.uint8)
                for column, (attempt_name, guess) in enumerate(
                    zip(
                        ("attempt_1", "attempt_2"),
                        diagnostic_guesses[:2],
                    ),
                    start=2,
                ):
                    guess = np.asarray(guess, dtype=np.uint8)
                    title, border_color = attempt_title(
                        attempt_name,
                        guess,
                        truth,
                    )
                    mismatch_mask = (
                        guess != truth if guess.shape == truth.shape else None
                    )
                    prediction_ax = figure.add_subplot(
                        training_grid[0, column]
                    )
                    draw_arc_grid(
                        prediction_ax,
                        guess,
                        f"Training example {row + 1} — {title}",
                        cmap,
                        norm,
                        border_color=border_color,
                        mismatch_mask=mismatch_mask,
                    )

        for test_index, test_case in enumerate(test_cases):
            base_key = f"{task_key}_{test_index}"
            report_output = (
                output_nll_report.get("outputs", {}).get(base_key, {})
                if output_nll_report is not None
                else {}
            )
            truth = (
                np.asarray(known_outputs[test_index], dtype=np.uint8)
                if known_outputs is not None
                else (
                    np.asarray(
                        report_output["ground_truth"],
                        dtype=np.uint8,
                    )
                    if report_output.get("ground_truth") is not None
                    else None
                )
            )
            test_row = len(training_examples) + test_index
            test_grid = grid_spec[test_row, 0].subgridspec(
                1,
                4,
                wspace=0.35,
            )
            test_axes = [
                figure.add_subplot(test_grid[0, column])
                for column in range(4)
            ]
            draw_arc_grid(
                test_axes[0],
                test_case["input"],
                f"Test {test_index + 1} — input",
                cmap,
                norm,
            )
            if truth is None:
                test_axes[1].axis("off")
                test_axes[1].text(
                    0.5,
                    0.5,
                    "Ground truth\nunavailable",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
            else:
                truth_title = f"Test {test_index + 1} — ground truth"
                if output_nll_report is not None:
                    truth_title += "\n" + output_nll_text(
                        find_output_nll_entry(
                            output_nll_report,
                            base_key,
                            truth,
                        )
                    )
                draw_arc_grid(
                    test_axes[1],
                    truth,
                    truth_title,
                    cmap,
                    norm,
                    border_color="#2ECC40",
                )

            for column, attempt_name in enumerate(
                ("attempt_1", "attempt_2"),
                start=2,
            ):
                guess = np.asarray(
                    submission[task_key][test_index][attempt_name],
                    dtype=np.uint8,
                )
                title, border_color = attempt_title(
                    attempt_name,
                    guess,
                    truth,
                )
                mismatch_mask = (
                    guess != truth
                    if truth is not None and guess.shape == truth.shape
                    else None
                )
                prediction_title = f"Test {test_index + 1} — {title}"
                if output_nll_report is not None:
                    prediction_title += "\n" + output_nll_text(
                        find_output_nll_entry(
                            output_nll_report,
                            base_key,
                            guess,
                        )
                    )
                draw_arc_grid(
                    test_axes[column],
                    guess,
                    prediction_title,
                    cmap,
                    norm,
                    border_color=border_color,
                    mismatch_mask=mismatch_mask,
                )

        figure.suptitle(
            f"ARC task {task_key} — {len(test_cases)} test output(s)",
            fontsize=14,
            fontweight="bold",
        )
        plt.show()
        plt.close(figure)
        displayed_count += 1

    return displayed_count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-status-file",
        default=os.getenv("ARC_RUN_STATUS_PATH"),
        help=(
            "Update the atomic private-rerun status JSON after submission "
            "generation (environment: ARC_RUN_STATUS_PATH)."
        ),
    )
    parser.add_argument(
        "--selection-algorithm",
        choices=tuple(SELECTION_ALGORITHMS),
        default=os.getenv(
            "ARC_SELECTION_ALGORITHM",
            DEFAULT_SELECTION_ALGORITHM,
        ),
        help=(
            "Candidate-ranking formula used for attempt_1 and attempt_2 "
            f"(default: {DEFAULT_SELECTION_ALGORITHM}; environment: "
            "ARC_SELECTION_ALGORITHM)."
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Display training pairs, test input, ground truth, and submitted "
            "attempts as inline comparison sheets."
        ),
    )
    parser.add_argument(
        "--predict-train",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("ARC_PREDICT_TRAIN", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Load leave-one-out training predictions and include them in "
            "--visualize worksheets (default: disabled; environment: "
            "ARC_PREDICT_TRAIN=1/0)."
        ),
    )
    parser.add_argument(
        "--simulate-8b-routing-zero",
        action=argparse.BooleanOptionalAction,
        default=os.getenv(
            "ARC_SIMULATE_8B_ROUTING_ZERO",
            "0",
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Simulate the 4B-to-8B confidence router by replacing both "
            "attempts for every routed task with [[0]]. This changes only "
            "submission.json, not inference artifacts (default: disabled; "
            "environment: ARC_SIMULATE_8B_ROUTING_ZERO=1/0)."
        ),
    )
    parser.add_argument(
        "--routing-kgmon-threshold",
        type=float,
        default=float(os.getenv("ARC_ROUTING_KGMON_THRESHOLD", "-5")),
        help=(
            "Route a task when its minimum test-output KGMon confidence is "
            "at or below this value (default: -5; environment: "
            "ARC_ROUTING_KGMON_THRESHOLD)."
        ),
    )
    parser.add_argument(
        "--routing-beam-threshold",
        type=float,
        default=float(os.getenv("ARC_ROUTING_BEAM_THRESHOLD", "0.7")),
        help=(
            "Route a task when its maximum test-output top-candidate beam "
            "NLL is at or above this value (default: 0.7; environment: "
            "ARC_ROUTING_BEAM_THRESHOLD)."
        ),
    )
    parser.add_argument(
        "--cascade-8b-inference-output-dir",
        default=os.getenv("ARC_8B_INFERENCE_OUTPUT_DIR"),
        metavar="DIR",
        help=(
            "Load independently trained 8B candidates for tasks selected by "
            "the 4B confidence router, then merge them into the final two "
            "attempts. The primary 4B store remains ARC_INFERENCE_OUTPUT_DIR "
            "(environment: ARC_8B_INFERENCE_OUTPUT_DIR)."
        ),
    )
    parser.add_argument(
        "--cascade-merge-policy",
        choices=CASCADE_MERGE_POLICIES,
        default=os.getenv("ARC_CASCADE_MERGE_POLICY", "one-each"),
        help=(
            "Candidate priority for routed outputs: one-each preserves the "
            "top candidate from both models; prefer-8b or prefer-4b keeps "
            "both candidates from that model when available (default: "
            "one-each; environment: ARC_CASCADE_MERGE_POLICY)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        args.simulate_8b_routing_zero
        and args.cascade_8b_inference_output_dir
    ):
        raise ValueError(
            "--simulate-8b-routing-zero cannot be combined with a real "
            "--cascade-8b-inference-output-dir"
        )
    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        data = ArcDataset.from_file(resolve_data_challenge_path(rerun_mode))
    else:
        data = ArcDataset.from_file(resolve_data_challenge_path(rerun_mode))
        data = data.load_replies(resolve_evaluation_solution_path())

    decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)

    selected_keys = select_task_keys(data.queries.keys())
    selected_base_keys = {
        f"{selected_key}_{test_index}"
        for selected_key in selected_keys
        for test_index in range(len(data.queries[selected_key]["test"]))
    }
    print(f"Using predictions only for tasks: {selected_keys}")
    print(f"Using selection algorithm: {args.selection_algorithm}")

    inference_output_dir = os.getenv(
        "ARC_INFERENCE_OUTPUT_DIR",
        "/kaggle/inference_outputs",
    )
    early_skipped = early_routed_task_keys(inference_output_dir, data.queries, selected_keys)
    active_base_keys = {
        f"{key}_{index}" for key in selected_keys if key not in early_skipped
        for index in range(len(data.queries[key]["test"]))
    }
    if early_skipped:
        print(f"*** Early-routed tasks stay zero (ignoring stale 4B candidates): {sorted(early_skipped)}")
    decoder.load_decoded_results(
        inference_output_dir,
        allowed_base_keys=active_base_keys,
    )
    # Skips must count as unsolved, not disappear from the accuracy denominator.
    decoder.allowed_base_keys = selected_base_keys
    output_nll_reports = load_output_nll_reports(
        inference_output_dir,
        [key for key in selected_keys if key not in early_skipped],
    )

    train_predictions = None
    if args.predict_train:
        selected_data = data.change_keys(selected_keys)
        train_prediction_data = selected_data.split_train_replies()
        train_prediction_decoder = ArcDecoder(
            train_prediction_data,
            n_guesses=2,
        )
        train_prediction_output_dir = (
            resolve_train_prediction_output_dir(inference_output_dir)
        )
        if os.path.isdir(train_prediction_output_dir):
            train_prediction_decoder.load_decoded_results(
                train_prediction_output_dir,
                allowed_base_keys={key for key in train_prediction_data.keys
                                   if key.split("_")[0] not in early_skipped},
            )
            train_predictions = train_prediction_decoder.run_selection_algo(
                args.selection_algorithm
            )
        else:
            train_predictions = {}
            print(
                "*** Training prediction directory is missing: "
                f"{train_prediction_output_dir}"
            )
        exact_top_two = sum(
            any(
                np.array_equal(guess, train_prediction_data.replies[key][0])
                for guess in train_predictions.get(key, [])[:2]
            )
            for key in train_prediction_data.keys
        )
        print(
            "*** Leave-one-out training prediction coverage: "
            f"{len(train_predictions)}/{len(train_prediction_data.keys)}; "
            f"top-2 exact: {exact_top_two}/"
            f"{len(train_prediction_data.keys)}"
        )

    primary_predictions = decoder.run_selection_algo(
        args.selection_algorithm
    )
    routing_decisions = None
    cascade_active = bool(args.cascade_8b_inference_output_dir)
    if cascade_active:
        secondary_output_dir = os.path.abspath(
            os.path.expanduser(args.cascade_8b_inference_output_dir)
        )
        if not os.path.isdir(secondary_output_dir):
            raise FileNotFoundError(
                "8B cascade inference-output directory does not exist: "
                f"{secondary_output_dir}"
            )
        routing_decisions = route_tasks_by_confidence(
            data.queries,
            decoder.decoded_results,
            selected_keys,
            kgmon_threshold=args.routing_kgmon_threshold,
            beam_threshold=args.routing_beam_threshold,
        )
        routed_task_keys = [
            decision.task_key
            for decision in routing_decisions
            if decision.route
        ]
        routed_base_keys = {
            f"{task_key}_{test_index}"
            for task_key in routed_task_keys
            for test_index in range(len(data.queries[task_key]["test"]))
        }
        secondary_decoder = ArcDecoder(
            data.split_multi_replies(),
            n_guesses=2,
        )
        secondary_decoder.load_decoded_results(
            secondary_output_dir,
            run_name=".8b",
            allowed_base_keys=routed_base_keys,
        )
        secondary_predictions = secondary_decoder.run_selection_algo(
            args.selection_algorithm
        )
        merged_predictions, merge_audit = merge_routed_predictions(
            data.queries,
            primary_predictions,
            secondary_predictions,
            routing_decisions,
            policy=args.cascade_merge_policy,
            n_guesses=2,
        )
        print_cascade_merge_summary(
            routing_decisions,
            merge_audit,
            policy=args.cascade_merge_policy,
        )
        submission = data.get_submission(merged_predictions)
    else:
        submission = data.get_submission(primary_predictions)
    if args.simulate_8b_routing_zero:
        routing_decisions = route_tasks_by_confidence(
            data.queries,
            decoder.decoded_results,
            selected_keys,
            kgmon_threshold=args.routing_kgmon_threshold,
            beam_threshold=args.routing_beam_threshold,
        )
        zero_routed_submission_tasks(submission, routing_decisions)
        print_routing_simulation_summary(
            routing_decisions,
            kgmon_threshold=args.routing_kgmon_threshold,
            beam_threshold=args.routing_beam_threshold,
        )
    if output_nll_reports:
        reported_outputs = print_output_nll_summary(
            output_nll_reports,
            submission,
            selected_keys,
        )
        print(
            f"*** Loaded output-NLL report(s) for "
            f"{len(output_nll_reports)} task(s)/"
            f"{reported_outputs} test output(s)"
        )

    with open("submission.json", "w") as file:
        json.dump(submission, file)

    if args.visualize:
        displayed_count = visualize_submission(
            data,
            submission,
            selected_keys,
            train_predictions=train_predictions,
            output_nll_reports=output_nll_reports,
        )
        print(f"*** Displayed {displayed_count} visualization(s)")

    if not rerun_mode:
        if cascade_active:
            print(
                "*** Skipping the primary-only selector benchmark in "
                "cascade mode; the reload score below evaluates the merged "
                "4B/8B submission."
            )
        else:
            decoder.benchmark_selection_algos()
        with open("submission.json", "r") as file:
            reload_submission = json.load(file)
        print("*** Reload score:", data.validate_submission(reload_submission))


if __name__ == "__main__":
    run_entrypoint_with_status(
        main,
        path=extract_run_status_path(),
        phase="submit",
        success_status=STATUS_SUCCESS,
        success_check=lambda: os.path.isfile("submission.json")
        and os.path.getsize("submission.json") > 0,
    )
