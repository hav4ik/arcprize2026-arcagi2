import os
import bz2
import pickle
import numpy as np


DEFAULT_SELECTION_ALGORITHM = "kgmon"


def hashable(guess):
    return tuple(map(tuple, guess))


def mean_augmented_nll(guesses):
    """Average the per-view sequence NLLs for one unique candidate grid."""
    return float(np.mean([
        np.mean(np.asarray(guess["score_aug"], dtype=np.float64))
        for guess in guesses
    ]))


def candidate_answer_token_count(solution):
    """Return digit, row-separator, and end-token count for an ARC grid."""
    solution = np.asarray(solution)
    if solution.ndim != 2 or not all(size > 0 for size in solution.shape):
        raise ValueError(
            f"Expected a non-empty 2D ARC grid, got shape {solution.shape}"
        )
    # QwenFormatter emits one token per cell, one newline between rows, and
    # one <|im_end|> token: H*W + (H-1) + 1 == H*(W+1).
    return int(solution.size + solution.shape[0])


def score_sum(guesses, getter):
    guess_list = list(guesses.values())
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores[h] = scores.get(h, [[], g["solution"]])
        x[0].append(g)
    scores = [(getter(sc), o) for sc, o in scores.values()]
    scores = sorted(scores, key=(lambda x: x[0]), reverse=True)
    ordered_outputs = [x[-1] for x in scores]
    return ordered_outputs


def getter_full_probmul_3(guesses, baseline=3):
    inf_score = np.sum([baseline-g["beam_score"] for g in guesses])
    aug_score = np.mean([np.sum([baseline-s for s in g["score_aug"]]) for g in guesses])
    return inf_score + aug_score


def score_full_probmul_3(guesses):
    return score_sum(guesses, getter_full_probmul_3)


def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = mean_augmented_nll(guesses)
    return inf_score - aug_score


def score_kgmon(guesses):
    return score_sum(guesses, getter_kgmon)


def getter_geometric_vote(guesses):
    """Combine occurrence count with geometric-mean sequence probability.

    If L is mean augmented NLL, geometric-mean probability is exp(-L).
    Ranking count * exp(-L) in stable log space gives log(count) - L.
    """
    return np.log(len(guesses)) - mean_augmented_nll(guesses)


def score_geometric_vote(guesses):
    return score_sum(guesses, getter_geometric_vote)


def getter_geometric_vote_per_token(guesses):
    """Length-normalized geometric probability with a log-count vote prior."""
    answer_tokens = candidate_answer_token_count(guesses[0]["solution"])
    return (
        np.log(len(guesses))
        - mean_augmented_nll(guesses) / answer_tokens
    )


def score_geometric_vote_per_token(guesses):
    return score_sum(guesses, getter_geometric_vote_per_token)


def getter_augmented_nll(guesses):
    return -mean_augmented_nll(guesses)


def score_augmented_nll(guesses):
    return score_sum(guesses, getter_augmented_nll)


def getter_votes_only(guesses):
    return len(guesses)


def score_votes_only(guesses):
    return score_sum(guesses, getter_votes_only)


def augmentation_rank_records(guesses):
    """Return shared-view rank statistics, or ``None`` when incompatible."""
    known_seeds = {
        guess["score_aug_seed"]
        for guess in guesses.values()
        if guess.get("score_aug_seed") is not None
    }
    has_unknown_seed = any(
        guess.get("score_aug_seed") is None
        for guess in guesses.values()
    )
    has_explicit_unknown_seed = any(
        "score_aug_seed" in guess
        and guess["score_aug_seed"] is None
        for guess in guesses.values()
    )
    if (
        has_explicit_unknown_seed
        or len(known_seeds) > 1
        or (known_seeds and has_unknown_seed)
    ):
        return None

    grouped = {}
    for guess in guesses.values():
        solution_key = hashable(guess["solution"])
        entry = grouped.setdefault(
            solution_key,
            {"guesses": [], "solution": guess["solution"]},
        )
        entry["guesses"].append(guess)

    if not grouped:
        return []

    candidates = list(grouped.values())
    score_vectors = []
    for candidate in candidates:
        vectors = [
            np.asarray(guess["score_aug"], dtype=np.float64).reshape(-1)
            for guess in candidate["guesses"]
        ]
        if not vectors or not vectors[0].size:
            return None
        if any(vector.shape != vectors[0].shape for vector in vectors[1:]):
            return None
        score_vectors.append(np.mean(np.stack(vectors), axis=0))

    if any(vector.shape != score_vectors[0].shape for vector in score_vectors[1:]):
        return None

    score_matrix = np.stack(score_vectors)
    if not np.all(np.isfinite(score_matrix)):
        return None
    best_per_view = np.min(score_matrix, axis=0, keepdims=True)
    view_winners = np.isclose(
        score_matrix,
        best_per_view,
        rtol=1e-10,
        atol=1e-12,
    )
    fractional_wins = view_winners / np.sum(
        view_winners,
        axis=0,
        keepdims=True,
    )
    # Competition rank: one plus the number of candidates with strictly lower
    # NLL on the same view. Exact ties therefore receive the same rank.
    view_ranks = 1 + np.sum(
        score_matrix[:, None, :] < score_matrix[None, :, :],
        axis=0,
    )

    return [
        {
            "solution": candidate["solution"],
            "augmentation_wins": float(wins),
            "worst_augmentation_rank": int(np.max(ranks)),
            "last_place_views": int(np.sum(ranks == len(candidates))),
            "nvarc_score": float(getter_kgmon(candidate["guesses"])),
        }
        for candidate, wins, ranks in zip(
            candidates,
            np.sum(fractional_wins, axis=1),
            view_ranks,
        )
    ]


def score_augmentation_wins(guesses):
    """Rank by per-augmentation wins, with the NVARC score as tie-breaker.

    Every unique solution is rescored with the same ordered augmentations.  A
    view awards one (possibly fractional, for exact ties) vote to the solution
    with the lowest NLL on that view.  This rank aggregation is insensitive to
    one augmentation having a much larger NLL scale than another.
    """
    records = augmentation_rank_records(guesses)
    if records is None:
        return score_kgmon(guesses)
    records.sort(
        key=lambda record: (
            record["augmentation_wins"],
            record["nvarc_score"],
        ),
        reverse=True,
    )
    return [record["solution"] for record in records]


def score_kgmon_diverse(guesses):
    """Keep the NVARC top-1 and diversify later attempts by view wins."""
    nvarc_order = score_kgmon(guesses)
    if not nvarc_order:
        return []
    first_key = hashable(nvarc_order[0])
    diverse_order = score_augmentation_wins(guesses)
    return [nvarc_order[0]] + [
        solution
        for solution in diverse_order
        if hashable(solution) != first_key
    ]


def score_kgmon_robust(
    guesses,
    specialist_win_threshold=2,
    max_specialist_last_place_views=1,
):
    """Preserve NVARC top-1 and choose a robust shared-view second attempt.

    A remaining candidate that wins at least two views without repeatedly
    ranking last is treated as a real specialist and selected by view wins. If
    every remaining candidate wins at most one view, select the candidate with
    the best worst-case view rank. If only brittle specialists are available,
    retain the original NVARC second attempt.
    """
    nvarc_order = score_kgmon(guesses)
    if len(nvarc_order) < 2:
        return nvarc_order
    records = augmentation_rank_records(guesses)
    if records is None:
        return nvarc_order

    first_key = hashable(nvarc_order[0])
    remaining = [
        record
        for record in records
        if hashable(record["solution"]) != first_key
    ]
    if not remaining:
        return nvarc_order

    specialists = [
        record
        for record in remaining
        if record["augmentation_wins"] >= specialist_win_threshold
    ]
    safe_specialists = [
        record
        for record in specialists
        if record["last_place_views"] <= max_specialist_last_place_views
    ]
    if safe_specialists:
        second = max(
            safe_specialists,
            key=lambda record: (
                record["augmentation_wins"],
                record["nvarc_score"],
            ),
        )
    elif specialists:
        return nvarc_order
    else:
        second = max(
            remaining,
            key=lambda record: (
                -record["worst_augmentation_rank"],
                record["nvarc_score"],
            ),
        )
    second_solution_key = hashable(second["solution"])
    return [nvarc_order[0], second["solution"]] + [
        solution
        for solution in nvarc_order[1:]
        if hashable(solution) != second_solution_key
    ]


SELECTION_ALGORITHMS = {
    "full_probmul_3": score_full_probmul_3,
    "kgmon": score_kgmon,
    "geometric_vote": score_geometric_vote,
    "geometric_vote_per_token": score_geometric_vote_per_token,
    "augmented_nll": score_augmented_nll,
    "votes_only": score_votes_only,
    "augmentation_wins": score_augmentation_wins,
    "kgmon_diverse": score_kgmon_diverse,
    "kgmon_robust": score_kgmon_robust,
}


selection_algorithms = [
    SELECTION_ALGORITHMS[name]
    for name in (
        "full_probmul_3",
        "kgmon",
        "geometric_vote",
        "geometric_vote_per_token",
        "augmented_nll",
        "votes_only",
        "augmentation_wins",
        "kgmon_diverse",
        "kgmon_robust",
    )
]


def get_selection_algorithm(name):
    try:
        return SELECTION_ALGORITHMS[name]
    except KeyError as error:
        choices = ", ".join(SELECTION_ALGORITHMS)
        raise ValueError(
            f"Unknown ARC selection algorithm {name!r}; choose from {choices}"
        ) from error


class ArcDecoder:

    def __init__(self, dataset, n_guesses):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}
        self.allowed_base_keys = None
        self.candidate_read_errors = []

    def load_decoded_results(self, store, run_name="", allowed_base_keys=None, *, skip_unreadable=False):
        if allowed_base_keys is not None:
            allowed_base_keys = set(allowed_base_keys)
            if self.allowed_base_keys is None:
                self.allowed_base_keys = set()
            self.allowed_base_keys.update(allowed_base_keys)
        for key in sorted(os.listdir(store)):
            result_path = os.path.join(store, key)
            if not os.path.isfile(result_path):
                continue
            base_key = key.split(".")[0]
            if allowed_base_keys is not None and base_key not in allowed_base_keys:
                continue
            try:
                with bz2.BZ2File(result_path) as f:
                    outputs = pickle.load(f)
            except (EOFError, OSError, pickle.UnpicklingError) as error:
                if not skip_unreadable:
                    raise
                # A deadline can interrupt a candidate-file write. Recovery is
                # opt-in and never deletes or mutates the original artifact.
                record = {"path": result_path, "error": f"{type(error).__name__}: {error}"}
                self.candidate_read_errors.append(record)
                print(f"*** Skipping unreadable candidate: {record}", flush=True)
                continue
            self.decoded_results[base_key] = self.decoded_results.get(base_key, {})
            for i, sample in enumerate(outputs):
                self.decoded_results[base_key][f"{key}{run_name}.out{i}"] = sample

    def run_selection_algo(self, selection_algorithm=DEFAULT_SELECTION_ALGORITHM):
        if isinstance(selection_algorithm, str):
            selection_algorithm = get_selection_algorithm(selection_algorithm)
        return {bk: selection_algorithm({k: g for k, g in v.items()}) for bk, v in self.decoded_results.items()}

    def benchmark_selection_algos(self):
        print("*** Benchmark selection algorithms...")

        labels = {}
        num_tasks_per_puzzle = {}
        num_solved_keys = 0
        num_total_keys = 0

        correct_beam_scores = []

        if self.allowed_base_keys is None:
            evaluated_base_keys = sorted(self.decoded_results)
        else:
            evaluated_base_keys = sorted(self.allowed_base_keys)
        evaluated_base_keys = [
            key for key in evaluated_base_keys
            if key in self.dataset.replies
        ]

        for basekey in evaluated_base_keys:
            basevalues = self.decoded_results.get(basekey, {})

            mult_key, mult_sub = basekey.split("_")
            num_tasks_per_puzzle[mult_key] = (
                num_tasks_per_puzzle.get(mult_key, 0) + 1
            )

            labels[basekey] = correct_solution = self.dataset.replies[basekey][0]

            for subkey, sample in basevalues.items():

                solution = sample["solution"]
                beam_score = sample["beam_score"]
                aug_mean = np.mean(sample["score_aug"])

                if np.shape(correct_solution) != np.shape(solution):
                    corr_str = "bad_xy_size"
                elif np.array_equal(correct_solution, solution):
                    corr_str = "ALL_CORRECT"
                    num_solved_keys += 1
                    correct_beam_scores.append(beam_score)
                else:
                    corr_str = "bad_content"

                output_len = f"{solution.shape[0]}x{solution.shape[1]}"

                if corr_str == "ALL_CORRECT":
                    print(f"{corr_str}:{beam_score:8.5f} - {aug_mean:8.5f} {output_len:5s} [{subkey}]")
                num_total_keys += 1

        print(f" subkeys: {num_solved_keys}/{num_total_keys}")
        if correct_beam_scores:
            print(f" avg correct beam score: {np.mean(correct_beam_scores):8.5f}")
            print(f" max correct beam score: {np.max(correct_beam_scores):8.5f}")
        else:
            print(" avg correct beam score: n/a")
            print(" max correct beam score: n/a")

        num_puzzles = len(num_tasks_per_puzzle)
        oracle_correct = {
            key
            for key, correct_solution in labels.items()
            if any(
                np.array_equal(sample["solution"], correct_solution)
                for sample in self.decoded_results.get(key, {}).values()
            )
        }
        oracle_score = sum(
            1 / num_tasks_per_puzzle[key.split("_")[0]]
            for key in oracle_correct
        )
        print(
            f" oracle candidate coverage: {oracle_score:5.1f}/"
            f"{num_puzzles:3} ({len(oracle_correct)} outputs)"
        )

        benchmark_results = {}
        for selection_algorithm in selection_algorithms:
            name = selection_algorithm.__name__
            selected = self.run_selection_algo(selection_algorithm)
            correct_puzzles = {
                key
                for key, correct_solution in labels.items()
                if any(
                    np.array_equal(guess, correct_solution)
                    for guess in selected.get(key, [])[:self.n_guesses]
                )
            }
            missed_selection = oracle_correct - correct_puzzles
            print(correct_puzzles)
            score = sum(
                1 / num_tasks_per_puzzle[key.split("_")[0]]
                for key in correct_puzzles
            )
            print(f" acc: {score:5.1f}/{num_puzzles:3} ('{name}')")
            if missed_selection:
                print(f" selection misses: {sorted(missed_selection)}")
            benchmark_results[name] = {
                "score": score,
                "num_puzzles": num_puzzles,
                "evaluated_outputs": len(labels),
                "correct_outputs": correct_puzzles,
                "selection_misses": missed_selection,
                "oracle_score": oracle_score,
                "oracle_correct_outputs": oracle_correct,
            }
        return benchmark_results
