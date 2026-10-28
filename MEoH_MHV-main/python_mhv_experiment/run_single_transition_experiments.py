"""Run independent benchmarks containing exactly one objective-set change.

This is the standalone two-environment chain:

1. optimize in the source environment;
2. let the selected algorithm construct/evaluate its change-response candidates;
3. optimize the resulting population in the target environment;
4. save the source and target MHV results.

The file has a normal ``main`` entry point and can be run directly from
PyCharm.  Its defaults use the paper's RAAT-NSGAII configuration for one
DTLZ2 transition; use command-line arguments to select another transition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from algorithms.registry import create_algorithm
from problems.d_mixed_seq import DMixedSeq
from run_experiments import parse_csv, save_result, validate_history
from settings import POPULATION_SIZE, PROBLEM_TAGS, SETTINGS, single_transition_setting


DEFAULT_ALGORITHMS = "RAATNSGAII"


def selected_settings(indices: tuple[int, ...]):
    for index in indices:
        if index < 1 or index > len(SETTINGS):
            raise ValueError(f"Setting index must be 1-{len(SETTINGS)}, got {index}.")
        yield index, SETTINGS[index - 1]


def run_transition(
    setting,
    algorithm_name: str,
    runs: int,
    n: int,
    source_gens: int,
    target_gens: int,
    seed: int,
    pf_count: int,
    progress_interval: int,
) -> list[np.ndarray]:
    max_fe = (source_gens + target_gens) * n
    histories: list[np.ndarray] = []
    for run_index in range(1, runs + 1):
        problem = DMixedSeq(
            taut=target_gens,
            seq_string=setting.seq,
            config_string=setting.config,
            frozen_string=setting.frozen,
            first_stage_gens=source_gens,
            n=n,
        )
        algorithm = create_algorithm(
            algorithm_name,
            seed=seed + run_index,
            pf_count=pf_count,
            progress_interval=progress_interval,
        )
        print(
            f"      [{algorithm_name}] Run {run_index}/{runs}: "
            f"source={setting.source_objectives} -> target={setting.target_objectives}",
            flush=True,
        )
        result = algorithm.run(problem, max_fe=max_fe, run_idx=run_index)
        validate_history(
            result.history,
            stage_count=2,
            tau=target_gens,
            algorithm=algorithm_name,
            run_idx=run_index,
            first_stage_gens=source_gens,
        )
        histories.append(result.history)
    return histories


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent source-to-target benchmark with exactly one environment change."
    )
    parser.add_argument("--runs", type=int, default=31)
    parser.add_argument("--n", type=int, default=POPULATION_SIZE)
    parser.add_argument("--source-gens", type=int, default=100)
    parser.add_argument("--target-gens", type=int, default=50)
    parser.add_argument("--settings", default="1")
    parser.add_argument(
        "--transitions",
        default="1",
        help="One-based adjacent transitions; 1 means original stage 1 -> stage 2.",
    )
    parser.add_argument("--problems", default="DTLZ2")
    parser.add_argument("--algorithms", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pf-count", type=int, default=5000)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--output-dir", default="Results_SingleTransition")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.runs < 1 or args.n < 2 or args.source_gens < 1 or args.target_gens < 1:
        raise ValueError("runs/source-gens/target-gens must be positive and n must be at least 2.")

    problems = tuple(tag.upper() for tag in parse_csv(args.problems))
    unknown = sorted(set(problems) - set(PROBLEM_TAGS))
    if unknown:
        raise ValueError(f"Unsupported problems: {unknown}; choose from {PROBLEM_TAGS}.")

    algorithms = parse_csv(args.algorithms)
    transitions = parse_csv(args.transitions, int)
    output_dir = Path(args.output_dir)

    for problem_tag in problems:
        for setting_index, template in selected_settings(parse_csv(args.settings, int)):
            for transition_index in transitions:
                setting = single_transition_setting(template, problem_tag, transition_index)
                print(
                    f"\n=== {problem_tag} | Setting {setting_index} | "
                    f"Transition {transition_index}: {setting.source_objectives} -> "
                    f"{setting.target_objectives} ===",
                    flush=True,
                )
                for algorithm_name in algorithms:
                    path = output_dir / (
                        f"MHV_Transition_{algorithm_name}_{problem_tag}_Set{setting_index}_"
                        f"T{transition_index}_Src{args.source_gens}_Dst{args.target_gens}.mat"
                    )
                    if path.exists() and not args.overwrite:
                        print(f"Skipping existing result: {path}", flush=True)
                        continue
                    histories = run_transition(
                        setting=setting,
                        algorithm_name=algorithm_name,
                        runs=args.runs,
                        n=args.n,
                        source_gens=args.source_gens,
                        target_gens=args.target_gens,
                        seed=args.seed,
                        pf_count=args.pf_count,
                        progress_interval=args.progress_interval,
                    )
                    save_result(path, histories, setting, args.target_gens, algorithm_name)
                    print(f"Saved: {path}", flush=True)


if __name__ == "__main__":
    main()
