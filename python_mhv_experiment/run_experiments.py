from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from algorithms.registry import AlgorithmNotPortedError, create_algorithm
from problems.d_mixed_seq import DMixedSeq
from settings import (
    FIRST_STAGE_GENS,
    POPULATION_SIZE,
    PROBLEM_TAGS,
    RUNS,
    SETTINGS,
    TAUS,
    setting_for_problem,
)


def parse_csv(values: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in values.split(",") if item.strip())


def selected_settings(indices: tuple[int, ...]):
    selected = []
    for index in indices:
        if index < 1 or index > len(SETTINGS):
            raise ValueError(f"Setting index must be 1-{len(SETTINGS)}, got {index}.")
        selected.append((index, SETTINGS[index - 1]))
    return selected


def save_result(path: Path, mhv_data: list[np.ndarray], setting, tau: int, algorithm: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mhv_cell = np.empty((1, len(mhv_data)), dtype=object)
    for idx, run_history in enumerate(mhv_data):
        mhv_cell[0, idx] = run_history
    try:
        from scipy.io import savemat
    except ImportError as exc:
        raise RuntimeError(
            "Saving MATLAB-compatible experiment results requires scipy. "
            "Install it with: pip install scipy"
        ) from exc

    temp_path = path.with_name(f".{path.stem}.tmp.mat")
    savemat(
        temp_path,
        {
            "MHV_Data": mhv_cell,
            "CurrentSetting": {
                "Name": setting.name,
                "ProbTag": setting.prob_tag,
                "Config": setting.config,
                "Seq": setting.seq,
                "Frozen": setting.frozen,
            },
            "tau": tau,
            "AlgoName": algorithm,
        },
    )
    temp_path.replace(path)


def result_is_complete(path: Path, runs: int, stages: int) -> bool:
    try:
        from scipy.io import loadmat

        mhv_data = loadmat(path)["MHV_Data"]
        return mhv_data.shape == (1, runs) and all(
            mhv_data[0, index].shape == (stages, 3) for index in range(runs)
        )
    except (ImportError, KeyError, OSError, ValueError):
        return False


def expected_target_gens(stage_count: int, tau: int, first_stage_gens: int = FIRST_STAGE_GENS) -> np.ndarray:
    targets = np.empty(stage_count, dtype=float)
    targets[0] = first_stage_gens - 1
    for stage in range(2, stage_count + 1):
        targets[stage - 1] = first_stage_gens + (stage - 1) * tau - 1
    return targets


def validate_history(
    history: np.ndarray,
    stage_count: int,
    tau: int,
    algorithm: str,
    run_idx: int,
    first_stage_gens: int = FIRST_STAGE_GENS,
) -> None:
    if history.shape != (stage_count, 3):
        raise RuntimeError(
            f"{algorithm} run {run_idx} recorded invalid MHV history shape "
            f"{history.shape}; expected ({stage_count}, 3)."
        )

    expected_targets = expected_target_gens(stage_count, tau, first_stage_gens)
    if not np.array_equal(history[:, 0], expected_targets):
        raise RuntimeError(
            f"{algorithm} run {run_idx} recorded wrong TargetGen values: "
            f"{history[:, 0].tolist()} expected {expected_targets.tolist()}."
        )

    trigger_gens = history[:, 1]
    if np.any(trigger_gens < expected_targets):
        raise RuntimeError(
            f"{algorithm} run {run_idx} detected a stage ending before its TargetGen: "
            f"{history[:, :2].tolist()}."
        )

    # TriggerGen is the global-FE detection time, not the generation in which
    # the saved population was evaluated. Algorithms such as DTAEA evaluate
    # 2N offspring per iteration, so detection can occur just after TargetGen.
    # It must still happen before the following stage itself ends.
    if stage_count > 1 and np.any(trigger_gens[:-1] >= expected_targets[1:]):
        raise RuntimeError(
            f"{algorithm} run {run_idx} detected a stage too late or skipped a stage: "
            f"{history[:, :2].tolist()}."
        )


def run_case(
    setting,
    tau: int,
    algorithm_name: str,
    runs: int,
    n: int,
    seed: int,
    pf_count: int,
    progress_interval: int,
) -> list[np.ndarray]:
    stages = len([part for part in setting.seq.split(";") if part.strip()])
    max_gen = FIRST_STAGE_GENS + (stages - 1) * tau
    max_fe = max_gen * n
    mhv_data: list[np.ndarray] = []

    for run_idx in range(1, runs + 1):
        print(
            f"      [{algorithm_name}] Run {run_idx}/{runs} started "
            f"(N={n}, MaxGen={max_gen}, MaxFE={max_fe}, PF={pf_count})",
            flush=True,
        )
        problem = DMixedSeq(
            taut=tau,
            seq_string=setting.seq,
            config_string=setting.config,
            frozen_string=setting.frozen,
            first_stage_gens=FIRST_STAGE_GENS,
            n=n,
        )
        algorithm = create_algorithm(
            algorithm_name,
            seed=seed + run_idx,
            pf_count=pf_count,
            progress_interval=progress_interval,
        )
        result = algorithm.run(problem, max_fe=max_fe, run_idx=run_idx)
        validate_history(result.history, stages, tau, algorithm_name, run_idx)
        mhv_data.append(result.history)
        print(
            f"      [{algorithm_name}] Run {run_idx}/{runs} finished "
            f"({result.history.shape[0]} stages recorded)",
            flush=True,
        )

    return mhv_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure Python MHV experiment runner.")
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--n", type=int, default=POPULATION_SIZE)
    parser.add_argument("--taus", default=",".join(str(tau) for tau in TAUS))
    parser.add_argument(
        "--algorithms",
        default=(
            "DTAEA,KTDMOEA,LEC,STA,ArchiveSPEA2SDE,TrustArchiveSPEA2SDE,ArchiveNSGAII,"
            "TrustArchiveNSGAII,RestartNSGAII"
        ),
    )
    parser.add_argument("--settings", default="1,2,3")
    parser.add_argument("--problems", default=",".join(PROBLEM_TAGS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pf-count", type=int, default=5000)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--output-dir", default="Results_StepMHV_Python")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    taus = parse_csv(args.taus, int)
    algorithms = parse_csv(args.algorithms, str)
    settings = selected_settings(parse_csv(args.settings, int))
    problems = tuple(tag.upper() for tag in parse_csv(args.problems, str))
    unknown = sorted(set(problems) - set(PROBLEM_TAGS))
    if unknown:
        raise ValueError(f"Unsupported problems: {unknown}; choose from {PROBLEM_TAGS}.")
    output_dir = Path(args.output_dir)

    print(
        "Experiment plan: "
        f"algorithms={','.join(algorithms)} | "
        f"settings={','.join(str(i) for i, _ in settings)} | "
        f"problems={','.join(problems)} | "
        f"taus={','.join(str(t) for t in taus)} | "
        f"runs={args.runs} | N={args.n} | pf_count={args.pf_count}",
        flush=True,
    )

    for problem_tag in problems:
        for setting_index, setting_template in settings:
            setting = setting_for_problem(setting_template, problem_tag)
            for tau in taus:
                stages = len([part for part in setting.seq.split(";") if part.strip()])
                print(f"\n=== {setting.name} [{setting.prob_tag}], Tau: {tau} (Stages: {stages}) ===", flush=True)
                for algorithm in algorithms:
                    out_file = output_dir / f"MHV_Step_{algorithm}_{setting.prob_tag}_Set{setting_index}_Tau{tau}.mat"
                    if out_file.exists() and not args.overwrite:
                        if result_is_complete(out_file, args.runs, stages):
                            print(f"Skipping {algorithm} (complete): {out_file}", flush=True)
                            continue
                        print(f"Re-running {algorithm} (incomplete result): {out_file}", flush=True)
                    print(f"   >>> Algorithm: {algorithm} | Output: {out_file}", flush=True)
                    try:
                        mhv_data = run_case(
                            setting,
                            tau,
                            algorithm,
                            args.runs,
                            args.n,
                            args.seed,
                            args.pf_count,
                            args.progress_interval,
                        )
                    except AlgorithmNotPortedError as exc:
                        print(f"Cannot run {algorithm}: {exc}", flush=True)
                        continue
                    save_result(out_file, mhv_data, setting, tau, algorithm)
                    print(f"   <<< Algorithm: {algorithm} done.", flush=True)


if __name__ == "__main__":
    main()
