"""Run selected algorithms and plot MHV against FE/N.

The ordinary experiment files store MHV only at the end of each environment.
This utility instead samples the current population during a new run, so it
can draw the recovery curves after every dynamic change.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from algorithms.registry import create_algorithm
from problems.d_mixed_seq import DMixedSeq
from settings import FIRST_STAGE_GENS, POPULATION_SIZE, PROBLEM_TAGS, SETTINGS, setting_for_problem


def parse_csv(value: str, cast=str) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def get_setting(index: int):
    if not 1 <= index <= len(SETTINGS):
        raise ValueError(f"--setting must be in 1-{len(SETTINGS)}, got {index}.")
    return SETTINGS[index - 1]


def run_problem(
    problem_tag: str,
    setting_index: int,
    tau: int,
    algorithms: tuple[str, ...],
    runs: int,
    n: int,
    sample_every: int,
    pf_count: int,
    seed: int,
) -> tuple[dict[str, list[tuple[int, int, float]]], int, list[int], str]:
    setting = setting_for_problem(get_setting(setting_index), problem_tag)
    stage_count = len([part for part in setting.seq.split(";") if part.strip()])
    max_gen = FIRST_STAGE_GENS + (stage_count - 1) * tau
    max_fe = max_gen * n
    change_gens = [FIRST_STAGE_GENS + stage * tau for stage in range(stage_count - 1)]
    traces: dict[str, list[tuple[int, int, float]]] = defaultdict(list)

    for algorithm_name in algorithms:
        print(f"\n=== {problem_tag}: {algorithm_name} ===", flush=True)
        for run_idx in range(1, runs + 1):
            samples: list[tuple[int, float, int]] = []

            def capture(generation: int, mhv: float, callback_run_idx: int) -> None:
                samples.append((generation, mhv, callback_run_idx))

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
                progress_interval=max(sample_every, 25),
            )
            algorithm.configure_trace(sample_every, capture)
            algorithm.run(problem, max_fe=max_fe, run_idx=run_idx)
            traces[algorithm_name].extend((run, generation, mhv) for generation, mhv, run in samples)
            print(
                f"   [{algorithm_name}] run {run_idx}/{runs}: {len(samples)} MHV samples",
                flush=True,
            )

    return traces, max_gen, change_gens, setting.name


def aggregate(points: list[tuple[int, int, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: dict[int, list[float]] = defaultdict(list)
    for _, generation, mhv in points:
        values[generation].append(mhv)
    generations = np.asarray(sorted(values), dtype=int)
    mean = np.asarray([np.mean(values[g]) for g in generations], dtype=float)
    std = np.asarray(
        [np.std(values[g], ddof=1) if len(values[g]) > 1 else 0.0 for g in generations],
        dtype=float,
    )
    return generations, mean, std


def save_trace_csv(path: Path, traces: dict[str, list[tuple[int, int, float]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["algorithm", "run", "generation_fe_over_n", "mhv"])
        for algorithm, points in traces.items():
            for run_idx, generation, mhv in sorted(points):
                writer.writerow([algorithm, run_idx, generation, f"{mhv:.16g}"])


def plot_trace(
    output_path: Path,
    traces: dict[str, list[tuple[int, int, float]]],
    max_gen: int,
    change_gens: list[int],
    setting_name: str,
    problem_tag: str,
    tau: int,
    show_std: bool,
    show_markers: bool,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib. Install it with: pip install matplotlib") from exc

    fig, axis = plt.subplots(figsize=(11, 6.4), constrained_layout=True)
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*")
    colors = plt.get_cmap("tab10")
    for index, (algorithm, points) in enumerate(traces.items()):
        generations, mean, std = aggregate(points)
        if not len(generations):
            continue
        color = colors(index % 10)
        line_options = dict(
            label=algorithm,
            color=color,
            linewidth=2.2,
        )
        if show_markers:
            line_options.update(
                marker=markers[index % len(markers)],
                markersize=4.8,
            )
        axis.plot(generations, mean, **line_options)
        if show_std and np.any(std > 0):
            axis.fill_between(generations, np.maximum(0.0, mean - std), np.minimum(1.0, mean + std),
                              color=color, alpha=0.14, linewidth=0)

    for generation in change_gens:
        axis.axvline(generation, color="0.35", linestyle="--", linewidth=1.0, alpha=0.55)
    axis.set_xlim(0, max_gen)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Iteration (FE/N)")
    axis.set_ylabel("MHV ratio (HV_pop/HV_PF)")
    axis.set_title(f"{problem_tag} — {setting_name}, Tau = {tau}")
    axis.grid(True, color="0.75", linewidth=0.8, alpha=0.55)
    axis.legend(loc="best", frameon=True, fancybox=False, edgecolor="0.2")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run algorithms and plot MHV-vs-FE/N curves.")
    parser.add_argument("--algorithms", required=True, help="Comma-separated registered algorithm names.")
    parser.add_argument("--problems", default="DTLZ1", help="Comma-separated problem tags, e.g. DTLZ1,WFG4.")
    parser.add_argument("--setting", type=int, default=1, help="Dynamic setting index (default: 1).")
    parser.add_argument("--tau", type=int, default=50)
    parser.add_argument("--runs", type=int, default=5, help="Independent runs per algorithm (default: 5).")
    parser.add_argument("--n", type=int, default=POPULATION_SIZE)
    parser.add_argument("--sample-every", type=int, default=10, help="MHV sampling interval in FE/N generations.")
    parser.add_argument("--pf-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="MHV_FE_Plots")
    parser.add_argument("--no-std-band", action="store_true", help="Do not draw the ±1 standard-deviation band.")
    parser.add_argument(
        "--show-markers",
        action="store_true",
        help="Show sampled MHV points as markers (hidden by default for clearer curves).",
    )
    args = parser.parse_args()

    if args.runs < 1 or args.n < 2 or args.tau < 1 or args.sample_every < 1:
        raise ValueError("--runs, --n, --tau, and --sample-every must be positive; --n must be at least 2.")
    algorithms = parse_csv(args.algorithms)
    problems = tuple(tag.upper() for tag in parse_csv(args.problems))
    unknown = sorted(set(problems) - set(PROBLEM_TAGS))
    if unknown:
        raise ValueError(f"Unsupported problems: {unknown}; choose from {PROBLEM_TAGS}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for problem_tag in problems:
        traces, max_gen, change_gens, setting_name = run_problem(
            problem_tag, args.setting, args.tau, algorithms, args.runs, args.n,
            args.sample_every, args.pf_count, args.seed,
        )
        stem = f"MHV_FE_{problem_tag}_Set{args.setting}_Tau{args.tau}"
        csv_path = output_dir / f"{stem}.csv"
        png_path = output_dir / f"{stem}.png"
        save_trace_csv(csv_path, traces)
        plot_trace(
            png_path, traces, max_gen, change_gens, setting_name, problem_tag,
            args.tau, not args.no_std_band, args.show_markers,
        )
        print(f"Saved trace data: {csv_path}", flush=True)
        print(f"Saved plot: {png_path}", flush=True)


if __name__ == "__main__":
    main()
