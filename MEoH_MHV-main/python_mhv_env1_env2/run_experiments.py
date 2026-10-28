"""Independent MHV recovery experiment for environments 1 and 2."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from types import MethodType

import numpy as np

from algorithms.registry import PORTED_ALGORITHMS, create_algorithm
from metrics.mhv_strict import mhv_strict
from problems.d_mixed_seq import DMixedSeq
from settings import FIRST_STAGE_GENS, POPULATION_SIZE, PROBLEM_TAGS, RUNS, SETTINGS, TAUS, setting_for_problem

DEFAULT_ALGORITHMS = (
    "DTAEA", "KTDMOEA", "LEC", "STA", "TrustArchiveSPEA2SDE",
)
LABELS = {
    "ArchiveSPEA2SDE": "UATS-SPEA2SDE", "ArchiveNSGAII": "UATS-NSGAII",
    "TrustArchiveSPEA2SDE": "RAAT-SPEA2SDE", "TrustArchiveNSGAII": "RAAT-NSGAII",
}
FIELDS = ("algorithm", "run", "environment", "generation_fe_over_n", "mhv")


def csv_items(value, cast=str):
    result = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise ValueError("Selections must not be empty.")
    return result


def install_window_trace(algorithm, problem, run_idx, sample_every, rows):
    """Sample the actual active environment; FE can cross a boundary first.

    Use elapsed FE/N instead of divisibility, since DTAEA consumes more than
    N evaluations per update and may never visit multiples of an interval.
    Both the initial environment and the first change response are sampled.
    """
    last_generation = {}

    def trace(self, population, current_problem, callback_run_idx, force=False):
        if population is None or not len(population):
            return
        active = tuple(current_problem.current_indices)
        stage = next((i for i, group in enumerate(current_problem.obj_groups, 1)
                      if tuple(group) == active), None)
        if stage not in (1, 2) or population.objs.shape[1] != len(active):
            return
        generation = max(0, current_problem.fe - 1) // current_problem.n
        start, end = environment_bounds(stage, current_problem.taut)
        # An update may consume >N evaluations and finish beyond the nominal
        # boundary while still holding the old environment's population.
        # For this forced endpoint, use its latest actual evaluation timestamp.
        if force and generation >= end and population.fes is not None:
            generation = int(np.max(population.fes)) // current_problem.n
        if not start <= generation < end:
            return
        previous = last_generation.get(stage)
        if previous is not None:
            if generation <= previous or (not force and generation - previous < sample_every):
                return
        score, _, _ = mhv_strict(population, current_problem, pf_count=self.pf_count)
        if not np.isfinite(score):
            raise RuntimeError(f"Non-finite MHV for {self.name}, run {run_idx}, FE/N={generation}")
        rows.append((self.name, run_idx, stage, generation, float(score)))
        last_generation[stage] = generation

    algorithm.try_trace = MethodType(trace, algorithm)


def environment_bounds(stage, tau):
    if stage == 1:
        return 0, FIRST_STAGE_GENS
    if stage == 2:
        return FIRST_STAGE_GENS, FIRST_STAGE_GENS + tau
    raise ValueError(f"Only environments 1 and 2 are included, got {stage}.")


def run_case(problem_tag, setting_index, tau, args):
    setting = setting_for_problem(SETTINGS[setting_index - 1], problem_tag)
    # Keep the full sequence: truncating it could change D/global_max_m/WFG
    # parameters. The FE budget alone stops the experiment after environment 2.
    max_fe = (FIRST_STAGE_GENS + tau) * args.n
    rows = []
    for name in args.algorithms:
        for run_idx in range(1, args.runs + 1):
            print(f"{problem_tag} | Set{setting_index} | tau={tau} | {name} | run {run_idx}/{args.runs}", flush=True)
            problem = DMixedSeq(tau, setting.seq, setting.config, setting.frozen,
                                first_stage_gens=FIRST_STAGE_GENS, n=args.n)
            algorithm = create_algorithm(name, seed=args.seed + run_idx,
                                         pf_count=args.pf_count, progress_interval=25)
            install_window_trace(algorithm, problem, run_idx, args.sample_every, rows)
            # Capture initialization before the first update, including FE/N=0.
            original_initialization = problem.initialization

            def traced_initialization(*initial_args, **initial_kwargs):
                population = original_initialization(*initial_args, **initial_kwargs)
                algorithm.try_trace(population, problem, run_idx)
                return population

            problem.initialization = traced_initialization
            result = algorithm.run(problem, max_fe=max_fe, run_idx=run_idx)
            algorithm.try_trace(result.last_population, problem, run_idx, force=True)
            stages = {row[2] for row in rows if row[0] == name and row[1] == run_idx}
            if stages != {1, 2}:
                raise RuntimeError(f"Missing environment traces: {name}, run {run_idx}, stages={stages}; increase tau.")
            counts = {stage: sum(row[0] == name and row[1] == run_idx and row[2] == stage
                                 for row in rows) for stage in (1, 2)}
            print(f"   [{name}] Run {run_idx} finished: Stage 1 MHV samples={counts[1]}, "
                  f"Stage 2 MHV samples={counts[2]}; FE={problem.fe}/{max_fe}", flush=True)
    return rows


def save_rows(path, rows):
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for name, run, stage, generation, score in sorted(rows):
            writer.writerow((name, run, stage, generation, f"{score:.16g}"))
    temporary.replace(path)


def read_rows(path, args, tau):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Invalid trace CSV header: {path}")
        rows = [(r["algorithm"], int(r["run"]), int(r["environment"]),
                 int(r["generation_fe_over_n"]), float(r["mhv"])) for r in reader]
    seen = set()
    for name, run, stage, generation, score in rows:
        start, end = environment_bounds(stage, tau)
        key = (name, run, stage, generation)
        if (name not in args.algorithms or not 1 <= run <= args.runs or stage not in (1, 2)
                or not start <= generation < end or not np.isfinite(score) or key in seen):
            raise ValueError(f"Invalid or duplicate trace row in {path}: {key}")
        seen.add(key)
    expected = {(name, run, stage) for name in args.algorithms
                for run in range(1, args.runs + 1) for stage in (1, 2)}
    if {(r[0], r[1], r[2]) for r in rows} != expected:
        raise ValueError(f"Incomplete trace CSV: {path}")
    return rows


def aggregate(rows, algorithm):
    groups = defaultdict(list)
    for name, _, stage, generation, score in rows:
        if name == algorithm:
            groups[(stage, generation)].append(score)
    keys = sorted(groups)
    x = np.asarray([key[1] for key in keys])
    mean = np.asarray([np.mean(groups[key]) for key in keys])
    std = np.asarray([np.std(groups[key], ddof=1) if len(groups[key]) > 1 else 0.0 for key in keys])
    return x, mean, std, keys, groups


def save_summary(path, rows, args):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("algorithm", "environment", "generation_fe_over_n", "mean_mhv", "std_mhv", "sample_count"))
        for name in args.algorithms:
            _, means, stds, keys, groups = aggregate(rows, name)
            for key, mean, std in zip(keys, means, stds):
                writer.writerow((name, *key, f"{mean:.16g}", f"{std:.16g}", len(groups[key])))


def draw_axis(axis, rows, setting_index, problem_tag, tau, args):
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("tab10")
    for index, name in enumerate(args.algorithms):
        x, mean, std, _, _ = aggregate(rows, name)
        color = colors(index % 10)
        axis.plot(x, mean, label=LABELS.get(name, name), color=color, linewidth=1.7)
        if not args.no_std_band:
            axis.fill_between(x, np.maximum(0, mean - std), np.minimum(1, mean + std),
                              color=color, alpha=0.14, linewidth=0)
    start, middle, end = 0, FIRST_STAGE_GENS, FIRST_STAGE_GENS + tau
    axis.axvline(middle, color="0.4", linestyle="--", linewidth=1, alpha=0.6)
    axis.set_xlim(start, end)
    axis.set_ylim(0, 1.02)
    axis.set_xticks(sorted(set(np.linspace(start, end, 5)) | {middle}))
    axis.set_xlabel("Iteration (FE/N)")
    axis.set_ylabel("MHV")
    axis.set_title(f"{problem_tag} — {SETTINGS[setting_index - 1].name}, tau={tau}")
    axis.text(middle / 2, 0.97, "Environment 1", ha="center", fontsize=9, color="0.35")
    axis.text(middle + tau / 2, 0.97, "Environment 2", ha="center", fontsize=9, color="0.35")
    axis.grid(True, color="0.8", alpha=0.55)


def save_figures(output_dir, problem_tag, tau, cases, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for setting_index, rows in cases.items():
        fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        draw_axis(axis, rows, setting_index, problem_tag, tau, args)
        axis.legend(loc="best", fontsize=8, frameon=True, fancybox=False)
        stem = f"MHV_Env1_Env2_{problem_tag}_Set{setting_index}_Tau{tau}"
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, len(cases), figsize=(6.0 * len(cases), 5.0),
                             sharey=True, squeeze=False)
    for axis, (setting_index, rows) in zip(axes[0], cases.items()):
        draw_axis(axis, rows, setting_index, problem_tag, tau, args)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=9)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    selection = "_".join(str(index) for index in cases)
    stem = f"MHV_Env1_Env2_{problem_tag}_Settings{selection}_Tau{tau}"
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", default=",".join(DEFAULT_ALGORITHMS),
                        help="Comma-separated algorithms (default: %(default)s).")
    parser.add_argument("--problems", default=",".join(PROBLEM_TAGS))
    parser.add_argument("--settings", default="1,2,3")
    parser.add_argument("--taus", default=",".join(map(str, TAUS)))
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--n", type=int, default=POPULATION_SIZE)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--pf-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--plot-only", action="store_true", help="Redraw saved traces without running algorithms.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing experiment traces.")
    parser.add_argument("--no-std-band", action="store_true")
    args = parser.parse_args()
    args.algorithms = csv_items(args.algorithms)
    args.problems = tuple(tag.upper() for tag in csv_items(args.problems))
    args.settings = csv_items(args.settings, int)
    args.taus = csv_items(args.taus, int)
    if args.runs < 1 or args.n < 2 or args.sample_every < 1 or args.pf_count < 1 or min(args.taus) < 1:
        parser.error("runs, taus, sample-every and pf-count must be positive; n must be at least 2.")
    for values, choices, label in ((args.algorithms, set(PORTED_ALGORITHMS) - {"RandomSearch"}, "algorithms"),
                                   (args.problems, PROBLEM_TAGS, "problems"),
                                   (args.settings, range(1, len(SETTINGS) + 1), "settings")):
        if set(values) - set(choices) or len(set(values)) != len(values):
            parser.error(f"Invalid or duplicate {label}: {values}; choose from {list(choices)}")
    if len(set(args.taus)) != len(args.taus):
        parser.error("Duplicate taus are not allowed.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Code version: env1-env2-stage2-fix-v1 | script: {Path(__file__).resolve()}", flush=True)
    print(f"Window: FE/N=0 to {FIRST_STAGE_GENS}+tau; settings={args.settings}; "
          f"problems={args.problems}; taus={args.taus}; runs={args.runs}; N={args.n}", flush=True)
    for problem_tag in args.problems:
        for tau in args.taus:
            cases = {}
            for setting_index in args.settings:
                stem = f"MHV_Env1_Env2_{problem_tag}_Set{setting_index}_Tau{tau}"
                path = args.output_dir / f"{stem}.csv"
                metadata_path = args.output_dir / f"{stem}.json"
                metadata = dict(problem=problem_tag, setting=setting_index, tau=tau, runs=args.runs,
                                n=args.n, seed=args.seed, pf_count=args.pf_count,
                                sample_every=args.sample_every, algorithms=list(args.algorithms),
                                first_stage_gens=FIRST_STAGE_GENS, sequence=SETTINGS[setting_index - 1].seq,
                                window=[0, FIRST_STAGE_GENS + tau], version=2)
                if args.plot_only or (path.exists() and not args.overwrite):
                    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if saved != metadata:
                        raise ValueError(f"Configuration differs from {metadata_path}. Use matching arguments, "
                                         "another output directory, or --overwrite.")
                    rows = read_rows(path, args, tau)
                    print(f"Loaded: {path}", flush=True)
                else:
                    rows = run_case(problem_tag, setting_index, tau, args)
                    save_rows(path, rows)
                    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
                save_summary(args.output_dir / f"{stem}_summary.csv", rows, args)
                cases[setting_index] = rows
            save_figures(args.output_dir, problem_tag, tau, cases, args)
            print(f"Saved figures and traces: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
