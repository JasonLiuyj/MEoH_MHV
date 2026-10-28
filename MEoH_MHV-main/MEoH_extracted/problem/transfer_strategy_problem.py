"""MEoH task: evolve archive-selection strategies for dynamic MOO transfer.

The base optimizer, environment transition, transfer ratio, evaluation budget,
and MHV metric are fixed.  MEoH evolves only ``score_archive_solutions``.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEOH_MHV_ROOT = PROJECT_ROOT.parent
BENCHMARK_ROOT = MEOH_MHV_ROOT / "python_mhv_env1_env2"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from algorithms.dynamic_algorithms import (
    best_solutions,
    spea2sde_environmental_selection,
    spea2sde_fitness,
    spea2sde_mating_pool,
)
from core.evolution import concat, operator_ga, subset
from llm4ad.base import Evaluation
from metrics.mhv_strict import mhv_strict
from problems.d_mixed_seq import DMixedSeq


TEMPLATE_PROGRAM = '''import numpy as np

def score_archive_solutions(
    archive_decs: np.ndarray,
    archive_objs: np.ndarray,
    old_active: np.ndarray,
    new_active: np.ndarray,
) -> np.ndarray:
    """Return one finite priority score per archived solution; larger is better."""
    ideal = np.min(archive_objs, axis=0)
    nadir = np.max(archive_objs, axis=0)
    normalized = (archive_objs - ideal) / (nadir - ideal + 1e-12)
    convergence = -np.sum(normalized, axis=1)
    diversity = np.linalg.norm(
        archive_decs - np.mean(archive_decs, axis=0, keepdims=True), axis=1
    )
    return convergence + 0.05 * diversity
'''


TASK_DESCRIPTION = r'''
Problem background:
We solve a dynamic multi-objective optimization problem.  In environment 1,
SPEA2+SDE has produced an archive of decision vectors and their objective
values.  When the environment changes, the number and identity of active
objectives may change: objectives may be added, removed, or replaced.  Thus,
the old archive objective vectors belong to the old objective space and must
not be treated as complete objective information for the new environment.
We must choose which old decisions are worth transferring, handle them under
the new objective set, fill any remaining population slots with random
solutions, and let the fixed SPEA2+SDE optimizer recover.

Design score_archive_solutions(archive_decs, archive_objs, old_active,
new_active).  It must return a one-dimensional finite NumPy array containing
exactly one priority score for every archive row.  Larger means that the old
solution should be transferred first.  The function may use only the supplied
arrays, must be deterministic, and must not perform file/network access or
call the objective function.  It should balance convergence, diversity, and
decision-space coverage.  It should use old_active and new_active to recognize
objectives that are retained, added, or removed.  Values of newly added
objectives are unknown before re-evaluation, while removed objectives no longer
belong to the new objective space.

Everything except this archive-ranking function is fixed.  Candidate
strategies are evaluated on two objectives: (1) the mean normalized area under
the MHV curve from the newly constructed environment-2 population through the
end of environment 2 (larger is better), and (2) archive scoring and selection
time (smaller is better).  MEoH maximizes every returned score, so the evaluator
returns negative selection time as its second score.  Evaluation uses multiple
deterministic dynamic DTLZ2 cases.  Do not change the function name or signature.
'''


@dataclass(frozen=True)
class TransferCase:
    setting_name: str
    sequence: str
    old_active: tuple[int, ...]
    new_active: tuple[int, ...]
    archive_decs: np.ndarray
    archive_objs: np.ndarray


class TransferStrategyProblem(Evaluation):
    """Evaluate MEoH-generated archive ranking strategies.

    Both returned objectives are maximized:

    * objective 1: mean MHV over the complete environment-2 curve;
    * objective 2: negative transfer selection time (MEoH maximizes it).
    """

    _SETTINGS = (
        ("mild", "2,4; 2,4,5; 1,2,4,5; 2,3,5; 3,5"),
        ("moderate", "2,7; 2,5,7,10; 1,2,5,6,7,10; 2,5,6,9; 5,6"),
        ("severe", "3,8; 2,3,6,7,8; 1,2,3,4,5,6,7,8,9,10; 2,3,6,7,8; 3,8"),
    )

    def __init__(
        self,
        *,
        population_size: int = 150,
        first_stage_generations: int = 100,
        tau: int = 25,
        case_seed: int = 2026,
        pf_count: int = 1000,
        timeout_seconds: float = 30.0,
        prepare_cases: bool = True,
    ) -> None:
        if population_size < 4:
            raise ValueError("population_size must be at least 4")
        super().__init__(
            template_program=TEMPLATE_PROGRAM,
            task_description=TASK_DESCRIPTION,
            timeout_seconds=timeout_seconds,
            safe_evaluate=True,
            fork_proc=False,
        )
        self.population_size = int(population_size)
        self.first_stage_generations = int(first_stage_generations)
        self.tau = int(tau)
        self.case_seed = int(case_seed)
        self.pf_count = int(pf_count)
        self._cases = self._prepare_cases() if prepare_cases else []

    @staticmethod
    def _config_for_sequence(sequence: str) -> str:
        max_objective = max(
            int(item)
            for stage in sequence.split(";")
            for item in stage.split(",")
            if item.strip()
        )
        return f"1-{max_objective}:DTLZ2"

    def _new_problem(self, sequence: str) -> DMixedSeq:
        return DMixedSeq(
            taut=self.tau,
            seq_string=sequence,
            config_string=self._config_for_sequence(sequence),
            first_stage_gens=self.first_stage_generations,
            n=self.population_size,
        )

    def _prepare_cases(self) -> list[TransferCase]:
        """Run environment 1 once per setting and cache its offspring archive."""
        cases: list[TransferCase] = []
        for setting_index, (setting_name, sequence) in enumerate(self._SETTINGS):
            rng = np.random.default_rng(self.case_seed + setting_index)
            problem = self._new_problem(sequence)
            population = problem.initialization(rng)
            fitness = spea2sde_fitness(population.objs)
            archive = None
            boundary_fe = self.first_stage_generations * self.population_size

            while problem.fe < boundary_fe:
                mating = spea2sde_mating_pool(rng, fitness, problem.n)
                offspring = operator_ga(rng, problem, subset(population, mating))
                population, fitness = spea2sde_environmental_selection(
                    concat(population, offspring), problem.n
                )
                archive = offspring if archive is None else concat(archive, offspring)

            source = archive if archive is not None and len(archive) else population
            pareto_archive = best_solutions(source)
            groups = problem.obj_groups
            cases.append(
                TransferCase(
                    setting_name=setting_name,
                    sequence=sequence,
                    old_active=tuple(groups[0]),
                    new_active=tuple(groups[1]),
                    archive_decs=pareto_archive.decs.copy(),
                    archive_objs=pareto_archive.objs.copy(),
                )
            )
        return cases

    @property
    def prepared_case_count(self) -> int:
        return len(self._cases)

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs):
        if not self._cases:
            return None
        stage2_curve_scores: list[float] = []
        selection_times: list[float] = []

        for case_index, case in enumerate(self._cases):
            # The strategy is short, so the median of five repeats reduces
            # timer noise.  Timing includes scoring, validation, ranking, and
            # index selection, but excludes re-evaluation and SPEA2+SDE.
            case_timings: list[float] = []
            selected = None
            for _ in range(5):
                start = time.perf_counter()
                try:
                    priorities = np.asarray(
                        callable_func(
                            case.archive_decs.copy(),
                            case.archive_objs.copy(),
                            np.asarray(case.old_active, dtype=int),
                            np.asarray(case.new_active, dtype=int),
                        ),
                        dtype=float,
                    )
                except Exception:
                    return None

                if (
                    priorities.ndim != 1
                    or priorities.shape[0] != case.archive_decs.shape[0]
                    or not np.all(np.isfinite(priorities))
                ):
                    return None
                order = np.argsort(-priorities, kind="stable")
                # Match ArchiveSPEA2SDE: transfer the complete old first
                # front when it fits; otherwise keep at most N ranked points.
                selected = order[: min(self.population_size, len(order))]
                case_timings.append(time.perf_counter() - start)

            selection_times.append(float(np.median(case_timings)))
            if selected.size == 0:
                return None

            problem = self._new_problem(case.sequence)
            # Enter environment 2 deterministically.  The first evaluation
            # below updates current_indices from the first to the second group.
            problem.fe = self.first_stage_generations * self.population_size
            problem.last_t = 0
            rng = np.random.default_rng(self.case_seed + 10_000 + case_index)

            transferred = problem.evaluate(case.archive_decs[selected])
            if len(transferred) < problem.n:
                population = concat(
                    transferred,
                    problem.initialization(rng, problem.n - len(transferred)),
                )
            else:
                population = subset(transferred, np.arange(problem.n))

            stage2_mhv = [mhv_strict(
                population, problem, pf_count=self.pf_count
            )[0]]
            stage2_x = [0]
            fitness = spea2sde_fitness(population.objs)
            stage2_end_fe = (
                self.first_stage_generations + self.tau
            ) * self.population_size
            relative_generation = 0
            while problem.fe < stage2_end_fe:
                mating = spea2sde_mating_pool(rng, fitness, problem.n)
                offspring = operator_ga(rng, problem, subset(population, mating))
                population, fitness = spea2sde_environmental_selection(
                    concat(population, offspring), problem.n
                )
                relative_generation += 1
                if relative_generation % 5 == 0 or problem.fe >= stage2_end_fe:
                    stage2_x.append(relative_generation)
                    stage2_mhv.append(mhv_strict(
                        population, problem, pf_count=self.pf_count
                    )[0])

            if not np.all(np.isfinite(stage2_mhv)):
                return None
            # Normalized trapezoidal AUC covers the complete second-stage
            # interval while sampling every five generations plus both ends.
            if stage2_x[-1] == 0:
                curve_score = stage2_mhv[0]
            else:
                curve_score = np.trapz(stage2_mhv, stage2_x) / stage2_x[-1]
            stage2_curve_scores.append(float(curve_score))

        return [
            float(np.mean(stage2_curve_scores)),
            -float(np.mean(selection_times)),
        ]

