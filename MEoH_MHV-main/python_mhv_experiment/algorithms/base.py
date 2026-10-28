from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.population import Population
from metrics.mhv_strict import mhv_strict


@dataclass
class RunResult:
    history: np.ndarray
    last_population: Population | None
    snapshots: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class StageInfo:
    index: int
    m: int
    active: list[int] = field(default_factory=list)
    frozen: list[int] = field(default_factory=list)


class DynamicAlgorithm:
    name = "DynamicAlgorithm"

    def __init__(self, seed: int = 1, pf_count: int = 5000) -> None:
        self.rng = np.random.default_rng(seed)
        self.pf_count = pf_count
        self.progress_interval = 25
        self._last_progress_gen = -1
        self.snapshots: dict[int, np.ndarray] = {}
        self.trace_interval = 0
        self.trace_callback = None
        self._last_trace_gen = -1

    def configure_trace(self, interval: int, callback) -> None:
        """Enable optional MHV sampling for an interactive comparison run."""
        self.trace_interval = max(0, int(interval))
        self.trace_callback = callback
        self._last_trace_gen = -1

    def try_trace(self, population: Population, problem, run_idx: int, force: bool = False) -> None:
        """Report current-population MHV at the configured FE/N interval.

        The method has no effect in ordinary experiments unless a plotting
        script explicitly installs a callback through ``configure_trace``.
        """
        if self.trace_callback is None or self.trace_interval <= 0 or population is None or len(population) == 0:
            return
        # Do not sample a transition transient.  The objective columns in a
        # pre-change population and the problem's current PF must describe
        # the same active set; otherwise an MHV point has no meaning.
        current_indices = getattr(problem, "current_indices", None)
        if current_indices and population.objs.shape[1] != len(current_indices):
            return
        generation = max(0, problem.fe - 1) // problem.n
        if generation == self._last_trace_gen:
            return
        if not force and generation % self.trace_interval != 0:
            return
        score, _, _ = mhv_strict(population, problem, pf_count=self.pf_count)
        self.trace_callback(int(generation), float(score), int(run_idx))
        self._last_trace_gen = generation

    def run(self, problem, max_fe: int, run_idx: int) -> RunResult:
        raise NotImplementedError

    def stage_signature(self, problem) -> tuple[int, int]:
        current_gen = problem.fe // problem.n
        if current_gen < problem.first_stage_gens:
            t = 0
        else:
            t = 1 + (current_gen - problem.first_stage_gens) // problem.taut
        obj_stage = min(t + 1, len(problem.obj_groups))
        frozen_stage = min(t + 1, len(problem.frozen_groups))
        return obj_stage, frozen_stage

    def stage_info(self, problem, population: Population, signature: tuple[int, int]) -> StageInfo:
        obj_stage, frozen_stage = signature
        active = problem.obj_groups[obj_stage - 1] if problem.obj_groups else []
        frozen = problem.frozen_groups[frozen_stage - 1] if problem.frozen_groups else []
        m = len(active) if active else population.objs.shape[1]
        return StageInfo(obj_stage, m, active, frozen)

    def try_record(self, history: list[list[float]], population: Population, problem, run_idx: int) -> None:
        self.try_trace(population, problem, run_idx)
        total_stages = len(problem.obj_groups)
        if len(history) >= total_stages or population is None or len(population) == 0:
            return
        stage_to_save = len(history) + 1
        current_gen = max(0, problem.fe - 1) // problem.n
        target_gen = problem.target_gen_for_stage(stage_to_save)
        current_stage = problem.stage_by_fe_minus_one()
        if current_stage != stage_to_save or current_gen < target_gen:
            return
        if history and history[-1][0] == target_gen:
            return
        score, hv_pop, hv_pf = mhv_strict(population, problem, pf_count=self.pf_count)
        print(
            f"   [Run {run_idx}] Stage {stage_to_save} Locked. "
            f"TargetGen {target_gen} | TriggerGen {current_gen} | "
            f"HV_Pop: {hv_pop:.4f} | HV_PF: {hv_pf:.4f} | MHV: {score:.4f}"
        )
        history.append([target_gen, current_gen, score])
        self.snapshots[stage_to_save] = population.objs.copy()

    def record_stage(
        self,
        history: list[list[float]],
        population: Population,
        problem,
        run_idx: int,
        stage_to_save: int,
        trigger_gen: int,
    ) -> None:
        if stage_to_save < 1 or stage_to_save > len(problem.obj_groups):
            return
        target_gen = problem.target_gen_for_stage(stage_to_save)
        if any(row[0] == target_gen for row in history):
            return
        old_fe = problem.fe
        problem.fe = target_gen * problem.n + 1
        score, hv_pop, hv_pf = mhv_strict(population, problem, pf_count=self.pf_count)
        problem.fe = old_fe
        print(
            f"   [Run {run_idx}] Stage {stage_to_save} Locked. "
            f"TargetGen {target_gen} | TriggerGen {trigger_gen} | "
            f"HV_Pop: {hv_pop:.4f} | HV_PF: {hv_pf:.4f} | MHV: {score:.4f}"
        )
        history.append([target_gen, trigger_gen, score])
        self.snapshots[stage_to_save] = population.objs.copy()

    def record_before_change(
        self,
        history: list[list[float]],
        population: Population,
        problem,
        run_idx: int,
        old_stage: int,
    ) -> None:
        """Lock the last available population before responding to a change."""
        expected_m = len(problem.obj_groups[old_stage - 1])
        if population.objs.shape[1] != expected_m:
            raise RuntimeError(
                f"{self.name} attempted to record stage {old_stage} with "
                f"{population.objs.shape[1]} objectives; expected {expected_m}."
            )
        # Preserve the final old-environment point even when it is not an
        # exact multiple of the requested plotting interval.
        self.try_trace(population, problem, run_idx, force=True)
        trigger_gen = max(0, problem.fe - 1) // problem.n
        self.record_stage(history, population, problem, run_idx, old_stage, trigger_gen)

    def validate_stage_transition(
        self,
        problem,
        old_signature: tuple[int, int],
        new_signature: tuple[int, int],
    ) -> None:
        if new_signature[0] != old_signature[0] + 1:
            raise RuntimeError(
                f"{self.name} skipped stages before change response: "
                f"{old_signature} -> {new_signature} at FE={problem.fe}."
            )

    def validate_response_stage(self, problem, expected_signature: tuple[int, int]) -> None:
        actual_signature = self.stage_signature(problem)
        if actual_signature != expected_signature:
            raise RuntimeError(
                f"{self.name} change response crossed a stage boundary: "
                f"expected {expected_signature}, got {actual_signature} at FE={problem.fe}. "
                "Increase tau or reduce response evaluation cost."
            )

    def report_progress(self, problem, run_idx: int, max_fe: int) -> None:
        current_gen = max(0, problem.fe - 1) // problem.n
        if current_gen == self._last_progress_gen:
            return
        if (
            self._last_progress_gen < 0
            or current_gen == 0
            or current_gen % self.progress_interval == 0
            or problem.fe >= max_fe
        ):
            stage = problem.stage_by_fe_minus_one()
            max_gen = max_fe // problem.n
            print(
                f"      [{self.name}] Run {run_idx} progress: "
                f"Gen {current_gen}/{max_gen}, FE {problem.fe}/{max_fe}, Stage {stage}",
                flush=True,
            )
            self._last_progress_gen = current_gen

    def final_record(self, history: list[list[float]], population: Population, problem, run_idx: int) -> None:
        total_stages = len(problem.obj_groups)
        if len(history) != total_stages - 1:
            return
        trigger_gen = max(0, problem.fe - 1) // problem.n
        self.record_stage(history, population, problem, run_idx, total_stages, trigger_gen)
