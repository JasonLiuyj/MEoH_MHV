from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.population import Population
from metrics.mhv_strict import mhv_strict


@dataclass
class RunResult:
    history: np.ndarray
    last_population: Population | None


class RandomSearch:
    """A tiny baseline used to verify the pure Python experiment pipeline.

    This is not a port of DTAEA, KTDMOEA, LEC, or STA.
    """

    name = "RandomSearch"

    def __init__(self, seed: int = 1, pf_count: int = 5000) -> None:
        self.rng = np.random.default_rng(seed)
        self.pf_count = pf_count

    def run(self, problem, max_fe: int, run_idx: int) -> RunResult:
        history: list[list[float]] = []
        last_population: Population | None = None
        total_stages = len(problem.obj_groups)

        while problem.fe < max_fe:
            decs = self.rng.random((problem.n, problem.d))
            population = problem.evaluate(decs)
            last_population = population
            self._try_record(history, population, problem, run_idx, total_stages)

        return RunResult(np.asarray(history, dtype=float), last_population)

    def _try_record(
        self,
        history: list[list[float]],
        population: Population,
        problem,
        run_idx: int,
        total_stages: int,
    ) -> None:
        if len(history) >= total_stages:
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
