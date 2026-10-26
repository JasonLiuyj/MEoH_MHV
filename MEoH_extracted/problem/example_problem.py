"""A small runnable example showing the Evaluation contract.

Replace this module with the evaluator for the new domain. MEoH maximizes every
returned objective, so minimization metrics must be negated.
"""
from __future__ import annotations

import time

import numpy as np

from llm4ad.base import Evaluation


TEMPLATE_PROGRAM = '''import numpy as np

def estimate_energy(values: np.ndarray) -> float:
    """Estimate sum(values ** 2)."""
    return float(np.sum(np.abs(values)))
'''


class ExampleProblem(Evaluation):
    """Two-objective toy problem: maximize negative error and negative runtime."""

    def __init__(self, *, timeout_seconds: float = 10.0, seed: int = 2026):
        super().__init__(
            template_program=TEMPLATE_PROGRAM,
            task_description=(
                "Design estimate_energy(values) to accurately compute the sum of "
                "squared values while keeping execution time low. Return one float."
            ),
            timeout_seconds=timeout_seconds,
            safe_evaluate=True,
        )
        rng = np.random.default_rng(seed)
        self._instances = [rng.normal(size=n) for n in (32, 64, 128, 256)]

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs):
        absolute_errors: list[float] = []
        elapsed_times: list[float] = []
        for values in self._instances:
            expected = float(np.sum(values ** 2))
            start = time.perf_counter()
            actual = float(callable_func(values.copy()))
            elapsed_times.append(time.perf_counter() - start)
            if not np.isfinite(actual):
                return None
            absolute_errors.append(abs(actual - expected))

        # MEoH treats larger values as better for every objective.
        return [-float(np.mean(absolute_errors)), -float(np.mean(elapsed_times))]
