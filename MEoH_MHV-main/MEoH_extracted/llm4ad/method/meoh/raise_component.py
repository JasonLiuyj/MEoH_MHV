"""RAISE Algorithm 1 lines 10-13, using a finite training-only candidate pool.

This implements hard-instance augmentation, not the paper's full adversarial EA.
"""
from __future__ import annotations

from .tide import valid_score


def select_worst_instance(candidates, evaluate):
    """Candidates are (stable ID, evaluator). Scores maximize task-defined quality.

    A returned -inf is a measured failure and ranks worst. None/NaN (e.g. a
    timeout without a score) are skipped rather than mistaken for measurements.
    """
    import numpy as np
    worst = None
    for identifier, evaluator in candidates:
        score = evaluate(evaluator)
        if score is None or np.asarray(score).size == 0:
            continue
        quality = float(np.asarray(score).flat[0])
        if np.isnan(quality) or quality == float('inf'):
            continue
        if worst is None or quality < worst[1]:
            worst = (identifier, quality)
    return worst


def refresh_population(population, evaluate):
    """Prepare fresh scores atomically, including pending offspring and elites.

    Caller holds the experiment lock. No old-dataset scores survive a commit.
    """
    funcs = list({id(f): f for f in
                  population.population + population.elitist + population._next_gen_pop}.values())
    results = [(f, evaluate(f)) for f in funcs]
    valid = [(f, result) for f, result in results if valid_score(result[0])]
    if not valid:
        raise RuntimeError('RAISE refresh produced no valid heuristics; training update rolled back')
    import numpy as np
    for func, (score, elapsed) in results:
        func.score = score
        func.evaluate_time = elapsed
    survivors = [f for f, _ in valid]
    valid_ids = {id(f) for f in survivors}
    population._population = [f for f in population.population if id(f) in valid_ids]
    if not population._population:
        population._population = survivors[:population._pop_size]
    population._next_gen_pop = [f for f in population._next_gen_pop if id(f) in valid_ids]
    population._elitist = [f for f in survivors if not any(
        np.all(np.asarray(g.score) >= f.score) and np.any(np.asarray(g.score) > f.score)
        for g in survivors if g is not f)]
    population._last_generation_transition = None
