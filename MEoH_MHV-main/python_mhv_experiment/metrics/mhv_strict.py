from __future__ import annotations

from functools import lru_cache
from math import comb, factorial, gamma, pi

import numpy as np

from core.ndsort import nondominated
from core.population import Population


def extract_objs(data: Population | np.ndarray) -> np.ndarray:
    if isinstance(data, Population):
        return np.asarray(data.objs, dtype=float)
    return np.asarray(data, dtype=float)


def hypervolume_2d(points: np.ndarray, ref_point: np.ndarray) -> float:
    points = points[np.argsort(points[:, 0])]
    volume = (ref_point[0] - points[-1, 0]) * (ref_point[1] - points[-1, 1])
    for i in range(points.shape[0] - 2, -1, -1):
        volume += (points[i + 1, 0] - points[i, 0]) * (ref_point[1] - points[i, 1])
    return float(max(volume, 0.0))


def hypervolume_3d_exact(points: np.ndarray, ref_point: np.ndarray) -> float:
    points = points[np.argsort(points[:, 0])]
    volume = 0.0
    for i in range(points.shape[0]):
        next_x = points[i + 1, 0] if i + 1 < points.shape[0] else ref_point[0]
        width = next_x - points[i, 0]
        if width > 0:
            yz = nondominated(points[: i + 1, 1:])
            volume += width * hypervolume_2d(yz, ref_point[1:])
    return float(max(volume, 0.0))


def hypervolume_mc(
    points: np.ndarray,
    ref_point: np.ndarray,
    sample_num: int,
    seed: int,
) -> float:
    min_value = np.min(points, axis=0)
    span = ref_point - min_value
    if np.any(span <= 0):
        return 0.0
    rng = np.random.default_rng(seed)
    samples = rng.random((sample_num, points.shape[1])) * span + min_value
    dominated = np.zeros(sample_num, dtype=bool)
    for point in points:
        dominated |= np.all(point <= samples, axis=1)
    return float(np.prod(span) * np.mean(dominated))


def cal_hv_hybrid(
    points: np.ndarray,
    ref_point: np.ndarray,
    is_pf: bool,
    pop_samples: int = 50_000,
    pf_samples: int = 1_000_000,
    seed: int = 1,
) -> float:
    points = nondominated(points)
    if points.size == 0:
        return 0.0
    if points.shape[1] == 2:
        return hypervolume_2d(points, ref_point)
    if points.shape[1] == 3 and points.shape[0] < 500:
        return hypervolume_3d_exact(points, ref_point)
    sample_num = pf_samples if is_pf else pop_samples
    return hypervolume_mc(points, ref_point, sample_num, seed)


def pf_signature(problem, filtered_pf: np.ndarray) -> tuple:
    current_indices = tuple(getattr(problem, "current_indices", ()))
    current_gen = max(0, getattr(problem, "fe", 0) - 1) // getattr(problem, "n", 1)
    rounded = np.round(filtered_pf, 10)
    return (
        type(problem).__name__,
        filtered_pf.shape[1],
        filtered_pf.shape[0],
        current_indices,
        current_gen,
        tuple(np.min(rounded, axis=0)),
        tuple(np.max(rounded, axis=0)),
        tuple(np.mean(rounded, axis=0)),
        tuple(np.std(rounded, axis=0)),
    )


def metric_active_indices(problem) -> list[int] | None:
    """Return the objective subset that produced the supplied population.

    ``DMixedSeq.get_optimum`` generates its PF from ``current_indices``.
    Therefore the MHV bounds and the analytical PF HV must use that same
    state.  Reconstructing a stage solely from FE is unsafe at a dynamic
    boundary: a population can still belong to the old environment while FE
    has already reached the first evaluation of the new one.
    """
    current_indices = getattr(problem, "current_indices", None)
    if current_indices:
        return list(current_indices)

    # Fallback for compatible static/problem wrappers that do not retain the
    # active subset explicitly.
    groups = getattr(problem, "obj_groups", None)
    if not groups:
        return None
    current_gen = max(0, getattr(problem, "fe", 0) - 1) // getattr(problem, "n", 1)
    first_stage_gens = getattr(problem, "first_stage_gens", 300)
    tau = getattr(problem, "taut", 1)
    stage = 0 if current_gen < first_stage_gens else 1 + (current_gen - first_stage_gens) // tau
    return groups[min(stage, len(groups) - 1)]


def true_normalized_pf_hv(problem, ref_point: np.ndarray) -> float | None:
    """Exact HV of the normalized true PF for supported minus benchmarks."""
    current_indices = metric_active_indices(problem)
    obj_map = getattr(problem, "obj_map", None)
    if not current_indices or obj_map is None:
        return None

    problem_name = obj_map[current_indices[0]]
    m = ref_point.size
    if not np.allclose(ref_point, ref_point[0]):
        return None
    margin = float(ref_point[0] - 1.0)
    if margin < 0:
        return None

    if problem_name == "DTLZ1":
        # y = 1-h, h >= 0, sum(h)=1
        unit_section = lambda k: 1.0 / factorial(k)
    elif problem_name in ("DTLZ2", "DTLZ3", "DTLZ4") or problem_name.startswith("WFG"):
        # y = 1-h, h >= 0, ||h||_2=1
        unit_section = lambda k: pi ** (k / 2) / gamma(k / 2 + 1) / (2**k)
    else:
        return None

    return float(
        sum(comb(m, k) * margin ** (m - k) * unit_section(k) for k in range(m + 1))
    )


def true_pf_bounds(problem, m: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return exact ideal/nadir points of supported minus benchmarks."""
    current_indices = metric_active_indices(problem)
    obj_map = getattr(problem, "obj_map", None)
    if not current_indices or obj_map is None or len(current_indices) != m:
        return None

    problem_name = obj_map[current_indices[0]]
    if any(obj_map[index] != problem_name for index in current_indices):
        return None

    if problem_name == "DTLZ1":
        k = problem.d - problem.global_max_m + 1
        g_max = 100 * k * (1 + 1.202602641454022)
        scale = 0.5 * (1 + g_max)
        return np.full(m, -scale), np.zeros(m)
    if problem_name in ("DTLZ2", "DTLZ4"):
        scale = 1 + 0.5**2 * 10
        return np.full(m, -scale), np.zeros(m)
    if problem_name == "DTLZ3":
        k = problem.d - problem.global_max_m + 1
        g_max = 100 * k * (1 + 1.202602641454022)
        return np.full(m, -(1 + g_max)), np.zeros(m)
    if problem_name.startswith("WFG"):
        problem_index = int(problem_name[3:])
        max_xm = 1.0
        if problem_index in (6, 9):
            c = np.ceil(problem.wfg_l / 2)
            max_xm = problem.wfg_l**2 / (c * (1 + 2 * problem.wfg_l - 2 * c))
        scales = problem.wfg_s[np.asarray(current_indices) - 1]
        return -(max_xm + scales), np.full(m, -max_xm)
    return None


@lru_cache(maxsize=128)
def _cached_pf_hv(signature: tuple, pf_bytes: bytes, shape: tuple[int, int], ref_tuple: tuple[float, ...]) -> float:
    filtered_pf = np.frombuffer(pf_bytes, dtype=np.float64).reshape(shape)
    return cal_hv_hybrid(filtered_pf, np.asarray(ref_tuple), is_pf=True, seed=17)


def mhv_strict(
    population: Population | np.ndarray,
    problem,
    pf_count: int = 5000,
    upper_bound: float = 1.1,
) -> tuple[float, float, float]:
    pop_obj = extract_objs(population)
    if pop_obj.size == 0:
        return 0.0, 0.0, 0.0
    pop_obj = np.asarray(pop_obj, dtype=float)

    pf = np.asarray(problem.get_optimum(pf_count), dtype=float)
    if pf.size == 0:
        return 0.0, 0.0, 0.0
    # Objective columns have meaning.  Truncating a PF from a different
    # environment merely because it has more columns compares two unrelated
    # objective spaces and can make MHV spuriously exceed one.
    if pf.shape[1] != pop_obj.shape[1]:
        return 0.0, 0.0, 0.0

    pf = pf[np.all(np.isfinite(pf), axis=1)]
    pf = np.unique(pf, axis=0)
    pf = nondominated(pf)
    if pf.size == 0:
        return 0.0, 0.0, 0.0

    exact_bounds = true_pf_bounds(problem, pop_obj.shape[1])
    if exact_bounds is None:
        z_star = np.min(pf, axis=0)
        z_nad = np.max(pf, axis=0)
    else:
        z_star, z_nad = exact_bounds
    gap = z_nad - z_star
    gap[gap < 1e-12] = 1e-12

    norm_pop = (pop_obj - z_star) / gap
    norm_pf = (pf - z_star) / gap
    norm_pop = norm_pop[np.all(np.isfinite(norm_pop), axis=1)]
    norm_pf = norm_pf[np.all(np.isfinite(norm_pf), axis=1)]
    if norm_pop.size == 0 or norm_pf.size == 0:
        return 0.0, 0.0, 0.0

    norm_pop = np.maximum(norm_pop, 0.0)
    norm_pf = np.maximum(norm_pf, 0.0)

    ref_point = np.ones(pop_obj.shape[1]) * upper_bound
    filtered_pop = norm_pop[np.all(norm_pop <= upper_bound, axis=1)]
    filtered_pf = norm_pf[np.all(norm_pf <= upper_bound, axis=1)]

    # MHV_Strict.m calls CalHV_Hybrid(..., true) for both population and PF.
    hv_pop = 0.0 if filtered_pop.size == 0 else cal_hv_hybrid(filtered_pop, ref_point, is_pf=True, seed=3)
    exact_pf_hv = true_normalized_pf_hv(problem, ref_point)
    if exact_pf_hv is not None:
        hv_pf = exact_pf_hv
    elif filtered_pf.size == 0:
        hv_pf = 0.0
    else:
        filtered_pf = np.ascontiguousarray(filtered_pf, dtype=np.float64)
        signature = pf_signature(problem, filtered_pf)
        hv_pf = _cached_pf_hv(signature, filtered_pf.tobytes(), filtered_pf.shape, tuple(ref_point))

    # MHV is a normalized quality ratio.  For the supported DTLZ/WFG fronts
    # a feasible population cannot dominate the true PF; clip only residual
    # floating-point / Monte-Carlo overshoot so reports remain in [0, 1].
    score = 0.0 if hv_pf < 1e-12 else float(np.clip(hv_pop / hv_pf, 0.0, 1.0))
    return float(score), float(hv_pop), float(hv_pf)
