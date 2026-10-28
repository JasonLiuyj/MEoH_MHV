from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from core.population import Population


def subset(pop: Population, indices) -> Population:
    return Population(pop.decs[indices], pop.objs[indices], None if pop.cons is None else pop.cons[indices], None if pop.fes is None else pop.fes[indices])


def concat(*pops: Population) -> Population:
    pops = [p for p in pops if p is not None and len(p) > 0]
    if not pops:
        return Population(np.empty((0, 0)), np.empty((0, 0)))
    decs = np.vstack([p.decs for p in pops])
    objs = np.vstack([p.objs for p in pops])
    cons = None if any(p.cons is None for p in pops) else np.vstack([p.cons for p in pops])
    fes = None if any(p.fes is None for p in pops) else np.vstack([p.fes for p in pops])
    return Population(decs, objs, cons, fes)


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def nd_sort(points: np.ndarray, n_sort: int | float = np.inf) -> tuple[np.ndarray, int]:
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    front_no = np.full(n, np.inf)
    max_front = 0
    while np.sum(np.isfinite(front_no)) < min(n_sort, n):
        max_front += 1
        remaining = np.where(~np.isfinite(front_no))[0]
        # A front must be determined against the complete set of currently
        # remaining solutions.  Assigning a rank during the scan removes an
        # early solution as a possible dominator and incorrectly promotes its
        # dominated successors into the same front.
        current_front = []
        for i in remaining:
            dominated = any(j != i and dominates(points[j], points[i]) for j in remaining)
            if not dominated:
                current_front.append(i)
        if not current_front:
            raise RuntimeError("Nondominated sorting could not construct a front.")
        front_no[np.asarray(current_front, dtype=int)] = max_front
    return front_no.astype(float), max_front


def crowding_distance(points: np.ndarray, front_no: np.ndarray | None = None) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    n, m = points.shape
    if front_no is None:
        front_no = np.ones(n)
    distance = np.zeros(n)
    for front in np.unique(front_no[np.isfinite(front_no)]):
        idx = np.where(front_no == front)[0]
        if idx.size == 0:
            continue
        f = points[idx]
        d = np.zeros(idx.size)
        for j in range(m):
            order = np.argsort(f[:, j])
            d[order[0]] = np.inf
            d[order[-1]] = np.inf
            span = f[order[-1], j] - f[order[0], j]
            if span <= 1e-12:
                continue
            for k in range(1, idx.size - 1):
                d[order[k]] += (f[order[k + 1], j] - f[order[k - 1], j]) / span
        distance[idx] = d
    return distance


def tournament_selection(rng: np.random.Generator, n_select: int, front_no: np.ndarray, crowd: np.ndarray) -> np.ndarray:
    selected = np.empty(n_select, dtype=int)
    n = len(front_no)
    for i in range(n_select):
        a, b = rng.integers(0, n, 2)
        if front_no[a] < front_no[b] or (front_no[a] == front_no[b] and crowd[a] >= crowd[b]):
            selected[i] = a
        else:
            selected[i] = b
    return selected


def environmental_selection(pop: Population, n_select: int) -> Population:
    front_no, max_front = nd_sort(pop.objs, n_select)
    next_mask = front_no < max_front
    last = np.where(front_no == max_front)[0]
    need = n_select - int(np.sum(next_mask))
    if need > 0 and last.size:
        crowd = crowding_distance(pop.objs[last])
        chosen_last = last[np.argsort(-crowd)[:need]]
        next_mask[chosen_last] = True
    chosen = np.where(next_mask)[0][:n_select]
    return subset(pop, chosen)


def polynomial_mutation(
    rng: np.random.Generator,
    decs: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    pro_m: float | None = None,
    dis_m: float = 20.0,
) -> np.ndarray:
    y = np.asarray(decs, dtype=float).copy()
    n, d = y.shape
    if pro_m is None:
        pro_m = 1.0 / d
    lower = np.broadcast_to(lower, (n, d))
    upper = np.broadcast_to(upper, (n, d))
    site = rng.random((n, d)) < pro_m
    mu = rng.random((n, d))
    delta1 = (y - lower) / np.maximum(upper - lower, 1e-12)
    delta2 = (upper - y) / np.maximum(upper - lower, 1e-12)
    mask = site & (mu <= 0.5)
    y[mask] += (upper[mask] - lower[mask]) * (
        (2 * mu[mask] + (1 - 2 * mu[mask]) * (1 - delta1[mask]) ** (dis_m + 1)) ** (1 / (dis_m + 1)) - 1
    )
    mask = site & (mu > 0.5)
    y[mask] += (upper[mask] - lower[mask]) * (
        1 - (2 * (1 - mu[mask]) + 2 * (mu[mask] - 0.5) * (1 - delta2[mask]) ** (dis_m + 1)) ** (1 / (dis_m + 1))
    )
    return np.clip(y, lower, upper)


def operator_ga(rng: np.random.Generator, problem, parents: Population) -> Population:
    decs = parents.decs
    if decs.shape[0] == 0:
        return problem.initialization(rng, problem.n)
    half = decs.shape[0] // 2
    p1 = decs[:half]
    p2 = decs[half : 2 * half]
    if half == 0:
        return problem.initialization(rng, problem.n)
    n, d = p1.shape
    pro_c, dis_c = 1.0, 20.0
    beta = np.zeros((n, d))
    mu = rng.random((n, d))
    beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (dis_c + 1))
    beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (dis_c + 1))
    beta *= np.where(rng.random((n, d)) < 0.5, 1, -1)
    beta[rng.random((n, d)) < 0.5] = 1
    beta[np.broadcast_to(rng.random((n, 1)) > pro_c, (n, d))] = 1
    c1 = (p1 + p2) / 2 + beta * (p1 - p2) / 2
    c2 = (p1 + p2) / 2 - beta * (p1 - p2) / 2
    offspring = np.vstack([c1, c2])
    offspring = np.clip(offspring, problem.lower, problem.upper)
    offspring = polynomial_mutation(rng, offspring, problem.lower, problem.upper)
    return problem.evaluate(offspring)


def uniform_point(n: int, m: int) -> np.ndarray:
    h1 = 1
    while comb(h1 + m, m - 1) <= n:
        h1 += 1
    rows = []
    for c in combinations(range(h1 + m - 1), m - 1):
        vec = np.diff((-1, *c, h1 + m - 1)) - 1
        rows.append(vec / h1)
    w = np.asarray(rows, dtype=float)
    if h1 < m:
        h2 = 0
        while comb(h1 + m - 1, m - 1) + comb(h2 + m, m - 1) <= n:
            h2 += 1
        if h2 > 0:
            rows2 = []
            for c in combinations(range(h2 + m - 1), m - 1):
                vec = np.diff((-1, *c, h2 + m - 1)) - 1
                rows2.append(vec / h2)
            w = np.vstack([w, np.asarray(rows2) / 2 + 1 / (2 * m)])
    w[w < 1e-6] = 1e-6
    return w


def latin_hypercube(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    result = np.empty((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        result[:, j] = (perm + rng.random(n)) / n
    return result


def lhs_population(rng: np.random.Generator, problem, count: int) -> Population:
    if count <= 0:
        return Population(np.empty((0, problem.d)), np.empty((0, problem.m)))
    decs = latin_hypercube(rng, count, problem.d) * (problem.upper - problem.lower) + problem.lower
    return problem.evaluate(decs)


def associate(norm_objs: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Associate each normalized objective vector with a weight vector.

    This follows DTAEA.m:GetAssociation: choose the weight vector whose
    projection line has the smallest perpendicular distance to the point.
    """
    x = np.asarray(norm_objs, dtype=float)
    weights = np.asarray(w, dtype=float)
    w_norm_sq = np.sum(weights * weights, axis=1)
    w_norm_sq[w_norm_sq < 1e-10] = 1e-10
    scalar = (x @ weights.T) / w_norm_sq
    projection = scalar[:, :, None] * weights[None, :, :]
    distance = np.linalg.norm(x[:, None, :] - projection, axis=2)
    return np.argmin(distance, axis=1)


def associate_cosine(norm_objs: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Cosine-style region association used by KTDMOEA/LEC helper logic.

    This mirrors MATLAB snippets such as:
        cosine = (sol * W') / norm(sol)
        [~, regionIdx] = max(cosine)
    """
    x = np.asarray(norm_objs, dtype=float)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1e-12
    cosine = x @ np.asarray(w, dtype=float).T / norm
    return np.argmax(cosine, axis=1)


def estimate_density(objs: np.ndarray, w: np.ndarray) -> np.ndarray:
    zmin = np.min(objs, axis=0)
    zmax = np.max(objs, axis=0)
    norm = (objs - zmin) / (zmax - zmin + 1e-6)
    region = associate(norm, w)
    counts = np.bincount(region, minlength=w.shape[0])
    return counts[region]


def generate_solution(rng: np.random.Generator, x: np.ndarray, direction: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    dvec = direction.copy()
    dvec[np.abs(dvec) < 1e-10] = 1e-10
    para = np.empty_like(x)
    pos = dvec > 0
    para[pos] = (upper[pos] - x[pos]) / dvec[pos]
    para[~pos] = (lower[~pos] - x[~pos]) / dvec[~pos]
    step = np.min(para)
    if not np.isfinite(step) or step < 0:
        step = 0.0
    return np.clip(x + step * rng.random() * dvec, lower, upper)


def pca_directions(decs: np.ndarray, count: int) -> np.ndarray:
    if decs.shape[0] < 2 or count <= 0:
        return np.empty((0, decs.shape[1]))
    centered = decs - np.mean(decs, axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[: min(count, vt.shape[0])]


def null_space(a: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    u, s, vh = np.linalg.svd(np.asarray(a, dtype=float), full_matrices=True)
    if s.size == 0:
        return np.eye(a.shape[1])
    tol = rtol * s[0]
    rank = int(np.sum(s > tol))
    return vh[rank:].T.copy()
