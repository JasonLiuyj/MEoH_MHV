from __future__ import annotations

import numpy as np

from algorithms.base import DynamicAlgorithm, RunResult
from core.evolution import (
    associate,
    associate_cosine,
    concat,
    crowding_distance,
    environmental_selection,
    estimate_density,
    generate_solution,
    latin_hypercube,
    lhs_population,
    nd_sort,
    operator_ga,
    pca_directions,
    polynomial_mutation,
    subset,
    tournament_selection,
    uniform_point,
)
from core.population import Population


def ensure_size(algorithm: DynamicAlgorithm, problem, population: Population, size: int) -> Population:
    if len(population) == size:
        return population
    if len(population) > size:
        front, max_front = nd_sort(population.objs, size)
        if np.all(np.isinf(front)):
            return subset(population, np.arange(size))
        return environmental_selection(population, size)
    fill = problem.initialization(algorithm.rng, size - len(population))
    return concat(population, fill)


def unique_by_objs(pop: Population) -> Population:
    _, idx = np.unique(pop.objs, axis=0, return_index=True)
    return subset(pop, np.sort(idx))


def random_mating_pool(rng: np.random.Generator, population_size: int, offspring_size: int) -> np.ndarray:
    return rng.integers(0, population_size, offspring_size)


def complete_old_population_evaluation(
    problem, old_pop: Population, ps_indices: np.ndarray, evaluated_ps: Population
) -> Population:
    """Finish re-evaluating old_pop without evaluating its PS twice."""
    other_indices = np.setdiff1d(np.arange(len(old_pop)), ps_indices, assume_unique=True)
    if other_indices.size == 0:
        return evaluated_ps
    return concat(evaluated_ps, problem.evaluate(old_pop.decs[other_indices]))


def lec_candidate_expansion_directions(
    rng: np.random.Generator, pst_decs: np.ndarray, old_m: int, population_size: int
) -> np.ndarray:
    """Implement LEC Algorithm 1's LHS-plus-linear-system construction."""
    d = pst_decs.shape[1]
    equation_count = old_m - 1
    free_count = d - equation_count
    if pst_decs.shape[0] < 2 or equation_count <= 0 or free_count <= 0:
        return np.empty((0, d))

    eigenvectors = pca_directions(pst_decs, equation_count)
    if eigenvectors.shape[0] != equation_count:
        return np.empty((0, d))

    # Algorithm 1 assigns the first n-m(t)+1 variables with LHS samples in
    # [0, 1], then solves the remaining m(t)-1 variables from D * ev_i = 0.
    coefficient_free = eigenvectors[:, :free_count]
    coefficient_dependent = eigenvectors[:, free_count:]
    if np.linalg.matrix_rank(coefficient_dependent) != equation_count:
        return np.empty((0, d))
    free_values = latin_hypercube(rng, population_size, free_count)
    try:
        dependent_values = np.linalg.solve(
            coefficient_dependent, -(coefficient_free @ free_values.T)
        ).T
    except np.linalg.LinAlgError:
        return np.empty((0, d))

    directions = np.hstack([free_values, dependent_values])
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions[norms[:, 0] > 1e-10] / norms[norms[:, 0] > 1e-10]


def solution_at_boundary(x: np.ndarray, direction: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """KTDMOEA Eq. (5): move from x along direction to the search boundary."""
    dvec = np.asarray(direction, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        para = np.where(dvec > 0, (upper - x) / dvec, (lower - x) / dvec)
    para[(para < 1e-6) | ~np.isfinite(para)] = np.inf
    step = np.min(para)
    if not np.isfinite(step):
        step = 0.0
    return np.clip(x + step * dvec, lower, upper)


def dtaea_ca_update(mixed: Population, w: np.ndarray, n: int) -> Population:
    front, _ = nd_sort(mixed.objs, len(mixed))
    selected: list[int] = []
    level = 1
    while len(selected) + int(np.sum(front == level)) <= n:
        selected.extend(np.where(front == level)[0].tolist())
        level += 1
    last = np.where(front == level)[0]
    candidates = subset(mixed, np.asarray(selected + last.tolist(), dtype=int))
    if len(candidates) <= n:
        return candidates

    zmin, zmax = np.min(candidates.objs, axis=0), np.max(candidates.objs, axis=0)
    norm = (candidates.objs - zmin) / (zmax - zmin + 1e-6)
    region = associate(norm, w)
    keep = np.ones(len(candidates), dtype=bool)
    while np.sum(keep) > n:
        counts = np.bincount(region[keep], minlength=w.shape[0])
        crowded = int(np.argmax(counts))
        members = np.where(keep & (region == crowded))[0]
        sub_w = np.maximum(w[crowded], 1e-6)
        tch = np.max(np.abs(norm[members]) / sub_w, axis=1)
        keep[members[np.argmax(tch)]] = False
    return subset(candidates, np.where(keep)[0])


def evenly_select_ps(pst: Population, count: int, population_size: int) -> Population:
    """Select a prescribed-size subset of a PS by DTAEA subspace density.

    KTDMOEA and LEC both require solutions to be selected *evenly* from
    the old PS in objective space.  Crowding distance is not the
    decomposition-based density estimator specified by those papers; CA's
    update operator supplies the required reference-vector selection.
    """
    if count <= 0:
        return subset(pst, np.empty(0, dtype=int))
    if len(pst) <= count:
        return pst
    w = uniform_point(population_size, pst.objs.shape[1])
    return dtaea_ca_update(pst, w, count)


def spea2sde_fitness(objs: np.ndarray) -> np.ndarray:
    pop_obj = np.asarray(objs, dtype=float)
    n = pop_obj.shape[0]
    if n == 0:
        return np.empty(0)

    dominate = np.all(pop_obj[:, None, :] <= pop_obj[None, :, :], axis=2) & np.any(
        pop_obj[:, None, :] < pop_obj[None, :, :], axis=2
    )
    np.fill_diagonal(dominate, False)
    strength = np.sum(dominate, axis=1)
    raw = strength @ dominate

    diff = np.minimum(pop_obj[:, None, :] - pop_obj[None, :, :], 0.0)
    distance = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distance, np.inf)

    sorted_distance = np.sort(distance, axis=1)
    k = max(0, int(np.floor(np.sqrt(n))) - 1)
    density = 1.0 / (sorted_distance[:, k] + 2.0)
    return raw + density


def spea2sde_truncation(objs: np.ndarray, k_delete: int) -> np.ndarray:
    pop_obj = np.asarray(objs, dtype=float)
    n = pop_obj.shape[0]
    diff = np.minimum(pop_obj[:, None, :] - pop_obj[None, :, :], 0.0)
    distance = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distance, np.inf)

    delete = np.zeros(n, dtype=bool)
    while int(np.sum(delete)) < k_delete:
        remain = np.where(~delete)[0]
        temp = np.sort(distance[np.ix_(remain, remain)], axis=1)
        order = np.lexsort(temp.T[::-1])
        delete[remain[order[0]]] = True
    return delete


def spea2sde_environmental_selection(population: Population, n: int) -> tuple[Population, np.ndarray]:
    fitness = spea2sde_fitness(population.objs)
    keep = fitness < 1
    if int(np.sum(keep)) < n:
        rank = np.argsort(fitness)
        keep[rank[:n]] = True
    elif int(np.sum(keep)) > n:
        selected = np.where(keep)[0]
        delete = spea2sde_truncation(population.objs[selected], int(np.sum(keep)) - n)
        keep[selected[delete]] = False
    selected_pop = subset(population, np.where(keep)[0])
    return selected_pop, fitness[keep]


def spea2sde_mating_pool(rng: np.random.Generator, fitness: np.ndarray, n_select: int) -> np.ndarray:
    n = len(fitness)
    selected = np.empty(n_select, dtype=int)
    for i in range(n_select):
        a, b = rng.integers(0, n, 2)
        selected[i] = a if fitness[a] <= fitness[b] else b
    return selected


def nsga2_environmental_selection(population: Population, n: int) -> tuple[Population, np.ndarray, np.ndarray]:
    """NSGA-II survival selection together with the next tournament state."""
    selected = environmental_selection(population, n)
    front, _ = nd_sort(selected.objs, len(selected))
    crowd = crowding_distance(selected.objs, front)
    return selected, front, crowd


def best_solutions(population: Population) -> Population:
    if population is None or len(population) == 0:
        return Population(np.empty((0, 0)), np.empty((0, 0)))
    front, _ = nd_sort(population.objs, 1)
    best = subset(population, np.where(front == 1)[0])
    return best if len(best) else population


class KTDMOEA(DynamicAlgorithm):
    name = "KTDMOEA"

    def __init__(self, seed: int = 1, pf_count: int = 5000, theta: int = 2) -> None:
        super().__init__(seed, pf_count)
        self.theta = theta

    def run(self, problem, max_fe: int, run_idx: int) -> RunResult:
        population = problem.initialization(self.rng)
        current_stage = self.stage_signature(problem)
        current_info = self.stage_info(problem, population, current_stage)
        w = uniform_point(problem.n, current_info.m)
        history: list[list[float]] = []

        while problem.fe < max_fe:
            self.report_progress(problem, run_idx, max_fe)
            next_stage = self.stage_signature(problem)
            if next_stage != current_stage:
                self.validate_stage_transition(problem, current_stage, next_stage)
                new_info = self.stage_info(problem, population, next_stage)
                self.record_before_change(history, population, problem, run_idx, current_info.index)
                w = uniform_point(problem.n, new_info.m)
                if new_info.m > current_info.m:
                    population = self.ps_expansion(population, problem, current_info.m)
                elif new_info.m < current_info.m:
                    population = self.ps_contraction(population, problem)
                else:
                    population = problem.evaluate(population.decs)
                self.validate_response_stage(problem, next_stage)
                current_stage, current_info = next_stage, new_info
                self.try_record(history, population, problem, run_idx)
                continue

            mating = random_mating_pool(self.rng, len(population), problem.n)
            offspring = operator_ga(self.rng, problem, subset(population, mating))
            population = dtaea_ca_update(concat(population, offspring), w, problem.n)
            self.try_record(history, population, problem, run_idx)

        self.final_record(history, population, problem, run_idx, max_fe=max_fe)
        return RunResult(np.asarray(history, dtype=float), population, dict(self.snapshots))

    def update_ca(self, mixed: Population, n: int) -> Population:
        mixed = unique_by_objs(mixed)
        front, _ = nd_sort(mixed.objs, 1)
        pop = subset(mixed, np.where(front == 1)[0])
        if len(pop) > n:
            crowd = crowding_distance(pop.objs)
            pop = subset(pop, np.argsort(-crowd)[:n])
        return pop if len(pop) else subset(mixed, np.arange(min(n, len(mixed))))

    def generate_solution_kt(self, x: np.ndarray, direction: np.ndarray, problem) -> np.ndarray:
        dvec = direction.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            diff_upper = (problem.upper - x) / dvec
            diff_lower = (problem.lower - x) / dvec
        candidates = np.concatenate([diff_upper, diff_lower])
        candidates[(candidates < 1e-6) | ~np.isfinite(candidates)] = np.inf
        step = np.min(candidates)
        if not np.isfinite(step):
            step = 1.0
        return np.clip(x + step * self.rng.random() * dvec, problem.lower, problem.upper)

    def search_expansion_directions(self, pst: Population, problem, pst_decs: np.ndarray) -> np.ndarray:
        if len(pst) == 0:
            return np.empty((0, problem.d))
        max_idx = np.unique(np.argmax(pst.objs, axis=0))
        xe = pst.decs[self.rng.choice(max_idx)]
        scouts = polynomial_mutation(self.rng, np.tile(xe, (problem.n, 1)), problem.lower, problem.upper)
        p_var = problem.evaluate(scouts)
        pst_new = problem.evaluate(pst_decs)
        # Algorithm 2 first removes detective solutions dominated by other
        # detective solutions.  Testing only against PSt would incorrectly
        # retain such points and turn them into spurious expansion directions.
        front, _ = nd_sort(p_var.objs, 1)
        p_var = subset(p_var, np.where(front == 1)[0])
        dominated = np.zeros(len(p_var), dtype=bool)
        for i, obj in enumerate(p_var.objs):
            dominated[i] = np.any(np.all(pst_new.objs <= obj, axis=1) & np.any(pst_new.objs < obj, axis=1))
        keep = np.where(~dominated)[0]
        if keep.size == 0:
            return np.empty((0, problem.d))
        p_non = subset(p_var, keep)
        w = uniform_point(problem.n, p_non.objs.shape[1])
        all_objs = np.vstack([pst_new.objs, p_non.objs])
        zmin, zmax = np.min(all_objs, axis=0), np.max(all_objs, axis=0)
        pst_norm = (pst_new.objs - zmin) / (zmax - zmin + 1e-6)
        non_norm = (p_non.objs - zmin) / (zmax - zmin + 1e-6)
        occupied = np.zeros(w.shape[0], dtype=bool)
        occupied[associate(pst_norm, w)] = True
        regions = associate(non_norm, w)
        p_non = subset(p_non, np.where(~occupied[regions])[0])
        if len(p_non) == 0:
            return np.empty((0, problem.d))
        dirs = p_non.decs - xe
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-10
        dirs = dirs[valid] / norms[valid]
        return np.unique(dirs, axis=0)

    def ps_expansion(self, old_pop: Population, problem, old_m: int) -> Population:
        front, _ = nd_sort(old_pop.objs, 1)
        ps_indices = np.where(front == 1)[0]
        pst = subset(old_pop, ps_indices)
        if len(pst) == 0:
            pst = old_pop
            ps_indices = np.arange(len(old_pop))
        dirs = self.search_expansion_directions(pst, problem, pst.decs)
        # Algorithm 1 in KTDMOEA transfers the old population unchanged
        # when no valid expansion direction is found.  Do not turn this
        # case into a restart by filling a (possibly small) first front with
        # random solutions.
        if dirs.size == 0:
            return problem.evaluate(old_pop.decs)
        transferred = []
        n_base = max(1, (problem.n - old_m) // max(1, dirs.shape[0] * self.theta))
        bases = evenly_select_ps(pst, min(len(pst), n_base), problem.n).decs
        for dvec in dirs:
            for x in bases:
                for _ in range(self.theta):
                    transferred.append(self.generate_solution_kt(x, dvec, problem))
        transferred = np.asarray(transferred).reshape(-1, problem.d) if transferred else np.empty((0, problem.d))
        need = problem.n - transferred.shape[0]
        if need > 0:
            # Algorithm 1 fills the remaining slots with solutions from the
            # old PS, rather than injecting newly random points.  old_pop has
            # N members even when its first front is smaller than N.
            # The paper fills with PSt (the old Pareto-optimal set), not with
            # dominated members from the complete old population.
            evenly = evenly_select_ps(pst, min(len(pst), need), problem.n)
            fill = evenly.decs[np.resize(np.arange(len(evenly)), need)]
            transferred = np.vstack([transferred, fill])
        return problem.evaluate(transferred[: problem.n])

    def ps_contraction(self, old_pop: Population, problem) -> Population:
        # Algorithm 3 takes the old Pareto set PSt as its input, rather than
        # all members of the current population.
        front_old, _ = nd_sort(old_pop.objs, 1)
        pst = subset(old_pop, np.where(front_old == 1)[0])
        if len(pst) == 0:
            pst = old_pop
        updated = problem.evaluate(pst.decs)
        front, _ = nd_sort(updated.objs, 1)
        p_non = subset(updated, np.where(front == 1)[0])
        if len(p_non) == 0:
            p_non = updated
        decs = [*p_non.decs]
        extreme = np.unique(np.argmax(p_non.objs, axis=0))
        for idx in extreme:
            xe = p_non.decs[idx]
            dist = np.linalg.norm(p_non.objs - p_non.objs[idx], axis=1)
            dist[idx] = np.inf
            close = p_non.decs[np.argmin(dist)]
            vec = xe - close
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                decs.append(solution_at_boundary(xe, vec / norm, problem.lower, problem.upper))
        while len(decs) < problem.n:
            pool = np.asarray(decs)
            if len(pool) >= 2:
                a, b = self.rng.choice(len(pool), 2, replace=False)
                x = pool[a] + self.rng.random() * (pool[a] - pool[b])
                decs.append(np.clip(x, problem.lower, problem.upper))
            else:
                decs.append(problem.lower + self.rng.random(problem.d) * (problem.upper - problem.lower))
        return problem.evaluate(np.asarray(decs[: problem.n]))


class STA(KTDMOEA):
    name = "STA"

    def __init__(self, seed: int = 1, pf_count: int = 5000, theta: int = 2, gamma: float = 0.2) -> None:
        super().__init__(seed, pf_count, theta)
        self.gamma = gamma
        self.archive: list[tuple[int, Population]] = []

    def run(self, problem, max_fe: int, run_idx: int) -> RunResult:
        population = problem.initialization(self.rng)
        current_stage = self.stage_signature(problem)
        current_info = self.stage_info(problem, population, current_stage)
        w = uniform_point(problem.n, current_info.m)
        history: list[list[float]] = []

        while problem.fe < max_fe:
            self.report_progress(problem, run_idx, max_fe)
            next_stage = self.stage_signature(problem)
            if next_stage != current_stage:
                self.validate_stage_transition(problem, current_stage, next_stage)
                new_info = self.stage_info(problem, population, next_stage)
                self.record_before_change(history, population, problem, run_idx, current_info.index)
                self.update_archive(current_info.m, population)
                w = uniform_point(problem.n, new_info.m)
                # STA is invoked for every environmental change.  Its
                # M_tr == M_c branch copies the selected historical source;
                # bypassing it merely because the objective count is equal
                # contradicts Algorithm 1 of the STA paper.
                population = self.similarity_transfer(problem, new_info.m)
                self.validate_response_stage(problem, next_stage)
                current_stage, current_info = next_stage, new_info
                self.try_record(history, population, problem, run_idx)
                continue
            mating = random_mating_pool(self.rng, len(population), problem.n)
            offspring = operator_ga(self.rng, problem, subset(population, mating))
            population = dtaea_ca_update(concat(population, offspring), w, problem.n)
            self.try_record(history, population, problem, run_idx)

        self.final_record(history, population, problem, run_idx, max_fe=max_fe)
        return RunResult(np.asarray(history, dtype=float), population, dict(self.snapshots))

    def update_archive(self, m: int, pop: Population) -> None:
        # STA Algorithm 1 stores the historical Pareto set PSt, rather than
        # the full population which can contain dominated solutions.
        front, _ = nd_sort(pop.objs, 1)
        ps = subset(pop, np.where(front == 1)[0])
        if len(ps) == 0:
            ps = pop
        self.archive.append(
            (
                m,
                Population(
                    ps.decs.copy(),
                    ps.objs.copy(),
                    None if ps.cons is None else ps.cons.copy(),
                    None if ps.fes is None else ps.fes.copy(),
                ),
            )
        )

    def similarity_transfer(self, problem, current_m: int) -> Population:
        if not self.archive:
            return problem.initialization(self.rng)
        source_m, source = min(self.archive, key=lambda item: abs(item[0] - current_m))
        if source_m < current_m:
            transferred, n_exp = self.ktdmoea_expansion_decs(source, problem, source_m)
            if n_exp < (current_m - source_m) / 2:
                n_random = round(self.gamma * problem.n)
                random_decs = problem.lower + self.rng.random((n_random, problem.d)) * (problem.upper - problem.lower)
                if n_random > 0 and transferred.shape[0] >= problem.n:
                    replace = self.rng.choice(transferred.shape[0], n_random, replace=False)
                    transferred[replace] = random_decs
                elif n_random > 0:
                    transferred = np.vstack([transferred, random_decs])
            decs = self.fit_decs_to_n(transferred, problem)
            return problem.evaluate(decs)
        if source_m > current_m:
            contracted = self.lec_contraction_population(source, problem, source_m)
            return self.fit_population_to_n(contracted, problem.n)
        decs = self.adjust_decs_dimension(source.decs, problem.d, problem.lower, problem.upper)
        return problem.evaluate(self.fit_decs_to_n(decs, problem))

    def fit_decs_to_n(self, decs: np.ndarray, problem) -> np.ndarray:
        decs = np.asarray(decs, dtype=float)
        if decs.ndim == 1:
            decs = decs.reshape(1, -1)
        decs = self.adjust_decs_dimension(decs, problem.d, problem.lower, problem.upper)
        if decs.shape[0] < problem.n:
            need = problem.n - decs.shape[0]
            if decs.shape[0] == 0:
                return problem.lower + self.rng.random((problem.n, problem.d)) * (problem.upper - problem.lower)
            # A transferred PSt can have fewer than N points.  Retain the
            # paper's knowledge-transfer semantics by reusing PSt members
            # instead of adding unrelated random decision vectors.
            decs = np.vstack([decs, decs[np.resize(np.arange(decs.shape[0]), need)]])
        elif decs.shape[0] > problem.n:
            decs = decs[self.rng.permutation(decs.shape[0])[: problem.n]]
        return decs

    @staticmethod
    def fit_population_to_n(population: Population, n: int) -> Population:
        """Resize an already evaluated transferred population without new FEs."""
        if len(population) == n:
            return population
        if len(population) > n:
            return subset(population, np.arange(n))
        if len(population) == 0:
            return population
        return concat(population, subset(population, np.resize(np.arange(len(population)), n - len(population))))

    def adjust_decs_dimension(self, old_decs: np.ndarray, target_d: int, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        old_decs = np.asarray(old_decs, dtype=float)
        n, old_d = old_decs.shape
        new = np.zeros((n, target_d))
        min_d = min(old_d, target_d)
        new[:, :min_d] = old_decs[:, :min_d]
        if target_d > old_d:
            new[:, old_d:] = self.rng.random((n, target_d - old_d)) * (upper[old_d:] - lower[old_d:]) + lower[old_d:]
        return np.clip(new, lower, upper)

    def ktdmoea_expansion_decs(self, source: Population, problem, source_m: int) -> tuple[np.ndarray, int]:
        dirs = self.search_expansion_directions(source, problem, source.decs)
        n_exp = dirs.shape[0]
        if n_exp == 0:
            return self.fit_decs_to_n(source.decs, problem), 0
        n_dir = dirs.shape[0]
        n_base = max(1, (problem.n - source_m) // max(1, n_dir * self.theta))
        base_sols = evenly_select_ps(source, min(len(source), n_base), problem.n).decs
        base_decs = self.adjust_decs_dimension(base_sols, problem.d, problem.lower, problem.upper)
        new = []
        for x in base_decs:
            for dvec in dirs:
                for _ in range(self.theta):
                    new.append(generate_solution(self.rng, x, dvec, problem.lower, problem.upper))
            if len(new) >= problem.n:
                break
        if len(new) < problem.n:
            rem = problem.n - len(new)
            source_decs = self.adjust_decs_dimension(source.decs, problem.d, problem.lower, problem.upper)
            evenly = evenly_select_ps(source, min(len(source), rem), problem.n)
            evenly_decs = self.adjust_decs_dimension(evenly.decs, problem.d, problem.lower, problem.upper)
            fill = evenly_decs[np.resize(np.arange(len(evenly_decs)), rem)]
            new.extend(fill)
        return np.asarray(new, dtype=float), n_exp

    def lec_contraction_population(self, source: Population, problem, source_m: int) -> Population:
        base_decs = self.adjust_decs_dimension(source.decs, problem.d, problem.lower, problem.upper)
        c_con = pca_directions(base_decs, max(0, source_m - 1))
        source_target = problem.evaluate(base_decs)
        if c_con.size == 0:
            return source_target
        x_base = subset(source_target, [self.rng.integers(0, len(source_target))])
        d_con = []
        for dvec in c_con:
            y_decs = [
                generate_solution(self.rng, x_base.decs[0], dvec, problem.lower, problem.upper)
                for _ in range(max(1, problem.n // max(1, source_m - 1)))
            ]
            y = problem.evaluate(np.asarray(y_decs))
            promising = np.any(np.all(y.objs <= x_base.objs[0], axis=1) & np.any(y.objs < x_base.objs[0], axis=1))
            if promising:
                d_con.append(dvec)
        if not d_con:
            # LEC (and consequently STA) copies the source population when
            # no promising contraction direction is learned.
            return source_target
        lambda_param = max(1, problem.n // 20)
        base = base_decs[self.rng.permutation(base_decs.shape[0])[: min(base_decs.shape[0], lambda_param)]]
        new = []
        ng = max(1, problem.n // max(1, len(d_con) * base.shape[0]))
        for x in base:
            for dvec in d_con:
                for _ in range(ng):
                    new.append(generate_solution(self.rng, x, dvec, problem.lower, problem.upper))
        generated = problem.evaluate(np.asarray(new, dtype=float))
        combined = concat(source_target, generated)
        w = uniform_point(problem.n, combined.objs.shape[1])
        return dtaea_ca_update(combined, w, problem.n)

    def get_expansion_dirs(self, source: Population, problem) -> tuple[np.ndarray, bool]:
        source_decs = self.adjust_decs_dimension(source.decs, problem.d, problem.lower, problem.upper)
        source_target = problem.evaluate(source_decs)
        extreme_decs = []
        for i in range(source.objs.shape[1]):
            rank = np.argsort(-source.objs[:, i])
            extreme_decs.append(source_decs[rank[0]])
        extreme_decs = np.unique(np.asarray(extreme_decs), axis=0)
        if extreme_decs.size == 0:
            return np.empty((0, problem.d)), False

        p_var_decs = []
        xe_list = []
        for xe in extreme_decs:
            mutants = polynomial_mutation(self.rng, np.tile(xe, (100, 1)), problem.lower, problem.upper)
            p_var_decs.append(mutants)
            xe_list.append(np.tile(xe, (100, 1)))
        p_var_decs = np.vstack(p_var_decs)
        xe_list = np.vstack(xe_list)
        p_var = problem.evaluate(p_var_decs)

        dominated = np.zeros(len(p_var), dtype=bool)
        for i, obj in enumerate(p_var.objs):
            dominated[i] = np.any(np.all(source_target.objs <= obj, axis=1) & np.any(source_target.objs < obj, axis=1))
        keep = np.where(~dominated)[0]
        if keep.size == 0:
            return np.empty((0, problem.d)), False

        w = uniform_point(problem.n, source_target.objs.shape[1])
        all_objs = np.vstack([source_target.objs, p_var.objs[keep]])
        zmin = np.min(all_objs, axis=0)
        zmax = np.max(all_objs, axis=0)
        zmax[zmax == zmin] += 1e-6
        source_norm = (source_target.objs - zmin) / (zmax - zmin)
        scout_norm = (p_var.objs[keep] - zmin) / (zmax - zmin)
        occupied = np.zeros(w.shape[0], dtype=bool)
        source_nonzero = np.linalg.norm(source_norm, axis=1) > 0
        occupied[associate_cosine(source_norm[source_nonzero], w)] = True
        scout_nonzero = np.linalg.norm(scout_norm, axis=1) > 0
        final_keep = np.ones(keep.shape[0], dtype=bool)
        final_keep[scout_nonzero] = ~occupied[associate_cosine(scout_norm[scout_nonzero], w)]
        valid_idx = keep[final_keep]
        if valid_idx.size == 0:
            return np.empty((0, problem.d)), False
        raw = p_var_decs[valid_idx] - xe_list[valid_idx]
        dirs = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-10)
        return np.unique(dirs, axis=0), True


class LEC(KTDMOEA):
    name = "LEC"

    def __init__(self, seed: int = 1, pf_count: int = 5000, lambda_param: int = 20) -> None:
        super().__init__(seed, pf_count)
        self.lambda_param = lambda_param
        self.w: np.ndarray | None = None
        self.neighbors: np.ndarray | None = None
        self.delta = 0.9
        self.t = 20

    def setup_weights(self, n: int, m: int) -> None:
        self.w = uniform_point(n, m)
        dist = np.linalg.norm(self.w[:, None, :] - self.w[None, :, :], axis=2)
        self.t = min(self.w.shape[0], 20)
        self.neighbors = np.argsort(dist, axis=1)[:, : self.t]

    def run(self, problem, max_fe: int, run_idx: int) -> RunResult:
        self.setup_weights(problem.n, problem.m)
        population = ensure_size(self, problem, problem.initialization(self.rng), problem.n)
        current_stage = self.stage_signature(problem)
        current_info = self.stage_info(problem, population, current_stage)
        history: list[list[float]] = []

        while problem.fe < max_fe:
            self.report_progress(problem, run_idx, max_fe)
            next_stage = self.stage_signature(problem)
            if next_stage != current_stage:
                self.validate_stage_transition(problem, current_stage, next_stage)
                new_info = self.stage_info(problem, population, next_stage)
                self.record_before_change(history, population, problem, run_idx, current_info.index)
                self.setup_weights(problem.n, new_info.m)
                if new_info.m > current_info.m:
                    population = self.ps_expansion(population, problem, current_info.m)
                elif new_info.m < current_info.m:
                    population = self.lec_contraction(population, problem, current_info.m)
                else:
                    population = problem.evaluate(population.decs)
                population = ensure_size(self, problem, population, problem.n)
                self.validate_response_stage(problem, next_stage)
                current_stage, current_info = next_stage, new_info
                self.try_record(history, population, problem, run_idx)
                continue

            population = ensure_size(self, problem, population, problem.n)
            pool = random_mating_pool(self.rng, len(population), problem.n)
            offspring = operator_ga(self.rng, problem, subset(population, pool))
            population = dtaea_ca_update(concat(population, offspring), self.w, problem.n)
            self.try_record(history, population, problem, run_idx)

        self.final_record(history, population, problem, run_idx, max_fe=max_fe)
        return RunResult(np.asarray(history, dtype=float), population, dict(self.snapshots))

    def ps_expansion(self, old_pop: Population, problem, old_m: int) -> Population:
        front, _ = nd_sort(old_pop.objs, 1)
        ps_indices = np.where(front == 1)[0]
        pst = subset(old_pop, ps_indices)
        if len(pst) == 0:
            pst = old_pop
            ps_indices = np.arange(len(old_pop))
        dirs_cand = lec_candidate_expansion_directions(self.rng, pst.decs, old_m, problem.n)

        old_new = problem.evaluate(pst.decs)
        d_exp = []
        if len(dirs_cand):
            # Algorithm 3 samples x from OldP, i.e. PSt re-evaluated in the
            # new environment.  This matters when the new environment also
            # projects/changes decision variables.
            x_base = old_new.decs[self.rng.integers(0, len(old_new))]
            for dvec in dirs_cand:
                y_dec = generate_solution(self.rng, x_base, dvec, problem.lower, problem.upper)
                y = problem.evaluate(y_dec.reshape(1, -1))
                dominated = np.any(np.all(old_new.objs <= y.objs[0], axis=1) & np.any(old_new.objs < y.objs[0], axis=1))
                if not dominated:
                    d_exp.append(dvec)

        if not d_exp:
            # Algorithm 3 of LEC: no promising direction means that the old
            # population is copied to the new environment.  old_new already
            # evaluated PSt, so evaluate only the complementary old members.
            return complete_old_population_evaluation(problem, old_pop, ps_indices, old_new)

        n_base = max(1, (problem.n - old_m) // max(1, len(d_exp) * self.theta))
        bases = evenly_select_ps(old_new, min(len(old_new), n_base), problem.n).decs
        new = []
        for dvec in d_exp:
            for x in bases:
                for _ in range(self.theta):
                    new.append(generate_solution(self.rng, x, dvec, problem.lower, problem.upper))
        if len(new) < problem.n:
            need = problem.n - len(new)
            # LEC's expansion fills from PSt, not from dominated members of
            # the complete old population.
            evenly = evenly_select_ps(pst, min(len(pst), need), problem.n)
            fill = evenly.decs[np.resize(np.arange(len(evenly)), need)]
            new.extend(fill)
        return problem.evaluate(np.asarray(new[: problem.n]))

    def lec_contraction(self, old_pop: Population, problem, old_m: int) -> Population:
        front, _ = nd_sort(old_pop.objs, 1)
        ps_indices = np.where(front == 1)[0]
        pst = subset(old_pop, ps_indices)
        if len(pst) == 0:
            pst = old_pop
            ps_indices = np.arange(len(old_pop))
        old_new = problem.evaluate(pst.decs)
        c_con = np.empty((0, problem.d))
        if len(pst) >= 2:
            c_con = pca_directions(pst.decs, max(0, old_m - 1))

        d_con = []
        if c_con.size and len(old_new):
            x_base = subset(old_new, [self.rng.integers(0, len(old_new))])
            for dvec in c_con:
                n_gen = max(1, problem.n // max(1, old_m - 1))
                y_decs = [
                    generate_solution(self.rng, x_base.decs[0], dvec, problem.lower, problem.upper)
                    for _ in range(n_gen)
                ]
                y = problem.evaluate(np.asarray(y_decs))
                promising = np.any(np.all(y.objs <= x_base.objs[0], axis=1) & np.any(y.objs < x_base.objs[0], axis=1))
                if promising:
                    d_con.append(dvec)

        if not d_con:
            return complete_old_population_evaluation(problem, old_pop, ps_indices, old_new)

        n_base = max(1, problem.n // self.lambda_param)
        base_idx = self.rng.permutation(len(pst))[: min(len(pst), n_base)]
        bases = pst.decs[base_idx]
        decs = []
        for x in bases:
            for dvec in d_con:
                ng = max(1, problem.n // max(1, len(d_con) * len(bases)))
                for _ in range(ng):
                    decs.append(generate_solution(self.rng, x, dvec, problem.lower, problem.upper))
        if not decs:
            return old_new
        generated = problem.evaluate(np.asarray(decs))
        return dtaea_ca_update(concat(old_new, generated), self.w, problem.n)


class RestartSPEA2SDE(DynamicAlgorithm):
    name = "RestartSPEA2SDE"

    def run(self, problem, max_fe: int, run_idx: int) -> RunResult:
        population = problem.initialization(self.rng)
        fitness = spea2sde_fitness(population.objs)
        current_stage = self.stage_signature(problem)
        current_info = self.stage_info(problem, population, current_stage)
        history: list[list[float]] = []

        while problem.fe < max_fe:
            self.report_progress(problem, run_idx, max_fe)
            next_stage = self.stage_signature(problem)
            if next_stage != current_stage:
                self.validate_stage_transition(problem, current_stage, next_stage)
                new_info = self.stage_info(problem, population, next_stage)
                self.record_before_change(history, population, problem, run_idx, current_info.index)
                population = problem.initialization(self.rng)
                fitness = spea2sde_fitness(population.objs)
                self.validate_response_stage(problem, next_stage)
                current_stage, current_info = next_stage, new_info
                self.try_record(history, population, problem, run_idx)
                continue

            mating = spea2sde_mating_pool(self.rng, fitness, problem.n)
            offspring = operator_ga(self.rng, problem, subset(population, mating))
            population, fitness = spea2sde_environmental_selection(concat(population, offspring), problem.n)
            self.try_record(history, population, problem, run_idx)

        self.final_record(history, population, problem, run_idx, max_fe=max_fe)
        return RunResult(np.asarray(history, dtype=float), population, dict(self.snapshots))


class RestartNSGAII(DynamicAlgorithm):
    """NSGA-II baseline that fully restarts at every environment change."""

    name = "RestartNSGAII"

    def run(self, problem, max_fe: int, run_idx: int) -> RunResult:
        population = problem.initialization(self.rng)
        population, front, crowd = nsga2_environmental_selection(population, problem.n)
        current_stage = self.stage_signature(problem)
        current_info = self.stage_info(problem, population, current_stage)
        history: list[list[float]] = []

        while problem.fe < max_fe:
            self.report_progress(problem, run_idx, max_fe)
            next_stage = self.stage_signature(problem)
            if next_stage != current_stage:
                self.validate_stage_transition(problem, current_stage, next_stage)
                new_info = self.stage_info(problem, population, next_stage)
                self.record_before_change(history, population, problem, run_idx, current_info.index)
                population = problem.initialization(self.rng)
                population, front, crowd = nsga2_environmental_selection(population, problem.n)
                self.validate_response_stage(problem, next_stage)
                current_stage, current_info = next_stage, new_info
                self.try_record(history, population, problem, run_idx)
                continue

            mating = tournament_selection(self.rng, problem.n, front, crowd)
            offspring = operator_ga(self.rng, problem, subset(population, mating))
            population, front, crowd = nsga2_environmental_selection(
                concat(population, offspring), problem.n
            )
            self.try_record(history, population, problem, run_idx)

        self.final_record(history, population, problem, run_idx, max_fe=max_fe)
        return RunResult(np.asarray(history, dtype=float), population, dict(self.snapshots))


class ArchiveSPEA2SDE(DynamicAlgorithm):
    name = "ArchiveSPEA2SDE"

    def run(self, problem, max_fe: int, run_idx: int) -> RunResult:
        population = problem.initialization(self.rng)
        fitness = spea2sde_fitness(population.objs)
        external_archive = Population(np.empty((0, problem.d)), np.empty((0, problem.m)))
        current_stage = self.stage_signature(problem)
        current_info = self.stage_info(problem, population, current_stage)
        history: list[list[float]] = []

        while problem.fe < max_fe:
            self.report_progress(problem, run_idx, max_fe)
            next_stage = self.stage_signature(problem)
            if next_stage != current_stage:
                self.validate_stage_transition(problem, current_stage, next_stage)
                new_info = self.stage_info(problem, population, next_stage)
                self.record_before_change(history, population, problem, run_idx, current_info.index)
                source = external_archive if len(external_archive) else population
                best = best_solutions(source)
                population = self.rebuild_population_at_change(problem, best, current_info, new_info)
                fitness = spea2sde_fitness(population.objs)
                external_archive = Population(np.empty((0, problem.d)), np.empty((0, population.objs.shape[1])))
                self.validate_response_stage(problem, next_stage)
                current_stage, current_info = next_stage, new_info
                self.try_record(history, population, problem, run_idx)
                continue

            mating = spea2sde_mating_pool(self.rng, fitness, problem.n)
            offspring = operator_ga(self.rng, problem, subset(population, mating))
            population, fitness = spea2sde_environmental_selection(concat(population, offspring), problem.n)
            external_archive = concat(external_archive, offspring)
            self.try_record(history, population, problem, run_idx)

        self.final_record(history, population, problem, run_idx, max_fe=max_fe)
        return RunResult(np.asarray(history, dtype=float), population, dict(self.snapshots))

    def rebuild_population_at_change(
        self,
        problem,
        best: Population,
        old_info,
        new_info,
    ) -> Population:
        inherited = self.inherit_best_solutions(problem, best, old_info, new_info)
        return self.fill_population_after_change(problem, inherited)

    def inherit_best_solutions(self, problem, best: Population, old_info, new_info) -> Population:
        if len(best) == 0:
            return Population(np.empty((0, problem.d)), np.empty((0, new_info.m)))

        obj_increase = new_info.m > old_info.m
        obj_decrease = new_info.m < old_info.m
        freeze_added = bool(set(new_info.frozen) - set(old_info.frozen))
        need_reeval = obj_increase or freeze_added

        if obj_decrease and self.can_project_objectives(old_info, new_info):
            inherited = self.project_best_solutions(best, old_info, new_info)
        else:
            inherited = best
            if old_info.active != new_info.active:
                need_reeval = True

        inherited = self.downsample_solutions(inherited, problem.n)
        if need_reeval and len(inherited):
            inherited = problem.evaluate(inherited.decs)
        return inherited

    @staticmethod
    def can_project_objectives(old_info, new_info) -> bool:
        return bool(old_info.active and new_info.active and all(idx in old_info.active for idx in new_info.active))

    def project_best_solutions(self, best: Population, old_info, new_info) -> Population:
        loc = [old_info.active.index(idx) for idx in new_info.active]
        projected = Population(best.decs.copy(), best.objs[:, loc].copy(), None if best.cons is None else best.cons.copy())
        return best_solutions(projected)

    def fill_population_after_change(self, problem, inherited: Population) -> Population:
        if len(inherited) == 0:
            return problem.initialization(self.rng)
        if len(inherited) >= problem.n:
            idx = self.rng.permutation(len(inherited))[: problem.n]
            return subset(inherited, idx)
        random_pop = problem.initialization(self.rng, problem.n - len(inherited))
        return concat(inherited, random_pop)

    def downsample_solutions(self, solutions: Population, n: int) -> Population:
        if len(solutions) <= n:
            return solutions
        idx = self.rng.permutation(len(solutions))[:n]
        return subset(solutions, idx)


class ArchiveNSGAII(ArchiveSPEA2SDE):
    """ArchiveSPEA2SDE's change response coupled with NSGA-II evolution.

    The archive contains every offspring produced within the current
    environment.  At a change, only the archive's old-environment first
    front is transferred, projected when the active objectives contract,
    re-evaluated when necessary, and completed by random solutions.  This
    is exactly the ArchiveSPEA2SDE response policy; NSGA-II replaces only
    its mating and environmental selection in stationary environments.
    """

    name = "ArchiveNSGAII"

    def run(self, problem, max_fe: int, run_idx: int) -> RunResult:
        population = problem.initialization(self.rng)
        population, front, crowd = nsga2_environmental_selection(population, problem.n)
        external_archive = Population(np.empty((0, problem.d)), np.empty((0, problem.m)))
        current_stage = self.stage_signature(problem)
        current_info = self.stage_info(problem, population, current_stage)
        history: list[list[float]] = []

        while problem.fe < max_fe:
            self.report_progress(problem, run_idx, max_fe)
            next_stage = self.stage_signature(problem)
            if next_stage != current_stage:
                self.validate_stage_transition(problem, current_stage, next_stage)
                new_info = self.stage_info(problem, population, next_stage)
                self.record_before_change(history, population, problem, run_idx, current_info.index)
                source = external_archive if len(external_archive) else population
                population = self.rebuild_population_at_change(
                    problem, best_solutions(source), current_info, new_info
                )
                population, front, crowd = nsga2_environmental_selection(population, problem.n)
                external_archive = Population(
                    np.empty((0, problem.d)), np.empty((0, new_info.m))
                )
                self.validate_response_stage(problem, next_stage)
                current_stage, current_info = next_stage, new_info
                self.try_record(history, population, problem, run_idx)
                continue

            mating = tournament_selection(self.rng, problem.n, front, crowd)
            offspring = operator_ga(self.rng, problem, subset(population, mating))
            population, front, crowd = nsga2_environmental_selection(
                concat(population, offspring), problem.n
            )
            external_archive = concat(external_archive, offspring)
            self.try_record(history, population, problem, run_idx)

        self.final_record(history, population, problem, run_idx, max_fe=max_fe)
        return RunResult(np.asarray(history, dtype=float), population, dict(self.snapshots))


class TrustArchiveSPEA2SDE(DynamicAlgorithm):
    """Archive-trust adaptive-transfer SPEA2+SDE.

    Python port of ``TrustArchiveSPEA2SDE.m``.  At a change it estimates
    archive quality from nondominated ratio and reference-vector coverage,
    then transfers 20--80% of the target population accordingly.
    """

    name = "TrustArchiveSPEA2SDE"

    def __init__(self, seed: int = 1, pf_count: int = 5000,
                 min_transfer_ratio: float = 0.2, max_transfer_ratio: float = 0.8) -> None:
        super().__init__(seed, pf_count)
        self.min_transfer_ratio = float(np.clip(min_transfer_ratio, 0.0, 1.0))
        self.max_transfer_ratio = float(np.clip(max_transfer_ratio, self.min_transfer_ratio, 1.0))

    @staticmethod
    def _empty(problem, m: int | None = None) -> Population:
        return Population(np.empty((0, problem.d)), np.empty((0, problem.m if m is None else m)))

    @staticmethod
    def _unique_decisions(population: Population) -> Population:
        if len(population) == 0:
            return population
        _, indices = np.unique(population.decs, axis=0, return_index=True)
        return subset(population, np.sort(indices))

    @staticmethod
    def _cosine_regions(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Match MATLAB ``pdist2(points, weights, 'cosine')`` association."""
        x = np.asarray(points, dtype=float)
        w = np.asarray(weights, dtype=float)
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        w_norm = np.linalg.norm(w, axis=1, keepdims=True).T
        x_norm[x_norm < 1e-12] = 1e-12
        w_norm[w_norm < 1e-12] = 1e-12
        cosine = (x @ w.T) / (x_norm * w_norm)
        # Minimizing cosine distance (1-cosine) is maximizing cosine.
        return np.argmax(cosine, axis=1)

    def _coverage(self, population: Population) -> float:
        if len(population) <= 1 or population.objs.shape[1] <= 1:
            return 1.0 if len(population) else 0.0
        objs = population.objs
        norm = (objs - np.min(objs, axis=0)) / (np.ptp(objs, axis=0) + 1e-12)
        w = uniform_point(max(min(len(population), 100), objs.shape[1] + 1), objs.shape[1])
        regions = self._cosine_regions(np.maximum(norm, 1e-12), w)
        return float(len(np.unique(regions)) / min(len(w), len(population)))

    def _trust(self, population: Population) -> float:
        if len(population) == 0:
            return 0.0
        nd_ratio = len(best_solutions(population)) / len(population)
        return float(np.clip(0.60 * nd_ratio + 0.40 * self._coverage(best_solutions(population)), 0.0, 1.0))

    def _transfer_number(self, n: int, trust: float) -> int:
        ratio = self.min_transfer_ratio + trust * (self.max_transfer_ratio - self.min_transfer_ratio)
        return int(np.clip(round(n * ratio), 0, n))

    def _reference_coverage_selection(self, population: Population, n: int) -> Population:
        if len(population) <= n:
            return population
        n = max(0, min(n, len(population)))
        if n == 0:
            return subset(population, np.empty(0, dtype=int))
        objs = population.objs
        if objs.shape[1] <= 1:
            return subset(population, np.argsort(spea2sde_fitness(objs))[:n])
        fitness = spea2sde_fitness(objs)
        norm = (objs - np.min(objs, axis=0)) / (np.ptp(objs, axis=0) + 1e-12)
        w = uniform_point(max(n, objs.shape[1] + 1), objs.shape[1])
        regions = self._cosine_regions(np.maximum(norm, 1e-12), w)
        chosen: list[int] = []
        for region in np.unique(regions):
            candidates = np.where(regions == region)[0]
            chosen.append(int(candidates[np.argmin(fitness[candidates])]))
        chosen = sorted(chosen, key=lambda index: fitness[index])[:n]
        if len(chosen) < n:
            chosen_set = set(chosen)
            chosen.extend(index for index in np.argsort(fitness) if index not in chosen_set)
        return subset(population, np.asarray(chosen[:n], dtype=int))

    @staticmethod
    def _decision_coverage_selection(population: Population, n: int) -> Population:
        if len(population) <= n:
            return population
        decs = population.decs
        norm = (decs - np.min(decs, axis=0)) / (np.ptp(decs, axis=0) + 1e-12)
        selected = [int(np.argmin(np.linalg.norm(norm - np.mean(norm, axis=0), axis=1)))]
        min_distance = np.linalg.norm(norm - norm[selected[0]], axis=1)
        while len(selected) < n:
            min_distance[selected] = -np.inf
            index = int(np.argmax(min_distance))
            selected.append(index)
            min_distance = np.minimum(min_distance, np.linalg.norm(norm - norm[index], axis=1))
        return subset(population, np.asarray(selected, dtype=int))

    def _comparable_source(self, source: Population, old_info, new_info) -> tuple[Population, bool]:
        common = [index for index in old_info.active if index in new_info.active]
        if common:
            locations = [old_info.active.index(index) for index in common]
            return Population(source.decs.copy(), source.objs[:, locations].copy()), False
        return Population(source.decs.copy(), source.decs.copy()), True

    @staticmethod
    def _can_project(old_info, new_info) -> bool:
        return bool(old_info.active and new_info.active and all(index in old_info.active for index in new_info.active))

    def _inherited_at_change(self, problem, source: Population, old_info, new_info) -> Population:
        if len(source) == 0:
            return self._empty(problem, new_info.m)
        max_transfer = min(problem.n, round(problem.n * self.max_transfer_ratio))
        if max_transfer == 0:
            return self._empty(problem, new_info.m)
        freeze_added = bool(set(new_info.frozen) - set(old_info.frozen))
        if new_info.m < old_info.m and self._can_project(old_info, new_info) and not freeze_added:
            locations = [old_info.active.index(index) for index in new_info.active]
            projected = Population(source.decs.copy(), source.objs[:, locations].copy())
            return self._reference_coverage_selection(
                projected, self._transfer_number(problem.n, self._trust(projected))
            )

        need_reeval = (
            new_info.m > old_info.m
            or freeze_added
            or old_info.active != new_info.active
        )
        if not need_reeval:
            return self._reference_coverage_selection(
                source, self._transfer_number(problem.n, self._trust(source))
            )

        comparable, use_decisions = self._comparable_source(source, old_info, new_info)
        # MATLAB keeps up to max(2*maxTransferN, N) candidates for new-stage
        # evaluation before applying the adaptive transfer-ratio decision.
        reeval_n = min(len(comparable), max(2 * max_transfer, problem.n))
        preselected = (
            self._decision_coverage_selection(comparable, reeval_n)
            if use_decisions else self._reference_coverage_selection(comparable, reeval_n)
        )
        evaluated = problem.evaluate(preselected.decs)
        return self._reference_coverage_selection(
            evaluated, self._transfer_number(problem.n, self._trust(evaluated))
        )

    def _fill_population(self, problem, inherited: Population) -> Population:
        if len(inherited) == 0:
            return problem.initialization(self.rng)
        if len(inherited) > problem.n:
            return spea2sde_environmental_selection(inherited, problem.n)[0]
        if len(inherited) < problem.n:
            return concat(inherited, problem.initialization(self.rng, problem.n - len(inherited)))
        return inherited

    def run(self, problem, max_fe: int, run_idx: int) -> RunResult:
        population = problem.initialization(self.rng)
        fitness = spea2sde_fitness(population.objs)
        external_archive = self._empty(problem)
        current_stage = self.stage_signature(problem)
        current_info = self.stage_info(problem, population, current_stage)
        history: list[list[float]] = []

        while problem.fe < max_fe:
            self.report_progress(problem, run_idx, max_fe)
            next_stage = self.stage_signature(problem)
            if next_stage != current_stage:
                self.validate_stage_transition(problem, current_stage, next_stage)
                new_info = self.stage_info(problem, population, next_stage)
                self.record_before_change(history, population, problem, run_idx, current_info.index)
                source = self._unique_decisions(concat(population, external_archive))
                inherited = self._inherited_at_change(problem, source, current_info, new_info)
                population = self._fill_population(problem, inherited)
                fitness = spea2sde_fitness(population.objs)
                external_archive = self._empty(problem, new_info.m)
                self.validate_response_stage(problem, next_stage)
                current_stage, current_info = next_stage, new_info
                self.try_record(history, population, problem, run_idx)
                continue

            mating = spea2sde_mating_pool(self.rng, fitness, problem.n)
            offspring = operator_ga(self.rng, problem, subset(population, mating))
            population, fitness = spea2sde_environmental_selection(concat(population, offspring), problem.n)
            external_archive = concat(external_archive, offspring)
            self.try_record(history, population, problem, run_idx)

        self.final_record(history, population, problem, run_idx, max_fe=max_fe)
        return RunResult(np.asarray(history, dtype=float), population, dict(self.snapshots))


class TrustArchiveNSGAII(TrustArchiveSPEA2SDE):
    """TrustArchiveSPEA2SDE's change-response archive with NSGA-II search.

    Archive construction, archive trust, objective projection, preselection,
    re-evaluation, and adaptive 20--80% transfer are deliberately inherited
    unchanged.  Tournament mating and within-environment survival use the
    standard NSGA-II rank-and-crowding operators instead of SPEA2+SDE.
    """

    name = "TrustArchiveNSGAII"

    def _fill_population(self, problem, inherited: Population) -> Population:
        if len(inherited) == 0:
            return problem.initialization(self.rng)
        if len(inherited) > problem.n:
            return environmental_selection(inherited, problem.n)
        if len(inherited) < problem.n:
            return concat(inherited, problem.initialization(self.rng, problem.n - len(inherited)))
        return inherited

    def run(self, problem, max_fe: int, run_idx: int) -> RunResult:
        population = problem.initialization(self.rng)
        population, front, crowd = nsga2_environmental_selection(population, problem.n)
        external_archive = self._empty(problem)
        current_stage = self.stage_signature(problem)
        current_info = self.stage_info(problem, population, current_stage)
        history: list[list[float]] = []

        while problem.fe < max_fe:
            self.report_progress(problem, run_idx, max_fe)
            next_stage = self.stage_signature(problem)
            if next_stage != current_stage:
                self.validate_stage_transition(problem, current_stage, next_stage)
                new_info = self.stage_info(problem, population, next_stage)
                self.record_before_change(history, population, problem, run_idx, current_info.index)
                source = self._unique_decisions(concat(population, external_archive))
                inherited = self._inherited_at_change(problem, source, current_info, new_info)
                population = self._fill_population(problem, inherited)
                population, front, crowd = nsga2_environmental_selection(population, problem.n)
                external_archive = self._empty(problem, new_info.m)
                self.validate_response_stage(problem, next_stage)
                current_stage, current_info = next_stage, new_info
                self.try_record(history, population, problem, run_idx)
                continue

            mating = tournament_selection(self.rng, problem.n, front, crowd)
            offspring = operator_ga(self.rng, problem, subset(population, mating))
            population, front, crowd = nsga2_environmental_selection(
                concat(population, offspring), problem.n
            )
            external_archive = concat(external_archive, offspring)
            self.try_record(history, population, problem, run_idx)

        self.final_record(history, population, problem, run_idx, max_fe=max_fe)
        return RunResult(np.asarray(history, dtype=float), population, dict(self.snapshots))


class DTAEA(DynamicAlgorithm):
    name = "DTAEA"

    def run(self, problem, max_fe: int, run_idx: int) -> RunResult:
        ca = problem.initialization(self.rng)
        da = problem.initialization(self.rng)
        current_stage = self.stage_signature(problem)
        current_info = self.stage_info(problem, ca, current_stage)
        w = uniform_point(problem.n, current_info.m)
        history: list[list[float]] = []

        while problem.fe < max_fe:
            self.report_progress(problem, run_idx, max_fe)
            next_stage = self.stage_signature(problem)
            if next_stage != current_stage:
                self.validate_stage_transition(problem, current_stage, next_stage)
                old_info = current_info
                new_info = self.stage_info(problem, ca, next_stage)
                self.record_before_change(history, ca, problem, run_idx, old_info.index)
                w = uniform_point(problem.n, new_info.m)
                ca = problem.evaluate(ca.decs)
                if new_info.m > old_info.m:
                    da = lhs_population(self.rng, problem, problem.n)
                elif new_info.m < old_info.m:
                    ca, da = self.respond_decrease(problem, ca, w)
                else:
                    # Only the same-NObj case retains the prior DA.
                    da = problem.evaluate(da.decs)
                self.validate_response_stage(problem, next_stage)
                current_stage, current_info = next_stage, new_info
                self.try_record(history, ca, problem, run_idx)
                continue

            if ca.objs.shape[1] != w.shape[1]:
                w = uniform_point(problem.n, ca.objs.shape[1])
            zmin, zmax = np.min(ca.objs, axis=0), np.max(ca.objs, axis=0)
            norm = (ca.objs - zmin) / (zmax - zmin + 1e-6)
            occupied = len(np.unique(associate(norm, w))) / w.shape[0]
            first_parents = []
            second_parents = []
            for _ in range(problem.n):
                p1 = self.rng.integers(0, len(ca))
                first_parents.append(ca.decs[p1])
                if self.rng.random() < occupied:
                    p2 = self.rng.integers(0, len(ca))
                    second_parents.append(ca.decs[p2])
                else:
                    p2 = self.rng.integers(0, len(da))
                    second_parents.append(da.decs[p2])
            parent_decs = np.vstack([first_parents, second_parents])
            parents = Population(parent_decs, np.zeros((parent_decs.shape[0], ca.objs.shape[1])))
            offspring = operator_ga(self.rng, problem, parents)
            ca = self.update_ca(ca, offspring, w, problem.n)
            da = self.update_da(ca, da, offspring, w, problem.n)
            self.try_record(history, ca, problem, run_idx)

        self.final_record(history, ca, problem, run_idx, max_fe=max_fe)
        return RunResult(np.asarray(history, dtype=float), ca, dict(self.snapshots))

    def respond_decrease(self, problem, ca: Population, w: np.ndarray) -> tuple[Population, Population]:
        front, _ = nd_sort(ca.objs, len(ca))
        next_ca = subset(ca, np.where(front == 1)[0])
        next_da = subset(ca, np.where(front > 1)[0])
        # DTAEA Algorithm 2 explicitly uses Latin-hypercube sampling to
        # complete DA after objective reduction.
        if len(next_da) < problem.n:
            next_da = concat(next_da, lhs_population(self.rng, problem, problem.n - len(next_da)))
        elif len(next_da) > problem.n:
            next_da = subset(next_da, np.arange(problem.n))
        if len(next_ca) == 0:
            next_ca = lhs_population(self.rng, problem, problem.n)
        while 0 < len(next_ca) < problem.n:
            # Algorithm 2 appends every mutant to CA before the following
            # tournament.  Recompute density on that enlarged CA, so the
            # next parent is genuinely taken from the current less-crowded
            # subspace rather than from a stale initial density ranking.
            density = estimate_density(next_ca.objs, w)
            a, b = self.rng.integers(0, len(next_ca), 2)
            if density[a] < density[b]:
                chosen = a
            elif density[a] > density[b]:
                chosen = b
            else:
                chosen = a if self.rng.random() < 0.5 else b
            mutant_dec = polynomial_mutation(
                self.rng, next_ca.decs[[chosen]], problem.lower, problem.upper
            )
            next_ca = concat(next_ca, problem.evaluate(mutant_dec))
        if len(next_ca) > problem.n:
            next_ca = subset(next_ca, np.arange(problem.n))
        return next_ca, next_da

    def update_ca(self, ca: Population, q: Population, w: np.ndarray, n: int) -> Population:
        return dtaea_ca_update(concat(ca, q), w, n)

    def update_da(self, ca: Population, da: Population, q: Population, w: np.ndarray, n: int) -> Population:
        r = concat(da, q)
        all_objs = np.vstack([ca.objs, r.objs])
        zmin, zmax = np.min(all_objs, axis=0), np.max(all_objs, axis=0)
        norm_r = (r.objs - zmin) / (zmax - zmin + 1e-6)
        norm_ca = (ca.objs - zmin) / (zmax - zmin + 1e-6)
        region_r = associate(norm_r, w)
        region_ca = associate(norm_ca, w)
        selected = []
        remaining = set(range(len(r)))
        itr = 1
        while len(selected) < n and remaining:
            for i in range(w.shape[0]):
                if len(selected) >= n:
                    break
                if np.sum(region_ca == i) < itr:
                    cand = [idx for idx in remaining if region_r[idx] == i]
                    if not cand:
                        continue
                    cand_objs = norm_r[cand]
                    front, _ = nd_sort(cand_objs, 1)
                    best = np.asarray(cand)[front == 1]
                    sub_w = np.maximum(w[i], 1e-6)
                    tch = np.max(np.abs(norm_r[best]) / sub_w, axis=1)
                    target = int(best[np.argmin(tch)])
                    selected.append(target)
                    remaining.remove(target)
            itr += 1
            if itr > n + 10:
                selected.extend(list(remaining)[: n - len(selected)])
                break
        return subset(r, np.asarray(selected[:n], dtype=int))
