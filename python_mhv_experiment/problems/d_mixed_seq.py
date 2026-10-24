from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.population import Population
from core.evolution import uniform_point


def parse_sequence_string(value: str) -> list[list[int]]:
    if not value.strip():
        return [[]]
    groups: list[list[int]] = []
    for stage in value.split(";"):
        stage = stage.strip()
        if not stage:
            groups.append([])
        else:
            groups.append([int(item.strip()) for item in stage.split(",") if item.strip()])
    return groups


def parse_config_string(value: str, max_m: int) -> list[str]:
    mapping = ["DTLZ2"] * (max_m + 1)
    for segment in value.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        range_text, problem_name = segment.split(":", 1)
        problem_name = problem_name.strip().upper()
        for part in range_text.split(","):
            part = part.strip()
            if "-" in part:
                start, end = [int(x) for x in part.split("-", 1)]
                indices = range(start, end + 1)
            else:
                indices = [int(part)]
            for idx in indices:
                if idx <= max_m:
                    mapping[idx] = problem_name
    return mapping


def uniform_sphere_points(n: int, m: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Deterministic positive orthant sphere samples for PF approximation."""
    if rng is None:
        rng = np.random.default_rng(1)
    points = rng.random((n, m))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms


@dataclass
class DMixedSeq:
    taut: int
    seq_string: str
    config_string: str
    frozen_string: str = ""
    first_stage_gens: int = 100
    n: int = 150
    d: int | None = None
    fe: int = 0
    last_t: int = -1
    obj_groups: list[list[int]] = field(init=False)
    obj_map: list[str] = field(init=False)
    frozen_groups: list[list[int]] = field(init=False)
    global_max_m: int = field(init=False)
    current_indices: list[int] = field(init=False)
    m: int = field(init=False)
    lower: np.ndarray = field(init=False)
    upper: np.ndarray = field(init=False)
    wfg_k: int = field(init=False)
    wfg_l: int = field(init=False)
    wfg_s: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.obj_groups = parse_sequence_string(self.seq_string)
        all_indices = [idx for group in self.obj_groups for idx in group]
        self.global_max_m = max(all_indices)
        self.obj_map = parse_config_string(self.config_string, self.global_max_m)
        if self.d is None:
            self.d = self.global_max_m + 9
        self.frozen_groups = parse_sequence_string(self.frozen_string)
        self.current_indices = self.obj_groups[0]
        self.m = len(self.current_indices)
        self.lower = np.zeros(self.d)
        self.upper = np.ones(self.d)
        self.wfg_k = self.global_max_m - 1
        self.wfg_l = self.d - self.wfg_k
        self.wfg_s = np.arange(2, 2 * self.global_max_m + 1, 2, dtype=float)

    def initialization(self, rng: np.random.Generator | None = None, count: int | None = None) -> Population:
        if rng is None:
            rng = np.random.default_rng()
        if count is None:
            count = self.n
        decs = rng.random((count, self.d)) * (self.upper - self.lower) + self.lower
        return self.evaluate(decs)

    def stage_from_gen(self, current_gen: int) -> tuple[int, bool]:
        if current_gen < self.first_stage_gens:
            t = 0
        else:
            t = 1 + (current_gen - self.first_stage_gens) // self.taut
        return t, t > self.last_t

    def update_active_indices(self, t: int) -> None:
        idx = min(t, len(self.obj_groups) - 1)
        self.current_indices = self.obj_groups[idx]
        self.m = len(self.current_indices)
        self.last_t = t

    def stage_by_fe_minus_one(self) -> int:
        current_gen = max(0, self.fe - 1) // self.n
        if current_gen < self.first_stage_gens:
            t = 0
        else:
            t = 1 + (current_gen - self.first_stage_gens) // self.taut
        return min(t + 1, len(self.obj_groups))

    def target_gen_for_stage(self, stage: int) -> int:
        if stage == 1:
            return self.first_stage_gens - 1
        return self.first_stage_gens + (stage - 1) * self.taut - 1

    def evaluate(self, pop_dec: np.ndarray) -> Population:
        pop_dec = np.asarray(pop_dec, dtype=float).copy()
        current_gen = self.fe // self.n
        t, changed = self.stage_from_gen(current_gen)
        if changed:
            self.update_active_indices(t)

        pop_dec = self.apply_variable_freezing(pop_dec, t)
        pop_obj = self.calculate_objectives(pop_dec) * -1.0
        pop_con = np.zeros((pop_dec.shape[0], 1))
        fes = np.full((pop_dec.shape[0], 1), self.fe, dtype=float)
        population = Population(pop_dec, pop_obj, pop_con, fes)
        self.fe += len(population)
        return population

    def apply_variable_freezing(self, pop_dec: np.ndarray, t: int) -> np.ndarray:
        if not self.frozen_groups:
            return pop_dec
        idx = min(t, len(self.frozen_groups) - 1)
        valid = [var - 1 for var in self.frozen_groups[idx] if 0 < var <= pop_dec.shape[1]]
        if valid:
            pop_dec[:, valid] = 0.5
        return pop_dec

    def calculate_objectives(self, pop_dec: np.ndarray) -> np.ndarray:
        active = self.current_indices
        problem_names = {self.obj_map[idx] for idx in active}
        pop_obj = np.zeros((pop_dec.shape[0], len(active)))
        for problem_name in problem_names:
            if problem_name.startswith("DTLZ"):
                full_obj = self.cal_dtlz(pop_dec, problem_name)
            elif problem_name.startswith("WFG"):
                full_obj = self.cal_wfg(pop_dec, problem_name)
            else:
                raise NotImplementedError(f"Unsupported problem: {problem_name}.")
            for local_col, global_idx in enumerate(active):
                if self.obj_map[global_idx] == problem_name:
                    pop_obj[:, local_col] = full_obj[:, global_idx - 1]
        return pop_obj

    def cal_dtlz(self, pop_dec: np.ndarray, problem_name: str) -> np.ndarray:
        m = self.global_max_m
        n, d = pop_dec.shape
        xm = pop_dec[:, m - 1 :]
        index = int(problem_name[4:])
        if index in (1, 3):
            g = 100 * (d - m + 1 + np.sum((xm - 0.5) ** 2 - np.cos(20 * np.pi * (xm - 0.5)), axis=1))
        else:
            g = np.sum((xm - 0.5) ** 2, axis=1)
        if index == 1:
            x = pop_dec[:, : m - 1]
            left = np.fliplr(np.cumprod(np.column_stack([np.ones(n), x]), axis=1))
            right = np.column_stack([np.ones(n), 1 - x[:, ::-1]])
            return 0.5 * (1 + g[:, None]) * left * right
        angles = pop_dec[:, : m - 1].copy()
        if index == 4:
            angles **= 100
        left = np.fliplr(np.cumprod(np.column_stack([np.ones(n), np.cos(angles * np.pi / 2)]), axis=1))
        right = np.column_stack([np.ones(n), np.sin(angles[:, ::-1] * np.pi / 2)])
        return (1 + g[:, None]) * left * right

    @staticmethod
    def wfg_s_linear(y, a):
        return np.abs(y - a) / np.abs(np.floor(a - y) + a)

    @staticmethod
    def wfg_s_decept(y, a, b, c):
        return 1 + (np.abs(y - a) - b) * (
            np.floor(y - a + b) * (1 - c + (a - b) / b) / (a - b)
            + np.floor(a + b - y) * (1 - c + (1 - a - b) / b) / (1 - a - b)
            + 1 / b
        )

    @staticmethod
    def wfg_s_multi(y, a, b, c):
        term = np.abs(y - c) / (2 * (np.floor(c - y) + c))
        return (1 + np.cos((4 * a + 2) * np.pi * (0.5 - term)) + 4 * b * term**2) / (b + 2)

    @staticmethod
    def wfg_r_sum(y, weights):
        return np.sum(y * weights, axis=1) / np.sum(weights)

    @staticmethod
    def wfg_b_param(y, u, a, b, c):
        v = a - (1 - 2 * u) * np.abs(np.floor(0.5 - u) + a)
        return y ** (b + (c - b) * v)

    @staticmethod
    def wfg_r_nonsep(y, a):
        n, d = y.shape
        blocks = int(np.ceil(d / a))
        out = np.zeros((n, blocks))
        for i in range(blocks):
            sub = y[:, i * a : min((i + 1) * a, d)]
            mean = np.sum(sub, axis=1, keepdims=True) / sub.shape[1]
            out[:, i] = np.sum(sub, axis=1) + np.sum(
                np.abs(sub - (1 + 2 * np.arange(sub.shape[1])) * mean), axis=1
            )
        c = np.ceil(a / 2)
        return out / c / (1 + 2 * a - 2 * c)

    @staticmethod
    def shape_concave(x):
        n, m = x.shape
        left = np.fliplr(np.cumprod(np.column_stack([np.ones(n), np.sin(x[:, : m - 1] * np.pi / 2)]), axis=1))
        right = np.column_stack([np.ones(n), np.cos(x[:, m - 2 :: -1] * np.pi / 2)])
        return left * right

    def cal_wfg(self, pop_dec: np.ndarray, problem_name: str) -> np.ndarray:
        m, k, l = self.global_max_m, self.wfg_k, self.wfg_l
        n, d = pop_dec.shape
        index = int(problem_name[3:])
        z = pop_dec.copy()
        if index == 4:
            t1 = self.wfg_s_multi(z, 30, 10, 0.35)
            t2 = t1
        elif index == 5:
            t1 = self.wfg_s_decept(z, 0.35, 0.001, 0.05)
            t2 = t1
        elif index == 6:
            t1 = z.copy()
            t1[:, k:] = self.wfg_s_linear(z[:, k:], 0.35)
            t2 = np.column_stack([t1[:, :k], self.wfg_r_nonsep(t1[:, k:], l)])
        elif index == 7:
            t1 = z.copy()
            for i in range(k):
                u = self.wfg_r_sum(z[:, i + 1 :], np.ones(d - i - 1))
                t1[:, i] = self.wfg_b_param(z[:, i], u, 0.98 / 49.98, 0.02, 50)
            t2 = t1.copy()
            t2[:, k:] = self.wfg_s_linear(t1[:, k:], 0.35)
        elif index == 8:
            t1 = z.copy()
            u = self.wfg_r_sum(z[:, :k], np.ones(k))
            for i in range(k, d):
                t1[:, i] = self.wfg_b_param(z[:, i], u, 0.98 / 49.98, 0.02, 50)
            t2 = t1.copy()
            t2[:, k:] = self.wfg_s_linear(t1[:, k:], 0.35)
        elif index == 9:
            t1 = np.zeros_like(z)
            for i in range(d - 1):
                u = self.wfg_r_sum(z[:, i + 1 :], np.ones(d - i - 1))
                t1[:, i] = self.wfg_b_param(z[:, i], u, 0.98 / 49.98, 0.02, 50)
            t1[:, -1] = z[:, -1]
            t2 = np.zeros_like(z)
            t2[:, :k] = self.wfg_s_decept(t1[:, :k], 0.35, 0.001, 0.05)
            t2[:, k:] = self.wfg_s_multi(t1[:, k:], 30, 10, 0.35)
        else:
            raise NotImplementedError(problem_name)

        t3 = np.zeros((n, m))
        for i in range(m - 1):
            start = i * k // (m - 1)
            end = (i + 1) * k // (m - 1)
            block = t2[:, start:end]
            if index == 9:
                t3[:, i] = self.wfg_r_nonsep(block, k // (m - 1))[:, 0]
            else:
                t3[:, i] = self.wfg_r_sum(block, np.ones(block.shape[1]))
        if index == 9:
            t3[:, -1] = self.wfg_r_nonsep(t2[:, k:], l)[:, 0]
        else:
            t3[:, -1] = self.wfg_r_sum(t2[:, k:], np.ones(t2.shape[1] - k))
        return t3[:, [-1]] + self.wfg_s * self.shape_concave(t3)

    def get_optimum(self, count: int = 5000) -> np.ndarray:
        # The PF must be generated for the objective columns that are
        # currently attached to evaluated populations.  FE alone is not a
        # reliable proxy at a dynamic boundary: change-response evaluations
        # can cross a nominal generation boundary before the old population
        # has been replaced.  ``evaluate`` updates current_indices whenever
        # an environment is actually entered, so it is the single source of
        # truth for both population objectives and the reference PF.
        real_indices = self.current_indices
        current_m = len(real_indices)
        problem_name = self.obj_map[real_indices[0]]
        r = uniform_point(count, current_m)
        if problem_name == "DTLZ1":
            k = self.d - self.global_max_m + 1
            g_max = 100 * k * (1 + 1.202602641454022)
            r = r / 2 * (1 + g_max)
        elif problem_name in ("DTLZ2", "DTLZ4"):
            r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
            r = r * (1 + 0.5**2 * 10)
        elif problem_name == "DTLZ3":
            r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
            k = self.d - self.global_max_m + 1
            g_max = 100 * k * (1 + 1.202602641454022)
            r = r * (1 + g_max)
        elif problem_name.startswith("WFG"):
            prob_index = int(problem_name[3:])
            max_xm = 1.0
            if prob_index in (6, 9):
                c = np.ceil(self.wfg_l / 2)
                max_xm = self.wfg_l**2 / (c * (1 + 2 * self.wfg_l - 2 * c))
            r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
            current_s = self.wfg_s[np.asarray(real_indices) - 1]
            r = r * current_s + max_xm
        else:
            raise NotImplementedError(problem_name)
        return -r
