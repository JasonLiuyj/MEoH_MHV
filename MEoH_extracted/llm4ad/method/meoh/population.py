from __future__ import annotations

import math
from threading import Lock
from typing import List
import numpy as np
import traceback

from ...base import *
from codebleu.syntax_match import calc_syntax_match
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def _stable_sigmoid(x: float) -> float:
    """数值稳定的 sigmoid，避免溢出。"""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        exp_x = np.exp(x)
        return exp_x / (1.0 + exp_x)


def _sample_power_law_indices(n: int, alpha: float) -> np.ndarray:
    """按幂律分布生成概率：rank i 的概率与 (i+1)^(-alpha) 成正比。alpha=0 为均匀。"""
    probs = np.array([(i + 1) ** (-alpha) for i in range(n)], dtype=float)
    if probs.sum() <= 0:
        probs = np.ones(n)
    return probs / probs.sum()


class Population:
    def __init__(self, pop_size, generation=0, pop: List[Function] | Population | None = None,
                 parent_selection_strategy: str = 'dominated',
                 exploitation_alpha: float = 1.0,
                 parent_selection_lambda: float = 10.0):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._pop_size = pop_size
        self._lock = Lock()
        self._next_gen_pop = []
        self._elitist = []
        self._generation = generation
        self._last_generation_transition = None
        # 父代选择策略: 'uniform' | 'power_law' | 'weighted' | 'dominated'(默认，原支配+语法差异)
        self._parent_selection_strategy = parent_selection_strategy
        self._exploitation_alpha = exploitation_alpha   # power_law 指数，0 为均匀
        self._parent_selection_lambda = parent_selection_lambda  # weighted 的 lambda

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def elitist(self):
        return self._elitist

    @property
    def generation(self):
        return self._generation

    def consume_generation_transition(self):
        try:
            self._lock.acquire()
            transition = self._last_generation_transition
            self._last_generation_transition = None
            return transition
        finally:
            self._lock.release()

    def register_function(self, func: Function):
        # we only accept valid functions
        if func.score is None:
            return
        try:
            self._lock.acquire()
            # register to next_gen
            if not self.has_duplicate_function(func):
                self._next_gen_pop.append(func)

            # update: perform survival if reach the pop size
            if len(self._next_gen_pop) >= self._pop_size:
                previous_population = list(self._population)
                pop = self._population + self._next_gen_pop

                pop_elitist = pop + self._elitist
                # Filter based on scores/objectives
                unique_pop = []
                seen_scores = set()

                for ind in pop_elitist:
                    # 确保 score 存在且有效（不包含inf）
                    if ind.score is None:
                        continue
                    score_array = np.array(ind.score)
                    if np.isinf(score_array).any():
                        continue
                    # Convert score list/array to tuple so it can be hashed
                    # 使用tolist()确保numpy数组被正确转换为Python原生类型，避免精度问题
                    score_list = score_array.tolist()
                    score_tuple = tuple(score_list)
                    if score_tuple not in seen_scores:
                        unique_pop.append(ind)
                        seen_scores.add(score_tuple)

                pop_elitist = unique_pop

                # 如果没有有效的个体，直接返回
                if len(pop_elitist) == 0:
                    self._next_gen_pop = []
                    return

                # 进行非支配排序，获取第一前沿（非支配解）
                # 确保所有score都是numpy数组格式
                objs = []
                for ind in pop_elitist:
                    score_array = np.array(ind.score)
                    objs.append(score_array)

                objs_array = -np.array(objs)  # 取负号，因为 NonDominatedSorting 默认最小化
                nondom_idx = NonDominatedSorting().do(objs_array, only_non_dominated_front=True)

                self._elitist = []
                for idx in nondom_idx.tolist():
                    self._elitist.append(pop_elitist[idx])

                crt_pop_size = len(pop)
                dominated_counts = np.zeros((crt_pop_size, crt_pop_size))
                for i in range(crt_pop_size):
                    for j in range(i + 1, crt_pop_size):
                        if (np.array(pop[i].score) >= np.array(pop[j].score)).all():
                            dominated_counts[i, j] = -calc_syntax_match([pop[i].entire_code], pop[j].entire_code,
                                                                        'python')
                        elif (np.array(pop[j].score) >= np.array(pop[i].score)).all():
                            dominated_counts[j, i] = -calc_syntax_match([pop[j].entire_code], pop[i].entire_code,
                                                                        'python')
                dominated_counts_ = dominated_counts.sum(0)
                self._population = [pop[i] for i in np.argsort(-dominated_counts_)[
                    :self._pop_size]]  # minus for descending, //5 for keep the original pop_size
                self._next_gen_pop = []
                self._generation += 1
                self._last_generation_transition = {
                    "generation": self._generation,
                    "previous_population": previous_population,
                    "current_population": list(self._population),
                    "current_elitist": list(self._elitist),
                }

        except Exception as e:
            # print(f"error in registering function to population: {e}")
            traceback.print_exc()
            return
        finally:
            self._lock.release()

    def has_duplicate_function(self, func: str | Function) -> bool:
        if func.score is None:
            return True

        for i in range(len(self._population)):
            f = self._population[i]
            if str(f) == str(func):
                if func.score[0] > f.score[0]:
                    self._population[i] = func
                    return True
                if func.score[0] == f.score[0] and func.score[1] > f.score[1]:
                    self._population[i] = func
                    return True

        for i in range(len(self._next_gen_pop)):
            f = self._next_gen_pop[i]
            if str(f) == str(func):
                if func.score[0] > f.score[0]:
                    self._next_gen_pop[i] = func
                    return True
                if func.score[0] == f.score[0] and func.score[1] > f.score[1]:
                    self._next_gen_pop[i] = func
                    return True
        return False

    def _get_valid_funcs(self) -> List[Function]:
        """返回 score 有效（非 None、无 inf）的个体列表。"""
        return [f for f in self._population if f.score is not None and not np.isinf(np.array(f.score)).any()]

    def _compute_dominated_counts_(self, funcs: List[Function]) -> np.ndarray:
        """
        对 funcs 计算支配关系+语法差异得到的适应度向量。
        返回 dominated_counts_ = dominated_counts.sum(0)，长度 len(funcs)，越大表示综合性能越好。
        """
        n = len(funcs)
        dominated_counts = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if (np.array(funcs[i].score) >= np.array(funcs[j].score)).all():
                    dominated_counts[i, j] = -calc_syntax_match(
                        [funcs[i].entire_code], funcs[j].entire_code, 'python'
                    )
                elif (np.array(funcs[j].score) >= np.array(funcs[i].score)).all():
                    dominated_counts[j, i] = -calc_syntax_match(
                        [funcs[j].entire_code], funcs[i].entire_code, 'python'
                    )
        return dominated_counts.sum(0)

    def selection(self) -> Function:
        funcs = self._get_valid_funcs()
        if len(funcs) == 0:
            if len(self._population) > 0:
                return self._population[np.random.randint(len(self._population))]
            raise RuntimeError("Population is empty, cannot select parent.")

        strategy = (self._parent_selection_strategy or 'dominated').lower()

        if strategy == 'uniform':
            return funcs[np.random.randint(len(funcs))]

        if strategy == 'power_law':
            # 按综合适应度 dominated_counts_ 降序排序，rank 0 最优
            dominated_counts_ = self._compute_dominated_counts_(funcs)
            order = np.argsort(-dominated_counts_)
            sorted_funcs = [funcs[i] for i in order]
            probs = _sample_power_law_indices(len(sorted_funcs), self._exploitation_alpha)
            idx = np.random.choice(len(sorted_funcs), p=probs)
            return sorted_funcs[idx]

        if strategy == 'weighted':
            # 适应度使用 dominated_counts_（综合性能），w_i = s_i * h_i; s_i = sigmoid(lambda * (fit_i - median)/scale); h_i = 1/(1+num_offspring)
            dominated_counts_ = self._compute_dominated_counts_(funcs)
            fits = np.asarray(dominated_counts_, dtype=float)
            alpha_0 = float(np.median(fits))
            deviations = np.abs(fits - alpha_0)
            mad = float(np.median(deviations)) if len(deviations) else 1.0
            scale_factor = max(mad, 1e-6)
            lam = self._parent_selection_lambda
            weights = []
            for i, f in enumerate(funcs):
                alpha_i = fits[i]
                n_i = getattr(f, 'num_offspring', 0)
                normalized_diff = (alpha_i - alpha_0) / scale_factor
                s_i = _stable_sigmoid(lam * normalized_diff)
                h_i = 1.0 / (1.0 + n_i)
                weights.append(s_i * h_i)
            w_sum = sum(weights)
            if w_sum <= 0:
                probs = np.ones(len(funcs)) / len(funcs)
            else:
                probs = np.array(weights, dtype=float) / w_sum
            return funcs[int(np.random.choice(len(funcs), p=probs))]

        # 默认: dominated（支配关系 + 语法差异，适应度=dominated_counts_）
        dominated_counts_ = self._compute_dominated_counts_(funcs)
        p = np.exp(dominated_counts_) / np.exp(dominated_counts_).sum()
        return funcs[np.random.choice(len(funcs), p=p)]

    def select_elitist_by_performance(self) -> List[Function]:
        """
        根据性能从elitist中选择算法，使用轮盘赌算法选择20%（向上取整）的算法。
        返回选中的算法列表。
        """
        if len(self._elitist) == 0:
            return []

        # 过滤掉无效的算法
        valid_elitist = [f for f in self._elitist if f.score is not None and not np.isinf(np.array(f.score)).any()]
        if len(valid_elitist) == 0:
            return []

        # 根据性能（score[0]）排序，性能越高越好
        sorted_elitist = sorted(valid_elitist, key=lambda x: x.score[0], reverse=True)

        # 计算需要选择的数量（20%向上取整）
        num_to_select = math.ceil(len(sorted_elitist) * 0.2)
        if num_to_select == 0:
            num_to_select = 1
        if num_to_select > len(sorted_elitist):
            num_to_select = len(sorted_elitist)

        # 使用轮盘赌选择
        # 计算适应度：性能越高，适应度越高
        performances = np.array([f.score[0] for f in sorted_elitist])
        # 处理负值：将所有值平移到非负
        min_perf = np.min(performances)
        if min_perf < 0:
            performances = performances - min_perf + 1
        else:
            performances = performances + 1

        # 计算概率
        probabilities = performances / np.sum(performances)

        # 使用轮盘赌选择
        selected_indices = np.random.choice(len(sorted_elitist), size=num_to_select, replace=False, p=probabilities)
        selected_funcs = [sorted_elitist[i] for i in selected_indices]

        return selected_funcs

    def select_elitist_by_time(self) -> List[Function]:
        """
        根据运行时间从elitist中选择算法，使用轮盘赌算法选择20%（向上取整）的算法。
        返回选中的算法列表。
        注意：score[1]通常是负的运行时间，所以越小越好（因为是负的）。
        """
        if len(self._elitist) == 0:
            return []

        # 过滤掉无效的算法
        valid_elitist = [f for f in self._elitist if f.score is not None and not np.isinf(np.array(f.score)).any()]
        if len(valid_elitist) == 0:
            return []

        # 根据运行时间（score[1]，注意是负值，所以越小越好）排序
        # 转换为正的时间值：-score[1]
        sorted_elitist = sorted(valid_elitist, key=lambda x: -x.score[1] if len(x.score) > 1 else float('inf'))

        # 计算需要选择的数量（20%向上取整）
        num_to_select = math.ceil(len(sorted_elitist) * 0.2)
        if num_to_select == 0:
            num_to_select = 1
        if num_to_select > len(sorted_elitist):
            num_to_select = len(sorted_elitist)

        # 使用轮盘赌选择
        # 计算适应度：运行时间越短（-score[1]越小），适应度越高
        # 转换为正的时间值
        times = np.array([-f.score[1] if len(f.score) > 1 else float('inf') for f in sorted_elitist])
        # 对于时间，越小越好，所以适应度应该是时间的倒数
        # 处理inf和0值
        times = np.where(np.isinf(times), np.max(times[~np.isinf(times)]) if np.any(~np.isinf(times)) else 1, times)
        times = np.where(times <= 0, 1e-10, times)
        # 适应度是时间的倒数
        fitness = 1.0 / times

        # 计算概率
        probabilities = fitness / np.sum(fitness)

        # 使用轮盘赌选择
        selected_indices = np.random.choice(len(sorted_elitist), size=num_to_select, replace=False, p=probabilities)
        selected_funcs = [sorted_elitist[i] for i in selected_indices]

        return selected_funcs
