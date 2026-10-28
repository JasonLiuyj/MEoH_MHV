"""将 ArchiveSPEA2SDE 明确拆成“基础进化”和“环境迁移”两部分。

这个文件保持 ``algorithms.dynamic_algorithms.ArchiveSPEA2SDE`` 的算法语义：

1. ``SPEA2SDEBaseOptimizer`` 只负责在一个不变环境中进化种群；
2. ``ArchiveInitialPopulationStrategy`` 只负责维护 archive，并在环境变化后
   从旧 archive 构造新环境的初始种群；
3. ``TwoPartArchiveSPEA2SDE`` 只负责检测环境变化并编排前两部分。

可直接在 PyCharm 中右键运行本文件执行一个小型演示。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 兼容“直接运行当前文件”和“从项目根目录导入”两种方式。
PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "algorithms":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.base import DynamicAlgorithm, RunResult, StageInfo
from algorithms.dynamic_algorithms import (
    best_solutions,
    spea2sde_environmental_selection,
    spea2sde_fitness,
    spea2sde_mating_pool,
)
from core.evolution import concat, operator_ga, subset
from core.population import Population


class SPEA2SDEBaseOptimizer:
    """基础算法：在当前静态环境中用 SPEA2+SDE 逼近 Pareto 前沿。"""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def initialize(self, problem) -> tuple[Population, np.ndarray]:
        """随机初始化当前环境的种群，并计算 SPEA2+SDE fitness。"""
        population = problem.initialization(self.rng)
        fitness = spea2sde_fitness(population.objs)
        return population, fitness

    def initialize_from_population(
        self, population: Population
    ) -> tuple[Population, np.ndarray]:
        """接收迁移策略产生的新环境初始种群。"""
        fitness = spea2sde_fitness(population.objs)
        return population, fitness

    def evolve_one_generation(
        self,
        problem,
        population: Population,
        fitness: np.ndarray,
    ) -> tuple[Population, np.ndarray, Population]:
        """完成一次选择、交叉变异和环境选择。

        返回值中的 ``offspring`` 会交给迁移策略保存到 archive；基础优化器
        本身不决定 archive 如何使用。
        """
        mating = spea2sde_mating_pool(self.rng, fitness, problem.n)
        parents = subset(population, mating)
        offspring = operator_ga(self.rng, problem, parents)
        next_population, next_fitness = spea2sde_environmental_selection(
            concat(population, offspring), problem.n
        )
        return next_population, next_fitness, offspring


class ArchiveInitialPopulationStrategy:
    """迁移策略：由旧环境 archive 生成新环境的初始种群。

    原 ArchiveSPEA2SDE 的策略如下：

    * archive 保存当前环境每一代产生的全部 offspring；
    * 环境变化时，从 archive 提取第一非支配前沿；
    * 若新目标是旧目标的子集，直接投影旧目标列；
    * 若目标增加、目标集合不兼容或新增冻结变量，则在新环境重新评价；
    * 迁移解超过 N 时随机下采样，不足 N 时用随机新解补齐。
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    @staticmethod
    def empty_archive(problem, objective_count: int | None = None) -> Population:
        m = problem.m if objective_count is None else objective_count
        return Population(np.empty((0, problem.d)), np.empty((0, m)))

    @staticmethod
    def update_archive(archive: Population, offspring: Population) -> Population:
        """保存环境内产生的候选解；不在这里提前丢弃潜在知识。"""
        return concat(archive, offspring)

    def create_initial_population(
        self,
        problem,
        current_population: Population,
        archive: Population,
        old_environment: StageInfo,
        new_environment: StageInfo,
    ) -> Population:
        """从旧 archive 中选解，并构造新环境初始种群。"""
        source = archive if len(archive) else current_population
        old_pareto_set = best_solutions(source)
        inherited = self._inherit_best_solutions(
            problem, old_pareto_set, old_environment, new_environment
        )
        return self._fill_to_population_size(problem, inherited)

    def _inherit_best_solutions(
        self,
        problem,
        old_pareto_set: Population,
        old_environment: StageInfo,
        new_environment: StageInfo,
    ) -> Population:
        if len(old_pareto_set) == 0:
            return self.empty_archive(problem, new_environment.m)

        objectives_increased = new_environment.m > old_environment.m
        objectives_decreased = new_environment.m < old_environment.m
        frozen_variables_added = bool(
            set(new_environment.frozen) - set(old_environment.frozen)
        )
        needs_reevaluation = objectives_increased or frozen_variables_added

        if objectives_decreased and self._can_project_objectives(
            old_environment, new_environment
        ):
            inherited = self._project_objectives(
                old_pareto_set, old_environment, new_environment
            )
        else:
            inherited = old_pareto_set
            if old_environment.active != new_environment.active:
                needs_reevaluation = True

        inherited = self._downsample(inherited, problem.n)
        if needs_reevaluation and len(inherited):
            inherited = problem.evaluate(inherited.decs)
        return inherited

    @staticmethod
    def _can_project_objectives(
        old_environment: StageInfo, new_environment: StageInfo
    ) -> bool:
        return bool(
            old_environment.active
            and new_environment.active
            and all(
                objective in old_environment.active
                for objective in new_environment.active
            )
        )

    @staticmethod
    def _project_objectives(
        population: Population,
        old_environment: StageInfo,
        new_environment: StageInfo,
    ) -> Population:
        locations = [
            old_environment.active.index(objective)
            for objective in new_environment.active
        ]
        projected = Population(
            population.decs.copy(),
            population.objs[:, locations].copy(),
            None if population.cons is None else population.cons.copy(),
        )
        # 投影后支配关系可能变化，因此再次取第一前沿。
        return best_solutions(projected)

    def _downsample(self, population: Population, size: int) -> Population:
        if len(population) <= size:
            return population
        indices = self.rng.permutation(len(population))[:size]
        return subset(population, indices)

    def _fill_to_population_size(
        self, problem, inherited: Population
    ) -> Population:
        if len(inherited) == 0:
            return problem.initialization(self.rng)
        if len(inherited) >= problem.n:
            # 与原实现保持严格一致：即使恰好有 N 个迁移解，也执行一次
            # permutation。除了打乱顺序，这还会保持后续 RNG 状态一致。
            indices = self.rng.permutation(len(inherited))[: problem.n]
            return subset(inherited, indices)
        random_population = problem.initialization(
            self.rng, problem.n - len(inherited)
        )
        return concat(inherited, random_population)


class TwoPartArchiveSPEA2SDE(DynamicAlgorithm):
    """编排器：组合基础 SPEA2+SDE 与 archive 初始解迁移策略。"""

    name = "TwoPartArchiveSPEA2SDE"

    def __init__(self, seed: int = 1, pf_count: int = 5000) -> None:
        super().__init__(seed=seed, pf_count=pf_count)
        self.base_optimizer = SPEA2SDEBaseOptimizer(self.rng)
        self.transfer_strategy = ArchiveInitialPopulationStrategy(self.rng)

    def run(self, problem, max_fe: int, run_idx: int = 1) -> RunResult:
        population, fitness = self.base_optimizer.initialize(problem)
        archive = self.transfer_strategy.empty_archive(problem)
        current_signature = self.stage_signature(problem)
        current_environment = self.stage_info(
            problem, population, current_signature
        )
        history: list[list[float]] = []

        while problem.fe < max_fe:
            self.report_progress(problem, run_idx, max_fe)
            next_signature = self.stage_signature(problem)

            if next_signature != current_signature:
                self.validate_stage_transition(
                    problem, current_signature, next_signature
                )
                next_environment = self.stage_info(
                    problem, population, next_signature
                )
                self.record_before_change(
                    history,
                    population,
                    problem,
                    run_idx,
                    current_environment.index,
                )

                population = self.transfer_strategy.create_initial_population(
                    problem,
                    population,
                    archive,
                    current_environment,
                    next_environment,
                )
                population, fitness = (
                    self.base_optimizer.initialize_from_population(population)
                )
                archive = self.transfer_strategy.empty_archive(
                    problem, next_environment.m
                )

                self.validate_response_stage(problem, next_signature)
                current_signature = next_signature
                current_environment = next_environment
                self.try_record(history, population, problem, run_idx)
                continue

            population, fitness, offspring = (
                self.base_optimizer.evolve_one_generation(
                    problem, population, fitness
                )
            )
            archive = self.transfer_strategy.update_archive(archive, offspring)
            self.try_record(history, population, problem, run_idx)

        self.final_record(
            history, population, problem, run_idx, max_fe=max_fe
        )
        return RunResult(
            np.asarray(history, dtype=float),
            population,
            dict(self.snapshots),
        )


def main() -> None:
    """PyCharm 右键运行用的小型两环境演示。"""
    from problems.d_mixed_seq import DMixedSeq

    population_size = 30
    tau = 25
    first_stage_generations = 100
    problem = DMixedSeq(
        taut=tau,
        seq_string="2,4; 2,4,5; 1,2,4,5; 2,3,5; 3,5",
        config_string="1-5:DTLZ2",
        first_stage_gens=first_stage_generations,
        n=population_size,
    )
    algorithm = TwoPartArchiveSPEA2SDE(seed=1, pf_count=1000)
    max_fe = (first_stage_generations + tau) * population_size
    result = algorithm.run(problem, max_fe=max_fe, run_idx=1)

    print("\nDemo finished")
    print(f"Final FE: {problem.fe}/{max_fe}")
    print(f"Final population shape: {result.last_population.objs.shape}")
    print(f"Recorded stage history:\n{result.history}")


if __name__ == "__main__":
    main()
