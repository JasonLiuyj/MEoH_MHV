"""Search an archive-to-new-environment transfer strategy with MEoH.

Open this project in PyCharm and right-click this file to run it.
"""
from __future__ import annotations

from pathlib import Path

from llm4ad.method.meoh import MEoH, MEoHProfiler
from llm4ad.tools.llm import HttpsApi, get_llm_config
from problem.transfer_strategy_problem import TransferStrategyProblem


PROJECT_ROOT = Path(__file__).resolve().parent


def build_llm() -> HttpsApi:
    config = get_llm_config()
    return HttpsApi(
        host=config.host,
        key=config.key,
        model=config.model,
        timeout=config.timeout,
        thinking={"type": "disabled"},
        temperature=1.0,
        max_tokens=4096,
    )


def main() -> None:
    experiment_name = "transfer_strategy_original_meoh"
    log_dir = PROJECT_ROOT / "logs" / experiment_name
    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(
            f"实验目录非空：{log_dir}。请修改 experiment_name，或清空旧目录后重试。"
        )

    problem = TransferStrategyProblem(
        population_size=150,
        first_stage_generations=100,
        tau=25,
        case_seed=2026,
        pf_count=1000,
        timeout_seconds=30,
    )
    print(
        f"Prepared {problem.prepared_case_count} transfer cases; "
        f"MEoH will evolve archive-ranking strategies.",
        flush=True,
    )

    method = MEoH(
        llm=build_llm(),
        evaluation=problem,
        profiler=MEoHProfiler(
            log_dir=str(log_dir),
            num_objs=2,
            log_style="simple",
            create_random_path=False,
        ),
        max_generations=10,
        max_sample_nums=100,
        pop_size=10,
        selection_num=2,
        num_samplers=1,
        num_evaluators=1,
        num_objs=2,
        parent_selection_strategy="dominated",
        # Strict original-MEoH operator set for a clean baseline.
        use_e2_operator=True,
        use_m1_operator=True,
        use_m2_operator=True,
        use_p1_operator=False,
        use_t1_operator=False,
        use_tide=False,
        use_raise=False,
        use_pfg=False,
        use_llm_cluster=False,
        use_flash_reflection=False,
        use_harmony_search=False,
        debug_mode=False,
    )
    method.run()


if __name__ == "__main__":
    main()

