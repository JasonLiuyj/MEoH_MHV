from __future__ import annotations

from dataclasses import dataclass

PROBLEM_TAGS = ("DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "WFG4", "WFG5", "WFG6", "WFG7", "WFG8", "WFG9")

# Default budget for the compact formal experiment.
POPULATION_SIZE = 150
FIRST_STAGE_GENS = 100
TAUS = (25, 50, 100)
RUNS = 31


@dataclass(frozen=True)
class Setting:
    name: str
    prob_tag: str
    config: str
    seq: str
    frozen: str = ""


@dataclass(frozen=True)
class SingleTransitionSetting:
    """One independent source-environment -> target-environment benchmark."""

    name: str
    prob_tag: str
    config: str
    seq: str
    transition_index: int
    source_objectives: tuple[int, ...]
    target_objectives: tuple[int, ...]
    frozen: str = ""


SETTINGS: tuple[Setting, ...] = (
    Setting(
        name="Setting I (Mild)",
        prob_tag="DTLZ4",
        config="1-5:DTLZ4",
        seq=(
            "2,4; "
            "2,4,5; "
            "1,2,4,5; "
            "2,3,5; "
            "3,5"
        ),
    ),
    Setting(
        name="Setting II (Moderate)",
        prob_tag="DTLZ4",
        config="1-10:DTLZ4",
        seq=(
            "2,7; "
            "2,5,7,10; "
            "1,2,5,6,7,10; "
            "2,5,6,9; "
            "5,6"
        ),
    ),
    Setting(
        name="Setting III (Severe)",
        prob_tag="DTLZ4",
        config="1-10:DTLZ4",
        seq=(
            "3,8; "
            "2,3,6,7,8; "
            "1,2,3,4,5,6,7,8,9,10; "
            "2,3,6,7,8; "
            "3,8"
        ),
    ),
)


def setting_for_problem(setting: Setting, problem_tag: str) -> Setting:
    problem_tag = problem_tag.upper()
    max_index = max(int(item) for stage in setting.seq.split(";") for item in stage.split(",") if item.strip())
    return Setting(setting.name, problem_tag, f"1-{max_index}:{problem_tag}", setting.seq, setting.frozen)


def _parse_stages(value: str) -> list[tuple[int, ...]]:
    return [
        tuple(int(item.strip()) for item in stage.split(",") if item.strip())
        for stage in value.split(";")
        if stage.strip()
    ]


def single_transition_setting(
    setting: Setting,
    problem_tag: str,
    transition_index: int,
) -> SingleTransitionSetting:
    """Extract one adjacent environment change from a multi-stage setting.

    ``transition_index`` is one-based: 1 means stage 1 -> stage 2.  The
    returned sequence always contains exactly two environments, so every run
    starts afresh in the source environment and performs exactly one change.
    """

    stages = _parse_stages(setting.seq)
    if transition_index < 1 or transition_index >= len(stages):
        raise ValueError(
            f"Transition index must be 1-{len(stages) - 1}, got {transition_index}."
        )
    source = stages[transition_index - 1]
    target = stages[transition_index]
    problem_tag = problem_tag.upper()
    max_index = max(source + target)
    seq = "; ".join(",".join(str(item) for item in stage) for stage in (source, target))

    frozen_stages = _parse_stages(setting.frozen) if setting.frozen.strip() else []
    frozen = ""
    if frozen_stages:
        if len(frozen_stages) != len(stages):
            raise ValueError("Frozen-variable stages must match objective stages.")
        frozen = "; ".join(
            ",".join(str(item) for item in stage)
            for stage in frozen_stages[transition_index - 1 : transition_index + 1]
        )

    return SingleTransitionSetting(
        name=f"{setting.name} / Transition {transition_index}",
        prob_tag=problem_tag,
        config=f"1-{max_index}:{problem_tag}",
        seq=seq,
        transition_index=transition_index,
        source_objectives=source,
        target_objectives=target,
        frozen=frozen,
    )
