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
