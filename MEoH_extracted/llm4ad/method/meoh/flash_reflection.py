from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import List

import numpy as np

from ...base import Function, LLM


FLASH_REFLECTION_SYSTEM_PROMPT = (
    "You are an expert in heuristic algorithm design and code review."
)

FLASH_REFLECTION_PROMPT = """### Task
{task_description}

### Candidate Heuristics
Below is a list of heuristic functions gathered from the current elitist set and the previous generation population.
The list is grouped from stronger to weaker examples based on their multi-objective score patterns.

{lst_method}

### Guide
- Compare the better heuristics against the weaker ones.
- Focus on algorithm structure, efficiency/performance tradeoffs, data reuse, and avoidable overhead.
- The response must be in Markdown and contain exactly the following sections:
"**Analysis:**
**Experience:**"
- In **Analysis:** summarize structural observations from the code comparisons.
- In **Experience:** provide short, actionable guidance for designing better heuristics in the next generation in under 80 words.
I'm going to tip $999K for a better heuristics! Let's think step by step.
"""

COMPREHENSIVE_REFLECTION_PROMPT = """Your task is to refine the current reflection into a compact and reusable guidance block.

### Current Reflection
{current_reflection}

### Effective Historical Reflections
{good_reflection}

### Ineffective Historical Reflections
{bad_reflection}

Return exactly 4 bullet points with these labels:
- Keywords:
- Advice:
- Avoid:
- Explanation:
Keep the whole response under 120 words.
I'm going to tip $999K for a better heuristics! Let's think step by step.
"""


@dataclass
class FlashReflectionMemory:
    analysis: str = ""
    experience: str = ""
    current_reflection: str = ""
    comprehensive_reflection: str = ""
    good_reflections: List[str] = field(default_factory=list)
    bad_reflections: List[str] = field(default_factory=list)
    source_generation: int = -1


class MEoHFlashReflection:
    def __init__(
        self,
        llm: LLM,
        task_description: str,
        *,
        max_good_reflections: int = 5,
        max_bad_reflections: int = 3,
    ):
        self._llm = llm
        self._task_description = task_description
        self._max_good_reflections = max_good_reflections
        self._max_bad_reflections = max_bad_reflections
        self._memory = FlashReflectionMemory()
        self._lock = Lock()

    @property
    def memory(self) -> FlashReflectionMemory:
        return self._memory

    def restore_memory(
        self,
        *,
        analysis: str = "",
        experience: str = "",
        current_reflection: str = "",
        comprehensive_reflection: str = "",
        good_reflections: List[str] | None = None,
        bad_reflections: List[str] | None = None,
        source_generation: int = -1,
    ) -> None:
        with self._lock:
            self._memory = FlashReflectionMemory(
                analysis=analysis or "",
                experience=experience or "",
                current_reflection=current_reflection or "",
                comprehensive_reflection=comprehensive_reflection or "",
                good_reflections=list(good_reflections or []),
                bad_reflections=list(bad_reflections or []),
                source_generation=source_generation,
            )

    def update(
        self,
        *,
        generation: int,
        current_elitist: List[Function],
        previous_population: List[Function],
        reflection_worked: bool | None,
    ) -> None:
        with self._lock:
            self._promote_previous_reflection(reflection_worked)

            candidates = self._prepare_candidates(current_elitist, previous_population)
            if len(candidates) < 2:
                return

            prompt = self._build_flash_reflection_prompt(candidates)
            response = self._llm.draw_sample(
                [
                    {"role": "system", "content": FLASH_REFLECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
            )
            analysis, experience = self._parse_flash_reflection(response)
            if not analysis and not experience:
                return

            self._memory.analysis = analysis
            self._memory.experience = experience
            self._memory.current_reflection = experience
            self._memory.source_generation = generation
            self._memory.comprehensive_reflection = self._build_comprehensive_reflection()

    def get_context_for_operator(self, operator_name: str) -> str:
        with self._lock:
            if not self._memory.current_reflection and not self._memory.analysis:
                return ""

            if operator_name in {"e1", "e2"}: # crossover
                parts = [
                    "Observed structural patterns:",
                    self._memory.analysis or "None",
                    "",
                    "Validated guidance:",
                    self._stringify_reflections(self._memory.good_reflections, default="None"),
                    "",
                    "Avoid these ineffective directions:",
                    self._stringify_reflections(self._memory.bad_reflections, default="None"),
                ]
                if self._memory.comprehensive_reflection:
                    parts.extend(["", "Compressed reflection:", self._memory.comprehensive_reflection])
                return "\n".join(parts)

            parts = [
                "Current actionable guidance:",
                self._memory.current_reflection or "None",
                "",
                "Validated guidance:",
                self._stringify_reflections(self._memory.good_reflections, default="None"),
                "",
                "Avoid these ineffective directions:",
                self._stringify_reflections(self._memory.bad_reflections, default="None"),
            ]
            if self._memory.comprehensive_reflection:
                parts.extend(["", "Compressed reflection:", self._memory.comprehensive_reflection])
            return "\n".join(parts)

    def _promote_previous_reflection(self, reflection_worked: bool | None) -> None:
        previous = self._memory.current_reflection.strip()
        if not previous or reflection_worked is None:
            return

        target = self._memory.good_reflections if reflection_worked else self._memory.bad_reflections
        if previous not in target:
            target.append(previous)

        if len(self._memory.good_reflections) > self._max_good_reflections:
            self._memory.good_reflections = self._memory.good_reflections[-self._max_good_reflections:]
        if len(self._memory.bad_reflections) > self._max_bad_reflections:
            self._memory.bad_reflections = self._memory.bad_reflections[-self._max_bad_reflections:]

    def _prepare_candidates(
        self,
        current_elitist: List[Function],
        previous_population: List[Function],
    ) -> List[Function]:
        valid_funcs: List[Function] = []
        seen = set()
        for func in list(current_elitist) + list(previous_population):
            if func is None or func.score is None:
                continue
            score = np.array(func.score)
            if np.isinf(score).any():
                continue
            signature = (str(func), tuple(score.tolist()))
            if signature in seen:
                continue
            seen.add(signature)
            valid_funcs.append(func)

        valid_funcs.sort(key=lambda f: (float(f.score[0]), float(f.score[1])), reverse=True)
        return valid_funcs

    def _build_flash_reflection_prompt(self, candidates: List[Function]) -> str:
        lst_method = []
        for idx, func in enumerate(candidates, start=1):
            lst_method.append(
                f"[Heuristic {idx}]\n"
                f"Algorithm: {getattr(func, 'algorithm', 'Unknown')}\n"
                f"Score: {np.array(func.score).tolist()}\n"
                f"Code:\n{str(func)}\n"
            )
        return FLASH_REFLECTION_PROMPT.format(
            task_description=self._task_description,
            lst_method="\n".join(lst_method),
        )

    def _parse_flash_reflection(self, response: str) -> tuple[str, str]:
        if not response:
            return "", ""
        analyze_start = response.find("**Analysis:**")
        experience_start = response.find("**Experience:**")
        if analyze_start == -1 or experience_start == -1:
            return "", response.strip()
        analysis = response[analyze_start + len("**Analysis:**"):experience_start].strip()
        experience = response[experience_start + len("**Experience:**"):].strip()
        return analysis, experience

    def _build_comprehensive_reflection(self) -> str:
        prompt = COMPREHENSIVE_REFLECTION_PROMPT.format(
            current_reflection=self._memory.current_reflection or "None",
            good_reflection=self._stringify_reflections(self._memory.good_reflections, default="None"),
            bad_reflection=self._stringify_reflections(self._memory.bad_reflections, default="None"),
        )
        return self._llm.draw_sample(
            [
                {"role": "system", "content": FLASH_REFLECTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        ).strip()

    @staticmethod
    def _stringify_reflections(reflections: List[str], *, default: str) -> str:
        if not reflections:
            return default
        return "\n".join(f"- {reflection}" for reflection in reflections)
