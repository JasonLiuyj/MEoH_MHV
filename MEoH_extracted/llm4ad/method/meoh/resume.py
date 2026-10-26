from __future__ import annotations

import copy
import json
import os
import re
from typing import Optional

from tqdm.auto import tqdm

from .meoh import MEoH
from .population import Population
from .raise_component import refresh_population
from ...base import Function, TextFunctionProgramConverter as tfpc


def _get_latest_pop_json(log_path: str):
    path = os.path.join(log_path, 'population')
    if not os.path.isdir(path):
        raise FileNotFoundError(f"population directory not found at {path}")

    orders = []
    for filename in os.listdir(path):
        match = re.fullmatch(r'pop_(\d+)\.json', filename)
        if match is not None:
            orders.append(int(match.group(1)))

    if len(orders) == 0:
        raise FileNotFoundError(f"no population checkpoint found in {path}")

    max_order = max(orders)
    return os.path.join(path, f'pop_{max_order}.json'), max_order


def _load_json_records(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data


def _normalize_score(score, num_objs: int, fill_value: float):
    if score is None:
        return None

    if num_objs >= 2:
        if isinstance(score, (int, float)):
            return [score] + [fill_value] * (num_objs - 1)

        try:
            score_list = list(score)
        except (TypeError, ValueError):
            return [score] + [fill_value] * (num_objs - 1)

        if len(score_list) < num_objs:
            score_list = score_list + [fill_value] * (num_objs - len(score_list))
        return score_list[:num_objs]

    if isinstance(score, (list, tuple)):
        return score[0] if len(score) > 0 else fill_value
    return score


def _record_to_function(record, num_objs: int) -> Function | None:
    func = tfpc.text_to_function(record['function'])
    if func is None:
        return None

    score = _normalize_score(record.get('score'), num_objs=num_objs, fill_value=-1e9)
    if score is None:
        return None

    func.score = score
    func.algorithm = record.get('algorithm', '')
    func.training_set_version = record.get('training_set_version', 0)
    flag = record.get('flag')
    if flag is not None:
        setattr(func, 'flag', flag)

    if not hasattr(func, 'entire_code') or func.entire_code is None:
        func.entire_code = record.get('program') or str(func)
    return func


def _resume_pop(log_path: str, pop_size: int, num_objs: int = 2) -> Population:
    _, max_gen = _get_latest_pop_json(log_path)
    pop_path = os.path.join(log_path, 'population', f'pop_{max_gen}.json')
    elitist_path = os.path.join(log_path, 'elitist', f'elitist_{max_gen}.json')
    print(f'RESUME MEoH: Generations: {max_gen}.', flush=True)

    population_funcs = []
    for record in _load_json_records(pop_path):
        func = _record_to_function(record, num_objs=num_objs)
        if func is not None:
            population_funcs.append(func)

    elitist_funcs = []
    for record in _load_json_records(elitist_path):
        func = _record_to_function(record, num_objs=num_objs)
        if func is not None:
            elitist_funcs.append(func)

    pop = Population(pop_size=pop_size, generation=max_gen, pop=population_funcs)
    pop._elitist = elitist_funcs.copy()
    return pop


def _resume_text2func(func_text, score, template_func: Function):
    temp = copy.deepcopy(template_func)
    func = tfpc.text_to_function(func_text)
    if func is None:
        temp.body = '    pass'
        temp.score = None
        return temp

    func.score = score
    if not hasattr(func, 'entire_code') or func.entire_code is None:
        func.entire_code = str(func)
    return func


def _get_all_samples_and_scores(path):
    """
    Collect all sampled functions/scores from log_path/samples.

    Robust to filenames like `samples_0~200.json` or `samples_123.json`.
    """
    samples_dir = os.path.join(path, "samples")
    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"samples directory not found at {samples_dir}")

    sample_files = [
        filename
        for filename in os.listdir(samples_dir)
        if filename.startswith("samples_") and filename.endswith(".json") and filename != "samples_best.json"
    ]

    def extract_number(filename: str) -> int:
        match = re.search(r"samples_(\d+)", filename)
        return int(match.group(1)) if match else -1

    def extract_range(filename: str) -> tuple[int, int]:
        match = re.search(r"samples_(\d+)~(\d+)", filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        number = extract_number(filename)
        return (number, number + 200) if number >= 0 else (-1, -1)

    sample_files = sorted(sample_files, key=extract_number)

    all_func = []
    all_score = []
    max_order = 0
    seen_sample_orders = set()

    for filename in sample_files:
        file_path = os.path.join(samples_dir, filename)
        data = _load_json_records(file_path)
        lower_bound, upper_bound = extract_range(filename)

        for sample in data:
            sample_order = sample.get("sample_order")
            if sample_order is not None:
                if lower_bound >= 0 and upper_bound >= 0 and not (lower_bound <= sample_order < upper_bound):
                    continue
                if sample_order in seen_sample_orders:
                    continue
                seen_sample_orders.add(sample_order)

            all_func.append(sample.get("function"))
            all_score.append(sample.get("score") if sample.get("score") is not None else float("-inf"))

            if sample_order is not None and sample_order > max_order:
                max_order = sample_order

        max_order = max(max_order, extract_number(filename))

    return all_func, all_score, max_order


def _resume_pf(log_path: str, profiler, template_func):
    _, db_max_order = _get_latest_pop_json(log_path)
    funcs, scores, _ = _get_all_samples_and_scores(log_path)
    print(f'RESUME MEoH: Sample order: {len(funcs)}.', flush=True)
    profiler.__class__._prog_db_order = db_max_order

    normalized_scores = [
        _normalize_score(score, num_objs=profiler._num_objs, fill_value=float('-inf'))
        for score in scores
    ]

    for i in tqdm(range(len(funcs)), desc='Resume MEoH Profiler'):
        func = _resume_text2func(funcs[i], normalized_scores[i], template_func)
        profiler.register_function(func, resume_mode=True)

    profiler._cur_gen = db_max_order


def _get_latest_flash_reflection_json(log_path: str):
    path = os.path.join(log_path, 'flash_reflection')
    if not os.path.isdir(path):
        return None, -1

    orders = []
    for filename in os.listdir(path):
        match = re.fullmatch(r'fr_(\d+)\.json', filename)
        if match is not None:
            orders.append(int(match.group(1)))

    if len(orders) == 0:
        return None, -1

    max_order = max(orders)
    return os.path.join(path, f'fr_{max_order}.json'), max_order


def _resume_flash_reflection(meoh: MEoH, log_path: str):
    if getattr(meoh, '_flash_reflection', None) is None:
        return

    fr_path, generation = _get_latest_flash_reflection_json(log_path)
    if fr_path is None:
        return

    with open(fr_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meoh._flash_reflection.restore_memory(
        analysis=data.get('analysis', ''),
        experience=data.get('experience', ''),
        current_reflection=data.get('current_reflection', ''),
        comprehensive_reflection=data.get('comprehensive_reflection', ''),
        good_reflections=data.get('good_reflections', []),
        bad_reflections=data.get('bad_reflections', []),
        source_generation=data.get('generation', generation),
    )
    meoh._last_reflection_generation = data.get('generation', generation)
    meoh._previous_reflection_baseline = meoh._summarize_elitist_scores(meoh._population.elitist)


def _resume_raise(meoh: MEoH, log_path: str, checkpoint_generation: int):
    """Restore only RAISE updates represented by the selected population checkpoint."""
    if not getattr(meoh, '_use_raise', False):
        return

    updates_path = os.path.join(log_path, 'raise_updates.jsonl')
    events = []
    if os.path.isfile(updates_path):
        with open(updates_path, 'r', encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid RAISE log at line {line_number}: {updates_path}') from exc
                if int(event.get('generation', -1)) <= checkpoint_generation:
                    events.append(event)

    prepare_restore = getattr(meoh._task_evaluation, 'prepare_raise_restore', None)
    if callable(prepare_restore):
        prepare_restore(events)
    available_ids = meoh._task_evaluation.available_raise_instance_ids()
    for event in events:
        identifier = event.get('instance_id')
        if identifier not in available_ids:
            raise ValueError(
                f"Cannot restore RAISE instance {identifier!r}. The training data, split seed, "
                "or candidate pool differs from the original run."
            )
        available_ids.remove(identifier)

    for event in events:
        meoh._task_evaluation.add_raise_instance(event['instance_id'])

    def reassess(func):
        source = tfpc.function_to_program(func, meoh._template_program)
        return meoh._evaluation_executor.submit(
            meoh._evaluator.evaluate_program_record_time, source
        ).result()

    refresh_population(meoh._population, reassess)
    export_datasets = getattr(meoh._task_evaluation, 'export_raise_datasets', None)
    if callable(export_datasets) and events:
        export_datasets(checkpoint_generation)
    version = len(events)
    for func in meoh._population.population + meoh._population.elitist + meoh._population._next_gen_pop:
        func.training_set_version = version
    meoh.component_stats['raise_updates'] = events
    meoh._last_raise_generation = checkpoint_generation
    meoh._previous_reflection_baseline = None

    # Historical profiler records may use older training sets. Reset its live best
    # summary to the scores that were just recomputed without appending a new event.
    profiler = meoh._profiler
    if profiler is not None and meoh._population.elitist:
        elites = meoh._population.elitist
        winners = [max(elites, key=lambda f: f.score[i]) for i in range(meoh._num_objs)]
        profiler._cur_best_function = winners if meoh._num_objs > 1 else winners[0]
        values = [float(f.score[i]) for i, f in enumerate(winners)]
        profiler._cur_best_program_score = values if meoh._num_objs > 1 else values[0]
        orders = [None] * meoh._num_objs
        profiler._cur_best_program_sample_order = orders if meoh._num_objs > 1 else None

    print(
        f'RESUME RAISE: restored {version} update(s) through generation '
        f'{checkpoint_generation} and re-scored the population.',
        flush=True,
    )


def resume_meoh(meoh: MEoH, path: Optional[str] = None):
    """
    Resume MEoH training from the specified log directory (or the default profiler log_dir).

    Args:
        meoh: The MEoH instance to resume.
        path: Optional custom log directory to resume from. If None, uses meoh._profiler._log_dir.
    """
    meoh._resume_mode = True
    profiler = meoh._profiler
    log_path = path or profiler._log_dir

    meoh._population = _resume_pop(log_path, meoh._pop_size, num_objs=meoh._num_objs)
    checkpoint_generation = meoh._population.generation
    _resume_pf(log_path, profiler, meoh._function_to_evolve)
    _resume_flash_reflection(meoh, log_path)
    _resume_raise(meoh, log_path, checkpoint_generation)

    _, _, sample_max_order = _get_all_samples_and_scores(log_path)
    meoh._tot_sample_nums = sample_max_order
