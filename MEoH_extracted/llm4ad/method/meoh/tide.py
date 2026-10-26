"""TIDE section 3.2.2: fixed-structure numerical refinement (not LLM training)."""
from __future__ import annotations

import ast
import copy
import json
import re
from dataclasses import dataclass

import numpy as np


def valid_score(score):
    return score is not None and np.asarray(score).size > 0 and np.isfinite(score).all()


def improves(candidate, incumbent):
    """Conservative multi-objective adaptation: accept strict Pareto improvement."""
    if not valid_score(candidate):
        return False
    if not valid_score(incumbent):
        return True
    a, b = np.asarray(candidate), np.asarray(incumbent)
    return bool(np.all(a >= b) and np.any(a > b))


@dataclass(frozen=True)
class TideConfig:
    population_size: int = 4
    iterations: int = 3
    scale: float = 0.5
    crossover: float = 0.7
    relative_radius: float = 0.5
    zero_radius: float = 0.1
    near_zero: float = 1e-8
    gaussian_sigma: float = 0.2
    seed: int = 2026

    def __post_init__(self):
        if self.population_size < 4 or self.iterations < 0:
            raise ValueError('TIDE requires population_size >= 4 and iterations >= 0')
        if not 0 < self.scale < 1 or not 0 <= self.crossover <= 1:
            raise ValueError('Invalid TIDE differential mutation/crossover settings')
        if not all(np.isfinite(v) and v > 0 for v in
                   (self.relative_radius, self.zero_radius, self.near_zero, self.gaussian_sigma)):
            raise ValueError('TIDE trust-region settings must be finite and positive')


class NumericTemplate:
    """Expose only real literal nodes; the LLM selects IDs, never rewrites code."""
    def __init__(self, source):
        self.tree = ast.parse(source)
        self.nodes = []

        def visit(node):
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                if isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float):
                    self.nodes.append(node)
                    return
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                self.nodes.append(node)
                return
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(self.tree)
        self.values = [ast.literal_eval(node) for node in self.nodes]

    def prompt(self, source):
        literals = [{'id': i, 'line': n.lineno, 'value': self.values[i]}
                    for i, n in enumerate(self.nodes)]
        return ('Identify continuous tunable heuristic coefficients only. Exclude array indices, '
                'dimensions, loop counts, seeds, boolean values, numerical stability epsilons and '
                'structural constants. Return ONLY a JSON list of literal IDs, e.g. [2, 5], '
                'or [] if none. Do not rewrite the code.\n'
                + json.dumps(literals) + '\n```python\n' + source + '\n```')

    def select(self, response):
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.strip())
        ids = json.loads(cleaned)
        if not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < len(self.nodes) for i in ids):
            raise ValueError('TIDE parameter selection must be a list of valid literal IDs')
        return list(dict.fromkeys(ids))

    def materialize(self, ids, values):
        tree = copy.deepcopy(self.tree)
        replacements = {(self.nodes[i].lineno, self.nodes[i].col_offset): float(v)
                        for i, v in zip(ids, values)}

        class Replace(ast.NodeTransformer):
            def visit(self, node):
                key = (getattr(node, 'lineno', None), getattr(node, 'col_offset', None))
                if key in replacements and isinstance(node, (ast.Constant, ast.UnaryOp)):
                    return ast.copy_location(ast.Constant(replacements[key]), node)
                return super().visit(node)

        return ast.unparse(ast.fix_missing_locations(Replace().visit(tree)))


class TideRefiner:
    def __init__(self, config=None):
        self.config = config or TideConfig()

    def refine(self, source, score, draw_sample, evaluate):
        """evaluate(source) -> score; original is always a fallback. Returns code, score, calls."""
        if not valid_score(score):
            return source, score, 0
        template = NumericTemplate(source)
        if not template.nodes:
            return source, score, 0
        ids = template.select(draw_sample(template.prompt(source)))
        if not ids:
            return source, score, 0
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        x = np.array([template.values[i] for i in ids], dtype=float)
        radius = np.where(np.abs(x) > cfg.near_zero, np.abs(x) * cfg.relative_radius, cfg.zero_radius)
        lower, upper = x - radius, x + radius
        vectors = np.clip(x + rng.normal(size=(cfg.population_size, len(x))) * radius * cfg.gaussian_sigma,
                          lower, upper)
        vectors[0] = x
        scores = [score] + [None] * (cfg.population_size - 1)
        best_source, best_score, calls = source, score, 0

        def assess(vector):
            nonlocal best_source, best_score, calls
            code = template.materialize(ids, vector)
            value = evaluate(code)
            calls += 1
            if improves(value, best_score):
                best_source, best_score = code, value
            return value

        for i in range(1, cfg.population_size):
            scores[i] = assess(vectors[i])
        for _ in range(cfg.iterations):
            previous = vectors.copy()
            for i in range(cfg.population_size):
                a, b, c = rng.choice([j for j in range(cfg.population_size) if j != i], 3, replace=False)
                mutant = np.clip(previous[a] + cfg.scale * (previous[b] - previous[c]), lower, upper)
                mask = rng.random(len(x)) < cfg.crossover
                mask[rng.integers(len(x))] = True
                trial = np.where(mask, mutant, previous[i])
                value = assess(trial)
                if improves(value, scores[i]):
                    vectors[i], scores[i] = trial, value
        return best_source, best_score, calls
