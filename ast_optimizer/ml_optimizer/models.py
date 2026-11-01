from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from ast_optimizer.analyzer.models import ASTFeatures, OptimizationCandidate, OptimizationType


@dataclass(frozen=True)
class CandidateScore:
    candidate: OptimizationCandidate
    predicted_speedup: float
    model_name: str
    explanation: str


class CostModel(Protocol):
    name: str

    def predict(self, features: Mapping[str, float]) -> float: ...


class HeuristicCostModel:
    name = "heuristic-baseline"
    _priors = {
        OptimizationType.CONSTANT_FOLDING.value: 1.04,
        OptimizationType.DEAD_BRANCH_ELIMINATION.value: 1.05,
        OptimizationType.LOOP_TO_COMPREHENSION.value: 1.02,
        OptimizationType.LISTCOMP_TO_GENERATOR.value: 1.03,
        OptimizationType.SIMPLIFY_NEGATION.value: 1.005,
    }

    def predict(self, features: Mapping[str, float]) -> float:
        prior = float(features.get("prior_speedup", 1.0))
        # Candidate count and program size are weak context signals, not proof
        # of runtime benefit. Keep this baseline deliberately conservative.
        complexity = max(0.0, float(features.get("cyclomatic_complexity", 1.0)))
        node_count = max(0.0, float(features.get("node_count", 0.0)))
        context = min(0.01, complexity / 10000.0 + node_count / 100000.0)
        return max(1.0, prior + context)

    @classmethod
    def prior_for(cls, candidate: OptimizationCandidate) -> float:
        return cls._priors.get(candidate.type.value, 1.0)


class LinearCostModel:
    def __init__(self, weights: Mapping[str, float], intercept: float = 1.0,
                 name: str = "linear-cost-model",
                 feature_means: Mapping[str, float] | None = None,
                 feature_scales: Mapping[str, float] | None = None,
                 min_speedup: float = 1.0) -> None:
        self.weights = {str(k): float(v) for k, v in weights.items()}
        self.intercept = float(intercept)
        self.name = name
        self.feature_means = {str(k): float(v) for k, v in (feature_means or {}).items()}
        self.feature_scales = {str(k): float(v) for k, v in (feature_scales or {}).items()}
        self.min_speedup = float(min_speedup)
        if self.min_speedup <= 0 or not __import__("math").isfinite(self.min_speedup):
            raise ValueError("min_speedup должен быть конечным положительным числом")

    @classmethod
    def from_json(cls, path: str | Path) -> "LinearCostModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("weights", {}), dict):
            raise ValueError("Ожидался JSON-объект с полем weights")
        return cls(payload.get("weights", {}), payload.get("intercept", 1.0),
                   payload.get("name", "linear-cost-model"),
                   payload.get("feature_means", {}), payload.get("feature_scales", {}),
                   payload.get("min_speedup", 1.0))

    def predict(self, features: Mapping[str, float]) -> float:
        import math
        prediction = self.intercept
        for key, weight in self.weights.items():
            value = float(features.get(key, 0.0))
            if key in self.feature_means:
                scale = self.feature_scales.get(key, 1.0) or 1.0
                value = (value - self.feature_means[key]) / scale
            prediction += weight * value
        if not math.isfinite(prediction):
            return self.min_speedup
        return max(self.min_speedup, prediction)
