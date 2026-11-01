from __future__ import annotations

from typing import Iterable, Optional

from ast_optimizer.analyzer.models import ASTFeatures, OptimizationCandidate

from .models import CandidateScore, CostModel, HeuristicCostModel
from .feature_vector import candidate_feature_vector


class MLOptimizer:
    def __init__(self, model: Optional[CostModel] = None) -> None:
        self.model = model or HeuristicCostModel()

    def score(self, candidate: OptimizationCandidate, ast_features: ASTFeatures,
              candidate_count: int = 1) -> CandidateScore:
        vector = candidate_feature_vector(candidate, ast_features, candidate_count)
        prediction = self.model.predict(vector)
        if prediction != prediction or prediction in (float("inf"), float("-inf")):
            prediction = 1.0
        explanation = ("Оценка эвристическая; требует подтверждения benchmark."
                       if isinstance(self.model, HeuristicCostModel)
                       else f"Оценка рассчитана моделью {self.model.name}; требует подтверждения benchmark.")
        return CandidateScore(candidate, max(0.01, float(prediction)), self.model.name, explanation)

    def rank(self, candidates: Iterable[OptimizationCandidate],
             ast_features: ASTFeatures) -> list[CandidateScore]:
        items = list(candidates)
        return sorted((self.score(c, ast_features, len(items)) for c in items),
                      key=lambda item: (-item.predicted_speedup, item.candidate.lineno,
                                        item.candidate.col_offset, item.candidate.node_index))

    def select_best(self, candidates: Iterable[OptimizationCandidate],
                    ast_features: ASTFeatures) -> Optional[CandidateScore]:
        ranked = self.rank(candidates, ast_features)
        if not ranked or ranked[0].predicted_speedup <= 1.0:
            return None
        return ranked[0]
