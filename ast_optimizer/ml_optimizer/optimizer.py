from __future__ import annotations

from typing import Iterable, Optional

from ast_optimizer.analyzer.models import ASTFeatures, OptimizationCandidate

from .models import CandidateScore, CostModel, HeuristicCostModel


class MLOptimizer:
    """Score and rank candidates through a replaceable cost-model interface."""
    def __init__(self, model: Optional[CostModel] = None) -> None:
        self.model = model or HeuristicCostModel()

    def score(self, candidate: OptimizationCandidate, ast_features: ASTFeatures) -> CandidateScore:
        vector = ast_features.to_vector()
        vector.update({
            "prior_speedup": HeuristicCostModel.prior_for(candidate),
            f"candidate.{candidate.type.value}": 1.0,
            "candidate.line": float(candidate.lineno),
            "candidate.column": float(candidate.col_offset),
        })
        prediction = self.model.predict(vector)
        if prediction != prediction or prediction in (float("inf"), float("-inf")):
            prediction = 1.0
        explanation = ("Оценка эвристическая; требует подтверждения benchmark."
                       if isinstance(self.model, HeuristicCostModel)
                       else f"Оценка рассчитана моделью {self.model.name}; требует подтверждения benchmark.")
        return CandidateScore(candidate, max(1.0, float(prediction)), self.model.name, explanation)

    def rank(self, candidates: Iterable[OptimizationCandidate],
             ast_features: ASTFeatures) -> list[CandidateScore]:
        return sorted((self.score(c, ast_features) for c in candidates),
                      key=lambda item: (-item.predicted_speedup, item.candidate.lineno,
                                        item.candidate.col_offset, item.candidate.node_index))

    def select_best(self, candidates: Iterable[OptimizationCandidate],
                    ast_features: ASTFeatures) -> Optional[CandidateScore]:
        ranked = self.rank(candidates, ast_features)
        return ranked[0] if ranked else None
