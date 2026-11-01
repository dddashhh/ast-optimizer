from __future__ import annotations

from ast_optimizer.analyzer.models import ASTFeatures, OptimizationCandidate

from .models import HeuristicCostModel


def candidate_feature_vector(
    candidate: OptimizationCandidate,
    ast_features: ASTFeatures,
    candidate_count: int = 1,
) -> dict[str, float]:
    features = ast_features.to_vector()
    features.update({
        "prior_speedup": HeuristicCostModel.prior_for(candidate),
        f"candidate.{candidate.type.value}": 1.0,
        f"node_type.{candidate.node_type}": 1.0,
        "candidate.line": float(candidate.lineno),
        "candidate.column": float(candidate.col_offset),
        "candidate.count": float(candidate_count),
    })
    return features
