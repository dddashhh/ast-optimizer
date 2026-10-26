import json

from ast_optimizer.analyzer.models import ASTFeatures, OptimizationCandidate, OptimizationType
from ast_optimizer.ml_optimizer import HeuristicCostModel, LinearCostModel, MLOptimizer


def candidate(kind=OptimizationType.CONSTANT_FOLDING, line=2):
    return OptimizationCandidate(kind, line, "BinOp", line, 0, "test candidate")


def test_heuristic_model_ranks_candidates_deterministically():
    optimizer = MLOptimizer()
    features = ASTFeatures(node_count=100, cyclomatic_complexity=5)
    ranked = optimizer.rank([candidate(OptimizationType.SIMPLIFY_NEGATION), candidate()], features)
    assert ranked[0].candidate.type is OptimizationType.CONSTANT_FOLDING
    assert ranked[0].predicted_speedup >= 1.0
    assert "эвристическая" in ranked[0].explanation


def test_select_best_returns_none_for_empty_candidates():
    assert MLOptimizer().select_best([], ASTFeatures()) is None


def test_linear_model_loads_json_and_predicts(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"name": "test-model", "intercept": 1.0,
                                "weights": {"node_count": 0.001}}), encoding="utf-8")
    model = LinearCostModel.from_json(path)
    assert model.name == "test-model"
    assert model.predict({"node_count": 10}) == 1.01
    assert model.predict({"node_count": -10000}) == 1.0


def test_model_can_be_injected():
    class FixedModel:
        name = "fixed"
        def predict(self, features):
            return 1.25
    score = MLOptimizer(FixedModel()).score(candidate(), ASTFeatures())
    assert score.predicted_speedup == 1.25
    assert score.model_name == "fixed"
