import ast

from ast_optimizer import InputPipeline
from ast_optimizer.analyzer import ASTAnalyzer, ASTFeatures, OptimizationType


def test_feature_extractor_counts_basic_nodes():
    tree = InputPipeline().process("def f(x):\n    for y in x:\n        if y:\n            print(y + 1)\n")
    result = ASTAnalyzer().analyze(tree)

    assert isinstance(result.features, ASTFeatures)
    assert result.features.function_count == 1
    assert result.features.loop_count == 1
    assert result.features.branch_count == 1
    assert result.features.call_count == 1
    assert result.features.binop_count == 1


def test_analyzer_finds_supported_optimizations_in_source_order():
    tree = InputPipeline().process("x = 2 + 3\nif True:\n    y = 1\nz = not (a in b)\n")
    result = ASTAnalyzer().analyze(tree)

    types = [candidate.type for candidate in result.candidates]
    assert OptimizationType.CONSTANT_FOLDING in types
    assert OptimizationType.DEAD_BRANCH_ELIMINATION in types
    assert OptimizationType.SIMPLIFY_NEGATION in types
    assert [(c.lineno, c.col_offset) for c in result.candidates] == sorted(
        (c.lineno, c.col_offset) for c in result.candidates
    )


def test_analyzer_does_not_modify_tree():
    tree = InputPipeline().process("x = 2 + 3")
    before = ast.dump(tree.root)

    ASTAnalyzer().analyze(tree)

    assert ast.dump(tree.root) == before


def test_analysis_result_groups_candidates_by_type():
    result = ASTAnalyzer().analyze(InputPipeline().process("x = 2 + 3\ny = 4 * 5"))
    candidates = result.by_type(OptimizationType.CONSTANT_FOLDING)

    assert len(candidates) == 2
    assert result.candidate_counts[OptimizationType.CONSTANT_FOLDING] == 2
