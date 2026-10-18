from __future__ import annotations

import ast
from typing import List

from ast_optimizer.common.ast_utils import try_fold
from ast_optimizer.common.patterns import (match_append_loop, match_dead_branch,
                                           match_listcomp_call, match_negated_membership)

from .interfaces import ICandidateDetector
from .models import AnalysisContext, OptimizationCandidate, OptimizationType


class ConstantFoldingDetector(ICandidateDetector):
    optimization_type = OptimizationType.CONSTANT_FOLDING

    def detect(self, ctx: AnalysisContext) -> List[OptimizationCandidate]:
        found: List[OptimizationCandidate] = []
        stack: List[ast.AST] = [ctx.tree.root]
        while stack:
            node = stack.pop()
            if try_fold(node) is not None:
                found.append(ctx.make_candidate(
                    self.optimization_type, node, "Выражение из констант вычисляется заранее"))
                continue
            stack.extend(ast.iter_child_nodes(node))
        return found


class DeadBranchDetector(ICandidateDetector):
    optimization_type = OptimizationType.DEAD_BRANCH_ELIMINATION

    def detect(self, ctx: AnalysisContext) -> List[OptimizationCandidate]:
        return [ctx.make_candidate(self.optimization_type, n, "Условие - константа, одна ветка недостижима")
                for n in ctx.nodes if match_dead_branch(n)]


class LoopToComprehensionDetector(ICandidateDetector):
    optimization_type = OptimizationType.LOOP_TO_COMPREHENSION

    def detect(self, ctx: AnalysisContext) -> List[OptimizationCandidate]:
        parents = ctx.tree.parents
        return [ctx.make_candidate(self.optimization_type, n, "Цикл с append заменяется списковым включением")
                for n in ctx.nodes if match_append_loop(n, parents) is not None]


class ListCompToGeneratorDetector(ICandidateDetector):
    optimization_type = OptimizationType.LISTCOMP_TO_GENERATOR

    def detect(self, ctx: AnalysisContext) -> List[OptimizationCandidate]:
        shadowed = ctx.shadowed_names
        return [ctx.make_candidate(self.optimization_type, n, "Списковое включение в аргументе -> генератор")
                for n in ctx.nodes if match_listcomp_call(n, shadowed)]


class SimplifyNegationDetector(ICandidateDetector):
    optimization_type = OptimizationType.SIMPLIFY_NEGATION

    def detect(self, ctx: AnalysisContext) -> List[OptimizationCandidate]:
        return [ctx.make_candidate(self.optimization_type, n, "not (a in b) -> a not in b")
                for n in ctx.nodes if match_negated_membership(n)]


def default_detectors() -> List[ICandidateDetector]:
    return [ConstantFoldingDetector(), DeadBranchDetector(), LoopToComprehensionDetector(),
            ListCompToGeneratorDetector(), SimplifyNegationDetector()]
