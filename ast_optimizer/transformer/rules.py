from __future__ import annotations

import ast
from typing import List

from ast_optimizer.analyzer.models import OptimizationType
from ast_optimizer.common.ast_utils import bound_names, replace_node, try_fold
from ast_optimizer.common.patterns import (match_append_loop, match_dead_branch,
                                           match_listcomp_call, match_negated_membership)

from .exceptions import StaleCandidateError
from .interfaces import IRewriteRule, RewriteContext


class ConstantFoldingRule(IRewriteRule):
    optimization_type = OptimizationType.CONSTANT_FOLDING

    def apply(self, node: ast.AST, ctx: RewriteContext) -> None:
        folded = try_fold(node)
        if folded is None:
            raise StaleCandidateError("Выражение больше не сворачивается")
        replace_node(ctx.parents, node, [ast.copy_location(folded, node)])


class DeadBranchRule(IRewriteRule):
    optimization_type = OptimizationType.DEAD_BRANCH_ELIMINATION

    def apply(self, node: ast.AST, ctx: RewriteContext) -> None:
        if not match_dead_branch(node):
            raise StaleCandidateError("Условие больше не является константой")
        taken: List[ast.stmt] = node.body if node.test.value else node.orelse
        replace_node(ctx.parents, node, taken)


class LoopToComprehensionRule(IRewriteRule):
    optimization_type = OptimizationType.LOOP_TO_COMPREHENSION

    def apply(self, node: ast.AST, ctx: RewriteContext) -> None:
        m = match_append_loop(node, ctx.parents)
        if m is None:
            raise StaleCandidateError("Шаблон 'цикл + append' не найден")
        comp = ast.ListComp(
            elt=m.elt,
            generators=[ast.comprehension(target=m.loop.target, iter=m.loop.iter,
                                          ifs=[m.cond] if m.cond is not None else [], is_async=0)])
        m.init.value = ast.copy_location(comp, m.init.value)
        replace_node(ctx.parents, m.loop, [])


class ListCompToGeneratorRule(IRewriteRule):
    optimization_type = OptimizationType.LISTCOMP_TO_GENERATOR

    def apply(self, node: ast.AST, ctx: RewriteContext) -> None:
        if not match_listcomp_call(node, bound_names(ctx.root)):
            raise StaleCandidateError("Вызов с списковым включением не найден")
        comp = node.args[0]
        node.args[0] = ast.copy_location(
            ast.GeneratorExp(elt=comp.elt, generators=comp.generators), comp)


class SimplifyNegationRule(IRewriteRule):
    optimization_type = OptimizationType.SIMPLIFY_NEGATION

    def apply(self, node: ast.AST, ctx: RewriteContext) -> None:
        if not match_negated_membership(node):
            raise StaleCandidateError("Шаблон 'not (a in b)' не найден")
        cmp_ = node.operand
        new_op = ast.NotIn() if isinstance(cmp_.ops[0], ast.In) else ast.IsNot()
        new = ast.Compare(left=cmp_.left, ops=[new_op], comparators=cmp_.comparators)
        replace_node(ctx.parents, node, [ast.copy_location(new, node)])


def default_rules() -> List[IRewriteRule]:
    return [ConstantFoldingRule(), DeadBranchRule(), LoopToComprehensionRule(),
            ListCompToGeneratorRule(), SimplifyNegationRule()]
