from __future__ import annotations

import ast
from typing import Dict, Optional, Sequence

from ast_optimizer.analyzer.models import OptimizationCandidate, OptimizationType
from ast_optimizer.input_layer.ast_builder import ASTBuilder
from ast_optimizer.input_layer.interfaces import IASTBuilder
from ast_optimizer.input_layer.models import ASTTree

from .exceptions import StaleCandidateError, TransformError, UnsupportedOptimizationError
from .interfaces import IASTTransformer, IRewriteRule, RewriteContext
from .rules import default_rules


class ASTTransformer(IASTTransformer):

    def __init__(self, builder: Optional[IASTBuilder] = None,
                 rules: Optional[Sequence[IRewriteRule]] = None) -> None:
        self.builder = builder or ASTBuilder()
        self._rules: Dict[OptimizationType, IRewriteRule] = {
            r.optimization_type: r for r in (rules if rules is not None else default_rules())}

    @property
    def supported_types(self):
        return set(self._rules)

    def transform(self, tree: ASTTree, candidate: OptimizationCandidate) -> ASTTree:
        rule = self._rules.get(candidate.type)
        if rule is None:
            raise UnsupportedOptimizationError(f"Нет правила для {candidate.type.value}")

        work = self.builder.clone(tree)
        nodes = list(ast.walk(work.root))
        if not 0 <= candidate.node_index < len(nodes):
            raise StaleCandidateError(f"Индекс узла вне дерева: {candidate.node_index}")
        node = nodes[candidate.node_index]
        if type(node).__name__ != candidate.node_type:
            raise StaleCandidateError(
                f"Ожидался {candidate.node_type}, найден {type(node).__name__}")

        rule.apply(node, RewriteContext(root=work.root, parents=work.parents))
        ast.fix_missing_locations(work.root)
        self._verify(work)
        return self.builder.build(work.source, work.root)

    @staticmethod
    def _verify(tree: ASTTree) -> None:
        try:
            compile(tree.root, tree.source.filename, "exec")
        except Exception as exc:
            raise TransformError(f"Преобразование дало некорректное AST: {exc}") from exc
