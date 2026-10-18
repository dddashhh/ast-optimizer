from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

from ast_optimizer.analyzer.models import OptimizationCandidate, OptimizationType
from ast_optimizer.input_layer.models import ASTTree


@dataclass
class RewriteContext:
    root: ast.Module
    parents: Dict[ast.AST, ast.AST]


class IRewriteRule(ABC):
    """Одно правило переписывания: мутирует узел в рабочей копии дерева."""

    @property
    @abstractmethod
    def optimization_type(self) -> OptimizationType: ...

    @abstractmethod
    def apply(self, node: ast.AST, ctx: RewriteContext) -> None: ...


class IASTTransformer(ABC):
    @abstractmethod
    def transform(self, tree: ASTTree, candidate: OptimizationCandidate) -> ASTTree:
        """Возвращает НОВОЕ дерево; исходное не изменяется."""
