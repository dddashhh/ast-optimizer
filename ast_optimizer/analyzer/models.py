from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import cached_property
from typing import Dict, List, Set

from ast_optimizer.common.ast_utils import bound_names
from ast_optimizer.input_layer.models import ASTTree


class OptimizationType(str, Enum):
    CONSTANT_FOLDING = "constant_folding"
    DEAD_BRANCH_ELIMINATION = "dead_branch_elimination"
    LOOP_TO_COMPREHENSION = "loop_to_comprehension"
    LISTCOMP_TO_GENERATOR = "listcomp_to_generator"
    SIMPLIFY_NEGATION = "simplify_negation"


@dataclass(frozen=True)
class OptimizationCandidate:
    type: OptimizationType
    node_index: int
    node_type: str
    lineno: int
    col_offset: int
    description: str

    @property
    def id(self) -> str:
        return f"{self.type.value}@{self.lineno}:{self.col_offset}#{self.node_index}"


@dataclass(frozen=True)
class ASTFeatures:
    node_count: int = 0
    line_count: int = 0
    max_depth: int = 0
    function_count: int = 0
    class_count: int = 0
    loop_count: int = 0
    max_loop_depth: int = 0
    branch_count: int = 0
    call_count: int = 0
    comprehension_count: int = 0
    binop_count: int = 0
    constant_count: int = 0
    cyclomatic_complexity: int = 1

    def to_vector(self) -> Dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class AnalysisResult:
    features: ASTFeatures
    candidates: List[OptimizationCandidate] = field(default_factory=list)

    @property
    def candidate_counts(self) -> Dict[OptimizationType, int]:
        counts: Dict[OptimizationType, int] = {}
        for c in self.candidates:
            counts[c.type] = counts.get(c.type, 0) + 1
        return counts

    def by_type(self, opt_type: OptimizationType) -> List[OptimizationCandidate]:
        return [c for c in self.candidates if c.type == opt_type]


@dataclass
class AnalysisContext:
    tree: ASTTree
    nodes: List[ast.AST]
    index: Dict[ast.AST, int]

    @classmethod
    def build(cls, tree: ASTTree) -> "AnalysisContext":
        nodes = list(ast.walk(tree.root))
        return cls(tree=tree, nodes=nodes, index={n: i for i, n in enumerate(nodes)})

    @cached_property
    def shadowed_names(self) -> Set[str]:
        return bound_names(self.tree.root)

    def make_candidate(self, opt_type: OptimizationType, node: ast.AST,
                       description: str) -> OptimizationCandidate:
        return OptimizationCandidate(
            type=opt_type, node_index=self.index[node], node_type=type(node).__name__,
            lineno=getattr(node, "lineno", 0), col_offset=getattr(node, "col_offset", 0),
            description=description)
