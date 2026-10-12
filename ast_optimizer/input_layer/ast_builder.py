from __future__ import annotations

import ast
import copy
from typing import Dict

from .interfaces import IASTBuilder
from .models import ASTTree, SourceCode


_SHARED_NODES = (ast.expr_context, ast.operator, ast.boolop, ast.unaryop, ast.cmpop)


class ASTBuilder(IASTBuilder):
    def build(self, source: SourceCode, root: ast.Module) -> ASTTree:
        ast.fix_missing_locations(root)
        parents: Dict[ast.AST, ast.AST] = {}
        count = 0
        for node in ast.walk(root):
            count += 1
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, _SHARED_NODES):
                    parents[child] = node
        return ASTTree(source=source, root=root, parents=parents, node_count=count)

    def clone(self, tree: ASTTree) -> ASTTree:
        return self.build(tree.source, copy.deepcopy(tree.root))