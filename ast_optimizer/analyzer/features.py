from __future__ import annotations

import ast

from ast_optimizer.input_layer.models import ASTTree

from .interfaces import IFeatureExtractor
from .models import ASTFeatures

_LOOPS = (ast.For, ast.AsyncFor, ast.While)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


class FeatureExtractor(IFeatureExtractor):

    def extract(self, tree: ASTTree) -> ASTFeatures:
        c = dict(function_count=0, class_count=0, loop_count=0, branch_count=0, call_count=0,
                 comprehension_count=0, binop_count=0, constant_count=0)
        max_depth = max_loop_depth = 0
        cyclomatic = 1
        stack = [(tree.root, 1, 0)]
        while stack:
            node, depth, loop_depth = stack.pop()
            max_depth = max(max_depth, depth)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                c["function_count"] += 1
            elif isinstance(node, ast.ClassDef):
                c["class_count"] += 1
            elif isinstance(node, (ast.If, ast.IfExp)):
                c["branch_count"] += 1
                cyclomatic += 1
            elif isinstance(node, ast.Call):
                c["call_count"] += 1
            elif isinstance(node, ast.BinOp):
                c["binop_count"] += 1
            elif isinstance(node, ast.Constant):
                c["constant_count"] += 1
            elif isinstance(node, _COMPREHENSIONS):
                c["comprehension_count"] += 1
            elif isinstance(node, ast.BoolOp):
                cyclomatic += len(node.values) - 1
            elif isinstance(node, ast.ExceptHandler):
                cyclomatic += 1
            elif isinstance(node, ast.comprehension):
                cyclomatic += len(node.ifs)

            if isinstance(node, _LOOPS):
                c["loop_count"] += 1
                cyclomatic += 1
                loop_depth += 1
                max_loop_depth = max(max_loop_depth, loop_depth)
            for child in ast.iter_child_nodes(node):
                stack.append((child, depth + 1, loop_depth))

        return ASTFeatures(
            node_count=tree.node_count, line_count=len(tree.source.text.splitlines()),
            max_depth=max_depth, max_loop_depth=max_loop_depth,
            cyclomatic_complexity=cyclomatic, **c)
