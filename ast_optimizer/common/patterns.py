from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, Optional, Set

from .ast_utils import contains_node, enclosing_scope, names_in, sibling_before

_SCOPE_BREAKERS = (ast.Yield, ast.YieldFrom, ast.Await, ast.NamedExpr)
_GENERATOR_FUNCS = {"sum", "min", "max", "any", "all"}



@dataclass
class AppendLoop:
    init: ast.Assign
    list_name: str
    loop: ast.For
    elt: ast.expr
    cond: Optional[ast.expr]


def _is_simple_target(target: ast.AST) -> bool:
    if isinstance(target, ast.Name):
        return True
    if isinstance(target, (ast.Tuple, ast.List)):
        return all(_is_simple_target(e) for e in target.elts)
    return False


def _as_append_call(stmt: ast.stmt):
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return None
    call = stmt.value
    func = call.func
    if (isinstance(func, ast.Attribute) and func.attr == "append" and isinstance(func.value, ast.Name)
            and len(call.args) == 1 and not call.keywords and not isinstance(call.args[0], ast.Starred)):
        return func.value.id, call.args[0]
    return None


def _is_empty_list_init(stmt: Optional[ast.AST], name: str) -> bool:
    return (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == name
            and isinstance(stmt.value, ast.List) and not stmt.value.elts)


def match_append_loop(loop: ast.AST, parents: Dict[ast.AST, ast.AST]) -> Optional[AppendLoop]:
    """
        res = []
        for x in it:            # (опционально: if cond:)
            res.append(expr)
    ->  res = [expr for x in it if cond]
    Все условия безопасности проверяются здесь.
    """
    if not isinstance(loop, ast.For) or loop.orelse or not _is_simple_target(loop.target):
        return None
    body, cond = loop.body, None
    if len(body) == 1 and isinstance(body[0], ast.If) and not body[0].orelse:
        cond, body = body[0].test, body[0].body
    if len(body) != 1:
        return None
    appended = _as_append_call(body[0])
    if appended is None:
        return None
    name, elt = appended

    init = sibling_before(loop, parents)
    if not _is_empty_list_init(init, name):
        return None

    scope = enclosing_scope(loop, parents)
    if scope is None or isinstance(scope, ast.ClassDef):
        return None

    parts = [loop.iter, elt] + ([cond] if cond is not None else [])
    for part in parts:
        if contains_node(part, _SCOPE_BREAKERS) or name in names_in(part):
            return None


    target_names = names_in(loop.target)
    inside = {id(n) for n in ast.walk(loop)}
    for n in ast.walk(scope):
        if isinstance(n, ast.Name) and id(n) not in inside and n.id in target_names:
            return None
    return AppendLoop(init=init, list_name=name, loop=loop, elt=elt, cond=cond)



def match_listcomp_call(call: ast.AST, shadowed: Set[str]) -> bool:
    """sum/min/max/any/all([... for ...]) -> то же с генератором."""
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
        return False
    fname = call.func.id
    if fname not in _GENERATOR_FUNCS or fname in shadowed:
        return False
    if len(call.args) != 1 or not isinstance(call.args[0], ast.ListComp):
        return False
    if call.keywords and fname not in ("min", "max"):
        return False
    comp = call.args[0]
    if contains_node(comp, (ast.Await,)):
        return False

    if fname in ("any", "all") and contains_node(comp, (ast.Call,)):
        return False
    return True



def match_negated_membership(node: ast.AST) -> bool:
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
            and isinstance(node.operand, ast.Compare) and len(node.operand.ops) == 1
            and isinstance(node.operand.ops[0], (ast.In, ast.Is)))



def match_dead_branch(node: ast.AST) -> bool:
    if not (isinstance(node, ast.If) and isinstance(node.test, ast.Constant)):
        return False
    removed = node.orelse if node.test.value else node.body

    breakers = (ast.Yield, ast.YieldFrom, ast.Global, ast.Nonlocal, ast.NamedExpr)
    return not any(contains_node(stmt, breakers) for stmt in removed)
