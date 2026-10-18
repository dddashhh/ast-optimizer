from __future__ import annotations

import ast
import math
import operator
from _ast import UnaryOp, Constant
from typing import Dict, Iterable, List, Optional, Set

MAX_FOLDED_SIZE = 4096                                                      
MAX_FOLDED_BITS = 8192                                                      
MAX_SHIFT_OR_POW = 1000                                                  

SCOPE_NODES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.LShift: operator.lshift, ast.RShift: operator.rshift,
    ast.BitOr: operator.or_, ast.BitXor: operator.xor, ast.BitAnd: operator.and_,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos,
              ast.Invert: operator.invert, ast.Not: operator.not_}
                                                                          
_CMP_OPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
}
_LITERAL_TYPES = (int, float, complex, str, bytes, bool, type(None))


class _NotFoldable(Exception):
    pass


                                                                             
def _check_size(value: object) -> None:
    if isinstance(value, (str, bytes)) and len(value) > MAX_FOLDED_SIZE:
        raise _NotFoldable
    if isinstance(value, int) and not isinstance(value, bool) and value.bit_length() > MAX_FOLDED_BITS:
        raise _NotFoldable


def _eval(node: ast.AST) -> object:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, _LITERAL_TYPES):
            raise _NotFoldable
        return node.value
    if isinstance(node, ast.UnaryOp):
        func = _UNARY_OPS.get(type(node.op))
        if func is None:
            raise _NotFoldable
        return func(_eval(node.operand))
    if isinstance(node, ast.BinOp):
        func = _BIN_OPS.get(type(node.op))
        if func is None:
            raise _NotFoldable
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, (ast.Pow, ast.LShift)) and isinstance(b, int) and abs(b) > MAX_SHIFT_OR_POW:
            raise _NotFoldable
        if isinstance(node.op, ast.Mult):
            for seq, n in ((a, b), (b, a)):
                if isinstance(seq, (str, bytes)) and isinstance(n, int) and len(seq) * n > MAX_FOLDED_SIZE:
                    raise _NotFoldable
        result = func(a, b)
        _check_size(result)
        return result
    if isinstance(node, ast.BoolOp):
        result = None
        is_and = isinstance(node.op, ast.And)
        for value_node in node.values:
            result = _eval(value_node)
            if bool(result) != is_and:                                                 
                break
        return result
    if isinstance(node, ast.Compare):
        left = _eval(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            func = _CMP_OPS.get(type(op))
            if func is None:
                raise _NotFoldable
            right = _eval(comparator)
            if not func(left, right):
                return False
            left = right
        return True
    raise _NotFoldable


def make_constant(value: object) -> UnaryOp | Constant:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        negative = value < 0 or (isinstance(value, float) and math.copysign(1.0, value) < 0)
        if negative:
            return ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=-value))
    return ast.Constant(value=value)


def _is_trivial(node: ast.AST) -> bool:
    """Уже каноничная запись: литерал или -литерал."""
    if isinstance(node, ast.Constant):
        return True
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float)))


def try_fold(node: ast.AST) -> Optional[ast.expr]:
    if not isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)) or _is_trivial(node):
        return None
    try:
        value = _eval(node)
    except _NotFoldable:
        return None
    except (ArithmeticError, TypeError, ValueError, RecursionError, MemoryError):
        return None                                                             
    if isinstance(value, complex):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return make_constant(value)


                                                                              
def enclosing_scope(node: ast.AST, parents: Dict[ast.AST, ast.AST]) -> Optional[ast.AST]:
    cur = parents.get(node)
    while cur is not None and not isinstance(cur, SCOPE_NODES):
        cur = parents.get(cur)
    return cur


def sibling_before(node: ast.AST, parents: Dict[ast.AST, ast.AST]) -> Optional[ast.AST]:
    parent = parents.get(node)
    if parent is None:
        return None
    for _, value in ast.iter_fields(parent):
        if isinstance(value, list):
            for i, item in enumerate(value):
                if item is node:
                    return value[i - 1] if i > 0 else None
    return None


def _may_be_empty(parent: ast.AST, field_name: str) -> bool:
    return isinstance(parent, ast.Module) or field_name in ("orelse", "finalbody")


def replace_node(parents: Dict[ast.AST, ast.AST], old: ast.AST, new_nodes: Iterable[ast.AST]) -> None:
    parent = parents.get(old)
    if parent is None:
        raise ValueError("У узла нет родителя")
    new_list: List[ast.AST] = list(new_nodes)
    for field_name, value in ast.iter_fields(parent):
        if isinstance(value, list):
            for i, item in enumerate(value):
                if item is old:
                    if not new_list and len(value) == 1 and not _may_be_empty(parent, field_name):
                        new_list = [ast.copy_location(ast.Pass(), old)]
                    value[i:i + 1] = new_list
                    return
        elif value is old:
            if len(new_list) != 1:
                raise ValueError("Одиночное поле можно заменить только одним узлом")
            setattr(parent, field_name, new_list[0])
            return
    raise ValueError("Узел не найден среди полей родителя")


                                                                          
def names_in(node: ast.AST) -> Set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def bound_names(root: ast.AST) -> Set[str]:
    result: Set[str] = set()
    for n in ast.walk(root):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            result.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result.add(n.name)
        elif isinstance(n, ast.arg):
            result.add(n.arg)
        elif isinstance(n, ast.alias):
            result.add((n.asname or n.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            result.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            result.update(n.names)
    return result


def contains_node(node: ast.AST, types: tuple) -> bool:
    return any(isinstance(n, types) for n in ast.walk(node))
