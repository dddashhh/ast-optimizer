from __future__ import annotations

import ast
import contextlib
import copy
import keyword
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

from ast_optimizer import SourceCode

FILE_SETTINGS = settings(max_examples=30)
FIXTURE_OK = settings(suppress_health_check=[HealthCheck.function_scoped_fixture])

BOM = b"\xef\xbb\xbf"

SHARED_NODES = (ast.expr_context, ast.operator, ast.boolop, ast.unaryop, ast.cmpop)

TEXT = st.text(alphabet=st.characters(blacklist_categories=["Cs"]))
FILENAMES = st.text(
    alphabet=st.characters(blacklist_categories=["Cs"]), min_size=1, max_size=20
)

names = st.from_regex(r"[a-z_][a-z0-9_]{0,6}", fullmatch=True).filter(
    lambda s: not keyword.iskeyword(s)
)

comment_lines = st.text(
    alphabet=st.characters(blacklist_categories=["Cs", "Cc", "Zl", "Zp"]), max_size=20
).map(lambda s: "# " + s)

blank_texts = st.text(
    alphabet=st.sampled_from(" \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f\x85\xa0\u2003\u2028\u2029\u3000"),
    max_size=50,
)

_leaf = st.one_of(
    st.integers(0, 9).map(lambda v: ast.Constant(value=v)),
    names.map(lambda n: ast.Name(id=n, ctx=ast.Load())),
)


def _extend_expr(inner):
    return st.one_of(
        inner.map(lambda e: ast.UnaryOp(op=ast.USub(), operand=e)),
        inner.map(lambda e: ast.UnaryOp(op=ast.Not(), operand=e)),
        st.builds(
            lambda l, op, r: ast.BinOp(left=l, op=op(), right=r),
            inner,
            st.sampled_from([ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod]),
            inner,
        ),
        st.builds(
            lambda l, op, r: ast.BoolOp(op=op(), values=[l, r]),
            inner,
            st.sampled_from([ast.And, ast.Or]),
            inner,
        ),
        st.builds(
            lambda l, op, r: ast.Compare(left=l, ops=[op()], comparators=[r]),
            inner,
            st.sampled_from([ast.Lt, ast.Eq, ast.GtE, ast.NotEq]),
            inner,
        ),
    )


exprs = st.recursive(_leaf, _extend_expr, max_leaves=12)

_simple_stmt = st.one_of(
    st.builds(
        lambda n, e: ast.Assign(targets=[ast.Name(id=n, ctx=ast.Store())], value=e),
        names,
        exprs,
    ),
    exprs.map(lambda e: ast.Expr(value=e)),
    st.builds(ast.Pass),
)


def _extend_stmt(inner):
    body = st.lists(inner, min_size=1, max_size=2)
    return st.one_of(
        st.builds(
            lambda t, b, o: ast.If(test=t, body=b, orelse=o),
            exprs,
            body,
            st.lists(inner, max_size=2),
        ),
        st.builds(lambda t, b: ast.While(test=t, body=b, orelse=[]), exprs, body),
    )


statements = st.recursive(_simple_stmt, _extend_stmt, max_leaves=8)

modules = st.lists(statements, min_size=1, max_size=4).map(
    lambda body: ast.Module(body=body, type_ignores=[])
)


def unparse(module: ast.Module) -> str:
    return ast.unparse(ast.fix_missing_locations(copy.deepcopy(module)))

clean_texts = modules.map(unparse)


@st.composite
def _valid_texts(draw):
    lines = draw(st.lists(comment_lines, max_size=2)) + draw(clean_texts).split("\n")
    eol = draw(st.sampled_from(["\n", "\r\n"]))
    return eol.join(lines) + (eol if draw(st.booleans()) else "")


valid_texts = _valid_texts()

_BROKEN_LINES = ["def f(:", "x = )", "class :", "if x", "x = 1 +"]


@st.composite
def _broken_texts(draw):
    prefix = draw(clean_texts)
    line = draw(st.sampled_from(_BROKEN_LINES))
    return prefix + "\n" + line, prefix.count("\n") + 2


broken_texts = _broken_texts()


@st.composite
def _null_texts(draw):
    text = draw(valid_texts)
    i = draw(st.integers(0, len(text)))
    return text[:i] + "\x00" + text[i:]


null_texts = _null_texts()

any_texts = st.one_of(
    valid_texts,
    broken_texts.map(lambda case: case[0]),
    null_texts,
    blank_texts,
    TEXT,
)


invalid_utf8 = st.builds(
    lambda head, tail: head + b"\xff" + tail,
    st.binary(max_size=20),
    st.binary(max_size=20),
)

wrong_types = st.one_of(
    st.none(),
    st.integers(),
    st.floats(),
    st.booleans(),
    st.lists(st.text(), max_size=3),
    st.tuples(st.text()),
)

@st.composite
def _ast_cases(draw):
    if draw(st.booleans()):
        module = draw(modules)
        return SourceCode(unparse(module)), module
    text = draw(valid_texts)
    return SourceCode(text), ast.parse(text)


ast_cases = _ast_cases()


@contextlib.contextmanager
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)