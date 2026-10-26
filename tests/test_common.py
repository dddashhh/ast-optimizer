import ast

from ast_optimizer import InputPipeline
from ast_optimizer.common.ast_utils import replace_node, try_fold
from ast_optimizer.common.patterns import (
    match_dead_branch,
    match_listcomp_call,
    match_negated_membership,
)


def test_try_fold_evaluates_safe_constant_expression():
    node = ast.parse("2 * (3 + 4)", mode="eval").body

    folded = try_fold(node)

    assert isinstance(folded, ast.Constant)
    assert folded.value == 14


def test_try_fold_leaves_unsafe_or_nonconstant_expression():
    assert try_fold(ast.parse("1 / 0", mode="eval").body) is None
    assert try_fold(ast.parse("x + 1", mode="eval").body) is None


def test_match_dead_branch_checks_constant_condition():
    assert match_dead_branch(ast.parse("if True:\n    x = 1").body[0])
    assert not match_dead_branch(ast.parse("if x:\n    x = 1").body[0])


def test_match_listcomp_call_respects_shadowed_names():
    call = ast.parse("sum([x for x in values])", mode="eval").body

    assert match_listcomp_call(call, set())
    assert not match_listcomp_call(call, {"sum"})


def test_match_negated_membership():
    assert match_negated_membership(ast.parse("not (x in values)", mode="eval").body)
    assert not match_negated_membership(ast.parse("not x", mode="eval").body)


def test_replace_node_inserts_pass_when_block_would_be_empty():
    tree = InputPipeline().process("if True:\n    x = 1")
    old = tree.root.body[0].body[0]
    replace_node(tree.parents, old, [])

    assert isinstance(tree.root.body[0].body[0], ast.Pass)


def test_match_dead_branch_rejects_assignment_that_changes_local_scope():
    source = """value = 'global'
def f():
    if False:
        value = 1
    return value
"""
    branch = ast.parse(source).body[1].body[0]

    assert not match_dead_branch(branch)


def test_match_listcomp_call_rejects_any_all_short_circuit_cases():
    any_call = ast.parse("any([obj.value for obj in values])", mode="eval").body
    all_call = ast.parse("all([obj[index] for index in values])", mode="eval").body

    assert not match_listcomp_call(any_call, set())
    assert not match_listcomp_call(all_call, set())
