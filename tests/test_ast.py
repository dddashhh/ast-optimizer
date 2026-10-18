from __future__ import annotations

import ast
import builtins
from pathlib import Path

import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from ast_optimizer import (
    ASTTree,
    CodeParser,
    EmptySourceError,
    InputPipeline,
    InvalidSourceError,
    PythonSyntaxError,
    SourceCode,
    SourceLoader,
    SourceTooLargeError,
    SourceValidator,
    ValidationResult,
)
from doubles import RejectAll, SpyParser
from strategies import (
    BOM,
    FILENAMES,
    FILE_SETTINGS,
    FIXTURE_OK,
    SHARED_NODES,
    TEXT,
    any_texts,
    ast_cases,
    blank_texts,
    broken_texts,
    clean_texts,
    comment_lines,
    invalid_utf8,
    modules,
    names,
    null_texts,
    tmpdir,
    unparse,
    valid_texts,
    wrong_types,
)


PY_FILE = "prog.py"


@given(TEXT, FILENAMES)
@example("", PY_FILE)
@example("x = 1", PY_FILE)
def test_from_string_keeps_text_and_filename(text, filename):
    loader = SourceLoader()
    assert loader.from_string(text, filename=filename) == SourceCode(text=text, filename=filename)
    assert loader.from_string(text).filename == "<input>"


@given(TEXT, st.booleans())
@example("\ufeffx = 1", True)
def test_from_bytes_decodes_utf8_and_strips_one_bom(text, with_bom):
    if not with_bom:
        assume(not text.startswith("\ufeff"))
    data = (BOM if with_bom else b"") + text.encode("utf-8")
    assert SourceLoader().from_string(data).text == text


@given(st.binary() | invalid_utf8)
def test_from_bytes_agrees_with_utf8_sig_decoding(data):
    loader = SourceLoader()
    try:
        expected = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        with pytest.raises(InvalidSourceError):
            loader.from_string(data)
    else:
        assert loader.from_string(data).text == expected


@given(wrong_types)
def test_from_string_rejects_wrong_type(bad):
    with pytest.raises(InvalidSourceError):
        SourceLoader().from_string(bad)


@given(FILENAMES, FILENAMES)
def test_from_string_rejects_null_in_filename(head, tail):
    with pytest.raises(InvalidSourceError):
        SourceLoader().from_string("x = 1", filename=head + "\x00" + tail)


@given(wrong_types)
def test_from_string_rejects_non_str_filename(bad):
    with pytest.raises(InvalidSourceError):
        SourceLoader().from_string("x = 1", filename=bad)


@given(wrong_types)
def test_from_file_rejects_wrong_path_type(bad):
    with pytest.raises(InvalidSourceError):
        SourceLoader().from_file(bad)


def test_from_file_os_error_becomes_invalid_source_error(tmp_path, monkeypatch):
    path = tmp_path / "prog.py"
    path.write_text("x = 1")

    def denied(self):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_bytes", denied)
    with pytest.raises(InvalidSourceError):
        SourceLoader().from_file(path)


@FILE_SETTINGS
@given(
    TEXT.filter(lambda t: "\r" not in t and not t.startswith("\ufeff")),
    st.sampled_from([str, Path]),
)
def test_from_file_reads_text_and_keeps_path(text, path_type):
    with tmpdir() as d:
        path = d / PY_FILE
        path.write_bytes(text.encode("utf-8"))
        src = SourceLoader().from_file(path_type(path))
    assert src.text == text
    assert src.filename == str(path)


@FILE_SETTINGS
@given(names)
def test_from_file_missing(stem):
    with tmpdir() as d:
        with pytest.raises(InvalidSourceError):
            SourceLoader().from_file(d / f"{stem}.py")


@FILE_SETTINGS
@given(names, st.sampled_from(["", ".txt", ".md", ".pyc", ".json"]))
def test_from_file_wrong_extension(stem, ext):
    with tmpdir() as d:
        path = d / f"{stem}{ext}"
        path.write_text("x = 1")
        with pytest.raises(InvalidSourceError):
            SourceLoader().from_file(path)


@FILE_SETTINGS
@given(names, st.sampled_from(["", ".py"]))
def test_from_file_directory(stem, ext):
    with tmpdir() as d:
        path = d / f"{stem}{ext}"
        path.mkdir()
        with pytest.raises(InvalidSourceError):
            SourceLoader().from_file(path)


@FILE_SETTINGS
@given(invalid_utf8)
def test_from_file_non_utf8(data):
    with tmpdir() as d:
        path = d / PY_FILE
        path.write_bytes(data)
        with pytest.raises(InvalidSourceError):
            SourceLoader().from_file(path)


@given(valid_texts)
def test_valid_code_is_accepted(text):
    res = SourceValidator().validate(SourceCode(text))
    assert res.is_valid and res.error is None


@given(st.lists(comment_lines, min_size=1, max_size=5))
def test_comment_only_is_valid(comments):
    assert SourceValidator().validate(SourceCode("\n".join(comments))).is_valid


@given(blank_texts)
def test_blank_source_is_empty_error(text):
    res = SourceValidator().validate(SourceCode(text))
    assert not res.is_valid
    assert isinstance(res.error, EmptySourceError)


@given(TEXT)
def test_non_blank_source_is_never_empty_error(text):
    assume(text.strip())
    res = SourceValidator().validate(SourceCode(text))
    assert not isinstance(res.error, EmptySourceError)


@given(st.integers(1, 200), st.integers(0, 200))
def test_size_limit_boundary(limit, extra):
    v = SourceValidator(max_size=limit)
    assert v.validate(SourceCode("x" * limit)).is_valid
    assert isinstance(v.validate(SourceCode("x" * (limit + extra + 1))).error, SourceTooLargeError)


@given(null_texts)
def test_null_byte_is_invalid_but_not_syntax_error(text):
    res = SourceValidator().validate(SourceCode(text))
    assert isinstance(res.error, InvalidSourceError)
    assert not isinstance(res.error, PythonSyntaxError)


@given(broken_texts)
def test_syntax_error_has_location(case):
    text, lineno = case
    res = SourceValidator().validate(SourceCode(text))
    assert isinstance(res.error, PythonSyntaxError)
    assert res.error.lineno == lineno
    assert f"строка {lineno}" in str(res.error)


@given(clean_texts)
def test_indentation_error_becomes_syntax_error(text):
    res = SourceValidator().validate(SourceCode("    " + text))
    assert isinstance(res.error, PythonSyntaxError)


@settings(max_examples=20)
@given(st.sampled_from(["()", "[]", "{}"]), st.integers(300, 5000))
def test_deep_nesting_is_rejected_without_crash(pair, depth):
    code = "x = " + pair[0] * depth + "1" + pair[1] * depth
    res = SourceValidator().validate(SourceCode(code))
    assert not res.is_valid
    assert isinstance(res.error, InvalidSourceError)


@given(names)
def test_validation_does_not_execute_code(name):
    attr = f"_executed_{name}"
    code = f"import builtins\nbuiltins.{attr} = True\nraise RuntimeError('executed')"
    assert SourceValidator().validate(SourceCode(code)).is_valid
    assert not hasattr(builtins, attr)


@given(any_texts)
def test_validator_result_is_consistent(text):
    res = SourceValidator().validate(SourceCode(text))
    assert isinstance(res, ValidationResult)
    assert res.is_valid == (res.error is None)
    if res.is_valid:
        res.raise_if_invalid()
    else:
        assert isinstance(res.error, InvalidSourceError)
        with pytest.raises(type(res.error)):
            res.raise_if_invalid()


def test_raise_if_invalid_without_error_still_raises():
    with pytest.raises(InvalidSourceError):
        ValidationResult(is_valid=False).raise_if_invalid()


@given(
    st.text(max_size=20),
    st.one_of(st.none(), st.integers(1, 10000)),
    st.one_of(st.none(), st.integers(1, 500)),
)
def test_python_syntax_error_str(message, lineno, offset):
    text = str(PythonSyntaxError(message, lineno, offset))
    if lineno is None:
        assert text == message
    elif offset is None:
        assert text == f"{message} (строка {lineno})"
    else:
        assert text == f"{message} (строка {lineno}, столбец {offset})"


@given(any_texts)
def test_validator_and_parser_agree(text):
    src = SourceCode(text)
    res = SourceValidator().validate(src)
    if res.is_valid:
        assert isinstance(CodeParser().parse(src), ast.Module)
    elif isinstance(res.error, PythonSyntaxError):
        with pytest.raises(PythonSyntaxError):
            CodeParser().parse(src)


@given(modules)
def test_parse_reproduces_generated_tree(module):
    parsed = CodeParser().parse(SourceCode(unparse(module)))
    assert isinstance(parsed, ast.Module)
    assert ast.dump(parsed) == ast.dump(module)


@given(valid_texts)
def test_parse_matches_stdlib_on_noisy_source(text):
    assert ast.dump(CodeParser().parse(SourceCode(text))) == ast.dump(ast.parse(text))


@given(broken_texts)
def test_parse_error_is_project_error_with_lineno(case):
    text, lineno = case
    with pytest.raises(PythonSyntaxError) as exc:
        CodeParser().parse(SourceCode(text))
    assert exc.value.lineno == lineno


@given(null_texts)
def test_parse_null_bytes_is_project_error(text):
    with pytest.raises(InvalidSourceError):
        CodeParser().parse(SourceCode(text))


@FIXTURE_OK
@given(case=ast_cases)
def test_build_counts_nodes_and_keeps_source(builder, case):
    src, root = case
    tree = builder.build(src, root)
    assert isinstance(tree, ASTTree)
    assert tree.source is src
    assert isinstance(tree.root, ast.Module)
    assert tree.node_count == sum(1 for _ in ast.walk(tree.root))


@FIXTURE_OK
@given(case=ast_cases)
def test_build_parents_map(builder, case):
    src, root = case
    tree = builder.build(src, root)
    regular = {n for n in ast.walk(tree.root) if not isinstance(n, SHARED_NODES)}
    assert set(tree.parents) == regular - {tree.root}
    for child, parent in tree.parents.items():
        assert child in list(ast.iter_child_nodes(parent))


@FIXTURE_OK
@given(case=ast_cases)
def test_build_fills_locations(builder, case):
    src, root = case
    tree = builder.build(src, root)
    assert all(
        hasattr(n, "lineno")
        for n in ast.walk(tree.root)
        if isinstance(n, (ast.stmt, ast.expr))
    )


@FIXTURE_OK
@given(case=ast_cases)
def test_clone_is_independent_copy(builder, case):
    src, root = case
    tree = builder.build(src, root)
    clone = builder.clone(tree)

    assert clone.root is not tree.root
    assert ast.dump(clone.root) == ast.dump(tree.root)
    assert clone.node_count == tree.node_count
    assert ast.dump(builder.clone(clone).root) == ast.dump(tree.root)

    clone_nodes = set(ast.walk(clone.root))
    original_nodes = set(ast.walk(tree.root))
    for child, parent in clone.parents.items():
        assert child in clone_nodes and parent in clone_nodes
        assert child not in original_nodes and parent not in original_nodes


@FIXTURE_OK
@given(case=ast_cases)
def test_clone_mutation_does_not_affect_original(builder, case):
    src, root = case
    tree = builder.build(src, root)
    before = ast.dump(tree.root)
    builder.clone(tree).root.body.clear()
    assert ast.dump(tree.root) == before


@given(valid_texts, FILENAMES, st.sampled_from(["str", "bytes", "source"]))
def test_process_valid_input(text, filename, kind):
    pipeline = InputPipeline()
    if kind == "str":
        tree = pipeline.process(text, filename=filename)
    elif kind == "bytes":
        tree = pipeline.process(text.encode("utf-8"), filename=filename)
    else:
        tree = pipeline.process(SourceCode(text, filename=filename))
    assert isinstance(tree, ASTTree)
    assert tree.node_count > 0
    assert tree.source.filename == filename
    assert ast.dump(tree.root) == ast.dump(ast.parse(text))


@FILE_SETTINGS
@given(valid_texts)
def test_process_file_matches_process(text):
    with tmpdir() as d:
        path = d / PY_FILE
        path.write_bytes(text.encode("utf-8"))
        tree = InputPipeline().process_file(path)
    assert tree.source.filename == str(path)
    assert ast.dump(tree.root) == ast.dump(InputPipeline().process(text).root)


_invalid_inputs = st.one_of(
    blank_texts.map(lambda t: (t, EmptySourceError)),
    broken_texts.map(lambda case: (case[0], PythonSyntaxError)),
    null_texts.map(lambda t: (t, InvalidSourceError)),
    invalid_utf8.map(lambda b: (b, InvalidSourceError)),
    wrong_types.map(lambda v: (v, InvalidSourceError)),
)


@given(_invalid_inputs)
def test_process_invalid_input_raises_matching_error(case):
    data, error_type = case
    with pytest.raises(error_type):
        InputPipeline().process(data)


@given(st.one_of(any_texts, st.binary(), invalid_utf8))
def test_try_process_is_wrapper_over_process(data):
    pipeline = InputPipeline()
    try:
        tree = pipeline.process(data)
    except InvalidSourceError as exc:
        tree2, err = pipeline.try_process(data)
        assert tree2 is None
        assert type(err) is type(exc)
    else:
        tree2, err = pipeline.try_process(data)
        assert err is None
        assert tree2 is not None
        assert ast.dump(tree2.root) == ast.dump(tree.root)


@given(any_texts)
def test_parser_is_called_if_validation_passes(text):
    is_valid = SourceValidator().validate(SourceCode(text)).is_valid
    spy = SpyParser()
    try:
        InputPipeline(parser=spy).process(text)
    except InvalidSourceError:
        pass
    assert spy.calls == (1 if is_valid else 0)


@given(valid_texts, st.text(min_size=1, max_size=30))
def test_validator_is_replaceable(text, message):
    with pytest.raises(InvalidSourceError) as exc:
        InputPipeline(validator=RejectAll(message)).process(text)
    assert message in str(exc.value)


@settings(max_examples=20)
@given(st.integers(1, 1500))
def test_deeply_nested_unary_never_leaks_recursion_error(depth):
    tree, err = InputPipeline().try_process("x = " + "-" * depth + "1")
    assert (tree is None) != (err is None)
