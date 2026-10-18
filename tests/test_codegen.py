import ast

import pytest

from ast_optimizer import InputPipeline, SourceCode
from ast_optimizer.codegen import CodeGenerationError, CodeGenerator


def test_generate_returns_valid_code_and_keeps_filename():
    tree = InputPipeline().process("x=1+2", filename="example.py")

    result = CodeGenerator().generate(tree)

    assert result.filename == "example.py"
    assert result.text.endswith("\n")
    assert ast.dump(ast.parse(result.text)) == ast.dump(ast.parse("x = 1 + 2"))


def test_generate_does_not_preserve_comments():
    tree = InputPipeline().process("x = 1  # note")

    assert "#" not in CodeGenerator().generate(tree).text


def test_generate_wraps_unparse_errors():
    tree = InputPipeline().process("x = 1")
    tree.root.body[0].value = ast.BinOp(left=ast.Constant(value=1), op=ast.Add(), right=None)

    with pytest.raises(CodeGenerationError):
        CodeGenerator().generate(tree)


def test_generate_without_verification_returns_source_code():
    tree = InputPipeline().process("x = 1")

    result = CodeGenerator(verify=False).generate(tree)

    assert isinstance(result, SourceCode)
    assert result.text == "x = 1\n"
