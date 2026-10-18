from __future__ import annotations

import ast
import sys

from ast_optimizer.input_layer.models import ASTTree, SourceCode

from .exceptions import CodeGenerationError
from .interfaces import ICodeGenerator


class CodeGenerator(ICodeGenerator):

    def __init__(self, verify: bool = True) -> None:
        if sys.version_info < (3, 9):
            raise RuntimeError("CodeGenerator требует Python 3.9+ (ast.unparse)")
        self.verify = verify

    def generate(self, tree: ASTTree) -> SourceCode:
        filename = tree.source.filename
        try:
            text = ast.unparse(tree.root)
        except (RecursionError, ValueError, TypeError, AttributeError) as exc:
            raise CodeGenerationError(f"Не удалось сгенерировать код: {exc}") from exc
        text += "\n"
        if self.verify:
            try:
                ast.parse(text, filename=filename, mode="exec")
            except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
                raise CodeGenerationError(f"Сгенерированный код некорректен: {exc}") from exc
        return SourceCode(text=text, filename=filename)
