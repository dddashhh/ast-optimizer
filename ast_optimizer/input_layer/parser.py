from __future__ import annotations

import ast

from .exceptions import InvalidSourceError, PythonSyntaxError
from .interfaces import ICodeParser
from .models import SourceCode


class CodeParser(ICodeParser):

    def parse(self, source: SourceCode) -> ast.Module:
        try:
            return ast.parse(source.text, filename=source.filename, mode="exec")
        except SyntaxError as exc:
            raise PythonSyntaxError(exc.msg, exc.lineno, exc.offset, exc.text)
        except (ValueError, RecursionError, MemoryError) as exc:
            raise InvalidSourceError(f"Код не может быть разобран: {exc}")