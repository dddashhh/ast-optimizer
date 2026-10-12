from __future__ import annotations

import ast

from .exceptions import (EmptySourceError, InvalidSourceError,
                         PythonSyntaxError, SourceTooLargeError)
from .interfaces import ISourceValidator
from .models import SourceCode, ValidationResult


class SourceValidator(ISourceValidator):

    DEFAULT_MAX_SIZE = 1000000

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE) -> None:
        self.max_size = max_size

    def validate(self, source: SourceCode) -> ValidationResult:
        try:
            self._check(source)
        except InvalidSourceError as exc:
            return ValidationResult(is_valid=False, error=exc)
        return ValidationResult(is_valid=True)

    def _check(self, source: SourceCode) -> None:
        text = source.text
        if not text.strip():
            raise EmptySourceError("Входной код пуст")
        if len(text) > self.max_size:
            raise SourceTooLargeError(
                f"Размер кода ({len(text)}) превышает лимит ({self.max_size})")
        if "\x00" in text:
            raise InvalidSourceError("Код содержит нулевые байты")
        try:
            compile(text, source.filename, "exec", flags=ast.PyCF_ONLY_AST)
        except SyntaxError as exc:
            raise PythonSyntaxError(exc.msg, exc.lineno, exc.offset, exc.text)
        except (ValueError, RecursionError, MemoryError) as exc:
            raise InvalidSourceError(f"Код не может быть разобран: {exc}")