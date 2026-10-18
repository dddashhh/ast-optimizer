from __future__ import annotations

import ast

from ast_optimizer import (
    ICodeParser,
    ISourceValidator,
    InvalidSourceError,
    ValidationResult,
)


class SpyParser(ICodeParser):

    def __init__(self):
        self.calls = 0

    def parse(self, source):
        self.calls += 1
        return ast.parse(source.text)


class RejectAll(ISourceValidator):

    def __init__(self, message):
        self.message = message

    def validate(self, source):
        return ValidationResult(False, InvalidSourceError(self.message))
