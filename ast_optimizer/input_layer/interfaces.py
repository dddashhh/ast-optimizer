from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from .models import ASTTree, SourceCode, ValidationResult


class ISourceLoader(ABC):
    """Приём кода и приведение к SourceCode."""

    @abstractmethod
    def from_string(self, text: Union[str, bytes], filename: str = "<input>") -> SourceCode: ...

    @abstractmethod
    def from_file(self, path: Union[str, Path]) -> SourceCode: ...


class ISourceValidator(ABC):
    """Проверка корректности кода"""

    @abstractmethod
    def validate(self, source: SourceCode) -> ValidationResult: ...


class ICodeParser(ABC):
    """Разбор кода в черновик синтаксического дерева."""

    @abstractmethod
    def parse(self, source: SourceCode) -> ast.Module: ...


class IASTBuilder(ABC):
    """Построение AST"""

    @abstractmethod
    def build(self, source: SourceCode, root: ast.Module) -> ASTTree: ...

    @abstractmethod
    def clone(self, tree: ASTTree) -> ASTTree:
        """Независимая копия"""