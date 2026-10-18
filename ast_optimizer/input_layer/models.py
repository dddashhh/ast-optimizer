from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, Optional

from .exceptions import InvalidSourceError


@dataclass(frozen=True)
class SourceCode:
    text: str
    filename: str = "<input>"


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    error: Optional[InvalidSourceError] = None

    def raise_if_invalid(self) -> None:
        if self.is_valid:
            return
        if self.error is not None:
            raise self.error
        raise InvalidSourceError("Код не прошёл валидацию")


@dataclass
class ASTTree:
    source: SourceCode
    root: ast.Module = field(repr=False)
    parents: Dict[ast.AST, ast.AST] = field(default_factory=dict, repr=False)
    node_count: int = 0
