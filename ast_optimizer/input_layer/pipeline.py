from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

from .ast_builder import ASTBuilder
from .exceptions import InvalidSourceError
from .interfaces import IASTBuilder, ICodeParser, ISourceLoader, ISourceValidator
from .loader import SourceLoader
from .models import ASTTree, SourceCode
from .parser import CodeParser
from .validator import SourceValidator


class InputPipeline:

    def __init__(self, loader: Optional[ISourceLoader] = None,
                 validator: Optional[ISourceValidator] = None,
                 parser: Optional[ICodeParser] = None,
                 builder: Optional[IASTBuilder] = None) -> None:
        self.loader = loader or SourceLoader()
        self.validator = validator or SourceValidator()
        self.parser = parser or CodeParser()
        self.builder = builder or ASTBuilder()

    def process(self, code: Union[str, bytes, SourceCode], filename: str = "<input>") -> ASTTree:
        source = code if isinstance(code, SourceCode) else self.loader.from_string(code, filename)
        self.validator.validate(source).raise_if_invalid()
        try:
            root = self.parser.parse(source)
            return self.builder.build(source, root)
        except RecursionError as exc:
            raise InvalidSourceError(f"Код слишком глубоко вложен: {exc}")

    def process_file(self, path: Union[str, Path]) -> ASTTree:
        return self.process(self.loader.from_file(path))

    def try_process(self, code: Union[str, bytes, SourceCode], filename: str = "<input>"
                    ) -> Tuple[Optional[ASTTree], Optional[InvalidSourceError]]:
        try:
            return self.process(code, filename), None
        except InvalidSourceError as exc:
            return None, exc