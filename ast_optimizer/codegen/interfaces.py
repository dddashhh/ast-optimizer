from __future__ import annotations

from abc import ABC, abstractmethod

from ast_optimizer.input_layer.models import ASTTree, SourceCode


class ICodeGenerator(ABC):

    @abstractmethod
    def generate(self, tree: ASTTree) -> SourceCode: ...
