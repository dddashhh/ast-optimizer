from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ast_optimizer.input_layer.models import ASTTree

from .models import (AnalysisContext, AnalysisResult, ASTFeatures,
                     OptimizationCandidate, OptimizationType)


class IFeatureExtractor(ABC):

    @abstractmethod
    def extract(self, tree: ASTTree) -> ASTFeatures: ...


class ICandidateDetector(ABC):

    @property
    @abstractmethod
    def optimization_type(self) -> OptimizationType: ...

    @abstractmethod
    def detect(self, ctx: AnalysisContext) -> List[OptimizationCandidate]: ...


class IASTAnalyzer(ABC):

    @abstractmethod
    def analyze(self, tree: ASTTree) -> AnalysisResult: ...
