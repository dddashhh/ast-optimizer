from __future__ import annotations

from typing import Optional, Sequence

from ast_optimizer.input_layer.models import ASTTree

from .detectors import default_detectors
from .features import FeatureExtractor
from .interfaces import IASTAnalyzer, ICandidateDetector, IFeatureExtractor
from .models import AnalysisContext, AnalysisResult


class ASTAnalyzer(IASTAnalyzer):

    def __init__(self, extractor: Optional[IFeatureExtractor] = None,
                 detectors: Optional[Sequence[ICandidateDetector]] = None) -> None:
        self.extractor = extractor or FeatureExtractor()
        self.detectors = list(detectors) if detectors is not None else default_detectors()

    def analyze(self, tree: ASTTree) -> AnalysisResult:
        ctx = AnalysisContext.build(tree)
        candidates = [c for d in self.detectors for c in d.detect(ctx)]
        candidates.sort(key=lambda c: (c.lineno, c.col_offset, c.node_index))
        return AnalysisResult(features=self.extractor.extract(tree), candidates=candidates)
