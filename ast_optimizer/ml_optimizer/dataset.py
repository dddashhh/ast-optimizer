from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from ast_optimizer.analyzer import ASTAnalyzer
from ast_optimizer.benchmark import _equivalent_results, _measure
from ast_optimizer.codegen import CodeGenerationError, CodeGenerator
from ast_optimizer.input_layer.pipeline import InputPipeline
from ast_optimizer.transformer import ASTTransformer, TransformError

from .feature_vector import candidate_feature_vector


@dataclass(frozen=True)
class TrainingExample:
    schema_version: int
    source_hash: str
    filename: str
    optimization_type: str
    candidate_id: str
    features: dict[str, float]
    baseline_median_seconds: float
    candidate_median_seconds: float
    actual_speedup: float
    python_version: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CollectionSummary:
    examples: tuple[TrainingExample, ...]
    candidates_seen: int
    rejected_transformations: int
    rejected_semantics: int
    measurement_failures: int


def collect_training_examples(
    source: str,
    filename: str = "<input>",
    repeats: int = 5,
    warmups: int = 1,
) -> CollectionSummary:
    """Measure each candidate independently and retain only equivalent results.

    The equivalence check compares one top-level ``result`` value; it is a
    useful filter, not a proof of semantic equivalence.
    """
    if not isinstance(source, str):
        raise TypeError("source должен быть строкой")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    pipeline = InputPipeline()
    tree = pipeline.process(source, filename=filename)
    analysis = ASTAnalyzer().analyze(tree)
    transformer = ASTTransformer()
    generator = CodeGenerator()
    examples: list[TrainingExample] = []
    rejected_transformations = rejected_semantics = measurement_failures = 0
    try:
        baseline_durations, _ = _measure(source, repeats, warmups, filename)
    except (OSError, RuntimeError, TimeoutError, ValueError):
        return CollectionSummary((), len(analysis.candidates), 0, 0, len(analysis.candidates))
    baseline_median = statistics.median(baseline_durations)
    if baseline_median <= 0:
        return CollectionSummary((), len(analysis.candidates), 0, 0, len(analysis.candidates))

    count = len(analysis.candidates)
    for candidate in analysis.candidates:
        try:
            transformed = transformer.transform(tree, candidate)
            candidate_source = generator.generate(transformed).text
            if candidate_source == source:
                rejected_transformations += 1
                continue
        except (CodeGenerationError, TransformError, ValueError, RuntimeError, TypeError):
            rejected_transformations += 1
            continue

        if not _equivalent_results(source, candidate_source, filename):
            rejected_semantics += 1
            continue
        try:
            durations, _ = _measure(candidate_source, repeats, warmups, filename)
        except (OSError, RuntimeError, TimeoutError, ValueError):
            measurement_failures += 1
            continue
        candidate_median = statistics.median(durations)
        if candidate_median <= 0 or not math.isfinite(candidate_median):
            measurement_failures += 1
            continue
        features = candidate_feature_vector(candidate, analysis.features, count)
        examples.append(TrainingExample(
            schema_version=1,
            source_hash=source_hash,
            filename=Path(filename).name if filename not in ("", "<input>") else "input.py",
            optimization_type=candidate.type.value,
            candidate_id=candidate.id,
            features=features,
            baseline_median_seconds=baseline_median,
            candidate_median_seconds=candidate_median,
            actual_speedup=baseline_median / candidate_median,
            python_version=platform.python_version(),
        ))
    return CollectionSummary(tuple(examples), count, rejected_transformations,
                             rejected_semantics, measurement_failures)


def append_jsonl(path: str | Path, examples: Iterable[TrainingExample]) -> int:
    """Append examples atomically per record; return number of records written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        for example in examples:
            stream.write(json.dumps(example.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
            count += 1
    return count


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Некорректный JSONL в строке {line_number}: {exc.msg}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("features"), dict):
                raise ValueError(f"Запись {line_number} должна содержать объект features")
            target = record.get("actual_speedup")
            if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target) or target <= 0:
                raise ValueError(f"Некорректный actual_speedup в строке {line_number}")
            for key, value in record["features"].items():
                if not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Некорректный признак в строке {line_number}")
            records.append(record)
    return records
