from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Sequence

from .dataset import load_jsonl
from .models import LinearCostModel


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            a[pivot][col] += 1e-8
        a[col], a[pivot] = a[pivot], a[col]
        divisor = a[col][col]
        if abs(divisor) < 1e-15:
            raise ValueError("Обучающая матрица вырождена; добавьте больше разнообразных примеров")
        for j in range(col, n + 1):
            a[col][j] /= divisor
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                a[row][j] -= factor * a[col][j]
    return [a[i][n] for i in range(n)]


def _fit(records: Sequence[dict], ridge: float = 1.0) -> tuple[LinearCostModel, dict]:
    if not records:
        raise ValueError("Нет записей для обучения")
    feature_names = sorted({key for row in records for key in row["features"]})
    if not feature_names:
        raise ValueError("В обучающих данных нет признаков")
    x_rows = [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in records]
    y = [float(row["actual_speedup"]) for row in records]
    means = {name: sum(row[i] for row in x_rows) / len(x_rows) for i, name in enumerate(feature_names)}
    scales = {}
    for i, name in enumerate(feature_names):
        variance = sum((row[i] - means[name]) ** 2 for row in x_rows) / len(x_rows)
        scales[name] = math.sqrt(variance) or 1.0
    z_rows = [[(row[i] - means[name]) / scales[name] for i, name in enumerate(feature_names)] for row in x_rows]
    # First column is intercept; regularize weights but not the intercept.
    design = [[1.0, *row] for row in z_rows]
    width = len(feature_names) + 1
    gram = [[sum(row[i] * row[j] for row in design) for j in range(width)] for i in range(width)]
    for i in range(1, width):
        gram[i][i] += ridge
    rhs = [sum(row[i] * target for row, target in zip(design, y)) for i in range(width)]
    coefficients = _solve_linear_system(gram, rhs)
    model = LinearCostModel(
        weights=dict(zip(feature_names, coefficients[1:])), intercept=coefficients[0],
        name="trained-linear-cost-model", feature_means=means, feature_scales=scales,
        min_speedup=0.01,
    )
    predictions = [model.predict(row["features"]) for row in records]
    metrics = _regression_metrics(y, predictions)
    metrics["sample_count"] = len(records)
    metrics["feature_count"] = len(feature_names)
    return model, metrics


def _regression_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float | None]:
    if not actual:
        return {"mae": None, "rmse": None, "r2": None}
    errors = [p - y for y, p in zip(actual, predicted)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    mean = sum(actual) / len(actual)
    total = sum((value - mean) ** 2 for value in actual)
    r2 = 1.0 - sum(error * error for error in errors) / total if total > 1e-12 else None
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_model(dataset_path: str | Path, output_path: str | Path,
                ridge: float = 1.0, seed: int = 0) -> dict:
    if ridge < 0 or not math.isfinite(ridge):
        raise ValueError("ridge должен быть конечным неотрицательным числом")
    records = load_jsonl(dataset_path)
    if len(records) < 3:
        raise ValueError("Для обучения нужно минимум 3 примера; желательно несколько разных программ")

    # Split by source, not individual candidates, to reduce leakage from the
    # same program appearing in both train and validation sets.
    groups: dict[str, list[dict]] = {}
    for row in records:
        groups.setdefault(str(row.get("source_hash") or row.get("filename") or "unknown"), []).append(row)
    validation_metrics = None
    validation_count = 0
    if len(groups) >= 3:
        keys = sorted(groups)
        random.Random(seed).shuffle(keys)
        validation_keys = set(keys[:max(1, round(len(keys) * 0.2))])
        train_rows = [row for key, rows in groups.items() if key not in validation_keys for row in rows]
        validation_rows = [row for key, rows in groups.items() if key in validation_keys for row in rows]
        if train_rows and validation_rows:
            validation_model, _ = _fit(train_rows, ridge)
            actual = [float(row["actual_speedup"]) for row in validation_rows]
            predicted = [validation_model.predict(row["features"]) for row in validation_rows]
            validation_metrics = _regression_metrics(actual, predicted)
            validation_count = len(validation_rows)
    model, train_metrics = _fit(records, ridge)
    training_info = {
        "dataset": str(dataset_path), "sample_count": len(records),
        "source_count": len(groups), "feature_count": train_metrics["feature_count"],
        "ridge": ridge, "seed": seed,
        "mae": train_metrics["mae"], "rmse": train_metrics["rmse"], "r2": train_metrics["r2"],
        "in_sample_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_sample_count": validation_count,
        "validation_note": None if validation_metrics is not None else
            "Недостаточно уникальных исходных программ для групповой holdout-валидации (нужно не менее 3)",
    }
    payload = {
        "schema_version": 1,
        "name": model.name,
        "intercept": model.intercept,
        "weights": model.weights,
        "feature_means": model.feature_means,
        "feature_scales": model.feature_scales,
        "min_speedup": model.min_speedup,
        "training": training_info,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload["training"]
