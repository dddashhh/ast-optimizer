from __future__ import annotations

import ast
import csv
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1

# Keys used by ACEOB-derived exports and common dataset wrappers.
_SLOW_KEYS = {
    "inefficientcode", "inefficient", "ic", "slowcode", "unoptimizedcode",
    "originalcode", "inputcode", "badcode", "codeinefficient", "sourcecode",
    "iccode", "icprogram", "icsolution", "inefficientprogram",
}
_FAST_KEYS = {
    "efficientcode", "efficient", "ec", "fastcode", "optimizedcode",
    "optimisedcode", "goodcode", "codeefficient", "targetcode", "outputcode",
    "eccode", "ecprogram", "ecsolution", "efficientprogram", "alternativeec", "alternativeecs",
}
_SLOW_LIST_KEYS = {"inefficientcodes", "slowcodes", "unoptimizedcodes", "ics"}
_FAST_LIST_KEYS = {"efficientcodes", "fastcodes", "optimizedcodes", "optimisedcodes", "ecs", "alternativeefficientcodes", "alternatives"}


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _code(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, Mapping):
        for key in ("code", "source", "text", "content", "program"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    return None


def _codes(value: Any) -> list[str]:
    one = _code(value)
    if one is not None:
        return [one]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_codes(item))
        return result
    return []


@dataclass(frozen=True)
class CodePair:
    inefficient_code: str
    efficient_code: str
    source_id: str = ""


def _record_pairs(record: Any, source_id: str = "") -> list[CodePair]:
    if isinstance(record, list):
        return [pair for i, item in enumerate(record) for pair in _record_pairs(item, f"{source_id}:{i}")]
    if not isinstance(record, Mapping):
        return []
    slow: list[str] = []
    fast: list[str] = []
    for raw_key, value in record.items():
        key = _key(raw_key)
        if key in _SLOW_KEYS:
            slow.extend(_codes(value))
        elif key in _FAST_KEYS:
            fast.extend(_codes(value))
        elif key in _SLOW_LIST_KEYS:
            slow.extend(_codes(value))
        elif key in _FAST_LIST_KEYS:
            fast.extend(_codes(value))
    # ACEOB exports may group multiple inefficient and efficient solutions
    # for one problem. Pair each side only within the same record.
    pairs = [CodePair(s, f, source_id) for s in slow for f in fast if s.strip() != f.strip()]
    if pairs:
        return pairs
    # Some wrappers nest the actual record under a metadata key.
    nested: list[CodePair] = []
    for key, value in record.items():
        if isinstance(value, (Mapping, list)):
            nested.extend(_record_pairs(value, f"{source_id}:{key}"))
    return nested


def _read_records(path: Path) -> Iterable[Any]:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [
                n for n in archive.namelist()
                if Path(n).suffix.lower() in {".json", ".jsonl", ".ndjson", ".csv"}
                and not any(marker in Path(n).stem.lower() for marker in ("test", "valid", "dev"))
            ]
            if not members:
                raise ValueError(f"В архиве нет JSON/JSONL/CSV: {path}")
            for member in members:
                raw = archive.read(member).decode("utf-8-sig")
                yield from _parse_text(raw, Path(member).suffix.lower(), f"{path.name}:{member}")
        return
    raw = path.read_text(encoding="utf-8-sig")
    yield from _parse_text(raw, suffix, str(path))


def _parse_text(raw: str, suffix: str, source_id: str) -> Iterable[Any]:
    if suffix in {".jsonl", ".ndjson"}:
        for i, line in enumerate(raw.splitlines()):
            if line.strip():
                yield json.loads(line), f"{source_id}:{i + 1}"
    elif suffix == ".csv":
        for i, row in enumerate(csv.DictReader(raw.splitlines())):
            yield row, f"{source_id}:{i + 2}"
    else:
        payload = json.loads(raw)
        if isinstance(payload, Mapping):
            # Prefer explicit dataset containers, otherwise process as a record.
            for key in ("data", "records", "pairs", "examples", "train", "test"):
                if key in payload and isinstance(payload[key], list):
                    for i, item in enumerate(payload[key]):
                        yield item, f"{source_id}:{key}:{i}"
                    return
        if isinstance(payload, list):
            for i, item in enumerate(payload):
                yield item, f"{source_id}:{i}"
        else:
            yield payload, source_id


def load_aceob_pairs(path: str | Path, limit: int | None = None) -> list[CodePair]:
    """Load code pairs from JSON, JSONL, CSV, or a ZIP containing these files."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Файл датасета не найден: {source}")
    pairs: list[CodePair] = []
    for record, record_id in _read_records(source):
        pairs.extend(_record_pairs(record, record_id))
        if limit is not None and len(pairs) >= limit:
            return pairs[:limit]
    if not pairs:
        raise ValueError(
            "Не найдены пары кода. Ожидаются поля вроде inefficient_code/efficient_code, "
            "IC/EC или original_code/optimized_code. Проверьте формат выгрузки ACEOB."
        )
    return pairs



def _read_text_member(name: str, raw: bytes) -> tuple[str, str, float | None] | None:
    """Return (role, code, runtime_ms) for ACEOB's labeled .txt files."""
    base = Path(name).name
    if base.startswith("._") or not base.lower().endswith(".txt"):
        return None
    normalized = _key(base)
    role = "efficient" if normalized.startswith("efficient") else (
        "inefficient" if normalized.startswith("inefficient") else "alternate"
    )
    if role == "alternate" and "alternateefficientcodes" not in _key(name):
        return None
    try:
        code = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        code = raw.decode("latin-1")
    match = re.search(r"(?:execution\s*time|time)\s*([0-9]+(?:\.[0-9]+)?)\s*ms", base, re.I)
    runtime = float(match.group(1)) if match else None
    return role, code, runtime


def _pairs_from_aceob_tree(entries: Iterable[tuple[str, bytes]], source: str) -> list[CodePair]:
    """Parse native ACEOB folder layout (pair folders + Alternate efficient codes)."""
    grouped: dict[str, dict[str, list[tuple[str, float | None]]]] = {}
    alternates: dict[str, list[tuple[str, float | None]]] = {}
    metadata: dict[str, str] = {}
    for name, raw in entries:
        clean = name.replace("\\", "/")
        parts = [part for part in clean.split("/") if part and part != "__MACOSX"]
        if not parts or any(part.startswith("._") for part in parts):
            continue
        parsed = _read_text_member(clean, raw)
        if parsed:
            role, code, runtime = parsed
            pair_folder = next((part for part in parts if part.startswith("Efficient-inefficient_code_pairs_")), None)
            if pair_folder:
                bucket = grouped.setdefault("/".join(parts[:-1]), {"efficient": [], "inefficient": []})
                if role in ("efficient", "inefficient"):
                    bucket[role].append((code, runtime))
            elif "Alternate efficient codes" in parts:
                root = parts[:parts.index("Alternate efficient codes")]
                alternates.setdefault("/".join(root), []).append((code, runtime))
        elif Path(clean).name.lower() == "metadata.txt":
            metadata["/".join(parts[:-1])] = raw.decode("utf-8-sig", errors="replace")

    result: list[CodePair] = []
    for folder, sides in grouped.items():
        efficient = sides["efficient"]
        inefficient = sides["inefficient"]
        if not efficient or not inefficient:
            continue
        for slow, _slow_time in inefficient:
            for fast, _fast_time in efficient:
                if slow.strip() != fast.strip():
                    result.append(CodePair(slow, fast, f"{source}:{folder}"))
            root = folder.split("/Efficient-inefficient_code_pairs_")[0]
            for fast, _fast_time in alternates.get(root, []):
                if slow.strip() != fast.strip():
                    result.append(CodePair(slow, fast, f"{source}:{root}:alternate"))
    return result


def import_aceob_folders(
    inputs: Iterable[str | Path], output_path: str | Path, *, split: str,
) -> dict[str, int | str]:
    """Import native ACEOB train/test directories or ZIPs into pairwise JSONL.

    Each input may be a directory or ZIP. The importer preserves split labels
    in the output and never merges train and test implicitly.
    """
    if split not in {"train", "test", "validation"}:
        raise ValueError("split должен быть train, validation или test")
    all_pairs: list[CodePair] = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_dir():
            entries = ((str(file.relative_to(path)), file.read_bytes()) for file in path.rglob("*.txt"))
            all_pairs.extend(_pairs_from_aceob_tree(entries, path.name))
        elif path.is_file() and path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                entries = ((name, archive.read(name)) for name in archive.namelist() if not name.endswith("/"))
                all_pairs.extend(_pairs_from_aceob_tree(entries, path.name))
        else:
            raise ValueError(f"Ожидалась папка или ZIP-архив ACEOB: {path}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as stream:
        for pair in all_pairs:
            record = {
                "inefficient_code": pair.inefficient_code,
                "efficient_code": pair.efficient_code,
                "source_id": pair.source_id,
                "split": split,
                "schema_version": SCHEMA_VERSION,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return {"split": split, "pairs_written": written, "output": str(output)}

def code_feature_vector(source: str) -> dict[str, float]:
    """Extract inexpensive, deterministic AST features from a Python snippet."""
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    counts: dict[str, float] = {
        "source_chars": float(len(source)),
        "source_lines": float(max(1, len(source.splitlines()))),
        "ast_nodes": float(len(nodes)),
        "max_ast_depth": 0.0,
        "loop_nesting": 0.0,
        "nested_loop_pairs": 0.0,
        "membership_tests": 0.0,
        "comprehension_filters": 0.0,
    }
    for node in nodes:
        name = type(node).__name__
        counts[f"node.{name}"] = counts.get(f"node.{name}", 0.0) + 1.0
        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else (node.func.attr if isinstance(node.func, ast.Attribute) else "<other>")
            counts[f"call.{func_name}"] = counts.get(f"call.{func_name}", 0.0) + 1.0
        if isinstance(node, ast.Compare):
            counts["membership_tests"] += sum(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            counts["comprehension_filters"] += sum(len(gen.ifs) for gen in node.generators)
    # Depth and nested loops are computed using an explicit stack.
    stack: list[tuple[ast.AST, int, int]] = [(tree, 1, 0)]
    while stack:
        node, depth, loop_depth = stack.pop()
        counts["max_ast_depth"] = max(counts["max_ast_depth"], float(depth))
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            loop_depth += 1
            counts["loop_nesting"] = max(counts["loop_nesting"], float(loop_depth))
            counts["nested_loop_pairs"] += float(max(0, loop_depth - 1))
        for child in ast.iter_child_nodes(node):
            stack.append((child, depth + 1, loop_depth))
    return counts


@dataclass
class ACEOBPreferenceModel:
    weights: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]
    name: str = "aceob-ast-pairwise-ranker"

    @classmethod
    def from_json(cls, path: str | Path) -> "ACEOBPreferenceModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_type") != "aceob_pairwise_ast_ranker" or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Неизвестный формат ACEOB-модели")
        return cls({k: float(v) for k, v in payload["weights"].items()},
                   {k: float(v) for k, v in payload["means"].items()},
                   {k: float(v) for k, v in payload["scales"].items()}, payload.get("name", "aceob-ast-pairwise-ranker"))

    def score(self, source: str) -> float:
        features = code_feature_vector(source)
        return sum(weight * ((features.get(key, 0.0) - self.means.get(key, 0.0)) / (self.scales.get(key, 1.0) or 1.0))
                   for key, weight in self.weights.items())

    def preference(self, inefficient_code: str, efficient_code: str) -> float:
        margin = self.score(efficient_code) - self.score(inefficient_code)
        if margin >= 0:
            return 1.0 / (1.0 + math.exp(-min(margin, 60.0)))
        exp_value = math.exp(max(margin, -60.0))
        return exp_value / (1.0 + exp_value)


def train_aceob_model(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    epochs: int = 80,
    learning_rate: float = 0.05,
    l2: float = 0.001,
) -> dict[str, Any]:
    """Train a pairwise logistic ranker from ACEOB efficient/inefficient pairs."""
    if epochs < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("epochs должен быть >= 1, learning_rate > 0, l2 >= 0")
    dataset_file = Path(dataset_path)
    if dataset_file.suffix.lower() in {".jsonl", ".ndjson"}:
        declared_splits: set[str] = set()
        for line_number, line in enumerate(dataset_file.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Некорректный JSONL на строке {line_number}: {exc}") from exc
            if isinstance(record, Mapping) and record.get("split"):
                declared_splits.add(str(record["split"]).lower())
        forbidden = declared_splits - {"train"}
        if forbidden:
            raise ValueError(
                "В обучающий файл попали split не train: " + ", ".join(sorted(forbidden)) +
                ". Обучайте модель только на train.jsonl; validation/test используйте отдельно."
            )
    pairs = load_aceob_pairs(dataset_path, limit=limit)
    # Ignore malformed Python examples rather than silently training on them.
    valid: list[tuple[dict[str, float], dict[str, float]]] = []
    valid_examples: list[CodePair] = []
    invalid = 0
    for pair in pairs:
        try:
            valid.append((code_feature_vector(pair.inefficient_code), code_feature_vector(pair.efficient_code)))
            valid_examples.append(pair)
        except (SyntaxError, ValueError, RecursionError):
            invalid += 1
    if not valid:
        raise ValueError("Нет валидных Python-пар для обучения после AST-разбора")
    keys = sorted({key for slow, fast in valid for key in (*slow.keys(), *fast.keys())})
    all_vectors = [vector for pair in valid for vector in pair]
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for key in keys:
        values = [v.get(key, 0.0) for v in all_vectors]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means[key] = mean
        scales[key] = math.sqrt(variance) or 1.0
    diffs: list[list[float]] = []
    for slow, fast in valid:
        diffs.append([((fast.get(k, 0.0) - slow.get(k, 0.0)) / scales[k]) for k in keys])
    weights = [0.0] * len(keys)
    for _ in range(epochs):
        gradients = [l2 * weight for weight in weights]
        for diff in diffs:
            margin = sum(w * x for w, x in zip(weights, diff))
            # d/dw softplus(-margin) = -sigmoid(-margin) * diff
            if margin >= 0:
                exp_neg = math.exp(-min(margin, 60.0))
                factor = -exp_neg / (1.0 + exp_neg)
            else:
                exp_margin = math.exp(max(margin, -60.0))
                factor = -1.0 / (1.0 + exp_margin)
            for i, value in enumerate(diff):
                gradients[i] += factor * value / len(diffs)
        for i in range(len(weights)):
            weights[i] -= learning_rate * gradients[i]
    model = ACEOBPreferenceModel(dict(zip(keys, weights)), means, scales)
    # Pairwise accuracy is a useful smoke metric, not a guarantee of generalization.
    correct = sum(model.score(pair.efficient_code) > model.score(pair.inefficient_code) for pair in valid_examples)
    metrics = {
        "pair_count": len(pairs), "valid_pair_count": len(valid), "invalid_pair_count": invalid,
        "feature_count": len(keys), "pairwise_accuracy_train": correct / len(valid),
        "epochs": epochs,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "model_type": "aceob_pairwise_ast_ranker",
        "name": model.name,
        "weights": model.weights,
        "means": model.means,
        "scales": model.scales,
        "training": metrics,
        "note": "ACEOB-trained AST-feature pairwise ranker; not an authors' pretrained model.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
