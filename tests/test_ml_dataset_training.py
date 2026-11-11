import json

import pytest

from ast_optimizer.analyzer.models import ASTFeatures, OptimizationCandidate, OptimizationType
from ast_optimizer.ml_optimizer import LinearCostModel, MLOptimizer
from ast_optimizer.ml_optimizer import dataset as dataset_module
from ast_optimizer.ml_optimizer.dataset import append_jsonl, collect_training_examples, load_jsonl
from ast_optimizer.ml_optimizer.training import train_model


def test_collects_candidate_with_measured_speedup(monkeypatch):
    monkeypatch.setattr(dataset_module, "_equivalent_results", lambda *args: True)

    def fake_measure(source, repeats, warmups, filename):
        return ((0.10, 0.11, 0.09) if "2 + 3" in source else (0.05, 0.06, 0.04)), 5

    monkeypatch.setattr(dataset_module, "_measure", fake_measure)
    summary = collect_training_examples("result = 2 + 3\n", repeats=3, warmups=0)
    assert summary.candidates_seen >= 1
    assert summary.examples
    example = summary.examples[0]
    assert example.actual_speedup > 1.0
    assert example.features["candidate.constant_folding"] == 1.0
    assert example.source_hash


def test_jsonl_round_trip_and_validation(tmp_path):
    path = tmp_path / "data.jsonl"
    row = {
        "schema_version": 1, "source_hash": "a", "filename": "x.py",
        "optimization_type": "constant_folding", "candidate_id": "constant_folding@1:0#1",
        "features": {"node_count": 3.0}, "baseline_median_seconds": 0.2,
        "candidate_median_seconds": 0.1, "actual_speedup": 2.0, "python_version": "3.x",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert load_jsonl(path)[0]["actual_speedup"] == 2.0
    path.write_text('{"features": {}, "actual_speedup": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Некорректный actual_speedup"):
        load_jsonl(path)


def test_training_exports_loadable_model_and_holdout_metrics(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    output_path = tmp_path / "model.json"
    rows = []
    for i, speedup in enumerate((0.6, 0.8, 1.2, 1.5, 1.7), 1):
        rows.append({
            "schema_version": 1, "source_hash": f"source-{i}", "filename": f"{i}.py",
            "optimization_type": "constant_folding", "candidate_id": f"candidate-{i}",
            "features": {"node_count": float(i), "candidate.constant_folding": 1.0},
            "baseline_median_seconds": 0.1, "candidate_median_seconds": 0.1 / speedup,
            "actual_speedup": speedup, "python_version": "3.x",
        })
    dataset_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metrics = train_model(dataset_path, output_path, ridge=0.1, seed=3)
    assert metrics["sample_count"] == 5
    assert metrics["feature_count"] == 2
    assert metrics["validation_metrics"] is not None
    model = LinearCostModel.from_json(output_path)
    assert model.name == "trained-linear-cost-model"
    assert model.predict({"node_count": 1.0, "candidate.constant_folding": 1.0}) < 1.0


def test_ml_optimizer_can_rank_predicted_slowdown():
    class SlowdownModel:
        name = "slowdown-model"

        def predict(self, features):
            return 0.7

    candidate = OptimizationCandidate(OptimizationType.CONSTANT_FOLDING, 0, "BinOp", 1, 0, "fold")
    score = MLOptimizer(SlowdownModel()).score(candidate, ASTFeatures())
    assert score.predicted_speedup == pytest.approx(0.7)


def test_select_best_declines_when_all_candidates_predict_slowdown():
    class SlowdownModel:
        name = "slowdown-model"

        def predict(self, features):
            return 0.8

    candidate = OptimizationCandidate(OptimizationType.CONSTANT_FOLDING, 0, "BinOp", 1, 0, "fold")
    assert MLOptimizer(SlowdownModel()).select_best([candidate], ASTFeatures()) is None


def test_aceob_pairwise_model_training_and_inference(tmp_path):
    import json
    from ast_optimizer.ml_optimizer.aceob import ACEOBPreferenceModel, load_aceob_pairs, train_aceob_model

    dataset = tmp_path / "aceob.jsonl"
    dataset.write_text(
        json.dumps({
            "inefficient_code": "result = []\nfor x in values:\n    result.append(x * 2)\n",
            "efficient_code": "result = [x * 2 for x in values]\n",
        }) + "\n" +
        json.dumps({
            "IC": "result = []\nfor x in values:\n    if x > 0:\n        result.append(x)\n",
            "EC": "result = [x for x in values if x > 0]\n",
        }),
        encoding="utf-8",
    )
    assert len(load_aceob_pairs(dataset)) == 2
    model_path = tmp_path / "aceob_model.json"
    metrics = train_aceob_model(dataset, model_path, epochs=25)
    model = ACEOBPreferenceModel.from_json(model_path)
    assert metrics["valid_pair_count"] == 2
    assert model.score("result = [x * 2 for x in values]") != model.score("result = []\nfor x in values:\n    result.append(x * 2)")


def test_aceob_pair_loader_accepts_csv(tmp_path):
    from ast_optimizer.ml_optimizer.aceob import load_aceob_pairs

    path = tmp_path / "pairs.csv"
    path.write_text('inefficient_code,efficient_code\n"result = []","result = [1]"\n', encoding="utf-8")
    pairs = load_aceob_pairs(path)
    assert len(pairs) == 1
    assert pairs[0].efficient_code == "result = [1]"



def test_native_aceob_zip_imports_many_pairs_and_keeps_split(tmp_path):
    import zipfile
    from ast_optimizer.ml_optimizer.aceob import import_aceob_folders, load_aceob_pairs

    archive_path = tmp_path / "0001.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("0001/Efficient-inefficient_code_pairs_000/Efficient,Single IO execution time 10 ms.txt", "print('fast')\n")
        archive.writestr("0001/Efficient-inefficient_code_pairs_000/Inefficient,Single IO execution time 20 ms.txt", "print('slow')\n")
        archive.writestr("0001/Alternate efficient codes/01,Single IO execution time 9 ms.txt", "print('alternate')\n")
    output = tmp_path / "train.jsonl"
    summary = import_aceob_folders([archive_path], output, split="train")
    pairs = load_aceob_pairs(output)
    assert summary["pairs_written"] == 2
    assert len(pairs) == 2
    assert all(pair.inefficient_code == "print('slow')\n" for pair in pairs)


def test_aceob_training_rejects_test_split(tmp_path):
    import json
    import pytest
    from ast_optimizer.ml_optimizer.aceob import train_aceob_model

    dataset = tmp_path / "test.jsonl"
    dataset.write_text(json.dumps({
        "inefficient_code": "print(1)", "efficient_code": "print(2)", "split": "test"
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="только на train"):
        train_aceob_model(dataset, tmp_path / "model.json")


def test_aceob_evaluation_reports_pairwise_metrics(tmp_path):
    from ast_optimizer.ml_optimizer.aceob import ACEOBPreferenceModel, evaluate_aceob_model

    dataset = tmp_path / "evaluation.jsonl"
    dataset.write_text(
        "\n".join((
            json.dumps({"inefficient_code": "x = 1", "efficient_code": "x = [1]"}),
            json.dumps({"inefficient_code": "x = [1]", "efficient_code": "x = 1"}),
            json.dumps({"inefficient_code": "x=1", "efficient_code": "y=2"}),
            json.dumps({"inefficient_code": "def broken(:", "efficient_code": "x = [1]"}),
        )),
        encoding="utf-8",
    )
    model = ACEOBPreferenceModel(
        weights={"source_chars": 1.0},
        means={"source_chars": 0.0},
        scales={"source_chars": 1.0},
    )

    metrics = evaluate_aceob_model(model, dataset)

    assert metrics["pair_count"] == 4
    assert metrics["valid_pair_count"] == 3
    assert metrics["invalid_pair_count"] == 1
    assert metrics["correct_pair_count"] == 1
    assert metrics["incorrect_pair_count"] == 1
    assert metrics["tied_pair_count"] == 1
    assert metrics["pairwise_accuracy"] == pytest.approx(1 / 3)
    assert metrics["mean_prediction_margin"] == pytest.approx(0.0)


def test_aceob_evaluation_handles_ties_and_rejects_invalid_limit(tmp_path):
    from ast_optimizer.ml_optimizer.aceob import ACEOBPreferenceModel, evaluate_aceob_model

    dataset = tmp_path / "evaluation.jsonl"
    dataset.write_text(
        json.dumps({"inefficient_code": "x = 1", "efficient_code": "y = 1"}) + "\n",
        encoding="utf-8",
    )
    model = ACEOBPreferenceModel(weights={}, means={}, scales={})

    metrics = evaluate_aceob_model(model, dataset)

    assert metrics["tied_pair_count"] == 1
    assert metrics["pairwise_accuracy"] == 0.0
    assert metrics["mean_prediction_margin"] == 0.0
    with pytest.raises(ValueError, match="limit должен быть >= 1"):
        evaluate_aceob_model(model, dataset, limit=0)


def test_aceob_evaluation_rejects_empty_dataset(tmp_path):
    from ast_optimizer.ml_optimizer.aceob import ACEOBPreferenceModel, evaluate_aceob_model

    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Не найдены пары кода"):
        evaluate_aceob_model(ACEOBPreferenceModel({}, {}, {}), dataset)


def test_aceob_evaluate_cli_prints_metrics_and_saves_report(tmp_path, monkeypatch, capsys):
    import json
    from ast_optimizer.ml_optimizer.__main__ import main

    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({
        "schema_version": 1,
        "model_type": "aceob_pairwise_ast_ranker",
        "name": "test-ranker",
        "weights": {"source_chars": 1.0},
        "means": {"source_chars": 0.0},
        "scales": {"source_chars": 1.0},
    }), encoding="utf-8")
    dataset_path = tmp_path / "test.jsonl"
    dataset_path.write_text(
        json.dumps({
            "inefficient_code": "x = 1",
            "efficient_code": "result = [value for value in values]",
        }) + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "reports" / "evaluation.json"
    monkeypatch.setattr("sys.argv", [
        "ast-optimizer", "aceob-evaluate", str(model_path), str(dataset_path),
        "--output", str(report_path),
    ])

    main()

    output = capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "ACEOB MODEL EVALUATION" in output
    assert "Pairwise accuracy:" in output
    assert report["pair_count"] == 1
    assert report["valid_pair_count"] == 1
    assert report["model"] == str(model_path)
    assert report["dataset"] == str(dataset_path)


def test_aceob_evaluate_cli_accepts_limit(tmp_path, monkeypatch, capsys):
    import json
    from ast_optimizer.ml_optimizer.__main__ import main

    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps({
        "schema_version": 1,
        "model_type": "aceob_pairwise_ast_ranker",
        "weights": {},
        "means": {},
        "scales": {},
    }), encoding="utf-8")
    dataset_path = tmp_path / "test.jsonl"
    records = [
        {"inefficient_code": "x = 1", "efficient_code": "y = 2"},
        {"inefficient_code": "a = 1", "efficient_code": "b = 2"},
    ]
    dataset_path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "ast-optimizer", "aceob-evaluate", str(model_path), str(dataset_path), "--limit", "1",
    ])

    main()

    assert "Pairs: 1" in capsys.readouterr().out
