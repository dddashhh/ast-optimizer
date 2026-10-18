import pytest

from ast_optimizer.benchmark import benchmark_file, benchmark_source, generate_synthetic_cases, optimize_source


@pytest.fixture
def synthetic_python_file(tmp_path):
    cases = generate_synthetic_cases(seed=17, size=25)
    path = tmp_path / "generated_case.py"
    path.write_text(cases[0][1], encoding="utf-8")
    return path


def test_full_pipeline_generates_valid_python(synthetic_python_file):
    source = synthetic_python_file.read_text(encoding="utf-8")
    result = optimize_source(source, str(synthetic_python_file))

    compile(result.optimized_source, result.filename, "exec")
    assert result.original_source == source
    assert result.candidate_count >= 1
    assert result.features["node_count"] > 0


def test_file_benchmark_runs_multiple_measurements(synthetic_python_file):
    result = benchmark_file(synthetic_python_file, repeats=3, warmups=1)

    assert len(result.baseline_seconds) == 3
    assert len(result.optimized_seconds) == 3
    assert result.baseline_result == result.optimized_result
    assert result.baseline_mean > 0
    assert result.optimized_mean > 0


def test_source_benchmark_preserves_result():
    result = benchmark_source("result = sum([x * x for x in range(30)])\n", repeats=2, warmups=0)

    assert result.baseline_result == result.optimized_result
    assert len(result.baseline_seconds) == 2


def test_file_benchmark_rejects_non_python_file(tmp_path):
    path = tmp_path / "program.txt"
    path.write_text("result = 1", encoding="utf-8")

    with pytest.raises(ValueError):
        benchmark_file(path)


def test_optimizer_applies_multiple_candidates_until_fixpoint():
    source = "result = sum([x * (2 + 3) for x in range(30)])\n"

    result = optimize_source(source)

    assert len(result.applied_optimizations) >= 2
    assert result.iterations == len(result.applied_optimizations)
    assert result.termination_reason in {"no_candidates", "no_verified_candidate"}
    assert result.optimized_source != source
    assert compile(result.optimized_source, result.filename, "exec")


def test_optimizer_stops_at_iteration_limit():
    source = "result = sum([x * (2 + 3) for x in range(30)])\n"

    result = optimize_source(source, max_iterations=1)

    assert result.iterations == 1
    assert result.termination_reason == "max_iterations"
    assert len(result.applied_optimizations) == 1


def test_optimizer_stops_when_no_candidates_exist():
    source = "result = len(\"abc\")\n"

    result = optimize_source(source)

    assert result.optimized_source == source
    assert result.applied_optimizations == ()
    assert result.termination_reason == "no_candidates"
