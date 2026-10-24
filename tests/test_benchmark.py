import pytest

from ast_optimizer.benchmark import generate_synthetic_cases, run_benchmarks


def test_synthetic_cases_are_reproducible_and_generated_on_demand():
    assert generate_synthetic_cases(seed=12, size=20) == generate_synthetic_cases(seed=12, size=20)
    assert len(generate_synthetic_cases(seed=12, size=20)) >= 3


def test_benchmark_runs_repeated_measurements_and_preserves_results():
    results = run_benchmarks(seed=3, size=30, repeats=3, warmups=1)

    assert len(results) == 3
    assert all(len(result.baseline_seconds) == 3 for result in results)
    assert all(len(result.optimized_seconds) == 3 for result in results)
    assert all(result.baseline_result == result.optimized_result for result in results)
    assert all(result.baseline_mean > 0 for result in results)


def test_benchmark_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        run_benchmarks(size=0)


def test_benchmark_collects_memory_and_call_profile():
    results = run_benchmarks(seed=5, size=20, repeats=2, warmups=0)

    for result in results:
        assert len(result.baseline_memory_peak_bytes) == 2
        assert len(result.optimized_memory_peak_bytes) == 2
        assert all(value >= 0 for value in result.baseline_memory_peak_bytes)
        assert all(value >= 0 for value in result.optimized_memory_peak_bytes)
        assert result.baseline_total_calls >= result.baseline_primitive_calls >= 0
        assert result.optimized_total_calls >= result.optimized_primitive_calls >= 0
        assert result.baseline_profile_seconds >= 0
        assert result.optimized_profile_seconds >= 0
        assert result.baseline_hotspots
        assert result.optimized_hotspots


def test_benchmark_memory_profile_is_separate_from_timing():
    from ast_optimizer.benchmark import benchmark_source

    result = benchmark_source("values = list(range(100))\nresult = sum(values)\n", repeats=2, warmups=0)

    assert len(result.baseline_seconds) == len(result.baseline_memory_peak_bytes) == 2
    assert result.baseline_memory_peak_mean > 0
    assert isinstance(result.memory_reduction_percent, float)
