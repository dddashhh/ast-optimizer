from __future__ import annotations

import argparse
import cProfile
import pstats
import random
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ast_optimizer import InputPipeline
from ast_optimizer.analyzer import ASTAnalyzer
from ast_optimizer.analyzer.models import OptimizationCandidate
from ast_optimizer.codegen import CodeGenerationError, CodeGenerator
from ast_optimizer.transformer import ASTTransformer, TransformError


@dataclass(frozen=True)
class PipelineResult:
    filename: str
    original_source: str
    optimized_source: str
    optimization: str
    candidate_count: int
    features: dict[str, float]


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    baseline_seconds: tuple[float, ...]
    optimized_seconds: tuple[float, ...]
    baseline_result: object
    optimized_result: object
    optimization: str
    baseline_memory_peak_bytes: tuple[int, ...]
    optimized_memory_peak_bytes: tuple[int, ...]
    baseline_current_memory_bytes: tuple[int, ...]
    optimized_current_memory_bytes: tuple[int, ...]
    baseline_total_calls: int
    optimized_total_calls: int
    baseline_primitive_calls: int
    optimized_primitive_calls: int
    baseline_profile_seconds: float
    optimized_profile_seconds: float
    baseline_hotspots: tuple[tuple[str, int, int, float, float], ...]
    optimized_hotspots: tuple[tuple[str, int, int, float, float], ...]

    @property
    def baseline_mean(self) -> float:
        return statistics.mean(self.baseline_seconds)

    @property
    def optimized_mean(self) -> float:
        return statistics.mean(self.optimized_seconds)

    @property
    def baseline_median(self) -> float:
        return statistics.median(self.baseline_seconds)

    @property
    def optimized_median(self) -> float:
        return statistics.median(self.optimized_seconds)

    @property
    def baseline_stdev(self) -> float:
        return statistics.stdev(self.baseline_seconds) if len(self.baseline_seconds) > 1 else 0.0

    @property
    def optimized_stdev(self) -> float:
        return statistics.stdev(self.optimized_seconds) if len(self.optimized_seconds) > 1 else 0.0

    @property
    def baseline_memory_peak_mean(self) -> float:
        return statistics.mean(self.baseline_memory_peak_bytes)

    @property
    def optimized_memory_peak_mean(self) -> float:
        return statistics.mean(self.optimized_memory_peak_bytes)

    @property
    def baseline_current_memory_mean(self) -> float:
        return statistics.mean(self.baseline_current_memory_bytes)

    @property
    def optimized_current_memory_mean(self) -> float:
        return statistics.mean(self.optimized_current_memory_bytes)

    @property
    def memory_reduction_percent(self) -> float:
        if self.baseline_memory_peak_mean == 0:
            return 0.0
        return (self.baseline_memory_peak_mean - self.optimized_memory_peak_mean) / self.baseline_memory_peak_mean * 100

    @property
    def speedup(self) -> float:
        if self.optimized_mean == 0:
            return 0.0
        return self.baseline_mean / self.optimized_mean


class MLOptimizerStub:
    def score(self, candidate: OptimizationCandidate) -> float:
        priorities = {
            "constant_folding": 0.9,
            "dead_branch_elimination": 0.85,
            "loop_to_comprehension": 0.8,
            "listcomp_to_generator": 0.75,
            "simplify_negation": 0.7,
        }
        return priorities.get(candidate.type.value, 0.0)

    def choose(self, candidates: Iterable[OptimizationCandidate]) -> OptimizationCandidate | None:
        return max(candidates, key=lambda item: (self.score(item), -item.lineno, -item.col_offset), default=None)


def optimize_source(source: str, filename: str = "<input>") -> PipelineResult:
    pipeline = InputPipeline()
    analyzer = ASTAnalyzer()
    transformer = ASTTransformer()
    generator = CodeGenerator()
    ml_stub = MLOptimizerStub()
    tree = pipeline.process(source, filename=filename)
    analysis = analyzer.analyze(tree)
    candidate = ml_stub.choose(analysis.candidates)
    optimized_source = source
    optimization = "не применена"
    if candidate is not None:
        try:
            transformed = transformer.transform(tree, candidate)
            optimized_source = generator.generate(transformed).text
            optimization = candidate.type.value
        except (CodeGenerationError, TransformError, ValueError, RuntimeError, TypeError):
            optimization = "не применена: ошибка трансформации"
    return PipelineResult(
        filename=filename,
        original_source=source,
        optimized_source=optimized_source,
        optimization=optimization,
        candidate_count=len(analysis.candidates),
        features=analysis.features.to_vector(),
    )


def generate_synthetic_cases(seed: int = 0, size: int = 5000) -> list[tuple[str, str]]:
    if size < 1:
        raise ValueError("size должен быть положительным")
    rng = random.Random(seed)
    values = [rng.randint(1, 100) for _ in range(size)]
    values_literal = repr(values)
    return [
        ("constant_folding", f"values = {values_literal}\nresult = sum(x * (2 + 3) for x in values)\n"),
        ("loop_to_comprehension", f"values = {values_literal}\nresult = []\nfor x in values:\n    if x % 2 == 0:\n        result.append(x * 2)\nresult = sum(result)\n"),
        ("listcomp_to_generator", f"values = {values_literal}\nresult = sum([x * x for x in values])\n"),
    ]


def _execute(code: object) -> object:
    namespace: dict[str, object] = {"__builtins__": __builtins__}
    exec(code, namespace, namespace)
    return namespace.get("result")


def _measure(source: str, repeats: int, warmups: int, filename: str = "<benchmark>") -> tuple[tuple[float, ...], object]:
    code = compile(source, filename, "exec")
    for _ in range(warmups):
        _execute(code)
    durations = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = _execute(code)
        durations.append(time.perf_counter() - start)
    return tuple(durations), result


def _measure_memory(source: str, repeats: int, filename: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    code = compile(source, filename, "exec")
    peaks: list[int] = []
    currents: list[int] = []
    for _ in range(repeats):
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tracemalloc.start()
        try:
            result = _execute(code)
            current, peak = tracemalloc.get_traced_memory()
            currents.append(current)
            peaks.append(peak)
        finally:
            tracemalloc.stop()
    return tuple(peaks), tuple(currents)


def _profile_calls(source: str, filename: str, limit: int = 8) -> tuple[int, int, float, tuple[tuple[str, int, int, float, float], ...]]:
    code = compile(source, filename, "exec")
    profiler = cProfile.Profile()
    profiler.enable()
    _execute(code)
    profiler.disable()
    stats = pstats.Stats(profiler)
    hotspots = sorted(
        ((f"{Path(path).name}:{line}({func})", values[1], values[0], values[2], values[3])
         for (path, line, func), values in stats.stats.items()
         if path != __file__ and "disable" not in func and "Profiler" not in func),
        key=lambda item: (item[4], item[1]),
        reverse=True,
    )[:limit]
    return stats.total_calls, stats.prim_calls, stats.total_tt, tuple(hotspots)


def benchmark_source(source: str, filename: str = "<input>", repeats: int = 7, warmups: int = 2) -> BenchmarkResult:
    if repeats < 1 or warmups < 0:
        raise ValueError("repeats должен быть положительным, warmups не может быть отрицательным")
    pipeline_result = optimize_source(source, filename)
    baseline_times, baseline_value = _measure(source, repeats, warmups, filename)
    optimized_source = pipeline_result.optimized_source
    optimized_times, optimized_value = _measure(optimized_source, repeats, warmups, filename)
    optimization = pipeline_result.optimization
    if baseline_value != optimized_value:
        optimized_source = source
        optimized_times, optimized_value = _measure(source, repeats, warmups, filename)
        optimization = "отклонена: результат отличается"
    baseline_memory_peak, baseline_memory_current = _measure_memory(source, repeats, filename)
    optimized_memory_peak, optimized_memory_current = _measure_memory(optimized_source, repeats, filename)
    baseline_calls, baseline_primitive, baseline_profile_time, baseline_hotspots = _profile_calls(source, filename)
    optimized_calls, optimized_primitive, optimized_profile_time, optimized_hotspots = _profile_calls(optimized_source, filename)
    return BenchmarkResult(
        name=Path(filename).name if filename != "<input>" else "input.py",
        baseline_seconds=baseline_times,
        optimized_seconds=optimized_times,
        baseline_result=baseline_value,
        optimized_result=optimized_value,
        optimization=optimization,
        baseline_memory_peak_bytes=baseline_memory_peak,
        optimized_memory_peak_bytes=optimized_memory_peak,
        baseline_current_memory_bytes=baseline_memory_current,
        optimized_current_memory_bytes=optimized_memory_current,
        baseline_total_calls=baseline_calls,
        optimized_total_calls=optimized_calls,
        baseline_primitive_calls=baseline_primitive,
        optimized_primitive_calls=optimized_primitive,
        baseline_profile_seconds=baseline_profile_time,
        optimized_profile_seconds=optimized_profile_time,
        baseline_hotspots=baseline_hotspots,
        optimized_hotspots=optimized_hotspots,
    )


def benchmark_file(path: str | Path, repeats: int = 7, warmups: int = 2) -> BenchmarkResult:
    file_path = Path(path)
    if file_path.suffix != ".py":
        raise ValueError("Входной файл должен иметь расширение .py")
    source = file_path.read_text(encoding="utf-8")
    return benchmark_source(source, str(file_path), repeats, warmups)


def run_benchmarks(seed: int = 0, size: int = 5000, repeats: int = 7, warmups: int = 2) -> list[BenchmarkResult]:
    if repeats < 1 or warmups < 0:
        raise ValueError("repeats должен быть положительным, warmups не может быть отрицательным")
    return [benchmark_source(source, f"{name}.py", repeats, warmups)
            for name, source in generate_synthetic_cases(seed=seed, size=size)]


def print_report(results: list[BenchmarkResult]) -> None:
    print("\nРЕЗУЛЬТАТЫ AST-ОПТИМИЗАТОРА")
    print("=" * 100)
    print(f"{'Тест':<25} {'Оптимизация':<30} {'Исходный, мс':>13} {'Новый, мс':>12} {'Ускорение':>10}")
    print("-" * 100)
    for result in results:
        print(f"{result.name:<25} {result.optimization[:30]:<30} {result.baseline_mean * 1000:>13.3f} "
              f"{result.optimized_mean * 1000:>12.3f} {result.speedup:>9.3f}x")
        print(f"  медиана: {result.baseline_median * 1000:.3f} / {result.optimized_median * 1000:.3f} мс; "
              f"σ: {result.baseline_stdev * 1000:.3f} / {result.optimized_stdev * 1000:.3f} мс")
        print(f"  пиковая память Python, среднее: {result.baseline_memory_peak_mean / 1024:.2f} / "
              f"{result.optimized_memory_peak_mean / 1024:.2f} KiB; изменение: {result.memory_reduction_percent:+.2f}%")
        print(f"  память после выполнения, среднее: {result.baseline_current_memory_mean / 1024:.2f} / "
              f"{result.optimized_current_memory_mean / 1024:.2f} KiB")
        print(f"  вызовы функций (cProfile): {result.baseline_total_calls} / {result.optimized_total_calls}; "
              f"примитивные: {result.baseline_primitive_calls} / {result.optimized_primitive_calls}")
        print(f"  время профилируемого запуска: {result.baseline_profile_seconds * 1000:.3f} / "
              f"{result.optimized_profile_seconds * 1000:.3f} мс (не использовать для сравнения скорости)")
        print("  горячие функции исходного кода:")
        for function, calls, primitive, internal, cumulative in result.baseline_hotspots[:5]:
            print(f"    {function}: вызовы={calls}, примитивные={primitive}, собственное={internal * 1000:.3f} мс, "
                  f"накопленное={cumulative * 1000:.3f} мс")
        print("  горячие функции оптимизированного кода:")
        for function, calls, primitive, internal, cumulative in result.optimized_hotspots[:5]:
            print(f"    {function}: вызовы={calls}, примитивные={primitive}, собственное={internal * 1000:.3f} мс, "
                  f"накопленное={cumulative * 1000:.3f} мс")
    if results:
        baseline = statistics.mean(item.baseline_mean for item in results)
        optimized = statistics.mean(item.optimized_mean for item in results)
        speedup = baseline / optimized if optimized else 0.0
        print("-" * 100)
        print(f"Среднее по тестам: исходный {baseline * 1000:.3f} мс; новый {optimized * 1000:.3f} мс; ускорение {speedup:.3f}x")
        print(f"Проверка результата: {'пройдена' if all(x.baseline_result == x.optimized_result for x in results) else 'ошибка'}")
        print(f"Измерений на версию: {len(results[0].baseline_seconds)}; прогревочных запусков: задаются параметром --warmups")


def main() -> None:
    parser = argparse.ArgumentParser(description="Полный цикл AST-оптимизации и сравнения производительности")
    parser.add_argument("input", nargs="?", help="Путь к Python-файлу для анализа и benchmark")
    parser.add_argument("--size", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="Куда сохранить сгенерированный оптимизированный код")
    args = parser.parse_args()
    if args.input:
        file_path = Path(args.input)
        result = optimize_source(file_path.read_text(encoding="utf-8"), str(file_path))
        benchmark_result = benchmark_source(result.original_source, result.filename, args.repeats, args.warmups)
        if args.output:
            output_source = result.optimized_source if benchmark_result.baseline_result == benchmark_result.optimized_result else result.original_source
            Path(args.output).write_text(output_source, encoding="utf-8")
        print(f"Файл: {result.filename}")
        print(f"Кандидатов найдено: {result.candidate_count}")
        print(f"Выбранная оптимизация: {benchmark_result.optimization}")
        print(f"Признаки AST: {result.features}")
        print_report([benchmark_result])
        if args.output:
            print(f"Результат сохранён: {args.output}")
    else:
        print_report(run_benchmarks(seed=args.seed, size=args.size, repeats=args.repeats, warmups=args.warmups))


if __name__ == "__main__":
    main()
