from __future__ import annotations

import argparse
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Any

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
    applied_optimizations: tuple[str, ...] = ()
    iterations: int = 0
    termination_reason: str = "no_candidates"
    candidates_tried: int = 0


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
    candidate_count: int
    applied_optimizations: tuple[str, ...]
    iterations: int
    termination_reason: str
    candidates_tried: int

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


EXECUTION_TIMEOUT_SECONDS = 3.0
MAX_SOURCE_BYTES = 1_000_000
MAX_REPEATS = 100
MAX_WARMUPS = 100
MAX_ITERATIONS = 500

_CHILD_RUNNER = r'''import cProfile, json, pstats, sys, time, tracemalloc
__file__ = "<ast_optimizer_child>"
payload = json.loads(sys.stdin.read())
source = payload["source"]
filename = payload.get("filename", "<benchmark>")
mode = payload.get("mode", "run")
def safe_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [safe_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: safe_value(v) for k, v in value.items()}
    return {"__unsupported_type__": type(value).__name__}
def execute():
    namespace = {"__builtins__": __builtins__, "__file__": filename, "__name__": "__main__"}
    exec(compile(source, filename, "exec"), namespace, namespace)
    return namespace.get("result")
try:
    code = compile(source, filename, "exec")
    if mode == "run":
        value = execute()
        result = {"ok": True, "value": safe_value(value)}
    elif mode == "timing":
        for _ in range(payload["warmups"]): execute()
        durations = []
        value = None
        for _ in range(payload["repeats"]):
            start = time.perf_counter(); value = execute(); durations.append(time.perf_counter() - start)
        result = {"ok": True, "durations": durations, "value": safe_value(value)}
    elif mode == "memory":
        peaks, currents = [], []
        for _ in range(payload["repeats"]):
            tracemalloc.start()
            try:
                execute(); current, peak = tracemalloc.get_traced_memory()
                currents.append(current); peaks.append(peak)
            finally:
                tracemalloc.stop()
        result = {"ok": True, "peaks": peaks, "currents": currents}
    elif mode == "profile":
        profiler = cProfile.Profile(); profiler.enable(); execute(); profiler.disable()
        stats = pstats.Stats(profiler)
        hotspots = sorted(
            ((f"{__import__('pathlib').Path(path).name}:{line}({func})", values[1], values[0], values[2], values[3])
             for (path, line, func), values in stats.stats.items()
             if path != __file__ and "disable" not in func and "Profiler" not in func),
            key=lambda item: (item[4], item[1]), reverse=True,
        )[:payload.get("limit", 8)]
        result = {"ok": True, "total_calls": stats.total_calls, "primitive_calls": stats.prim_calls,
                  "profile_seconds": stats.total_tt, "hotspots": hotspots}
    else:
        raise ValueError("unsupported mode")
except BaseException as exc:
    result = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:500]}
# Prefix makes incidental print() calls by the target program distinguishable.
print("ASTOPT_RESULT:" + json.dumps(result, allow_nan=False))
''' 


def _isolated_run(source: str, filename: str = "<benchmark>", *, mode: str = "run", **options):
    if not isinstance(source, str):
        raise TypeError("source должен быть строкой")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError(f"Исходный код превышает лимит {MAX_SOURCE_BYTES} байт")
    payload = {"source": source, "filename": filename, "mode": mode, **options}
    with tempfile.TemporaryDirectory(prefix="ast-optimizer-") as workdir:
        command = [sys.executable, "-I", "-c", _CHILD_RUNNER]
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=workdir,
            env={"PATH": os.defpath, "PYTHONIOENCODING": "utf-8"},
            start_new_session=(os.name == "posix"),
        )
        try:
            stdout, stderr = process.communicate(json.dumps(payload), timeout=EXECUTION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.communicate()
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            raise TimeoutError(f"Выполнение превысило лимит {EXECUTION_TIMEOUT_SECONDS:g} с") from exc
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
        marker = "ASTOPT_RESULT:"
        lines = [line[len(marker):] for line in stdout.splitlines() if line.startswith(marker)]
        if process.returncode != 0 or not lines:
            detail = stderr[-500:] or "дочерний процесс завершился без результата"
            raise RuntimeError(f"Изолированный запуск завершился с ошибкой: {detail}")
        try:
            result = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise RuntimeError("Не удалось прочитать результат дочернего процесса") from exc
        if not result.get("ok"):
            return result
        return result


def _execute_for_verification(source: str, filename: str) -> tuple[bool, object]:
    try:
        result = _isolated_run(source, filename)
        if not result.get("ok"):
            return False, (result.get("error_type", "ExecutionError"), result.get("error", "ошибка"))
        value = result.get("value")
        if isinstance(value, dict) and "__unsupported_type__" in value:
            return False, None
        return True, value
    except (TimeoutError, RuntimeError, ValueError) as exc:
        return False, (type(exc).__name__, str(exc))

def _equivalent_results(original_source: str, candidate_source: str, filename: str) -> bool:
    original_ok, original_result = _execute_for_verification(original_source, filename)
    candidate_ok, candidate_result = _execute_for_verification(candidate_source, filename)
    return original_ok and candidate_ok and original_result == candidate_result


def optimize_source(source: str, filename: str = "<input>", max_iterations: int = 100) -> PipelineResult:
    if not 1 <= max_iterations <= MAX_ITERATIONS:
        raise ValueError(f"max_iterations должен быть в диапазоне 1..{MAX_ITERATIONS}")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError(f"Исходный код превышает лимит {MAX_SOURCE_BYTES} байт")
    pipeline = InputPipeline()
    analyzer = ASTAnalyzer()
    transformer = ASTTransformer()
    generator = CodeGenerator()
    ml_stub = MLOptimizerStub()
    original_tree = pipeline.process(source, filename=filename)
    current_tree = original_tree
    current_source = source
    initial_analysis = analyzer.analyze(current_tree)
    applied: list[str] = []
    candidates_tried = 0
    iterations = 0
    termination_reason = "max_iterations"

    while iterations < max_iterations:
        analysis = analyzer.analyze(current_tree)
        if not analysis.candidates:
            termination_reason = "no_candidates"
            break
        ranked = sorted(
            analysis.candidates,
            key=lambda candidate: (
                ml_stub.score(candidate),
                -candidate.lineno,
                -candidate.col_offset,
                -candidate.node_index,
            ),
            reverse=True,
        )
        accepted = False
        for candidate in ranked:
            candidates_tried += 1
            try:
                transformed = transformer.transform(current_tree, candidate)
                generated = generator.generate(transformed).text
            except (CodeGenerationError, TransformError, ValueError, RuntimeError, TypeError):
                continue
            if generated == current_source:
                continue
            if not _equivalent_results(source, generated, filename):
                continue
            current_tree = pipeline.process(generated, filename=filename)
            current_source = generated
            applied.append(candidate.type.value)
            iterations += 1
            accepted = True
            break
        if not accepted:
            termination_reason = "no_verified_candidate"
            break
    else:
        termination_reason = "max_iterations"

    if iterations >= max_iterations and not analyzer.analyze(current_tree).candidates:
        termination_reason = "no_candidates"
    return PipelineResult(
        filename=filename,
        original_source=source,
        optimized_source=current_source,
        optimization=", ".join(applied) if applied else "не применена",
        candidate_count=len(initial_analysis.candidates),
        features=initial_analysis.features.to_vector(),
        applied_optimizations=tuple(applied),
        iterations=iterations,
        termination_reason=termination_reason,
        candidates_tried=candidates_tried,
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


def _measure(source: str, repeats: int, warmups: int, filename: str = "<benchmark>") -> tuple[tuple[float, ...], object]:
    _validate_run_options(repeats, warmups, 100)
    result = _isolated_run(source, filename, mode="timing", repeats=repeats, warmups=warmups)
    if not result.get("ok"):
        raise RuntimeError(f"Ошибка выполнения программы: {result.get('error_type')}: {result.get('error')}")
    return tuple(result["durations"]), result.get("value")


def _measure_memory(source: str, repeats: int, filename: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    _validate_run_options(repeats, 0, 100)
    result = _isolated_run(source, filename, mode="memory", repeats=repeats)
    if not result.get("ok"):
        raise RuntimeError(f"Ошибка измерения памяти: {result.get('error_type')}: {result.get('error')}")
    return tuple(result["peaks"]), tuple(result["currents"])


def _profile_calls(source: str, filename: str, limit: int = 8) -> tuple[Any, Any, Any, tuple[tuple[Any], ...]]:
    result = _isolated_run(source, filename, mode="profile", limit=limit)
    if not result.get("ok"):
        raise RuntimeError(f"Ошибка профилирования: {result.get('error_type')}: {result.get('error')}")
    return (result["total_calls"], result["primitive_calls"], result["profile_seconds"],
            tuple(tuple(item) for item in result["hotspots"]))


def _validate_run_options(repeats: int, warmups: int, max_iterations: int) -> None:
    if not 1 <= repeats <= MAX_REPEATS:
        raise ValueError(f"repeats должен быть в диапазоне 1..{MAX_REPEATS}")
    if not 0 <= warmups <= MAX_WARMUPS:
        raise ValueError(f"warmups должен быть в диапазоне 0..{MAX_WARMUPS}")
    if not 1 <= max_iterations <= MAX_ITERATIONS:
        raise ValueError(f"max_iterations должен быть в диапазоне 1..{MAX_ITERATIONS}")

def benchmark_source(
    source: str,
    filename: str = "<input>",
    repeats: int = 7,
    warmups: int = 2,
    max_iterations: int = 100,
) -> BenchmarkResult:
    _validate_run_options(repeats, warmups, max_iterations)
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError(f"Исходный код превышает лимит {MAX_SOURCE_BYTES} байт")
    pipeline_result = optimize_source(source, filename, max_iterations=max_iterations)
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
        candidate_count=pipeline_result.candidate_count,
        applied_optimizations=pipeline_result.applied_optimizations,
        iterations=pipeline_result.iterations,
        termination_reason=pipeline_result.termination_reason,
        candidates_tried=pipeline_result.candidates_tried,
    )


def benchmark_file(
    path: str | Path, repeats: int = 7, warmups: int = 2, max_iterations: int = 100
) -> BenchmarkResult:
    file_path = Path(path)
    if file_path.suffix != ".py":
        raise ValueError("Входной файл должен иметь расширение .py")
    _validate_run_options(repeats, warmups, max_iterations)
    if not file_path.is_file():
        raise ValueError("Входной путь должен указывать на существующий обычный файл")
    if file_path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError(f"Входной файл превышает лимит {MAX_SOURCE_BYTES} байт")
    source = file_path.read_text(encoding="utf-8")
    return benchmark_source(source, str(file_path), repeats, warmups, max_iterations)


def run_benchmarks(
    seed: int = 0,
    size: int = 5000,
    repeats: int = 7,
    warmups: int = 2,
    max_iterations: int = 100,
) -> list[BenchmarkResult]:
    _validate_run_options(repeats, warmups, max_iterations)
    if not 1 <= size <= 100_000:
        raise ValueError("size должен быть в диапазоне 1..100000")
    return [
        benchmark_source(source, f"{name}.py", repeats, warmups, max_iterations)
        for name, source in generate_synthetic_cases(seed=seed, size=size)
    ]


def print_report(results: list[BenchmarkResult]) -> None:
    print("\nРЕЗУЛЬТАТЫ AST-ОПТИМИЗАТОРА")
    print("=" * 100)
    print(f"{'Тест':<25} {'Оптимизация':<30} {'Исходный, мс':>13} {'Новый, мс':>12} {'Ускорение':>10}")
    print("-" * 100)
    for result in results:
        print(f"{result.name:<25} {result.optimization[:30]:<30} {result.baseline_mean * 1000:>13.3f} "
              f"{result.optimized_mean * 1000:>12.3f} {result.speedup:>9.3f}x")
        print(f"  кандидатов найдено: {result.candidate_count}; проверено: {result.candidates_tried}; "
              f"итераций: {result.iterations}; остановка: {result.termination_reason}")
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
        print(f"Smoke check результата: {'пройдена' if all(x.baseline_result == x.optimized_result for x in results) else 'ошибка'}")
        print(f"Измерений на версию: {len(results[0].baseline_seconds)}; прогревочных запусков: задаются параметром --warmups")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Полный цикл AST-оптимизации и сравнения производительности",
    )
    parser.add_argument("input", nargs="?", help="Путь к Python-файлу для анализа и benchmark")
    parser.add_argument("--size", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="Куда сохранить сгенерированный оптимизированный код")
    args = parser.parse_args()
    try:
        _validate_run_options(args.repeats, args.warmups, args.max_iterations)
        if not 1 <= args.size <= 100_000:
            raise ValueError("size должен быть в диапазоне 1..100000")
        if args.input:
            file_path = Path(args.input)
            if file_path.suffix != ".py" or not file_path.is_file():
                raise ValueError("input должен быть существующим файлом с расширением .py")
            if file_path.stat().st_size > MAX_SOURCE_BYTES:
                raise ValueError(f"Входной файл превышает лимит {MAX_SOURCE_BYTES} байт")
            if args.output and Path(args.output).resolve() == file_path.resolve():
                raise ValueError("Файл вывода не должен совпадать с исходным файлом")
            source = file_path.read_text(encoding="utf-8")
            result = optimize_source(source, str(file_path), max_iterations=args.max_iterations)
            benchmark_result = benchmark_source(
                result.original_source, result.filename, args.repeats, args.warmups, args.max_iterations
            )
            if args.output:
                output_source = (result.optimized_source
                                 if benchmark_result.baseline_result == benchmark_result.optimized_result
                                 else result.original_source)
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(output_source, encoding="utf-8")
            print(f"Файл: {result.filename}")
            print(f"Кандидатов найдено: {result.candidate_count}")
            print(f"Применённые оптимизации: {benchmark_result.optimization}")
            print(f"Итераций: {result.iterations}; кандидатов проверено: {result.candidates_tried}; остановка: {result.termination_reason}")
            print(f"Признаки AST: {result.features}")
            print_report([benchmark_result])
            if args.output:
                print(f"Результат сохранён: {args.output}")
        else:
            print_report(run_benchmarks(
                seed=args.seed, size=args.size, repeats=args.repeats,
                warmups=args.warmups, max_iterations=args.max_iterations,
            ))
    except (OSError, UnicodeError, ValueError, RuntimeError, TimeoutError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
