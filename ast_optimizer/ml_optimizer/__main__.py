from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import append_jsonl, collect_training_examples
from .training import train_model
from .aceob import train_aceob_model, import_aceob_folders


def main() -> None:
    parser = argparse.ArgumentParser(description="Сбор benchmark-данных и обучение ML-модели AST-оптимизатора")
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="измерить кандидатов и добавить примеры в JSONL")
    collect.add_argument("inputs", nargs="+", help="Python-файлы для сбора данных")
    collect.add_argument("--output", default="data/optimization_dataset.jsonl")
    collect.add_argument("--repeats", type=int, default=5)
    collect.add_argument("--warmups", type=int, default=1)

    train = commands.add_parser("train", help="обучить линейную cost-модель из JSONL")
    train.add_argument("dataset", help="путь к JSONL-датасету")
    train.add_argument("--output", default="models/linear_cost_model.json")
    train.add_argument("--ridge", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=0)

    aceob = commands.add_parser("aceob-train", help="обучить pairwise AST-модель на парах ACEOB")
    aceob.add_argument("dataset", help="ACEOB JSON/JSONL/CSV/ZIP с парами кода")
    aceob.add_argument("--output", default="models/aceob_ast_ranker.json")
    aceob.add_argument("--limit", type=int, help="лимит пар для быстрого пробного обучения")
    aceob.add_argument("--epochs", type=int, default=80)
    aceob.add_argument("--learning-rate", type=float, default=0.05)
    aceob.add_argument("--l2", type=float, default=0.001)

    importer = commands.add_parser("aceob-import", help="обработать нативные папки/ZIP ACEOB и создать pairwise JSONL")
    importer.add_argument("inputs", nargs="+", help="папки или ZIP-архивы с задачами ACEOB")
    importer.add_argument("--split", choices=("train", "validation", "test"), required=True)
    importer.add_argument("--output", required=True, help="выходной JSONL-файл")

    args = parser.parse_args()
    try:
        if args.command == "collect":
            if not 1 <= args.repeats <= 100 or not 0 <= args.warmups <= 100:
                raise ValueError("repeats должен быть 1..100, warmups — 0..100")
            total_written = total_seen = total_transform = total_semantics = total_measure = 0
            for name in args.inputs:
                path = Path(name)
                if path.suffix != ".py" or not path.is_file():
                    raise ValueError(f"Вход должен быть существующим .py-файлом: {name}")
                source = path.read_text(encoding="utf-8")
                summary = collect_training_examples(source, str(path), args.repeats, args.warmups)
                written = append_jsonl(args.output, summary.examples)
                total_written += written
                total_seen += summary.candidates_seen
                total_transform += summary.rejected_transformations
                total_semantics += summary.rejected_semantics
                total_measure += summary.measurement_failures
                print(f"{name}: кандидатов={summary.candidates_seen}, примеров={written}, "
                      f"отклонено трансформаций={summary.rejected_transformations}, "
                      f"проверкой результата={summary.rejected_semantics}, ошибок измерения={summary.measurement_failures}")
            print(f"Итого: записано {total_written} примеров из {total_seen} кандидатов в {args.output}")
            if total_transform + total_semantics + total_measure:
                print("Важно: не прошедшие трансформацию/проверку кандидаты не включены в датасет.")
        elif args.command == "aceob-import":
            summary = import_aceob_folders(args.inputs, args.output, split=args.split)
            print(f"ACEOB {summary['split']}: записано пар {summary['pairs_written']} в {summary['output']}")
        elif args.command == "train":
            metrics = train_model(args.dataset, args.output, args.ridge, args.seed)
            print(f"Модель сохранена: {args.output}")
            print(f"Обучающих примеров: {metrics['sample_count']}; признаков: {metrics['feature_count']}")
            print(f"In-sample MAE: {metrics['mae']:.6f}; RMSE: {metrics['rmse']:.6f}; R²: {metrics['r2']}")
            validation = metrics.get("validation_metrics")
            if validation:
                print(f"Holdout MAE: {validation['mae']:.6f}; RMSE: {validation['rmse']:.6f}; R²: {validation['r2']}")
            else:
                print(metrics.get("validation_note", "Holdout-валидация не выполнена"))
        else:
            metrics = train_aceob_model(
                args.dataset, args.output, limit=args.limit, epochs=args.epochs,
                learning_rate=args.learning_rate, l2=args.l2,
            )
            print(f"ACEOB-модель сохранена: {args.output}")
            print(f"Пар всего: {metrics['pair_count']}; валидных: {metrics['valid_pair_count']}; "
                  f"невалидных пар: {metrics['invalid_pair_count']}")
            print(f"Признаков: {metrics['feature_count']}; train pairwise accuracy: "
                  f"{metrics['pairwise_accuracy_train']:.3f}")
            print("Важно: train accuracy не заменяет отдельную проверку на отложенных задачах.")
    except (OSError, UnicodeError, ValueError, RuntimeError, TimeoutError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
