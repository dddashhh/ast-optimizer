class TransformError(Exception):
    """Базовая ошибка преобразования AST."""


class UnsupportedOptimizationError(TransformError):
    """Для данного вида оптимизации нет правила."""


class StaleCandidateError(TransformError):
    """Кандидат не соответствует дереву (дерево изменилось после анализа)."""
