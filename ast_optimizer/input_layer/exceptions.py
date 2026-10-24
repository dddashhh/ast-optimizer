from typing import Optional


class InvalidSourceError(Exception):
    """Базовая ошибка исходного кода."""


class EmptySourceError(InvalidSourceError):
    """Код пустой или состоит только из пробелов."""


class SourceTooLargeError(InvalidSourceError):
    """Размер кода превышает предел."""


class PythonSyntaxError(InvalidSourceError):

    def __init__(self, message: str, lineno: Optional[int] = None,
                 offset: Optional[int] = None, text: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.lineno = lineno
        self.offset = offset
        self.text = text

    def __str__(self) -> str:
        if not self.lineno:
            return self.message
        if self.offset is None:
            return f"{self.message} (строка {self.lineno})"
        return f"{self.message} (строка {self.lineno}, столбец {self.offset})"
