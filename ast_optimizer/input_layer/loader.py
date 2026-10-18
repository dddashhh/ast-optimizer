from __future__ import annotations

from pathlib import Path
from typing import Union

from .exceptions import InvalidSourceError
from .interfaces import ISourceLoader
from .models import SourceCode


class SourceLoader(ISourceLoader):
    def from_string(self, text: Union[str, bytes], filename: str = "<input>") -> SourceCode:
        if not isinstance(filename, str) or "\x00" in filename:
            raise InvalidSourceError("Недопустимое имя файла")
        if isinstance(text, bytes):
            text = self._decode(text)
        if not isinstance(text, str):
            raise InvalidSourceError(
                f"Ожидалась строка с кодом, получен {type(text).__name__}")
        return SourceCode(text=text, filename=filename)

    def from_file(self, path: Union[str, Path]) -> SourceCode:
        p = self._to_path(path)
        try:
            is_file = p.is_file()
        except OSError as exc:
            raise InvalidSourceError(f"Файл недоступен: {p} ({exc})")
        if not is_file:
            raise InvalidSourceError(f"Файл не найден: {p}")
        if p.suffix != ".py":
            raise InvalidSourceError(
                f"Ожидался файл .py, получен: {p.suffix or 'без расширения'}")
        try:
            data = p.read_bytes()
        except OSError as exc:
            raise InvalidSourceError(f"Не удалось прочитать файл {p}: {exc}")
        return SourceCode(text=self._decode(data), filename=str(p))

    @staticmethod
    def _to_path(path: Union[str, Path]) -> Path:
        try:
            return Path(path)
        except TypeError as exc:
            raise InvalidSourceError(f"Недопустимый путь: {path!r}")

    @staticmethod
    def _decode(data: bytes) -> str:
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InvalidSourceError(f"Код не в кодировке UTF-8: {exc}")
