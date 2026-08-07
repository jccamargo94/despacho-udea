from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator


class LocalStorage:
    """Storage backed by the local filesystem, rooted at `root`."""

    def __init__(self, root: str):
        self.root = Path(root)

    def _resolve(self, path: str) -> Path:
        return self.root / path

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    @contextmanager
    def open(self, path: str, mode: str = "r", encoding: str | None = None) -> Iterator[IO]:
        p = self._resolve(path)
        if "w" in mode or "a" in mode:
            p.parent.mkdir(parents=True, exist_ok=True)
        f = open(p, mode, encoding=encoding)
        try:
            yield f
        finally:
            f.close()

    def list_dir(self, path: str) -> list[str]:
        p = self._resolve(path)
        if not p.is_dir():
            return []
        return [entry.name for entry in p.iterdir()]
