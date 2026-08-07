from __future__ import annotations

from typing import IO, ContextManager, Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    def exists(self, path: str) -> bool: ...
    def open(
        self, path: str, mode: str = "r", encoding: str | None = None
    ) -> ContextManager[IO]: ...
    def list_dir(self, path: str) -> list[str]: ...
