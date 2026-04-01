from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any


class LRUCache:
    def __init__(self, maxsize: int = 256) -> None:
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._maxsize = maxsize

    @staticmethod
    def _key(query: str, mode: str) -> str:
        raw = f"{mode}::{query}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str, mode: str) -> Any | None:
        key = self._key(query, mode)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, query: str, mode: str, value: Any) -> None:
        key = self._key(query, mode)
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
