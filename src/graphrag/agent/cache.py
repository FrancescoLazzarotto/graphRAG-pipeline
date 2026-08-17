from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict
from typing import Any


class LRUCache:
    """Retrieval-result cache that hands out copies, never its own objects.

    The cached value is a dict of retrieved nodes and triples that the caller
    merges into LangGraph state, where downstream nodes are free to mutate it.
    Storing and returning it by reference meant one in-place edit poisoned every
    later turn that hit the same key. See docs/code_audit_2026-08-15.md §1.11.
    """

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
            return copy.deepcopy(self._cache[key])
        return None

    def put(self, query: str, mode: str, value: Any) -> None:
        key = self._key(query, mode)
        self._cache[key] = copy.deepcopy(value)
        self._cache.move_to_end(key)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
