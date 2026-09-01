from __future__ import annotations

import copy
import hashlib
import threading
from collections import OrderedDict
from typing import Any


class LRUCache:
    """Retrieval-result cache that hands out copies, never its own objects.

    The cached value is a dict of retrieved nodes and triples that the caller
    merges into LangGraph state, where downstream nodes are free to mutate it.
    Storing and returning it by reference meant one in-place edit poisoned every
    later turn that hit the same key. See docs/code_audit_2026-08-15.md §1.11.

    Locked, because in the Streamlit demo one agent — and so one of these — is
    shared by every browser session through `@st.cache_resource`, and Streamlit
    runs each session in its own thread. `OrderedDict` is not safe against
    that: `get` used to look a key up, move it to the end and then read it,
    which raises `KeyError` if another thread evicted it in between, and two
    concurrent `put` calls could each see the size over the limit and evict
    twice. The content is public to every user of the demo, so nothing leaked;
    what it cost was a spurious failed turn for whoever lost the race.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._maxsize = maxsize
        # Held across the copy too: releasing it earlier would let an eviction
        # land between the lookup and the read, which is the race itself.
        self._lock = threading.Lock()

    @staticmethod
    def _key(query: str, mode: str) -> str:
        raw = f"{mode}::{query}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, query: str, mode: str) -> Any | None:
        key = self._key(query, mode)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return copy.deepcopy(self._cache[key])
            return None

    def put(self, query: str, mode: str, value: Any) -> None:
        key = self._key(query, mode)
        # Copied before the lock: the value belongs to the caller and nothing
        # else can reach it yet, so the copy needs no protection and the lock
        # is held only for the dictionary work.
        stored = copy.deepcopy(value)
        with self._lock:
            self._cache[key] = stored
            self._cache.move_to_end(key)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
