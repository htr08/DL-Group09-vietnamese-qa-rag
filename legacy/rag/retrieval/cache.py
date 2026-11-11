"""Simple in-memory TTL cache for retrieval results."""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from typing import Any, Hashable, Optional


class RetrievalCache:
    """LRU cache with TTL, storing deep-copiable values."""

    def __init__(self, max_size: int = 256, ttl_seconds: int = 120) -> None:
        self.max_size = max(1, int(max_size))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._store: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()

    def get(self, key: Hashable) -> Optional[Any]:
        self._prune_expired()
        value = self._store.get(key)
        if value is None:
            return None
        expires_at, payload = value
        if expires_at <= time.time():
            self._store.pop(key, None)
            return None
        self._store.move_to_end(key)
        return copy.deepcopy(payload)

    def set(self, key: Hashable, payload: Any) -> None:
        self._prune_expired()
        self._store[key] = (time.time() + self.ttl_seconds, copy.deepcopy(payload))
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def _prune_expired(self) -> None:
        now = time.time()
        to_remove = [key for key, (expires_at, _) in self._store.items() if expires_at <= now]
        for key in to_remove:
            self._store.pop(key, None)

