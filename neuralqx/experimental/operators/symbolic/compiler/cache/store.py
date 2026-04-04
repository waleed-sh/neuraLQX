# Copyright (c) 2026 The neuraLQX Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Artifact cache stores for the symbolic compiler."""

from __future__ import annotations

import abc
import threading

from neuralqx.experimental.operators.symbolic.compiler.core.artifact import (
    SymbolicCompiledArtifact,
)
from neuralqx.experimental.operators.symbolic.compiler.core.signature import (
    SymbolicCacheKey,
)


class AbstractSymbolicArtifactStore(abc.ABC):
    """
    Abstract interface for compiled symbolic operator artifact stores.

    Stores are keyed by :class:`~neuralqx.experimental.operators.symbolic.compiler.core.signature.SymbolicCacheKey`
    (a namespace + SHA-256 token pair) and return
    :class:`~neuralqx.experimental.operators.symbolic.compiler.core.artifact.SymbolicCompiledArtifact`
    instances.
    """

    @abc.abstractmethod
    def get(self, key: SymbolicCacheKey) -> SymbolicCompiledArtifact | None:
        """
        Returns the cached artifact for *key*, or ``None`` on a miss.

        Args:
            key: Cache lookup key.

        Returns:
            Cached artifact, or ``None``.
        """

    @abc.abstractmethod
    def put(self, key: SymbolicCacheKey, artifact: SymbolicCompiledArtifact) -> None:
        """
        Stores *artifact* under *key*.

        Implementations may evict entries or silently ignore duplicates.

        Args:
            key: Cache storage key.
            artifact: Compiled artifact to store.
        """

    @abc.abstractmethod
    def invalidate(self, key: SymbolicCacheKey) -> bool:
        """
        Removes *key* from the store.

        Args:
            key: Key to remove.

        Returns:
            ``True`` if the key was present and removed, ``False`` otherwise.
        """

    @abc.abstractmethod
    def clear(self) -> None:
        """Removes all entries from this store."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the number of entries in the store."""


class InMemorySymbolicArtifactStore(AbstractSymbolicArtifactStore):
    """
    Thread-safe in-process artifact store backed by a Python dict.

    This is the default store used by
    :class:`~neuralqx.experimental.operators.symbolic.compiler.compiler.SymbolicCompiler`.
    It lives for the duration of the Python process and provides O(1) lookup
    keyed by the SHA-256 cache token.

    Args:
        max_entries: Optional soft upper bound on the number of entries.
            When exceeded the oldest entry (insertion order) is evicted.
            ``None`` (default) means no limit.
    """

    def __init__(self, max_entries: int | None = None) -> None:
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be a positive integer or None.")
        self._max_entries = max_entries
        self._store: dict[str, SymbolicCompiledArtifact] = {}
        self._lock = threading.Lock()

    def _composite_key(self, key: SymbolicCacheKey) -> str:
        """Internal storage key: ``namespace/token``."""
        return f"{key.namespace}/{key.token}"

    def get(self, key: SymbolicCacheKey) -> SymbolicCompiledArtifact | None:
        ck = self._composite_key(key)
        with self._lock:
            return self._store.get(ck)

    def put(self, key: SymbolicCacheKey, artifact: SymbolicCompiledArtifact) -> None:
        ck = self._composite_key(key)
        with self._lock:
            if self._max_entries is not None and len(self._store) >= self._max_entries:
                # Evict oldest entry (dict preserves insertion order in Python 3.7+)
                oldest = next(iter(self._store))
                del self._store[oldest]
            self._store[ck] = artifact

    def invalidate(self, key: SymbolicCacheKey) -> bool:
        ck = self._composite_key(key)
        with self._lock:
            if ck in self._store:
                del self._store[ck]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        with self._lock:
            count = len(self._store)
        return (
            f"InMemorySymbolicArtifactStore("
            f"entries={count}, "
            f"max_entries={self._max_entries!r})"
        )


__all__ = [
    "AbstractSymbolicArtifactStore",
    "InMemorySymbolicArtifactStore",
]
