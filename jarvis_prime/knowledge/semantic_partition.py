"""
Semantic Partition — L2 learned UI patterns for the Knowledge Fabric.

Stores app-specific UI patterns with 24h TTL. In-memory dict storage
with keyword search. Optional ChromaDB vector search (graceful fallback
to keyword match when unavailable).

Global IDs: kg://semantic/pattern/{name}

Default TTL is 86400 seconds (24h), configurable via the environment
variable ``SEMANTIC_PARTITION_TTL_S``.  All expiry timestamps use
``time.monotonic()`` for drift-free measurement.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional


_DEFAULT_TTL_S: float = float(os.environ.get("SEMANTIC_PARTITION_TTL_S", 86400))


class SemanticPartition:
    """In-memory L2 partition for learned UI patterns.

    Entities are stored with an expiry timestamp derived from their TTL.
    Reads and searches check expiry and prune stale entries lazily, so no
    background sweep thread is needed.

    Parameters
    ----------
    default_ttl_seconds:
        Seconds until a written entry expires when the caller does not
        supply an explicit TTL.  Defaults to ``SEMANTIC_PARTITION_TTL_S``
        env var, falling back to 86400 (24 h).
    """

    def __init__(self, default_ttl_seconds: Optional[float] = None) -> None:
        self._default_ttl: float = (
            default_ttl_seconds if default_ttl_seconds is not None else _DEFAULT_TTL_S
        )
        # Payload store: entity_id -> data dict (shallow copy on write)
        self._store: Dict[str, dict] = {}
        # Expiry timestamps: entity_id -> monotonic expiry time (seconds)
        self._expiry: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(
        self,
        entity_id: str,
        data: dict,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        """Store *data* under *entity_id* with an optional custom TTL.

        If the entity already exists its data and expiry are both
        refreshed (upsert semantics).

        Parameters
        ----------
        entity_id:
            Full ``kg://`` entity ID string (e.g.
            ``"kg://semantic/pattern/gmail-compose"``).
        data:
            Arbitrary dict payload to store.  A shallow copy is kept so
            the caller's original dict cannot mutate stored state.
        ttl_seconds:
            Seconds until this entry expires.  ``None`` uses the
            partition-level default (``_default_ttl``).
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        self._store[entity_id] = dict(data)
        self._expiry[entity_id] = time.monotonic() + ttl

    def read(self, entity_id: str) -> Optional[dict]:
        """Return stored data for *entity_id*, or ``None`` on miss/expiry.

        Expired entries are pruned on access (lazy eviction).

        Parameters
        ----------
        entity_id:
            Full ``kg://`` entity ID string.
        """
        if entity_id not in self._store:
            return None
        if time.monotonic() >= self._expiry[entity_id]:
            del self._store[entity_id]
            del self._expiry[entity_id]
            return None
        return self._store[entity_id]

    def search(
        self,
        app: Optional[str] = None,
        element_contains: Optional[str] = None,
    ) -> List[dict]:
        """Return all live entries matching the given keyword filters.

        Expired entries encountered during the scan are lazily pruned.
        At least one filter keyword must be supplied; passing neither
        raises ``ValueError``.

        Parameters
        ----------
        app:
            Exact match against the entry's ``"app"`` field.
        element_contains:
            Case-insensitive substring match against the entry's
            ``"element"`` field.

        Returns
        -------
        list[dict]
            Shallow copies of all matching, non-expired data dicts.
        """
        if app is None and element_contains is None:
            raise ValueError("search() requires at least one filter keyword.")

        now = time.monotonic()
        results: List[dict] = []
        expired_keys: List[str] = []

        for eid, data in self._store.items():
            if now >= self._expiry[eid]:
                expired_keys.append(eid)
                continue

            # Apply app filter
            if app is not None and data.get("app") != app:
                continue

            # Apply element substring filter
            if element_contains is not None:
                element_value = data.get("element", "")
                if element_contains.lower() not in element_value.lower():
                    continue

            results.append(dict(data))

        # Lazy eviction of entries found expired during scan
        for eid in expired_keys:
            self._store.pop(eid, None)
            self._expiry.pop(eid, None)

        return results

    def clear(self) -> None:
        """Remove all entries from the partition (including expired ones)."""
        self._store.clear()
        self._expiry.clear()

    def entry_count(self) -> int:
        """Return the number of live (non-expired) entries.

        This is an O(n) scan; prefer for debugging and tests rather than
        hot paths.
        """
        now = time.monotonic()
        return sum(1 for eid in self._store if now < self._expiry[eid])

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Synonym for :py:meth:`entry_count`."""
        return self.entry_count()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SemanticPartition("
            f"live={self.entry_count()}, "
            f"total_slots={len(self._store)}, "
            f"default_ttl={self._default_ttl}s)"
        )
