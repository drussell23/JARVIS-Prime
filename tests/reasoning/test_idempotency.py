"""Tests for SQLite-backed idempotency store."""
import pytest
from jarvis_prime.reasoning.idempotency_store import IdempotencyStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_idempotency.db"
    return IdempotencyStore(db_path=str(db_path))


class TestIdempotencyStore:
    def test_store_and_retrieve(self, store):
        store.store("req-001", "sess-001", '{"status": "plan_ready"}')
        cached = store.get("req-001")
        assert cached is not None
        assert '"plan_ready"' in cached

    def test_miss_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_duplicate_returns_first(self, store):
        store.store("req-002", "sess-001", '{"first": true}')
        store.store("req-002", "sess-001", '{"second": true}')
        cached = store.get("req-002")
        assert '"first"' in cached  # first write wins (INSERT OR IGNORE)

    def test_prune_removes_old(self, store):
        store.store("req-old", "sess-001", '{"old": true}')
        store._conn.execute(
            "UPDATE idempotency SET created_at = datetime('now', '-25 hours') WHERE request_id = 'req-old'"
        )
        store._conn.commit()
        store.prune(window_hours=24)
        assert store.get("req-old") is None

    def test_prune_keeps_recent(self, store):
        store.store("req-new", "sess-001", '{"new": true}')
        store.prune(window_hours=24)
        assert store.get("req-new") is not None

    def test_max_entries_eviction(self, store):
        store._max_entries = 5
        for i in range(10):
            store.store(f"req-{i:03d}", "sess", f'{{"i": {i}}}')
        count = store._conn.execute("SELECT COUNT(*) FROM idempotency").fetchone()[0]
        assert count <= 6  # 5 + some slack from eviction timing

    def test_persists_across_instances(self, tmp_path):
        db_path = str(tmp_path / "persist.db")
        store1 = IdempotencyStore(db_path=db_path)
        store1.store("req-persist", "sess", '{"data": 1}')
        store1.close()

        store2 = IdempotencyStore(db_path=db_path)
        assert store2.get("req-persist") is not None
        store2.close()

    def test_close_and_reopen(self, tmp_path):
        db_path = str(tmp_path / "close_test.db")
        store = IdempotencyStore(db_path=db_path)
        store.store("req-x", "sess", '{"x": 1}')
        store.close()
        # Reopen should work
        store2 = IdempotencyStore(db_path=db_path)
        assert store2.get("req-x") is not None
        store2.close()
