"""Tests for semantic partition — learned UI patterns."""
import time
import pytest
from jarvis_prime.knowledge.semantic_partition import SemanticPartition


@pytest.fixture
def partition():
    return SemanticPartition()


class TestBasicCRUD:
    def test_write_and_read(self, partition):
        partition.write("kg://semantic/pattern/gmail-compose", {
            "app": "Gmail",
            "element": "compose button",
            "typical_position": (520, 190),
            "success_rate": 0.97,
        })
        result = partition.read("kg://semantic/pattern/gmail-compose")
        assert result is not None
        assert result["app"] == "Gmail"
        assert result["typical_position"] == (520, 190)

    def test_read_miss(self, partition):
        assert partition.read("kg://semantic/pattern/nonexistent") is None

    def test_update_overwrites(self, partition):
        partition.write("kg://semantic/pattern/test", {"version": 1})
        partition.write("kg://semantic/pattern/test", {"version": 2})
        result = partition.read("kg://semantic/pattern/test")
        assert result["version"] == 2


class TestTTL:
    def test_default_ttl_24h(self, partition):
        partition.write("kg://semantic/pattern/test", {"data": True})
        # Should still be readable immediately
        assert partition.read("kg://semantic/pattern/test") is not None

    def test_custom_ttl_expiry(self, partition):
        partition.write("kg://semantic/pattern/short", {"data": True}, ttl_seconds=0.01)
        time.sleep(0.02)
        assert partition.read("kg://semantic/pattern/short") is None


class TestSearch:
    def test_search_by_app(self, partition):
        partition.write("kg://semantic/pattern/gmail-compose", {"app": "Gmail", "element": "compose"})
        partition.write("kg://semantic/pattern/gmail-inbox", {"app": "Gmail", "element": "inbox"})
        partition.write("kg://semantic/pattern/slack-message", {"app": "Slack", "element": "message"})

        results = partition.search(app="Gmail")
        assert len(results) == 2

    def test_search_by_element(self, partition):
        partition.write("kg://semantic/pattern/gmail-compose", {"app": "Gmail", "element": "compose button"})
        partition.write("kg://semantic/pattern/slack-compose", {"app": "Slack", "element": "compose message"})

        results = partition.search(element_contains="compose")
        assert len(results) == 2

    def test_search_no_results(self, partition):
        results = partition.search(app="NonexistentApp")
        assert len(results) == 0


class TestClear:
    def test_clear_all(self, partition):
        partition.write("kg://semantic/pattern/a", {"data": 1})
        partition.write("kg://semantic/pattern/b", {"data": 2})
        partition.clear()
        assert partition.read("kg://semantic/pattern/a") is None
        assert partition.read("kg://semantic/pattern/b") is None


class TestStats:
    def test_entry_count(self, partition):
        partition.write("kg://semantic/pattern/a", {"data": 1})
        partition.write("kg://semantic/pattern/b", {"data": 2})
        assert partition.entry_count() == 2
