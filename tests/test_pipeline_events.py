# tests/test_pipeline_events.py
"""Tests for the pipeline event logger in reactor_core_watcher (v1.1).

Verifies that emit_pipeline_event writes structured JSONL events
with correct fields and handles errors gracefully.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def events_dir(tmp_path):
    """Provide a temporary directory for pipeline events."""
    events_path = tmp_path / "events"
    events_path.mkdir()
    return events_path


@pytest.fixture
def emit_fn(events_dir):
    """Return emit_pipeline_event patched to use the temp directory."""
    with patch("jarvis_prime.docker.reactor_core_watcher._PIPELINE_EVENTS_DIR", events_dir), \
         patch("jarvis_prime.docker.reactor_core_watcher._PIPELINE_EVENTS_FILE", events_dir / "pipeline_events.jsonl"):
        from jarvis_prime.docker.reactor_core_watcher import emit_pipeline_event
        yield emit_pipeline_event


# =========================================================================
# Tests
# =========================================================================

class TestEmitPipelineEvent:
    """Verify emit_pipeline_event writes correct JSONL events."""

    def test_writes_event_to_file(self, emit_fn, events_dir):
        """Should write a JSONL line to the events file."""
        event_id = emit_fn(
            topic="model.deployed",
            payload={"model_id": "test-model-v1"},
            correlation_id="job-123",
        )

        assert event_id is not None
        events_file = events_dir / "pipeline_events.jsonl"
        assert events_file.exists()

        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 1

        event = json.loads(lines[0])
        assert event["topic"] == "model.deployed"
        assert event["source"] == "prime"
        assert event["correlation_id"] == "job-123"
        assert event["payload"]["model_id"] == "test-model-v1"
        assert event["event_id"] == event_id

    def test_multiple_events_append(self, emit_fn, events_dir):
        """Multiple calls should append lines to the same file."""
        emit_fn(topic="probation.started", payload={"model_id": "m1"})
        emit_fn(topic="probation.committed", payload={"model_id": "m1"})
        emit_fn(topic="probation.rollback", payload={"model_id": "m2"})

        events_file = events_dir / "pipeline_events.jsonl"
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) == 3

        topics = [json.loads(line)["topic"] for line in lines]
        assert topics == ["probation.started", "probation.committed", "probation.rollback"]

    def test_event_has_required_fields(self, emit_fn, events_dir):
        """Each event should have all required fields."""
        emit_fn(
            topic="model.deployed",
            payload={"model_id": "m1"},
            correlation_id="corr-1",
            causation_id="cause-1",
        )

        events_file = events_dir / "pipeline_events.jsonl"
        event = json.loads(events_file.read_text().strip())

        required_fields = ["event_id", "topic", "source", "timestamp",
                           "correlation_id", "causation_id", "payload"]
        for field in required_fields:
            assert field in event, f"Missing field: {field}"

        assert event["causation_id"] == "cause-1"

    def test_returns_event_id(self, emit_fn):
        """Should return a non-empty event_id string."""
        event_id = emit_fn(topic="probation.started")
        assert event_id is not None
        assert len(event_id) > 0

    def test_default_correlation_id_is_event_id(self, emit_fn, events_dir):
        """When no correlation_id is provided, it should default to event_id."""
        event_id = emit_fn(topic="model.deployed")

        events_file = events_dir / "pipeline_events.jsonl"
        event = json.loads(events_file.read_text().strip())
        assert event["correlation_id"] == event_id

    def test_survives_write_error(self, events_dir):
        """Should return None (not crash) when write fails."""
        bad_dir = Path("/nonexistent/path/that/does/not/exist")
        with patch("jarvis_prime.docker.reactor_core_watcher._PIPELINE_EVENTS_DIR", bad_dir), \
             patch("jarvis_prime.docker.reactor_core_watcher._PIPELINE_EVENTS_FILE", bad_dir / "events.jsonl"):
            from jarvis_prime.docker.reactor_core_watcher import emit_pipeline_event
            result = emit_pipeline_event(topic="model.deployed")
            # Should not crash, returns None
            assert result is None

    def test_source_is_prime(self, emit_fn, events_dir):
        """Events from jarvis-prime should have source='prime'."""
        emit_fn(topic="probation.committed", payload={"model_id": "m1"})

        events_file = events_dir / "pipeline_events.jsonl"
        event = json.loads(events_file.read_text().strip())
        assert event["source"] == "prime"
