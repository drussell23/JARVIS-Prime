"""Tests for deployment feedback writing after model deployment.

Validates that write_deployment_feedback() correctly writes
deployment_status.json to the cross-repo directory after
Prime deploys (or fails to deploy) a GGUF model from Reactor-Core.
"""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch


@pytest.fixture
def tmp_cross_repo(tmp_path):
    """Temporary cross-repo directory."""
    cross_repo = tmp_path / "cross_repo"
    cross_repo.mkdir()
    return cross_repo


class TestWriteDeploymentFeedback:
    """Verify write_deployment_feedback writes correct JSON to disk."""

    def test_write_deployment_status_on_success(self, tmp_cross_repo):
        """Successful deployment should write deployment_status.json with all fields."""
        from jarvis_prime.docker.reactor_core_watcher import write_deployment_feedback

        write_deployment_feedback(
            cross_repo_dir=tmp_cross_repo,
            model_id="qwen2.5-coder-7b-v3",
            status="success",
            previous_model="qwen2.5-coder-7b-v2",
            reactor_job_id="job-123",
            first_inference_latency_ms=3200.0,
        )

        status_file = tmp_cross_repo / "deployment_status.json"
        assert status_file.exists(), "deployment_status.json should be created"
        data = json.loads(status_file.read_text())
        assert data["model_id"] == "qwen2.5-coder-7b-v3"
        assert data["deployment_status"] == "success"
        assert data["previous_model"] == "qwen2.5-coder-7b-v2"
        assert data["reactor_job_id"] == "job-123"
        assert data["health_check_passed"] is True
        assert data["first_inference_latency_ms"] == 3200.0
        assert "deployed_at" in data
        assert data["error"] is None

    def test_write_deployment_status_on_failure(self, tmp_cross_repo):
        """Failed deployment should write deployment_status.json with error details."""
        from jarvis_prime.docker.reactor_core_watcher import write_deployment_feedback

        write_deployment_feedback(
            cross_repo_dir=tmp_cross_repo,
            model_id="qwen2.5-coder-7b-v3",
            status="failed",
            previous_model="qwen2.5-coder-7b-v2",
            error="GGUF header invalid",
        )

        status_file = tmp_cross_repo / "deployment_status.json"
        data = json.loads(status_file.read_text())
        assert data["deployment_status"] == "failed"
        assert data["error"] == "GGUF header invalid"
        assert data["health_check_passed"] is False
        assert data["first_inference_latency_ms"] is None

    def test_write_deployment_status_on_rollback(self, tmp_cross_repo):
        """Rollback deployment should write status='rollback'."""
        from jarvis_prime.docker.reactor_core_watcher import write_deployment_feedback

        write_deployment_feedback(
            cross_repo_dir=tmp_cross_repo,
            model_id="qwen2.5-coder-7b-v3",
            status="rollback",
            previous_model="qwen2.5-coder-7b-v2",
            error="Health check failed after hot-swap",
        )

        status_file = tmp_cross_repo / "deployment_status.json"
        data = json.loads(status_file.read_text())
        assert data["deployment_status"] == "rollback"
        assert data["health_check_passed"] is False
        assert "Health check failed" in data["error"]

    def test_atomic_write_no_partial_file(self, tmp_cross_repo):
        """Write should be atomic - no partial deployment_status.json visible."""
        from jarvis_prime.docker.reactor_core_watcher import write_deployment_feedback

        write_deployment_feedback(
            cross_repo_dir=tmp_cross_repo,
            model_id="test-model",
            status="success",
        )

        # Temp file should not remain
        tmp_file = tmp_cross_repo / "deployment_status.tmp"
        assert not tmp_file.exists(), "Temp file should be cleaned up after atomic write"

        # Final file should be valid JSON
        status_file = tmp_cross_repo / "deployment_status.json"
        data = json.loads(status_file.read_text())
        assert data["model_id"] == "test-model"

    def test_overwrites_previous_status(self, tmp_cross_repo):
        """Subsequent writes should overwrite the previous deployment_status.json."""
        from jarvis_prime.docker.reactor_core_watcher import write_deployment_feedback

        # First write
        write_deployment_feedback(
            cross_repo_dir=tmp_cross_repo,
            model_id="model-v1",
            status="success",
        )

        # Second write
        write_deployment_feedback(
            cross_repo_dir=tmp_cross_repo,
            model_id="model-v2",
            status="failed",
            error="Out of memory",
        )

        status_file = tmp_cross_repo / "deployment_status.json"
        data = json.loads(status_file.read_text())
        assert data["model_id"] == "model-v2"
        assert data["deployment_status"] == "failed"

    def test_creates_cross_repo_dir_if_missing(self, tmp_path):
        """Should create the cross_repo_dir if it doesn't exist yet."""
        from jarvis_prime.docker.reactor_core_watcher import write_deployment_feedback

        nonexistent = tmp_path / "deep" / "nested" / "cross_repo"
        assert not nonexistent.exists()

        write_deployment_feedback(
            cross_repo_dir=nonexistent,
            model_id="test-model",
            status="success",
        )

        assert nonexistent.exists()
        status_file = nonexistent / "deployment_status.json"
        assert status_file.exists()

    def test_deployed_at_is_iso_format(self, tmp_cross_repo):
        """deployed_at should be a valid ISO 8601 datetime string."""
        from jarvis_prime.docker.reactor_core_watcher import write_deployment_feedback
        from datetime import datetime

        write_deployment_feedback(
            cross_repo_dir=tmp_cross_repo,
            model_id="test-model",
            status="success",
        )

        status_file = tmp_cross_repo / "deployment_status.json"
        data = json.loads(status_file.read_text())
        # Should not raise
        parsed = datetime.fromisoformat(data["deployed_at"])
        assert parsed.year >= 2026

    def test_minimal_call_with_defaults(self, tmp_cross_repo):
        """Calling with only required args should still produce valid JSON."""
        from jarvis_prime.docker.reactor_core_watcher import write_deployment_feedback

        write_deployment_feedback(
            cross_repo_dir=tmp_cross_repo,
            model_id="minimal-model",
            status="success",
        )

        status_file = tmp_cross_repo / "deployment_status.json"
        data = json.loads(status_file.read_text())
        assert data["model_id"] == "minimal-model"
        assert data["deployment_status"] == "success"
        assert data["previous_model"] is None
        assert data["reactor_job_id"] is None
        assert data["first_inference_latency_ms"] is None
        assert data["error"] is None
        assert data["health_check_passed"] is True


class TestDeploymentFeedbackIntegration:
    """Verify _deploy_model and _handle_failed_deployment call write_deployment_feedback."""

    def test_deploy_model_calls_feedback_on_success(self, tmp_cross_repo, tmp_path):
        """_deploy_model should call write_deployment_feedback on success."""
        from jarvis_prime.docker.reactor_core_watcher import ReactorCoreWatcher

        watch_dir = tmp_path / "watch"
        models_dir = tmp_path / "models"
        watch_dir.mkdir()
        models_dir.mkdir()

        watcher = ReactorCoreWatcher(
            watch_dir=watch_dir,
            models_dir=models_dir,
            cross_repo_dir=tmp_cross_repo,
        )

        # Create a fake GGUF file in pending
        pending_dir = watch_dir / "pending"
        model_file = pending_dir / "test-model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 2048)

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            watcher._deploy_model(model_file, None)
        )

        status_file = tmp_cross_repo / "deployment_status.json"
        assert status_file.exists(), "deployment feedback should be written on success"
        data = json.loads(status_file.read_text())
        assert data["deployment_status"] == "success"
        assert data["model_id"] == "test-model"

    def test_handle_failed_deployment_calls_feedback(self, tmp_cross_repo, tmp_path):
        """_handle_failed_deployment should call write_deployment_feedback on failure."""
        from jarvis_prime.docker.reactor_core_watcher import ReactorCoreWatcher

        watch_dir = tmp_path / "watch"
        models_dir = tmp_path / "models"
        watch_dir.mkdir()
        models_dir.mkdir()

        watcher = ReactorCoreWatcher(
            watch_dir=watch_dir,
            models_dir=models_dir,
            cross_repo_dir=tmp_cross_repo,
        )

        pending_dir = watch_dir / "pending"
        model_file = pending_dir / "bad-model.gguf"
        model_file.write_bytes(b"GGUF" + b"\x00" * 2048)

        import asyncio
        asyncio.get_event_loop().run_until_complete(
            watcher._handle_failed_deployment(model_file, None, "Checksum mismatch")
        )

        status_file = tmp_cross_repo / "deployment_status.json"
        assert status_file.exists(), "deployment feedback should be written on failure"
        data = json.loads(status_file.read_text())
        assert data["deployment_status"] == "failed"
        assert data["error"] == "Checksum mismatch"


class TestCrossRepoDirResolution:
    """Verify cross_repo_dir is resolved from env or default."""

    def test_default_cross_repo_dir(self, tmp_path):
        """Without env var, cross_repo_dir defaults to ~/.jarvis/cross_repo."""
        from jarvis_prime.docker.reactor_core_watcher import resolve_cross_repo_dir

        with patch.dict("os.environ", {}, clear=False):
            # Remove JARVIS_CROSS_REPO_DIR if set
            import os
            env_copy = os.environ.copy()
            env_copy.pop("JARVIS_CROSS_REPO_DIR", None)
            with patch.dict("os.environ", env_copy, clear=True):
                result = resolve_cross_repo_dir()
                assert result == Path.home() / ".jarvis" / "cross_repo"

    def test_env_var_overrides_default(self, tmp_path):
        """JARVIS_CROSS_REPO_DIR env var should override default."""
        from jarvis_prime.docker.reactor_core_watcher import resolve_cross_repo_dir

        custom_dir = str(tmp_path / "custom_cross_repo")
        with patch.dict("os.environ", {"JARVIS_CROSS_REPO_DIR": custom_dir}):
            result = resolve_cross_repo_dir()
            assert result == Path(custom_dir)
