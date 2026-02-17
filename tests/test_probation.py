"""Tests for post-deployment probation monitoring.

Validates that ProbationMonitor correctly:
- Starts a probation window after successful deployment
- Probes health at configured intervals
- Commits the deployment when health_score >= commit_threshold
- Rolls back when health_score < rollback_threshold
- Triggers emergency rollback when error rate > 5x baseline
- Persists probation state for crash recovery
- Backs up previous model to ~/.jarvis/reactor/models/previous/
- Updates deployment_status.json on commit/rollback
"""

import asyncio
import json
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from jarvis_prime.docker.reactor_core_watcher import (
    ProbationConfig,
    ProbationMonitor,
    ProbationStatus,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def probation_config():
    """Short-duration config for fast tests."""
    return ProbationConfig(
        probation_duration_s=10.0,
        probe_interval_s=1.0,
        commit_threshold=0.8,
        rollback_threshold=0.5,
        emergency_error_rate_multiplier=5.0,
        min_probes_before_decision=3,
    )


@pytest.fixture
def tmp_dirs(tmp_path):
    """Create temporary directory structure for probation tests."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    previous_dir = tmp_path / "reactor" / "models" / "previous"
    previous_dir.mkdir(parents=True)
    cross_repo_dir = tmp_path / "cross_repo"
    cross_repo_dir.mkdir()
    state_dir = tmp_path / "reactor"
    return {
        "models_dir": models_dir,
        "previous_dir": previous_dir,
        "cross_repo_dir": cross_repo_dir,
        "state_dir": state_dir,
        "tmp_path": tmp_path,
    }


@pytest.fixture
def make_model_file(tmp_dirs):
    """Helper to create a fake GGUF model file."""
    def _make(name: str = "test-model.gguf", content: bytes = b"GGUF" + b"\x00" * 4096) -> Path:
        path = tmp_dirs["models_dir"] / name
        path.write_bytes(content)
        return path
    return _make


@pytest.fixture
def monitor(probation_config, tmp_dirs):
    """Create a ProbationMonitor with test config."""
    return ProbationMonitor(
        config=probation_config,
        health_endpoint="http://localhost:8000/health",
        cross_repo_dir=tmp_dirs["cross_repo_dir"],
        previous_model_dir=tmp_dirs["previous_dir"],
        state_file=tmp_dirs["state_dir"] / "probation_state.json",
    )


# ============================================================================
# ProbationConfig Tests
# ============================================================================

class TestProbationConfig:
    """Test ProbationConfig defaults and customization."""

    def test_default_config(self):
        """Default config should have 30-min duration, 60s probes."""
        cfg = ProbationConfig()
        assert cfg.probation_duration_s == 1800.0
        assert cfg.probe_interval_s == 60.0
        assert cfg.commit_threshold == 0.8
        assert cfg.rollback_threshold == 0.5
        assert cfg.emergency_error_rate_multiplier == 5.0
        assert cfg.min_probes_before_decision == 3

    def test_custom_config(self, probation_config):
        """Custom config should override defaults."""
        assert probation_config.probation_duration_s == 10.0
        assert probation_config.probe_interval_s == 1.0


# ============================================================================
# ProbationStatus Tests
# ============================================================================

class TestProbationStatus:
    """Test ProbationStatus enum values."""

    def test_status_values(self):
        """All expected statuses should exist."""
        assert ProbationStatus.MONITORING.value == "monitoring"
        assert ProbationStatus.COMMITTED.value == "committed"
        assert ProbationStatus.ROLLING_BACK.value == "rolling_back"
        assert ProbationStatus.ROLLED_BACK.value == "rolled_back"


# ============================================================================
# Health Probe Tests
# ============================================================================

class TestHealthProbe:
    """Test ProbationMonitor.probe() health check logic."""

    async def test_probe_healthy_endpoint(self, monitor):
        """Healthy endpoint should return health_score near 1.0."""
        mock_response = {
            "status": "healthy",
            "phase": "ready",
            "ready_for_inference": True,
        }
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            result = await monitor.probe()

        assert result is not None
        assert "health_score" in result
        assert result["health_score"] >= 0.8
        assert result["reachable"] is True

    async def test_probe_unreachable_endpoint(self, monitor):
        """Unreachable endpoint should return low health_score."""
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.get = MagicMock(side_effect=Exception("Connection refused"))
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            result = await monitor.probe()

        assert result is not None
        assert result["health_score"] == 0.0
        assert result["reachable"] is False

    async def test_probe_error_status(self, monitor):
        """Endpoint returning error status should get low health_score."""
        mock_response = {
            "status": "error",
            "error": "Model loading failed",
        }
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_resp = AsyncMock()
            mock_resp.status = 503
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            result = await monitor.probe()

        assert result is not None
        assert result["health_score"] < 0.5

    async def test_probe_slow_response(self, monitor):
        """Slow response should reduce health_score via latency penalty."""
        mock_response = {
            "status": "healthy",
            "phase": "ready",
        }
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)

            mock_session = AsyncMock()
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            # Patch time.monotonic to simulate 5 seconds latency
            with patch("time.monotonic", side_effect=[0.0, 5.0]):
                result = await monitor.probe()

        assert result is not None
        # With 5s latency, health should be penalized
        assert result["health_score"] < 1.0
        assert result["latency_s"] == 5.0


# ============================================================================
# Evaluation Tests
# ============================================================================

class TestEvaluate:
    """Test ProbationMonitor.evaluate() decision logic."""

    def test_evaluate_healthy_probes_commits(self, monitor):
        """Consistently healthy probes should result in COMMITTED."""
        # Simulate 5 healthy probes
        for _ in range(5):
            monitor._probe_history.append({
                "health_score": 0.95,
                "reachable": True,
                "latency_s": 0.05,
                "error_rate": 0.0,
                "timestamp": time.time(),
            })

        decision = monitor.evaluate()
        assert decision == ProbationStatus.COMMITTED

    def test_evaluate_unhealthy_probes_rolls_back(self, monitor):
        """Consistently unhealthy probes should result in ROLLING_BACK."""
        for _ in range(5):
            monitor._probe_history.append({
                "health_score": 0.3,
                "reachable": True,
                "latency_s": 2.0,
                "error_rate": 0.4,
                "timestamp": time.time(),
            })

        decision = monitor.evaluate()
        assert decision == ProbationStatus.ROLLING_BACK

    def test_evaluate_mixed_probes_continues_monitoring(self, monitor):
        """Mixed probes should continue MONITORING."""
        # Some good, some bad
        monitor._probe_history.append({
            "health_score": 0.9, "reachable": True,
            "latency_s": 0.05, "error_rate": 0.0, "timestamp": time.time(),
        })
        monitor._probe_history.append({
            "health_score": 0.4, "reachable": True,
            "latency_s": 1.5, "error_rate": 0.3, "timestamp": time.time(),
        })
        monitor._probe_history.append({
            "health_score": 0.85, "reachable": True,
            "latency_s": 0.1, "error_rate": 0.0, "timestamp": time.time(),
        })

        decision = monitor.evaluate()
        assert decision == ProbationStatus.MONITORING

    def test_evaluate_needs_min_probes(self, monitor):
        """Should not decide with fewer than min_probes_before_decision."""
        # Only 1 probe (min is 3)
        monitor._probe_history.append({
            "health_score": 0.95, "reachable": True,
            "latency_s": 0.05, "error_rate": 0.0, "timestamp": time.time(),
        })

        decision = monitor.evaluate()
        assert decision == ProbationStatus.MONITORING

    def test_evaluate_emergency_rollback(self, monitor):
        """Error rate > 5x baseline should trigger emergency ROLLING_BACK."""
        # Set a baseline error rate
        monitor._baseline_error_rate = 0.01
        # Add probes with 6x baseline error rate
        for _ in range(3):
            monitor._probe_history.append({
                "health_score": 0.6,
                "reachable": True,
                "latency_s": 0.5,
                "error_rate": 0.06,  # 6x the 0.01 baseline
                "timestamp": time.time(),
            })

        decision = monitor.evaluate()
        assert decision == ProbationStatus.ROLLING_BACK


# ============================================================================
# Commit Tests
# ============================================================================

class TestCommit:
    """Test ProbationMonitor.commit() behavior."""

    async def test_commit_updates_status(self, monitor):
        """commit() should set status to COMMITTED."""
        monitor._status = ProbationStatus.MONITORING
        monitor._model_id = "test-model-v1"

        await monitor.commit()

        assert monitor._status == ProbationStatus.COMMITTED

    async def test_commit_writes_deployment_feedback(self, monitor, tmp_dirs):
        """commit() should write deployment_status.json with success."""
        monitor._model_id = "test-model-v1"
        monitor._status = ProbationStatus.MONITORING

        await monitor.commit()

        status_file = tmp_dirs["cross_repo_dir"] / "deployment_status.json"
        assert status_file.exists()
        data = json.loads(status_file.read_text())
        assert data["deployment_status"] == "success"
        assert data["model_id"] == "test-model-v1"
        assert "probation" in data
        assert data["probation"]["result"] == "committed"

    async def test_commit_cleans_up_state_file(self, monitor, tmp_dirs):
        """commit() should remove the probation state file."""
        state_file = tmp_dirs["state_dir"] / "probation_state.json"
        state_file.write_text("{}")

        monitor._model_id = "test-model-v1"
        monitor._status = ProbationStatus.MONITORING

        await monitor.commit()

        assert not state_file.exists()


# ============================================================================
# Rollback Tests
# ============================================================================

class TestRollback:
    """Test ProbationMonitor.rollback() behavior."""

    async def test_rollback_restores_previous_model(self, monitor, tmp_dirs, make_model_file):
        """rollback() should copy previous model back to models dir."""
        # Create current and previous models
        current = make_model_file("current-model.gguf")
        previous = tmp_dirs["previous_dir"] / "previous-model.gguf"
        previous.write_bytes(b"GGUF" + b"\x00" * 2048)

        monitor._model_id = "current-model"
        monitor._deployed_model_path = current
        monitor._previous_model_path = previous
        monitor._status = ProbationStatus.ROLLING_BACK

        await monitor.rollback()

        assert monitor._status == ProbationStatus.ROLLED_BACK

    async def test_rollback_writes_deployment_feedback(self, monitor, tmp_dirs, make_model_file):
        """rollback() should write deployment_status.json with rollback status."""
        previous = tmp_dirs["previous_dir"] / "previous-model.gguf"
        previous.write_bytes(b"GGUF" + b"\x00" * 2048)

        monitor._model_id = "test-model-v2"
        monitor._previous_model_path = previous
        monitor._deployed_model_path = tmp_dirs["models_dir"] / "test-model-v2.gguf"
        monitor._status = ProbationStatus.ROLLING_BACK

        await monitor.rollback()

        status_file = tmp_dirs["cross_repo_dir"] / "deployment_status.json"
        assert status_file.exists()
        data = json.loads(status_file.read_text())
        assert data["deployment_status"] == "rollback"
        assert data["model_id"] == "test-model-v2"

    async def test_rollback_cleans_up_state_file(self, monitor, tmp_dirs):
        """rollback() should remove probation state file after completion."""
        state_file = tmp_dirs["state_dir"] / "probation_state.json"
        state_file.write_text("{}")

        monitor._model_id = "test-model-v1"
        monitor._previous_model_path = None  # No previous model case
        monitor._deployed_model_path = None
        monitor._status = ProbationStatus.ROLLING_BACK

        await monitor.rollback()

        assert not state_file.exists()

    async def test_rollback_without_previous_model(self, monitor, tmp_dirs):
        """rollback() should still succeed even without a previous model to restore."""
        monitor._model_id = "test-model-v1"
        monitor._previous_model_path = None
        monitor._deployed_model_path = None
        monitor._status = ProbationStatus.ROLLING_BACK

        await monitor.rollback()

        assert monitor._status == ProbationStatus.ROLLED_BACK


# ============================================================================
# Model Backup Tests
# ============================================================================

class TestModelBackup:
    """Test previous model backup functionality."""

    async def test_start_backs_up_previous_model(self, monitor, tmp_dirs, make_model_file):
        """start() should copy the previous model to previous/ dir."""
        current = make_model_file("current-model.gguf")
        previous_src = make_model_file("previous-model.gguf", b"GGUF" + b"\x01" * 2048)

        monitor._status = ProbationStatus.MONITORING
        await monitor.backup_previous_model(previous_src)

        expected = tmp_dirs["previous_dir"] / "previous-model.gguf"
        assert expected.exists()
        # Should be a copy, original should still exist
        assert previous_src.exists()

    async def test_backup_creates_directory_if_missing(self, monitor, tmp_dirs, make_model_file):
        """Backup should create the previous/ directory if it doesn't exist."""
        import shutil
        shutil.rmtree(tmp_dirs["previous_dir"])
        assert not tmp_dirs["previous_dir"].exists()

        model = make_model_file("model.gguf")
        await monitor.backup_previous_model(model)

        assert tmp_dirs["previous_dir"].exists()


# ============================================================================
# State Persistence Tests
# ============================================================================

class TestStatePersistence:
    """Test probation state persistence for crash recovery."""

    async def test_persist_state(self, monitor, tmp_dirs):
        """State should be written to probation_state.json."""
        monitor._model_id = "test-model-v1"
        monitor._status = ProbationStatus.MONITORING
        monitor._started_at = time.time()
        monitor._probe_history.append({
            "health_score": 0.9, "reachable": True,
            "latency_s": 0.05, "error_rate": 0.0,
            "timestamp": time.time(),
        })

        await monitor.persist_state()

        state_file = tmp_dirs["state_dir"] / "probation_state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["model_id"] == "test-model-v1"
        assert data["status"] == "monitoring"
        assert len(data["probe_history"]) == 1

    async def test_restore_state(self, monitor, tmp_dirs):
        """ProbationMonitor should be able to restore from persisted state."""
        state = {
            "model_id": "restored-model",
            "status": "monitoring",
            "started_at": time.time() - 300,
            "probe_history": [
                {"health_score": 0.9, "reachable": True,
                 "latency_s": 0.05, "error_rate": 0.0,
                 "timestamp": time.time() - 60},
            ],
            "baseline_error_rate": 0.01,
            "deployed_model_path": str(tmp_dirs["models_dir"] / "restored-model.gguf"),
            "previous_model_path": None,
        }
        state_file = tmp_dirs["state_dir"] / "probation_state.json"
        state_file.write_text(json.dumps(state))

        restored = await monitor.restore_state()

        assert restored is True
        assert monitor._model_id == "restored-model"
        assert monitor._status == ProbationStatus.MONITORING
        assert len(monitor._probe_history) == 1

    async def test_restore_state_no_file(self, monitor, tmp_dirs):
        """restore_state() should return False when no state file exists."""
        # Make sure state file does not exist
        state_file = tmp_dirs["state_dir"] / "probation_state.json"
        if state_file.exists():
            state_file.unlink()

        restored = await monitor.restore_state()
        assert restored is False


# ============================================================================
# Health Score Calculation Tests
# ============================================================================

class TestHealthScoreCalculation:
    """Test the health_score formula."""

    def test_perfect_health(self, monitor):
        """No errors and fast response should score near 1.0."""
        score = monitor.calculate_health_score(
            http_status=200,
            response_data={"status": "healthy", "phase": "ready"},
            latency_s=0.05,
        )
        assert score >= 0.9

    def test_error_status_penalizes(self, monitor):
        """Error HTTP status should heavily penalize the score."""
        score = monitor.calculate_health_score(
            http_status=503,
            response_data={"status": "error"},
            latency_s=0.05,
        )
        assert score < 0.5

    def test_high_latency_penalizes(self, monitor):
        """High latency should reduce score."""
        fast_score = monitor.calculate_health_score(
            http_status=200,
            response_data={"status": "healthy"},
            latency_s=0.05,
        )
        slow_score = monitor.calculate_health_score(
            http_status=200,
            response_data={"status": "healthy"},
            latency_s=10.0,
        )
        assert slow_score < fast_score

    def test_starting_status_reduced_score(self, monitor):
        """'starting' status should get a reduced score compared to 'healthy'."""
        healthy_score = monitor.calculate_health_score(
            http_status=200,
            response_data={"status": "healthy"},
            latency_s=0.1,
        )
        starting_score = monitor.calculate_health_score(
            http_status=200,
            response_data={"status": "starting"},
            latency_s=0.1,
        )
        # Starting should score less than healthy due to status penalty
        assert starting_score < healthy_score
        # But still positive since HTTP 200 and low latency
        assert starting_score > 0.5


# ============================================================================
# Start / Full Lifecycle Tests
# ============================================================================

class TestProbationLifecycle:
    """Test the full probation lifecycle."""

    async def test_start_initializes_state(self, monitor, make_model_file):
        """start() should initialize monitoring state."""
        model = make_model_file("new-model.gguf")

        await monitor.start(
            model_id="new-model",
            deployed_model_path=model,
            previous_model_path=None,
        )

        assert monitor._model_id == "new-model"
        assert monitor._status == ProbationStatus.MONITORING
        assert monitor._started_at is not None
        assert len(monitor._probe_history) == 0

    async def test_start_with_previous_model_backs_up(self, monitor, tmp_dirs, make_model_file):
        """start() with a previous model should back it up."""
        new_model = make_model_file("new-model.gguf")
        old_model = make_model_file("old-model.gguf", b"GGUF" + b"\x02" * 2048)

        await monitor.start(
            model_id="new-model",
            deployed_model_path=new_model,
            previous_model_path=old_model,
        )

        # Previous model should be backed up
        backup = tmp_dirs["previous_dir"] / "old-model.gguf"
        assert backup.exists()
        # Original should still exist (copy, not move)
        assert old_model.exists()

    async def test_full_lifecycle_commit(self, monitor, make_model_file, tmp_dirs):
        """Full lifecycle: start -> probe (healthy) -> evaluate -> commit."""
        model = make_model_file("lifecycle-model.gguf")

        await monitor.start(
            model_id="lifecycle-model",
            deployed_model_path=model,
            previous_model_path=None,
        )

        # Simulate healthy probes
        for _ in range(5):
            monitor._probe_history.append({
                "health_score": 0.92, "reachable": True,
                "latency_s": 0.05, "error_rate": 0.0,
                "timestamp": time.time(),
            })

        decision = monitor.evaluate()
        assert decision == ProbationStatus.COMMITTED

        await monitor.commit()
        assert monitor._status == ProbationStatus.COMMITTED

        # deployment_status.json should reflect success
        status_file = tmp_dirs["cross_repo_dir"] / "deployment_status.json"
        data = json.loads(status_file.read_text())
        assert data["deployment_status"] == "success"

    async def test_full_lifecycle_rollback(self, monitor, make_model_file, tmp_dirs):
        """Full lifecycle: start -> probe (unhealthy) -> evaluate -> rollback."""
        current = make_model_file("bad-model.gguf")
        previous = make_model_file("good-model.gguf", b"GGUF" + b"\x03" * 2048)

        await monitor.start(
            model_id="bad-model",
            deployed_model_path=current,
            previous_model_path=previous,
        )

        # Simulate unhealthy probes
        for _ in range(5):
            monitor._probe_history.append({
                "health_score": 0.3, "reachable": True,
                "latency_s": 3.0, "error_rate": 0.3,
                "timestamp": time.time(),
            })

        decision = monitor.evaluate()
        assert decision == ProbationStatus.ROLLING_BACK

        await monitor.rollback()
        assert monitor._status == ProbationStatus.ROLLED_BACK

        # deployment_status.json should reflect rollback
        status_file = tmp_dirs["cross_repo_dir"] / "deployment_status.json"
        data = json.loads(status_file.read_text())
        assert data["deployment_status"] == "rollback"
