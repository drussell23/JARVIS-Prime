from __future__ import annotations

from pathlib import Path


RUN_SERVER_PATH = Path(__file__).resolve().parents[1] / "run_server.py"


def _source() -> str:
    return RUN_SERVER_PATH.read_text(encoding="utf-8")


def test_background_init_initializes_reactor_and_training_pipeline():
    source = _source()

    assert 'log_step("initializing_reactor_pipeline", 4)' in source
    assert "from jarvis_prime.core.reactor_core_bridge import get_reactor_core_bridge" in source
    assert "from jarvis_prime.core.training_data_pipeline import get_training_data_pipeline" in source
    assert "_reactor_bridge = await get_reactor_core_bridge()" in source
    assert "_training_pipeline = await get_training_data_pipeline()" in source


def test_capture_interaction_routes_into_training_pipeline():
    source = _source()

    assert "if not _telemetry_hook and not _training_pipeline:" in source
    assert "await _training_pipeline.capture_conversation(" in source


def test_shutdown_cleans_up_training_and_reactor_bridges():
    source = _source()

    assert "from jarvis_prime.core.training_data_pipeline import shutdown_training_data_pipeline" in source
    assert "from jarvis_prime.core.reactor_core_bridge import shutdown_reactor_core_bridge" in source
    assert "await shutdown_training_data_pipeline()" in source
    assert "await shutdown_reactor_core_bridge()" in source


def test_background_init_initializes_jarvis_bridge_after_agi_hub():
    source = _source()

    assert 'log_step("initializing_jarvis_bridge", 6)' in source
    assert "from jarvis_prime.core.jarvis_bridge import get_jarvis_bridge" in source
    assert "_jarvis_bridge = await get_jarvis_bridge()" in source
    assert "if bridge_enabled and _agi_hub:" in source


def test_shutdown_cleans_up_jarvis_bridge():
    source = _source()

    assert "from jarvis_prime.core.jarvis_bridge import shutdown_jarvis_bridge" in source
    assert "await shutdown_jarvis_bridge()" in source
