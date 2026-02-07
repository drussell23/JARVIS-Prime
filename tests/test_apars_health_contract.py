import json

from jarvis_prime.server import StartupState


def test_health_includes_apars_from_file(tmp_path, monkeypatch):
    apars_payload = {
        "phase": 4,
        "phase_progress": 60,
        "total_progress": 55,
        "checkpoint": "checking_model_cache",
        "ready_for_inference": False,
    }
    apars_path = tmp_path / "apars.json"
    apars_path.write_text(json.dumps(apars_payload))

    monkeypatch.setenv("JARVIS_APARS_FILE", str(apars_path))

    state = StartupState()
    status = state.get_status()

    assert "apars" in status, "health response should include APARS data when file is set"
    assert status["apars"]["total_progress"] == 55
    assert status["apars"]["checkpoint"] == "checking_model_cache"
