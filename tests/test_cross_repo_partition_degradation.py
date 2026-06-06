"""jarvis-prime recoverable partition-degradation tests (Slice 98 Phase 3).

The posture is a PURE FUNCTION of now vs last-verified-handshake: stale → soft
OBSERVATION_ONLY; a fresh ripple recovers to NORMAL automatically (no reset, no
irreversible sever). Cold start is NORMAL (never false-paranoia).
"""

from __future__ import annotations

import time

import pytest

from cross_repo_mesh import partition_degradation as D


@pytest.fixture(autouse=True)
def _state(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_PRIME_PARTITION_DEGRADATION_ENABLED", "true")
    monkeypatch.setenv("JARVIS_PRIME_PARTITION_STALENESS_S", "300")
    monkeypatch.setenv("JARVIS_PRIME_LAST_VERIFIED_PATH", str(tmp_path / "lastv"))
    yield


def test_cold_start_is_normal_never_false_paranoia():
    # Never received a ripple → NORMAL (not partition).
    assert D.read_last_verified() is None
    assert D.partition_posture(now_unix=time.time()) == D.POSTURE_NORMAL


def test_fresh_handshake_is_normal():
    now = time.time()
    D.record_verified(now)
    assert D.partition_posture(now_unix=now + 10) == D.POSTURE_NORMAL


def test_stale_handshake_degrades_to_observation_only():
    now = time.time()
    D.record_verified(now)
    # 301s later with no fresh ripple → partition → soft degrade.
    assert D.partition_posture(now_unix=now + 301) == D.POSTURE_OBSERVATION_ONLY


def test_recovers_automatically_on_fresh_ripple():
    now = time.time()
    D.record_verified(now)
    # go stale
    assert D.partition_posture(now_unix=now + 400) == D.POSTURE_OBSERVATION_ONLY
    # a fresh ripple lands → record_verified moves freshness forward → recover.
    D.record_verified(now + 400)
    assert D.partition_posture(now_unix=now + 405) == D.POSTURE_NORMAL  # no reset call


def test_transient_blip_recovers_purely_from_time():
    now = time.time()
    D.record_verified(now)
    assert D.partition_posture(now_unix=now + 350) == D.POSTURE_OBSERVATION_ONLY  # blip
    D.record_verified(now + 360)                                                  # handshake resumes
    assert D.partition_posture(now_unix=now + 360) == D.POSTURE_NORMAL            # recovered


def test_master_off_is_always_normal(monkeypatch):
    monkeypatch.delenv("JARVIS_PRIME_PARTITION_DEGRADATION_ENABLED", raising=False)
    now = time.time()
    D.record_verified(now)
    assert D.partition_posture(now_unix=now + 99999) == D.POSTURE_NORMAL  # inert


def test_clock_skew_future_stamp_is_fresh():
    now = time.time()
    D.record_verified(now + 1000)  # future stamp
    assert D.partition_posture(now_unix=now) == D.POSTURE_NORMAL  # age<=0 → fresh, no crash


def test_explicit_last_verified_arg_overrides_state():
    now = time.time()
    assert D.partition_posture(now_unix=now, last_verified_unix=now - 400) == D.POSTURE_OBSERVATION_ONLY
    assert D.partition_posture(now_unix=now, last_verified_unix=now - 10) == D.POSTURE_NORMAL


def test_no_irreversible_constructs_never_raises():
    # garbage state file → never raises, returns NORMAL (cold-start-like)
    assert D.read_last_verified("/nonexistent/dir/x") is None
    assert D.partition_posture(now_unix=time.time(), state_path="/nonexistent/dir/x") == D.POSTURE_NORMAL
