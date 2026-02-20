from __future__ import annotations

import pytest

import jarvis_prime.server as prime_server


@pytest.mark.asyncio
async def test_main_delegates_to_authoritative_entrypoint(monkeypatch):
    calls = {"authoritative": 0, "legacy": 0}

    async def fake_authoritative_main():
        calls["authoritative"] += 1
        return "authoritative"

    async def fake_legacy_main():
        calls["legacy"] += 1
        return "legacy"

    monkeypatch.setattr(prime_server, "_load_authoritative_main", lambda: fake_authoritative_main)
    monkeypatch.setattr(prime_server, "_legacy_main", fake_legacy_main)

    result = await prime_server.main()

    assert result == "authoritative"
    assert calls["authoritative"] == 1
    assert calls["legacy"] == 0


@pytest.mark.asyncio
async def test_main_fails_fast_when_authoritative_missing(monkeypatch):
    calls = {"legacy": 0}

    async def fake_legacy_main():
        calls["legacy"] += 1
        return "legacy"

    monkeypatch.setattr(prime_server, "_load_authoritative_main", lambda: None)
    monkeypatch.setattr(prime_server, "_legacy_main", fake_legacy_main)
    monkeypatch.delenv("JARVIS_PRIME_ALLOW_LEGACY_SERVER_FALLBACK", raising=False)

    with pytest.raises(RuntimeError, match="Authoritative Prime entrypoint unavailable"):
        await prime_server.main()

    assert calls["legacy"] == 0


@pytest.mark.asyncio
async def test_main_allows_legacy_fallback_when_explicitly_enabled(monkeypatch):
    calls = {"legacy": 0}

    async def fake_legacy_main():
        calls["legacy"] += 1
        return "legacy"

    monkeypatch.setattr(prime_server, "_load_authoritative_main", lambda: None)
    monkeypatch.setattr(prime_server, "_legacy_main", fake_legacy_main)
    monkeypatch.setenv("JARVIS_PRIME_ALLOW_LEGACY_SERVER_FALLBACK", "true")

    result = await prime_server.main()

    assert result == "legacy"
    assert calls["legacy"] == 1
