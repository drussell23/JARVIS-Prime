"""Tests for Prime managed-mode behavior."""
import os
import pytest


class TestManagedModeRestart:
    def test_auto_restart_disabled_when_managed(self, monkeypatch):
        monkeypatch.setenv("JARVIS_ROOT_MANAGED", "true")
        from managed_mode import is_root_managed
        assert is_root_managed()

    def test_auto_restart_enabled_when_not_managed(self, monkeypatch):
        monkeypatch.delenv("JARVIS_ROOT_MANAGED", raising=False)
        from managed_mode import is_root_managed
        assert not is_root_managed()
