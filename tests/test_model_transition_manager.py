"""Tests for model_transition_manager.py — FSM executor + VRAMBudgetAuthority."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from jarvis_prime.core.model_transition_manager import (
    LeaseState,
    VRAMGrant,
    VRAMPriority,
    VRAMBudgetAuthority,
    TransitionState,
    TransitionEpoch,
    TransitionPolicy,
    ModelTransitionManager,
)


class TestVRAMBudgetAuthority:

    @pytest.mark.asyncio
    async def test_grant_within_budget(self):
        auth = VRAMBudgetAuthority(total_vram_bytes=23_000_000_000)
        grant = await auth.request("model-iq2m", 11_000_000_000, VRAMPriority.NORMAL)
        assert grant is not None
        assert grant.state == LeaseState.GRANTED

    @pytest.mark.asyncio
    async def test_deny_exceeds_budget(self):
        auth = VRAMBudgetAuthority(total_vram_bytes=23_000_000_000)
        grant = await auth.request("model-q4km", 25_000_000_000, VRAMPriority.NORMAL)
        assert grant is None

    @pytest.mark.asyncio
    async def test_commit_transitions_to_active(self):
        auth = VRAMBudgetAuthority(total_vram_bytes=23_000_000_000)
        grant = await auth.request("model-test", 10_000_000_000, VRAMPriority.NORMAL)
        assert grant is not None
        await grant.commit(10_500_000_000)
        assert grant.state == LeaseState.ACTIVE

    @pytest.mark.asyncio
    async def test_release_frees_budget(self):
        auth = VRAMBudgetAuthority(total_vram_bytes=23_000_000_000)
        g1 = await auth.request("model-a", 15_000_000_000, VRAMPriority.NORMAL)
        assert g1 is not None
        # Second grant would exceed budget
        g2 = await auth.request("model-b", 15_000_000_000, VRAMPriority.NORMAL)
        assert g2 is None
        # Release first, try again
        await g1.release()
        g3 = await auth.request("model-b", 15_000_000_000, VRAMPriority.NORMAL)
        assert g3 is not None

    @pytest.mark.asyncio
    async def test_rollback_frees_budget(self):
        auth = VRAMBudgetAuthority(total_vram_bytes=23_000_000_000)
        grant = await auth.request("model-test", 10_000_000_000, VRAMPriority.NORMAL)
        assert grant is not None
        await grant.rollback("test failure")
        assert grant.state == LeaseState.ROLLED_BACK
        assert auth.allocated_bytes == 0


class TestTransitionEpoch:

    def test_advance_model_increments(self):
        epoch = TransitionEpoch()
        assert epoch.model_epoch == 0
        val = epoch.advance_model()
        assert val == 1
        assert epoch.model_epoch == 1

    def test_monotonic(self):
        epoch = TransitionEpoch()
        for i in range(5):
            val = epoch.advance_model()
            assert val == i + 1


class TestTransitionPolicy:

    def test_default_values(self):
        policy = TransitionPolicy()
        assert policy.min_cooldown_s == 90.0
        assert policy.max_swaps_per_hour == 4


class TestModelTransitionManager:

    @pytest.mark.asyncio
    async def test_accept_startup_proposal(self, mock_executor, tmp_models_dir, fake_gguf_files):
        from jarvis_prime.core.adaptive_model_selector import (
            ModelVariant, ModelSelectionProposal, scan_inventory, propose_optimal,
        )

        auth = VRAMBudgetAuthority(total_vram_bytes=23_034 * 1024 * 1024)
        mgr = ModelTransitionManager(
            executor=mock_executor,
            vram_authority=auth,
            model_dir=tmp_models_dir,
        )

        families = await scan_inventory(tmp_models_dir)
        proposal = await propose_optimal(
            families=families,
            vram_budget_bytes=23_034 * 1024 * 1024,
            target_context=8192,
            task_complexity="medium",
            trigger="startup",
        )
        assert proposal is not None
        result = await mgr.accept(proposal)
        assert result is True
        assert mgr.state == TransitionState.IDLE
        assert mgr.epoch.model_epoch == 1

    @pytest.mark.asyncio
    async def test_reject_concurrent_transition(self, mock_executor, tmp_models_dir):
        auth = VRAMBudgetAuthority(total_vram_bytes=23_034 * 1024 * 1024)
        mgr = ModelTransitionManager(
            executor=mock_executor,
            vram_authority=auth,
            model_dir=tmp_models_dir,
        )
        # Force non-IDLE state
        mgr._state = TransitionState.DRAIN
        from unittest.mock import MagicMock
        fake_proposal = MagicMock()
        fake_proposal.trigger = "pressure"
        result = await mgr.accept(fake_proposal)
        assert result is False

    @pytest.mark.asyncio
    async def test_rollback_on_validation_failure(self, mock_executor, tmp_models_dir, fake_gguf_files):
        """Validation failure should trigger ROLLBACK and restore previous model."""
        from jarvis_prime.core.adaptive_model_selector import scan_inventory, propose_optimal

        mock_executor.validate = AsyncMock(return_value=False)
        auth = VRAMBudgetAuthority(total_vram_bytes=23_034 * 1024 * 1024)
        mgr = ModelTransitionManager(
            executor=mock_executor, vram_authority=auth, model_dir=tmp_models_dir,
        )
        families = await scan_inventory(tmp_models_dir)
        proposal = await propose_optimal(
            families=families, vram_budget_bytes=23_034 * 1024 * 1024,
            target_context=8192, task_complexity="medium", trigger="startup",
        )
        result = await mgr.accept(proposal)
        assert result is False
        assert mgr.state == TransitionState.IDLE

    @pytest.mark.asyncio
    async def test_rollback_on_load_failure(self, mock_executor, tmp_models_dir, fake_gguf_files):
        """Load failure should trigger ROLLBACK."""
        from jarvis_prime.core.adaptive_model_selector import scan_inventory, propose_optimal

        mock_executor.load = AsyncMock(side_effect=RuntimeError("OOM"))
        auth = VRAMBudgetAuthority(total_vram_bytes=23_034 * 1024 * 1024)
        mgr = ModelTransitionManager(
            executor=mock_executor, vram_authority=auth, model_dir=tmp_models_dir,
        )
        families = await scan_inventory(tmp_models_dir)
        proposal = await propose_optimal(
            families=families, vram_budget_bytes=23_034 * 1024 * 1024,
            target_context=8192, task_complexity="medium", trigger="startup",
        )
        result = await mgr.accept(proposal)
        assert result is False

    @pytest.mark.asyncio
    async def test_reject_during_cold_start_lockout(self, mock_executor, tmp_models_dir):
        """Swaps should be rejected during cold start lockout."""
        auth = VRAMBudgetAuthority(total_vram_bytes=23_034 * 1024 * 1024)
        mgr = ModelTransitionManager(
            executor=mock_executor, vram_authority=auth, model_dir=tmp_models_dir,
        )
        # Cold start lockout = 120s, just started
        mgr._started_at = time.monotonic()
        from unittest.mock import MagicMock
        fake_proposal = MagicMock()
        fake_proposal.trigger = "pressure"  # Not "startup" — startup bypasses cooldown
        result = await mgr.accept(fake_proposal)
        assert result is False

    @pytest.mark.asyncio
    async def test_swap_reservation_same_size(self):
        """Two-grant swap reservation should allow same-size model swaps."""
        auth = VRAMBudgetAuthority(total_vram_bytes=23_000_000_000)
        g1 = await auth.request("model-a", 11_000_000_000, VRAMPriority.NORMAL)
        assert g1 is not None
        await g1.commit(11_000_000_000)
        # Without releasing_grant_id, this would fail (11+11 > 23)
        g2 = await auth.request(
            "model-b", 11_000_000_000, VRAMPriority.NORMAL,
            releasing_grant_id=g1.grant_id,
        )
        assert g2 is not None

    @pytest.mark.asyncio
    async def test_reject_stale_inventory_digest(self, mock_executor, tmp_models_dir, fake_gguf_files):
        """Proposal with stale inventory digest should be rejected."""
        from jarvis_prime.core.adaptive_model_selector import scan_inventory, propose_optimal
        auth = VRAMBudgetAuthority(total_vram_bytes=23_034 * 1024 * 1024)
        mgr = ModelTransitionManager(
            executor=mock_executor, vram_authority=auth, model_dir=tmp_models_dir,
        )
        families = await scan_inventory(tmp_models_dir)
        proposal = await propose_optimal(
            families=families, vram_budget_bytes=23_034 * 1024 * 1024,
            target_context=8192, task_complexity="medium", trigger="startup",
            model_dir=tmp_models_dir,
        )
        # Mutate inventory after proposal
        new_file = tmp_models_dir / "NewModel-7B-Q4_K_M.gguf"
        new_file.write_bytes(b"\0" * 100)
        result = await mgr.accept(proposal)
        assert result is False

    def test_drain_admission_gate(self, mock_executor, tmp_models_dir):
        auth = VRAMBudgetAuthority(total_vram_bytes=23_034 * 1024 * 1024)
        mgr = ModelTransitionManager(
            executor=mock_executor, vram_authority=auth, model_dir=tmp_models_dir,
        )
        assert mgr.admit_request() is True
        mgr._draining = True
        assert mgr.admit_request() is False
