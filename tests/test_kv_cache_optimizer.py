"""Tests for kv_cache_optimizer.py — context window maximizer."""
from __future__ import annotations

import pytest

from jarvis_prime.core.kv_cache_optimizer import (
    KVCacheType,
    KVCacheProfile,
    ModelArchitectureParams,
    compute_kv_bytes_per_token,
    compute_feasible_profiles,
)


class TestModelArchitectureParams:
    """Test architecture parameter handling."""

    def test_qwen_32b_params(self):
        params = ModelArchitectureParams(
            n_layers=64, n_heads=40, n_kv_heads=8, head_dim=128, vocab_size=152064,
        )
        assert params.n_layers == 64
        assert params.n_kv_heads == 8

    def test_frozen(self):
        params = ModelArchitectureParams(
            n_layers=64, n_heads=40, n_kv_heads=8, head_dim=128, vocab_size=152064,
        )
        with pytest.raises(AttributeError):
            params.n_layers = 32


class TestKVBytesPerToken:
    """Test per-token KV cache size computation."""

    def test_f16_qwen_32b(self):
        """Qwen2.5-32B f16 KV: 2 × 64 × 8 × 128 × 2 = 262,144 bytes."""
        params = ModelArchitectureParams(
            n_layers=64, n_heads=40, n_kv_heads=8, head_dim=128, vocab_size=152064,
        )
        result = compute_kv_bytes_per_token(params, KVCacheType.F16)
        assert result == 262_144

    def test_q8_halves_size(self):
        params = ModelArchitectureParams(
            n_layers=64, n_heads=40, n_kv_heads=8, head_dim=128, vocab_size=152064,
        )
        f16 = compute_kv_bytes_per_token(params, KVCacheType.F16)
        q8 = compute_kv_bytes_per_token(params, KVCacheType.Q8_0)
        assert q8 == f16 // 2

    def test_q4_quarters_size(self):
        params = ModelArchitectureParams(
            n_layers=64, n_heads=40, n_kv_heads=8, head_dim=128, vocab_size=152064,
        )
        f16 = compute_kv_bytes_per_token(params, KVCacheType.F16)
        q4 = compute_kv_bytes_per_token(params, KVCacheType.Q4_0)
        assert q4 == f16 // 4


class TestComputeFeasibleProfiles:
    """Test feasible KV cache profile generation."""

    def test_returns_profiles_sorted_by_quality(self):
        params = ModelArchitectureParams(
            n_layers=64, n_heads=40, n_kv_heads=8, head_dim=128, vocab_size=152064,
        )
        profiles = compute_feasible_profiles(
            model_params=params,
            model_weight_bytes=11_000_000_000,  # 11GB IQ2_M
            total_vram_bytes=23_034 * 1024 * 1024,
        )
        assert len(profiles) >= 1
        # First should be best quality (lowest quality_impact)
        for i in range(len(profiles) - 1):
            assert profiles[i].quality_impact <= profiles[i + 1].quality_impact

    def test_excludes_profiles_below_min_context(self):
        params = ModelArchitectureParams(
            n_layers=64, n_heads=40, n_kv_heads=8, head_dim=128, vocab_size=152064,
        )
        profiles = compute_feasible_profiles(
            model_params=params,
            model_weight_bytes=22_000_000_000,  # Barely fits
            total_vram_bytes=23_034 * 1024 * 1024,
            min_context=2048,
        )
        for p in profiles:
            assert p.max_context_tokens >= 2048

    def test_iq2_m_on_l4_supports_8k_context(self):
        """IQ2_M (11GB) on L4 (23GB) should support 8192 context."""
        params = ModelArchitectureParams(
            n_layers=64, n_heads=40, n_kv_heads=8, head_dim=128, vocab_size=152064,
        )
        profiles = compute_feasible_profiles(
            model_params=params,
            model_weight_bytes=11_000_000_000,
            total_vram_bytes=23_034 * 1024 * 1024,
            target_context=8192,
        )
        assert any(p.max_context_tokens >= 8192 for p in profiles)

    def test_no_vram_returns_empty(self):
        params = ModelArchitectureParams(
            n_layers=64, n_heads=40, n_kv_heads=8, head_dim=128, vocab_size=152064,
        )
        profiles = compute_feasible_profiles(
            model_params=params,
            model_weight_bytes=25_000_000_000,  # Exceeds VRAM
            total_vram_bytes=23_034 * 1024 * 1024,
        )
        assert profiles == []
