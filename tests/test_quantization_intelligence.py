"""Tests for quantization_intelligence.py — rate-distortion scoring engine."""
from __future__ import annotations

import pytest

from jarvis_prime.core.quantization_intelligence import (
    CalibrationData,
    CalibrationPoint,
    QuantizationProfile,
    QuantizationQualityScore,
    KNOWN_PROFILES,
    score_quantization,
    rank_quantizations,
    estimate_throughput,
)


class TestQuantizationProfile:
    """Test QuantizationProfile frozen dataclass."""

    def test_iq2_m_profile_exists(self):
        profile = KNOWN_PROFILES["IQ2_M"]
        assert profile.bits_per_weight == 2.70
        assert profile.uses_importance_matrix is True
        assert 0.0 < profile.quality_floor < profile.quality_ceiling <= 1.0

    def test_q4_k_m_profile_exists(self):
        profile = KNOWN_PROFILES["Q4_K_M"]
        assert profile.bits_per_weight == 4.83
        assert profile.uses_importance_matrix is False

    def test_profile_is_frozen(self):
        profile = KNOWN_PROFILES["IQ2_M"]
        with pytest.raises(AttributeError):
            profile.bits_per_weight = 5.0

    def test_all_profiles_have_valid_ranges(self):
        for name, p in KNOWN_PROFILES.items():
            assert p.bits_per_weight > 0, f"{name}: bpw must be positive"
            assert 0.0 < p.compression_ratio < 1.0, f"{name}: compression_ratio must be (0,1)"
            assert p.quality_floor <= p.quality_ceiling, f"{name}: floor > ceiling"


class TestEstimateThroughput:
    """Test roofline model throughput estimation."""

    def test_7b_q4_throughput(self):
        """7B Q4_K_M on L4 should estimate ~40-60 tok/s."""
        tok_s = estimate_throughput(
            model_params_billions=7.0,
            bits_per_weight=4.83,
            gpu_memory_bandwidth_gbps=300.0,
            gpu_compute_tflops=30.3,
        )
        assert 30.0 < tok_s < 80.0

    def test_32b_iq2_throughput(self):
        """32B IQ2_M on L4 should estimate ~8-20 tok/s."""
        tok_s = estimate_throughput(
            model_params_billions=32.0,
            bits_per_weight=2.70,
            gpu_memory_bandwidth_gbps=300.0,
            gpu_compute_tflops=30.3,
        )
        assert 5.0 < tok_s < 30.0

    def test_higher_bpw_means_lower_throughput_same_model(self):
        """Same model size, higher bpw → lower throughput (more memory to read)."""
        tok_s_q4 = estimate_throughput(32.0, 4.83, 300.0, 30.3)
        tok_s_iq2 = estimate_throughput(32.0, 2.70, 300.0, 30.3)
        assert tok_s_iq2 > tok_s_q4


class TestScoreQuantization:
    """Test composite quality scoring."""

    def test_q4_k_m_scores_higher_quality_than_iq2_m(self):
        """Higher bpw → higher quality score."""
        q4 = score_quantization(
            profile=KNOWN_PROFILES["Q4_K_M"],
            model_family="qwen2.5-coder-32b",
            model_size_bytes=19_000_000_000,
            total_vram_bytes=23_034 * 1024 * 1024,
        )
        iq2 = score_quantization(
            profile=KNOWN_PROFILES["IQ2_M"],
            model_family="qwen2.5-coder-32b",
            model_size_bytes=11_000_000_000,
            total_vram_bytes=23_034 * 1024 * 1024,
        )
        assert q4.quality_score > iq2.quality_score

    def test_model_that_doesnt_fit_gets_zero_fitness(self):
        """Model larger than VRAM should have fitness_score = 0."""
        score = score_quantization(
            profile=KNOWN_PROFILES["Q4_K_M"],
            model_family="qwen2.5-coder-32b",
            model_size_bytes=19_000_000_000,
            total_vram_bytes=15_000_000_000,  # Only 15GB VRAM
        )
        assert score.fitness_score == 0.0

    def test_scoring_basis_without_calibration(self):
        """Without calibration data, basis should be extrapolated."""
        score = score_quantization(
            profile=KNOWN_PROFILES["IQ2_M"],
            model_family="qwen2.5-coder-32b",
            model_size_bytes=11_000_000_000,
            total_vram_bytes=23_034 * 1024 * 1024,
        )
        assert score.scoring_basis in ("interpolated", "extrapolated")

    def test_scoring_with_calibration(self):
        """With calibration data, basis should be empirical."""
        cal = CalibrationData(
            model_family="qwen2.5-coder-32b",
            measurements={
                "IQ2_M": CalibrationPoint(
                    quant_name="IQ2_M",
                    measured_tok_s=12.5,
                    measured_perplexity=None,
                    measured_vram_bytes=21_474 * 1024 * 1024,
                    context_size=8192,
                    timestamp=1710400000.0,
                ),
            },
        )
        score = score_quantization(
            profile=KNOWN_PROFILES["IQ2_M"],
            model_family="qwen2.5-coder-32b",
            model_size_bytes=11_000_000_000,
            total_vram_bytes=23_034 * 1024 * 1024,
            calibration_data=cal,
        )
        assert score.scoring_basis == "empirical"
        assert abs(score.estimated_tok_s - 12.5) < 0.01

    def test_perplexity_ratio_always_gte_one(self):
        for name, profile in KNOWN_PROFILES.items():
            score = score_quantization(
                profile=profile,
                model_family="qwen2.5-coder-7b",
                model_size_bytes=4_400_000_000,
                total_vram_bytes=23_034 * 1024 * 1024,
            )
            assert score.estimated_perplexity_ratio >= 1.0, f"{name}: ppl ratio < 1.0"


class TestRankQuantizations:
    """Test ranking of multiple quantization variants."""

    def test_rank_excludes_models_that_dont_fit(self):
        """Models exceeding VRAM should not appear in ranking."""
        available = [
            (KNOWN_PROFILES["Q4_K_M"], 19_000_000_000),
            (KNOWN_PROFILES["IQ2_M"], 11_000_000_000),
            (KNOWN_PROFILES["Q8_0"], 30_000_000_000),  # Too large
        ]
        ranked = rank_quantizations(
            available=available,
            model_family="qwen2.5-coder-32b",
            total_vram_bytes=23_034 * 1024 * 1024,
        )
        names = [r.profile.name for r in ranked]
        assert "Q8_0" not in names

    def test_rank_returns_best_first(self):
        """First result should have highest fitness_score."""
        available = [
            (KNOWN_PROFILES["Q4_K_M"], 19_000_000_000),
            (KNOWN_PROFILES["IQ2_M"], 11_000_000_000),
        ]
        ranked = rank_quantizations(
            available=available,
            model_family="qwen2.5-coder-32b",
            total_vram_bytes=23_034 * 1024 * 1024,
        )
        assert len(ranked) >= 1
        for i in range(len(ranked) - 1):
            assert ranked[i].fitness_score >= ranked[i + 1].fitness_score

    def test_empty_available_returns_empty(self):
        ranked = rank_quantizations(
            available=[],
            model_family="qwen2.5-coder-32b",
            total_vram_bytes=23_034 * 1024 * 1024,
        )
        assert ranked == []
