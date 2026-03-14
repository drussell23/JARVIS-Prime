"""
Quantization Intelligence — Rate-Distortion Scoring Engine
==========================================================

Pure, deterministic, side-effect-free module that scores quantization
variants using information-theoretic metrics.

Mathematical foundation:
  R(D) = minimum bit-rate to achieve distortion ≤ D
  ppl(bpw) ≈ ppl_fp16 × (1 + α × (fp16_bpw / bpw)^β)

Where:
  α ≈ 0.015 (model-family coefficient)
  β ≈ 2.1   (distortion exponent)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass(frozen=True)
class QuantizationProfile:
    """Immutable descriptor for a quantization method."""
    name: str
    bits_per_weight: float
    compression_ratio: float       # relative to FP16 (0.0-1.0)
    uses_importance_matrix: bool   # True for IQ variants
    quality_floor: float           # Minimum quality estimate (0.0-1.0)
    quality_ceiling: float         # Maximum quality estimate (0.0-1.0)


@dataclass(frozen=True)
class CalibrationPoint:
    """Empirical measurement for a specific quant variant."""
    quant_name: str
    measured_tok_s: float
    measured_perplexity: Optional[float]
    measured_vram_bytes: int
    context_size: int
    timestamp: float


@dataclass(frozen=True)
class CalibrationData:
    """Empirical measurements that override theoretical estimates."""
    model_family: str
    measurements: Dict[str, CalibrationPoint]   # quant_name → point


@dataclass(frozen=True)
class QuantizationQualityScore:
    """Computed quality assessment for a specific model+quant combination."""
    profile: QuantizationProfile
    model_family: str
    estimated_perplexity_ratio: float   # ppl(quant) / ppl(fp16), ≥1.0
    quality_score: float                # 0.0-1.0
    vram_bytes: int                     # Model weight footprint
    estimated_tok_s: float              # Throughput estimate
    context_headroom_tokens: int        # Max context with f16 KV cache
    fitness_score: float                # Composite score
    scoring_basis: str                  # "empirical" | "interpolated" | "extrapolated"


# =============================================================================
# KNOWN PROFILES
# =============================================================================

KNOWN_PROFILES: Dict[str, QuantizationProfile] = {
    "IQ2_XXS": QuantizationProfile(
        name="IQ2_XXS", bits_per_weight=2.06, compression_ratio=0.129,
        uses_importance_matrix=True, quality_floor=0.55, quality_ceiling=0.70,
    ),
    "IQ2_M": QuantizationProfile(
        name="IQ2_M", bits_per_weight=2.70, compression_ratio=0.169,
        uses_importance_matrix=True, quality_floor=0.65, quality_ceiling=0.80,
    ),
    "Q2_K": QuantizationProfile(
        name="Q2_K", bits_per_weight=2.96, compression_ratio=0.185,
        uses_importance_matrix=False, quality_floor=0.60, quality_ceiling=0.75,
    ),
    "Q3_K_S": QuantizationProfile(
        name="Q3_K_S", bits_per_weight=3.50, compression_ratio=0.219,
        uses_importance_matrix=False, quality_floor=0.70, quality_ceiling=0.85,
    ),
    "Q3_K_M": QuantizationProfile(
        name="Q3_K_M", bits_per_weight=3.89, compression_ratio=0.243,
        uses_importance_matrix=False, quality_floor=0.75, quality_ceiling=0.88,
    ),
    "Q4_K_S": QuantizationProfile(
        name="Q4_K_S", bits_per_weight=4.58, compression_ratio=0.286,
        uses_importance_matrix=False, quality_floor=0.82, quality_ceiling=0.93,
    ),
    "Q4_K_M": QuantizationProfile(
        name="Q4_K_M", bits_per_weight=4.83, compression_ratio=0.302,
        uses_importance_matrix=False, quality_floor=0.85, quality_ceiling=0.95,
    ),
    "Q5_K_M": QuantizationProfile(
        name="Q5_K_M", bits_per_weight=5.69, compression_ratio=0.356,
        uses_importance_matrix=False, quality_floor=0.90, quality_ceiling=0.97,
    ),
    "Q6_K": QuantizationProfile(
        name="Q6_K", bits_per_weight=6.56, compression_ratio=0.410,
        uses_importance_matrix=False, quality_floor=0.93, quality_ceiling=0.98,
    ),
    "Q8_0": QuantizationProfile(
        name="Q8_0", bits_per_weight=8.50, compression_ratio=0.531,
        uses_importance_matrix=False, quality_floor=0.97, quality_ceiling=0.99,
    ),
}


# =============================================================================
# MODEL FAMILY COEFFICIENTS
# =============================================================================

_MODEL_FAMILY_COEFFICIENTS: Dict[str, Tuple[float, float]] = {
    # (alpha, beta) for ppl(bpw) ≈ ppl_fp16 × (1 + α × (16/bpw)^β)
    "qwen2.5-coder-32b": (0.015, 2.1),
    "qwen2.5-coder-14b": (0.018, 2.0),
    "qwen2.5-coder-7b": (0.022, 1.9),
    "deepseek-r1-qwen-7b": (0.020, 2.0),
    "llama-3.2-1b": (0.030, 1.8),
}

_DEFAULT_COEFFICIENTS: Tuple[float, float] = (0.020, 2.0)

# L4 GPU specs
_L4_MEMORY_BANDWIDTH_GBPS: float = 300.0
_L4_COMPUTE_TFLOPS: float = 30.3
_FP16_BPW: float = 16.0
_OVERHEAD_BYTES: int = 500_000_000  # 500MB CUDA/framework overhead


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def _get_coefficients(model_family: str) -> Tuple[float, float]:
    """Get alpha, beta for a model family."""
    key = model_family.lower().replace(" ", "-")
    for family_key, coeffs in _MODEL_FAMILY_COEFFICIENTS.items():
        if family_key in key:
            return coeffs
    return _DEFAULT_COEFFICIENTS


def _estimate_perplexity_ratio(
    bits_per_weight: float,
    alpha: float,
    beta: float,
) -> float:
    """
    Estimate ppl(quant) / ppl(fp16) using power law.

    ppl(bpw) ≈ ppl_fp16 × (1 + α × (16/bpw)^β)
    ratio = ppl(bpw) / ppl_fp16 = 1 + α × (16/bpw)^β
    """
    ratio = 1.0 + alpha * (_FP16_BPW / bits_per_weight) ** beta
    return max(ratio, 1.0)


def _perplexity_ratio_to_quality(ratio: float) -> float:
    """
    Map perplexity ratio to a 0-1 quality score.

    Uses exponential decay: quality = exp(-k × (ratio - 1))
    where k controls sensitivity. At ratio=1.0 → quality=1.0.
    """
    k = 8.0  # Tuned so ratio=1.10 → quality≈0.45, ratio=1.02 → quality≈0.85
    return math.exp(-k * (ratio - 1.0))


def _estimate_context_headroom(
    model_size_bytes: int,
    total_vram_bytes: int,
    n_layers: int = 64,
    n_kv_heads: int = 8,
    head_dim: int = 128,
) -> int:
    """Estimate max context tokens with f16 KV cache."""
    available = total_vram_bytes - model_size_bytes - _OVERHEAD_BYTES
    if available <= 0:
        return 0
    # KV per token = 2 × n_layers × n_kv_heads × head_dim × 2 (f16)
    kv_per_token = 2 * n_layers * n_kv_heads * head_dim * 2
    if kv_per_token == 0:
        return 0
    return max(0, int(available / kv_per_token))


def estimate_throughput(
    model_params_billions: float,
    bits_per_weight: float,
    gpu_memory_bandwidth_gbps: float,
    gpu_compute_tflops: float,
) -> float:
    """
    Estimate tok/s using roofline model.

    LLM decode is memory-bandwidth-bound:
      tok/s ≈ memory_bandwidth / (model_params × bpw / 8)
    """
    model_bytes = model_params_billions * 1e9 * bits_per_weight / 8.0
    if model_bytes <= 0:
        return 0.0
    bandwidth_bytes_per_s = gpu_memory_bandwidth_gbps * 1e9
    tok_s = bandwidth_bytes_per_s / model_bytes
    return tok_s


def score_quantization(
    profile: QuantizationProfile,
    model_family: str,
    model_size_bytes: int,
    total_vram_bytes: int,
    target_context: int = 8192,
    task_complexity: str = "medium",
    calibration_data: Optional[CalibrationData] = None,
) -> QuantizationQualityScore:
    """
    Score a quantization variant for the given hardware and task.
    Pure function — no I/O, no state mutation.
    """
    alpha, beta = _get_coefficients(model_family)

    # Check if model fits
    fits = (model_size_bytes + _OVERHEAD_BYTES) < total_vram_bytes

    # Perplexity ratio
    ppl_ratio = _estimate_perplexity_ratio(profile.bits_per_weight, alpha, beta)
    quality = _perplexity_ratio_to_quality(ppl_ratio)

    # Context headroom
    context_headroom = _estimate_context_headroom(
        model_size_bytes, total_vram_bytes,
    ) if fits else 0

    # Throughput estimate
    scoring_basis = "extrapolated"
    tok_s = estimate_throughput(
        model_params_billions=model_size_bytes / (profile.bits_per_weight / 8.0) / 1e9,
        bits_per_weight=profile.bits_per_weight,
        gpu_memory_bandwidth_gbps=_L4_MEMORY_BANDWIDTH_GBPS,
        gpu_compute_tflops=_L4_COMPUTE_TFLOPS,
    )

    # Override with calibration if available
    if calibration_data and profile.name in calibration_data.measurements:
        cal = calibration_data.measurements[profile.name]
        tok_s = cal.measured_tok_s
        scoring_basis = "empirical"

    # Task complexity weights
    complexity_weights = {
        "trivial": {"quality": 0.2, "throughput": 0.6, "context": 0.2},
        "light": {"quality": 0.3, "throughput": 0.5, "context": 0.2},
        "medium": {"quality": 0.4, "throughput": 0.3, "context": 0.3},
        "heavy": {"quality": 0.6, "throughput": 0.1, "context": 0.3},
        "complex": {"quality": 0.7, "throughput": 0.1, "context": 0.2},
    }
    weights = complexity_weights.get(task_complexity, complexity_weights["medium"])

    # Fitness score
    if not fits:
        fitness = 0.0
    else:
        # Normalize throughput (0-1 scale, assuming max ~60 tok/s on L4)
        norm_tok_s = min(tok_s / 60.0, 1.0)
        # Normalize context (0-1 scale relative to target)
        norm_ctx = min(context_headroom / max(target_context, 1), 1.0)
        fitness = (
            weights["quality"] * quality
            + weights["throughput"] * norm_tok_s
            + weights["context"] * norm_ctx
        )

    return QuantizationQualityScore(
        profile=profile,
        model_family=model_family,
        estimated_perplexity_ratio=ppl_ratio,
        quality_score=quality,
        vram_bytes=model_size_bytes,
        estimated_tok_s=tok_s,
        context_headroom_tokens=context_headroom,
        fitness_score=fitness,
        scoring_basis=scoring_basis,
    )


def rank_quantizations(
    available: List[Tuple[QuantizationProfile, int]],
    model_family: str,
    total_vram_bytes: int,
    target_context: int = 8192,
    task_complexity: str = "medium",
    calibration_data: Optional[CalibrationData] = None,
) -> List[QuantizationQualityScore]:
    """
    Rank all available quantizations by fitness_score.
    Returns sorted list, best first. Excludes variants that won't fit.
    """
    scores = []
    for profile, file_size in available:
        score = score_quantization(
            profile=profile,
            model_family=model_family,
            model_size_bytes=file_size,
            total_vram_bytes=total_vram_bytes,
            target_context=target_context,
            task_complexity=task_complexity,
            calibration_data=calibration_data,
        )
        if score.fitness_score > 0.0:
            scores.append(score)

    scores.sort(key=lambda s: s.fitness_score, reverse=True)
    return scores
