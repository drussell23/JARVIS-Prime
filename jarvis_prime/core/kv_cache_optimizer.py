"""
KV Cache Optimizer — Context Window Maximizer
==============================================

Pure, deterministic, side-effect-free module that computes feasible
KV cache configurations given model architecture and VRAM constraints.

KV cache memory per token:
  kv_per_token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_element
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

class KVCacheType(Enum):
    F16 = "f16"     # Full precision, best quality
    Q8_0 = "q8_0"   # 50% savings, ~0.1% quality loss
    Q4_0 = "q4_0"   # 75% savings, ~0.5% quality loss


# Bytes per element for each cache type
_BYTES_PER_ELEMENT = {
    KVCacheType.F16: 2,
    KVCacheType.Q8_0: 1,
    KVCacheType.Q4_0: 0.5,
}

# Quality impact (0.0 = no impact, higher = worse)
_QUALITY_IMPACT = {
    KVCacheType.F16: 0.0,
    KVCacheType.Q8_0: 0.001,
    KVCacheType.Q4_0: 0.005,
}


@dataclass(frozen=True)
class ModelArchitectureParams:
    """Model architecture parameters for KV cache computation."""
    n_layers: int
    n_heads: int
    n_kv_heads: int     # For GQA models (< n_heads)
    head_dim: int
    vocab_size: int


@dataclass(frozen=True)
class KVCacheProfile:
    """Feasible KV cache configuration for given constraints."""
    cache_type_k: KVCacheType
    cache_type_v: KVCacheType
    max_context_tokens: int
    vram_bytes: int              # Total KV cache VRAM at max_context
    quality_impact: float        # 0.0 = no impact, 1.0 = severe
    recommendation: str          # Human-readable


# =============================================================================
# KNOWN ARCHITECTURES
# =============================================================================

KNOWN_ARCHITECTURES = {
    "qwen2.5-coder-32b": ModelArchitectureParams(
        n_layers=64, n_heads=40, n_kv_heads=8, head_dim=128, vocab_size=152064,
    ),
    "qwen2.5-coder-14b": ModelArchitectureParams(
        n_layers=48, n_heads=40, n_kv_heads=8, head_dim=128, vocab_size=152064,
    ),
    "qwen2.5-coder-7b": ModelArchitectureParams(
        n_layers=28, n_heads=28, n_kv_heads=4, head_dim=128, vocab_size=152064,
    ),
    "deepseek-r1-qwen-7b": ModelArchitectureParams(
        n_layers=28, n_heads=28, n_kv_heads=4, head_dim=128, vocab_size=152064,
    ),
    "llama-3.2-1b": ModelArchitectureParams(
        n_layers=16, n_heads=32, n_kv_heads=8, head_dim=64, vocab_size=128256,
    ),
}


# =============================================================================
# COMPUTATION FUNCTIONS
# =============================================================================

def compute_kv_bytes_per_token(
    params: ModelArchitectureParams,
    cache_type: KVCacheType,
) -> int:
    """
    Compute KV cache bytes per token.

    Formula: 2 × n_layers × n_kv_heads × head_dim × bytes_per_element
    The factor of 2 is for K and V caches.
    """
    bpe = _BYTES_PER_ELEMENT[cache_type]
    return int(2 * params.n_layers * params.n_kv_heads * params.head_dim * bpe)


def compute_feasible_profiles(
    model_params: ModelArchitectureParams,
    model_weight_bytes: int,
    total_vram_bytes: int,
    overhead_bytes: int = 500_000_000,
    target_context: int = 8192,
    min_context: int = 2048,
) -> List[KVCacheProfile]:
    """
    Compute all feasible KV cache profiles.

    Returns profiles sorted by quality (best first), filtered to those
    that achieve at least min_context tokens.
    """
    available_for_kv = total_vram_bytes - model_weight_bytes - overhead_bytes
    if available_for_kv <= 0:
        return []

    # Generate profiles for all K×V type combinations
    profiles: List[KVCacheProfile] = []

    # Only consider symmetric K/V types (k=v) for simplicity
    for cache_type in KVCacheType:
        bpt = compute_kv_bytes_per_token(model_params, cache_type)
        if bpt <= 0:
            continue

        max_tokens = int(available_for_kv / bpt)
        if max_tokens < min_context:
            continue

        # Cap at a reasonable maximum
        max_tokens = min(max_tokens, 131072)

        impact = _QUALITY_IMPACT[cache_type]
        vram_at_target = min(max_tokens, target_context) * bpt

        # Build recommendation string
        if max_tokens >= target_context:
            rec = f"{cache_type.value}: full {target_context}-token context ({max_tokens} max)"
        else:
            rec = f"{cache_type.value}: reduced to {max_tokens} tokens (target: {target_context})"

        profiles.append(KVCacheProfile(
            cache_type_k=cache_type,
            cache_type_v=cache_type,
            max_context_tokens=max_tokens,
            vram_bytes=vram_at_target,
            quality_impact=impact,
            recommendation=rec,
        ))

    # Sort by quality impact (lower = better)
    profiles.sort(key=lambda p: p.quality_impact)
    return profiles
