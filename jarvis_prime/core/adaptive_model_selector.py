"""
Adaptive Model Selector — Proposal Engine
==========================================

Scans model directory, groups by family, and proposes optimal
model selections. Read-only I/O. Proposes plans — never executes them.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from jarvis_prime.core.quantization_intelligence import (
    KNOWN_PROFILES,
    CalibrationData,
    QuantizationProfile,
    QuantizationQualityScore,
    rank_quantizations,
    score_quantization,
)
from jarvis_prime.core.kv_cache_optimizer import (
    KVCacheProfile,
    KVCacheType,
    KNOWN_ARCHITECTURES,
    compute_feasible_profiles,
)

logger = logging.getLogger(__name__)


# =============================================================================
# FILENAME PARSING
# =============================================================================

# Matches quant suffix before .gguf: -Q4_K_M, -IQ2_M, -Q8_0, etc.
_QUANT_PATTERN = re.compile(
    r"-("
    r"IQ[12345]_(?:XXS|XS|S|M|L|XL)"
    r"|Q[2345678]_(?:K_S|K_M|K_L|K|0)"
    r"|Q8_0"
    r"|F16|F32|BF16"
    r")\.gguf$",
    re.IGNORECASE,
)

# Extract parameter count from name
_PARAM_PATTERN = re.compile(r"(\d+(?:\.\d+)?)[Bb]", re.IGNORECASE)


def parse_gguf_filename(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a GGUF filename into (base_model, quant_name).

    Example: "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"
             → ("qwen2.5-coder-32b-instruct", "Q4_K_M")
    """
    match = _QUANT_PATTERN.search(filename)
    if not match:
        return (filename.replace(".gguf", "").lower(), None)

    quant_name = match.group(1).upper()
    # Normalize: IQ2_M stays IQ2_M, Q4_K_M stays Q4_K_M
    base = filename[:match.start()].lower()
    return (base, quant_name)


def _extract_param_count(name: str) -> float:
    """Extract parameter count in billions from model name."""
    match = _PARAM_PATTERN.search(name)
    if match:
        return float(match.group(1))
    return 0.0


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass(frozen=True)
class ModelVariant:
    """A single GGUF file with parsed metadata."""
    path: Path
    size_bytes: int
    base_model: str
    quant_name: str
    quant_profile: QuantizationProfile
    sha256: Optional[str] = None
    provenance: str = "local"


@dataclass(frozen=True)
class ModelFamily:
    """All quantization variants of one base model."""
    base_model: str
    variants: Tuple[ModelVariant, ...]
    parameter_count: float


@dataclass(frozen=True)
class ModelSelectionProposal:
    """A proposed model change — advisory only, not executed."""
    proposal_id: str
    selected_variant: ModelVariant
    quality_score: QuantizationQualityScore
    kv_cache_profile: Optional[KVCacheProfile]
    reason: str
    trigger: str
    inventory_digest: str
    timestamp: float


# =============================================================================
# INVENTORY SCANNING
# =============================================================================

def _compute_inventory_digest(model_dir: Path) -> str:
    """SHA256 of sorted (filename, size) pairs."""
    entries = []
    for p in sorted(model_dir.glob("*.gguf")):
        if p.is_file():
            entries.append(f"{p.name}:{p.stat().st_size}")
    return hashlib.sha256("|".join(entries).encode()).hexdigest()[:16]


async def scan_inventory(model_dir: Path) -> List[ModelFamily]:
    """Scan model directory, group by family. Read-only."""
    families_map: Dict[str, List[ModelVariant]] = {}

    for path in sorted(model_dir.glob("*.gguf")):
        if not path.is_file() or path.name.startswith("."):
            continue

        base, quant_name = parse_gguf_filename(path.name)
        if not quant_name or quant_name.upper() not in KNOWN_PROFILES:
            logger.debug(f"[ModelSelector] Skipping {path.name}: unknown quant {quant_name}")
            continue

        profile = KNOWN_PROFILES[quant_name.upper()]
        variant = ModelVariant(
            path=path,
            size_bytes=path.stat().st_size,
            base_model=base,
            quant_name=quant_name.upper(),
            quant_profile=profile,
        )

        families_map.setdefault(base, []).append(variant)

    families = []
    for base, variants in families_map.items():
        # Sort by quality descending (higher bpw = better quality)
        variants.sort(key=lambda v: v.quant_profile.bits_per_weight, reverse=True)
        param_count = _extract_param_count(base)
        families.append(ModelFamily(
            base_model=base,
            variants=tuple(variants),
            parameter_count=param_count,
        ))

    return families


# =============================================================================
# PROPOSAL GENERATION
# =============================================================================

async def propose_optimal(
    families: List[ModelFamily],
    vram_budget_bytes: int,
    target_context: int = 8192,
    task_complexity: str = "medium",
    current_model: Optional[ModelVariant] = None,
    calibration: Optional[CalibrationData] = None,
    trigger: str = "startup",
    model_dir: Optional[Path] = None,
) -> Optional[ModelSelectionProposal]:
    """Propose the best model for current conditions. Does NOT execute."""
    # Collect all variants across families
    all_available: List[Tuple[QuantizationProfile, int, ModelVariant]] = []
    for family in families:
        for variant in family.variants:
            all_available.append((variant.quant_profile, variant.size_bytes, variant))

    if not all_available:
        return None

    # Score and rank
    ranked_pairs: List[Tuple[QuantizationQualityScore, ModelVariant]] = []
    for profile, size, variant in all_available:
        score = score_quantization(
            profile=profile,
            model_family=variant.base_model,
            model_size_bytes=size,
            total_vram_bytes=vram_budget_bytes,
            target_context=target_context,
            task_complexity=task_complexity,
            calibration_data=calibration,
        )
        if score.fitness_score > 0.0:
            ranked_pairs.append((score, variant))

    if not ranked_pairs:
        return None

    ranked_pairs.sort(key=lambda p: p[0].fitness_score, reverse=True)
    best_score, best_variant = ranked_pairs[0]

    # Compute KV cache profile
    kv_profile = None
    arch_key = best_variant.base_model.replace("-instruct", "")
    for key, params in KNOWN_ARCHITECTURES.items():
        if key in arch_key:
            profiles = compute_feasible_profiles(
                model_params=params,
                model_weight_bytes=best_variant.size_bytes,
                total_vram_bytes=vram_budget_bytes,
                target_context=target_context,
            )
            if profiles:
                kv_profile = profiles[0]  # Best quality
            break

    # Build reason
    reason = (
        f"Selected {best_variant.path.name} "
        f"(fitness={best_score.fitness_score:.3f}, "
        f"quality={best_score.quality_score:.3f}, "
        f"tok/s≈{best_score.estimated_tok_s:.1f}, "
        f"ctx≈{best_score.context_headroom_tokens})"
    )

    digest = _compute_inventory_digest(best_variant.path.parent) if model_dir else "unknown"

    return ModelSelectionProposal(
        proposal_id=f"prop-{uuid.uuid4().hex[:12]}",
        selected_variant=best_variant,
        quality_score=best_score,
        kv_cache_profile=kv_profile,
        reason=reason,
        trigger=trigger,
        inventory_digest=digest,
        timestamp=time.time(),
    )
