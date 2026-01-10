"""
JARVIS-Prime Voice Integration - Trinity Voice Coordinator Bridge
==================================================================

Provides intelligent voice announcements for JARVIS-Prime model lifecycle events.
Integrates with Trinity Voice Coordinator (JARVIS Body repo) for cross-repo coordination.

v1.0 Features:
- Model load success/failure announcements
- Tier 0/1 routing announcements
- Fallback to cloud announcements
- Health status change announcements
- Zero hardcoding (environment-driven)
- Async/parallel execution
- Graceful degradation if voice unavailable

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│              JARVIS-Prime Voice Integration                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  J-Prime Event               Trinity Voice Context             │
│  ─────────────               ──────────────────                 │
│  • Model Load     ──────────▶ TRINITY context                  │
│  • Inference      ──────────▶ RUNTIME context                  │
│  • Cloud Fallback ──────────▶ NARRATOR context                 │
│  • Health Change  ──────────▶ ALERT/SUCCESS context            │
│                                                                 │
│           ▼                                                     │
│  ┌─────────────────────────────────────────┐                   │
│  │   Trinity Voice Coordinator              │                   │
│  │   (backend.core.trinity_voice_coordinator)│                  │
│  └─────────────────────────────────────────┘                   │
│           │                                                     │
│           ▼                                                     │
│  Multi-engine TTS (MacOS Say → pyttsx3 → Edge TTS)            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Usage:
    from jarvis_prime.core.voice_integration import (
        announce_model_loaded,
        announce_inference_complete,
        announce_cloud_fallback,
    )

    # After model load
    await announce_model_loaded(
        model_name="jarvis-prime-v1.1",
        load_time_seconds=3.2,
    )

    # After tier routing decision
    await announce_tier_routing(
        tier="tier_0",
        reason="Low complexity query",
    )

Author: JARVIS-Prime Trinity v1.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Add JARVIS body repo to path for Trinity Voice Coordinator import
JARVIS_BODY_PATH = os.getenv(
    "JARVIS_BODY_PATH",
    str(Path(__file__).parent.parent.parent.parent / "JARVIS-AI-Agent")
)
if JARVIS_BODY_PATH and Path(JARVIS_BODY_PATH).exists():
    sys.path.insert(0, JARVIS_BODY_PATH)


# =============================================================================
# Trinity Voice Coordinator Import (with graceful fallback)
# =============================================================================

_VOICE_AVAILABLE = False
_VOICE_COORDINATOR = None

try:
    from backend.core.trinity_voice_coordinator import (
        announce as trinity_announce,
        get_voice_coordinator,
        VoiceContext,
        VoicePriority,
    )
    _VOICE_AVAILABLE = True
    logger.info("✅ Trinity Voice Coordinator available for J-Prime announcements")
except ImportError as e:
    logger.debug(f"Trinity Voice Coordinator not available: {e}")
    # Create dummy implementations for graceful degradation
    class VoiceContext:
        STARTUP = "startup"
        TRINITY = "trinity"
        RUNTIME = "runtime"
        NARRATOR = "narrator"
        ALERT = "alert"
        SUCCESS = "success"

    class VoicePriority:
        CRITICAL = 0
        HIGH = 1
        NORMAL = 2
        LOW = 3
        BACKGROUND = 4

    async def trinity_announce(*args, **kwargs):
        return False

    async def get_voice_coordinator():
        return None


# =============================================================================
# Configuration
# =============================================================================

class JPrimeVoiceConfig:
    """Configuration for J-Prime voice announcements (environment-driven)."""

    def __init__(self):
        # Enable/disable voice announcements
        self.enabled = os.getenv("JPRIME_VOICE_ENABLED", "true").lower() == "true"

        # Announcement granularity
        self.announce_model_load = os.getenv("JPRIME_VOICE_MODEL_LOAD", "true").lower() == "true"
        self.announce_tier_routing = os.getenv("JPRIME_VOICE_TIER_ROUTING", "false").lower() == "true"
        self.announce_cloud_fallback = os.getenv("JPRIME_VOICE_CLOUD_FALLBACK", "true").lower() == "true"
        self.announce_health_changes = os.getenv("JPRIME_VOICE_HEALTH", "true").lower() == "true"

        # Source identifier for cross-repo tracking
        self.source_id = os.getenv("JPRIME_VOICE_SOURCE", "jarvis_prime")


_config = JPrimeVoiceConfig()


# =============================================================================
# Voice Announcement Functions
# =============================================================================

async def announce_model_loaded(
    model_name: str,
    load_time_seconds: float,
    success: bool = True,
    error_message: Optional[str] = None,
) -> bool:
    """
    Announce that a model has been loaded (or failed to load).

    Args:
        model_name: Name/version of the model
        load_time_seconds: Time taken to load
        success: Whether load was successful
        error_message: Error message if failed

    Returns:
        True if announcement was queued, False otherwise
    """
    if not _config.enabled or not _config.announce_model_load:
        return False

    if not _VOICE_AVAILABLE:
        logger.debug("[J-Prime Voice] Trinity coordinator unavailable, skipping announcement")
        return False

    try:
        if success:
            message = (
                f"JARVIS Prime model {model_name} loaded successfully "
                f"in {load_time_seconds:.1f} seconds. Ready for local inference."
            )
            context = VoiceContext.TRINITY
            priority = VoicePriority.HIGH
        else:
            message = (
                f"JARVIS Prime model load failed: {error_message}. "
                f"Falling back to cloud inference."
            )
            context = VoiceContext.ALERT
            priority = VoicePriority.HIGH

        return await trinity_announce(
            message=message,
            context=context,
            priority=priority,
            source=_config.source_id,
            metadata={
                "event": "model_load",
                "model_name": model_name,
                "load_time": load_time_seconds,
                "success": success,
            }
        )

    except Exception as e:
        logger.error(f"[J-Prime Voice] Failed to announce model load: {e}")
        return False


async def announce_tier_routing(
    tier: str,
    reason: str,
    complexity_score: Optional[float] = None,
) -> bool:
    """
    Announce tier routing decision (Tier 0 local vs Tier 1 cloud).

    Args:
        tier: "tier_0" or "tier_1"
        reason: Reason for routing decision
        complexity_score: Complexity score that influenced decision

    Returns:
        True if announcement was queued, False otherwise
    """
    if not _config.enabled or not _config.announce_tier_routing:
        return False

    if not _VOICE_AVAILABLE:
        return False

    try:
        if tier == "tier_0":
            message = f"Processing locally via JARVIS Prime. {reason}"
        elif tier == "tier_1":
            message = f"Routing to cloud for advanced processing. {reason}"
        else:
            message = f"Processing via {tier}. {reason}"

        return await trinity_announce(
            message=message,
            context=VoiceContext.NARRATOR,
            priority=VoicePriority.LOW,  # Low priority - don't interrupt
            source=_config.source_id,
            metadata={
                "event": "tier_routing",
                "tier": tier,
                "reason": reason,
                "complexity_score": complexity_score,
            }
        )

    except Exception as e:
        logger.error(f"[J-Prime Voice] Failed to announce tier routing: {e}")
        return False


async def announce_cloud_fallback(
    reason: str,
    local_error: Optional[str] = None,
) -> bool:
    """
    Announce fallback from local (Tier 0) to cloud (Tier 1).

    Args:
        reason: Reason for fallback
        local_error: Error that caused fallback

    Returns:
        True if announcement was queued, False otherwise
    """
    if not _config.enabled or not _config.announce_cloud_fallback:
        return False

    if not _VOICE_AVAILABLE:
        return False

    try:
        if local_error:
            message = (
                f"Local model unavailable: {local_error}. "
                f"Falling back to cloud inference."
            )
        else:
            message = f"Falling back to cloud inference. {reason}"

        return await trinity_announce(
            message=message,
            context=VoiceContext.NARRATOR,
            priority=VoicePriority.NORMAL,
            source=_config.source_id,
            metadata={
                "event": "cloud_fallback",
                "reason": reason,
                "error": local_error,
            }
        )

    except Exception as e:
        logger.error(f"[J-Prime Voice] Failed to announce cloud fallback: {e}")
        return False


async def announce_health_change(
    status: str,
    details: Optional[str] = None,
) -> bool:
    """
    Announce health status change.

    Args:
        status: "healthy", "degraded", "unhealthy"
        details: Additional context

    Returns:
        True if announcement was queued, False otherwise
    """
    if not _config.enabled or not _config.announce_health_changes:
        return False

    if not _VOICE_AVAILABLE:
        return False

    try:
        if status == "healthy":
            message = "JARVIS Prime is healthy. All models operational."
            context = VoiceContext.SUCCESS
            priority = VoicePriority.NORMAL
        elif status == "degraded":
            message = f"JARVIS Prime performance degraded. {details or 'Some models unavailable.'}"
            context = VoiceContext.ALERT
            priority = VoicePriority.HIGH
        elif status == "unhealthy":
            message = f"JARVIS Prime unhealthy. {details or 'Models not responding.'}"
            context = VoiceContext.ALERT
            priority = VoicePriority.HIGH
        else:
            message = f"JARVIS Prime status changed to {status}. {details or ''}"
            context = VoiceContext.RUNTIME
            priority = VoicePriority.NORMAL

        return await trinity_announce(
            message=message,
            context=context,
            priority=priority,
            source=_config.source_id,
            metadata={
                "event": "health_change",
                "status": status,
                "details": details,
            }
        )

    except Exception as e:
        logger.error(f"[J-Prime Voice] Failed to announce health change: {e}")
        return False


async def announce_manager_started(
    models_loaded: int = 0,
    startup_time_seconds: float = 0.0,
) -> bool:
    """
    Announce that the PrimeModelManager has started successfully.

    Args:
        models_loaded: Number of models loaded
        startup_time_seconds: Time taken to start

    Returns:
        True if announcement was queued, False otherwise
    """
    if not _config.enabled:
        return False

    if not _VOICE_AVAILABLE:
        return False

    try:
        if models_loaded > 0:
            message = (
                f"JARVIS Prime initialized successfully in {startup_time_seconds:.1f} seconds. "
                f"{models_loaded} model{'s' if models_loaded > 1 else ''} ready for local inference."
            )
        else:
            message = (
                f"JARVIS Prime initialized in {startup_time_seconds:.1f} seconds. "
                f"Cloud inference available."
            )

        return await trinity_announce(
            message=message,
            context=VoiceContext.TRINITY,
            priority=VoicePriority.HIGH,
            source=_config.source_id,
            metadata={
                "event": "manager_started",
                "models_loaded": models_loaded,
                "startup_time": startup_time_seconds,
            }
        )

    except Exception as e:
        logger.error(f"[J-Prime Voice] Failed to announce manager start: {e}")
        return False


# =============================================================================
# Utility Functions
# =============================================================================

def is_voice_available() -> bool:
    """Check if Trinity Voice Coordinator is available."""
    return _VOICE_AVAILABLE


def get_config() -> JPrimeVoiceConfig:
    """Get current voice configuration."""
    return _config


async def test_voice_integration() -> bool:
    """Test voice integration by sending a test announcement."""
    if not _VOICE_AVAILABLE:
        logger.warning("[J-Prime Voice] Trinity coordinator unavailable for testing")
        return False

    try:
        success = await trinity_announce(
            message="JARVIS Prime voice integration test successful.",
            context=VoiceContext.RUNTIME,
            priority=VoicePriority.LOW,
            source=_config.source_id,
            metadata={"event": "test"}
        )

        if success:
            logger.info("[J-Prime Voice] ✅ Voice integration test successful")
        else:
            logger.warning("[J-Prime Voice] ⚠️  Voice integration test returned False")

        return success

    except Exception as e:
        logger.error(f"[J-Prime Voice] ❌ Voice integration test failed: {e}")
        return False
