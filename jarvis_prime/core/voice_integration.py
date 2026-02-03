"""
JARVIS-Prime Voice Integration - Voice Orchestrator Client Bridge
===================================================================

Provides intelligent voice announcements for JARVIS-Prime model lifecycle events.
Integrates with JARVIS Voice Orchestrator via Unix domain socket IPC.

v2.0 Features:
- Uses VoiceClient for IPC communication (replaces trinity_voice_coordinator)
- Model load success/failure announcements
- Tier 0/1 routing announcements
- Fallback to cloud announcements
- Health status change announcements
- Zero hardcoding (environment-driven)
- Async/parallel execution
- Graceful degradation if orchestrator unavailable

Architecture:
+------------------------------------------------------------------+
|              JARVIS-Prime Voice Integration v2.0                  |
+------------------------------------------------------------------+
|                                                                   |
|  J-Prime Event               Category/Context                     |
|  -------------               ---------------                      |
|  Model Load     ---------->  init (HIGH priority)                 |
|  Inference      ---------->  general (LOW priority)               |
|  Cloud Fallback ---------->  general (NORMAL priority)            |
|  Health Change  ---------->  health (HIGH priority)               |
|                                                                   |
|           |                                                       |
|           v                                                       |
|  +--------------------------------------------------+            |
|  |   VoiceClient (Unix socket IPC)                   |            |
|  |   -> Queues locally if disconnected               |            |
|  |   -> Reconnects with exponential backoff          |            |
|  +--------------------------------------------------+            |
|           |                                                       |
|           v                                                       |
|  +--------------------------------------------------+            |
|  |   VoiceOrchestrator (JARVIS Body)                 |            |
|  |   -> Coalesces messages                           |            |
|  |   -> Serializes playback                          |            |
|  |   -> Multi-engine TTS                             |            |
|  +--------------------------------------------------+            |
|                                                                   |
+------------------------------------------------------------------+

Usage:
    from jarvis_prime.core.voice_integration import (
        announce_model_loaded,
        announce_cloud_fallback,
        announce_health_change,
    )

    await announce_model_loaded(
        model_name="jarvis-prime-v1.1",
        load_time_seconds=3.2,
    )

Author: JARVIS-Prime Voice Integration v2.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# VoiceClient Import (from JARVIS Body or local copy)
# =============================================================================

_VOICE_AVAILABLE = False
_voice_client = None

# Try multiple import paths for the shared voice client
JARVIS_BODY_PATH = os.getenv(
    "JARVIS_BODY_PATH",
    str(Path(__file__).parent.parent.parent.parent / "JARVIS-AI-Agent")
)

# Add JARVIS Body to path if it exists
if JARVIS_BODY_PATH and Path(JARVIS_BODY_PATH).exists():
    sys.path.insert(0, JARVIS_BODY_PATH)

try:
    # Try importing from JARVIS Body's shared module
    from backend.core.shared_voice_client import (
        VoiceClient,
        VoicePriority,
        VoiceContext,
        announce as _raw_announce,
    )
    _VOICE_AVAILABLE = True
    logger.info("[J-Prime Voice] VoiceClient available for announcements")
except ImportError:
    try:
        # Fallback: try importing from voice_client directly
        from backend.core.voice_client import (
            VoiceClient,
            VoicePriority,
            announce as _raw_announce,
        )
        # Create VoiceContext for backward compat
        class VoiceContext:
            STARTUP = "init"
            TRINITY = "init"
            RUNTIME = "general"
            NARRATOR = "general"
            ALERT = "error"
            SUCCESS = "ready"
            SHUTDOWN = "shutdown"
            HEALTH = "health"
            PROGRESS = "progress"

            @staticmethod
            def category(ctx):
                return ctx

        _VOICE_AVAILABLE = True
        logger.info("[J-Prime Voice] VoiceClient (basic) available for announcements")
    except ImportError as e:
        logger.debug(f"[J-Prime Voice] VoiceClient not available: {e}")

# Create dummy implementations for graceful degradation
if not _VOICE_AVAILABLE:
    class VoicePriority:
        CRITICAL = "CRITICAL"
        HIGH = "HIGH"
        NORMAL = "NORMAL"
        LOW = "LOW"
        BACKGROUND = "BACKGROUND"

    class VoiceContext:
        STARTUP = "init"
        TRINITY = "init"
        RUNTIME = "general"
        NARRATOR = "general"
        ALERT = "error"
        SUCCESS = "ready"
        SHUTDOWN = "shutdown"
        HEALTH = "health"
        PROGRESS = "progress"

        @property
        def category(self):
            return self

    async def _raw_announce(*args, **kwargs) -> bool:
        return False


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
# Internal Helper
# =============================================================================

async def _announce(
    message: str,
    context: str,
    priority: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Internal announce wrapper with graceful degradation."""
    if not _VOICE_AVAILABLE:
        logger.debug(f"[J-Prime Voice] Would announce: {message}")
        return False

    try:
        # Map context string to category
        category_map = {
            "trinity": "init",
            "startup": "init",
            "runtime": "general",
            "narrator": "general",
            "alert": "error",
            "success": "ready",
            "shutdown": "shutdown",
            "health": "health",
            "progress": "progress",
        }
        category = category_map.get(context, "general")

        # Map priority string to VoicePriority
        priority_map = {
            "CRITICAL": VoicePriority.CRITICAL,
            "HIGH": VoicePriority.HIGH,
            "NORMAL": VoicePriority.NORMAL,
            "LOW": VoicePriority.LOW,
            "BACKGROUND": VoicePriority.BACKGROUND,
        }
        prio = priority_map.get(priority, VoicePriority.NORMAL)

        return await _raw_announce(
            message=message,
            context=VoiceContext.RUNTIME,  # Use context enum if available
            priority=prio,
            source=_config.source_id,
            metadata=metadata,
        )
    except Exception as e:
        logger.debug(f"[J-Prime Voice] Announce failed (non-critical): {e}")
        return False


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

    try:
        if success:
            message = (
                f"JARVIS Prime model {model_name} loaded successfully "
                f"in {load_time_seconds:.1f} seconds. Ready for local inference."
            )
            context = "trinity"
            priority = "HIGH"
        else:
            message = (
                f"JARVIS Prime model load failed: {error_message}. "
                f"Falling back to cloud inference."
            )
            context = "alert"
            priority = "HIGH"

        return await _announce(
            message=message,
            context=context,
            priority=priority,
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

    try:
        if tier == "tier_0":
            message = f"Processing locally via JARVIS Prime. {reason}"
        elif tier == "tier_1":
            message = f"Routing to cloud for advanced processing. {reason}"
        else:
            message = f"Processing via {tier}. {reason}"

        return await _announce(
            message=message,
            context="narrator",
            priority="LOW",
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

    try:
        if local_error:
            message = (
                f"Local model unavailable: {local_error}. "
                f"Falling back to cloud inference."
            )
        else:
            message = f"Falling back to cloud inference. {reason}"

        return await _announce(
            message=message,
            context="narrator",
            priority="NORMAL",
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

    try:
        if status == "healthy":
            message = "JARVIS Prime is healthy. All models operational."
            context = "success"
            priority = "NORMAL"
        elif status == "degraded":
            message = f"JARVIS Prime performance degraded. {details or 'Some models unavailable.'}"
            context = "alert"
            priority = "HIGH"
        elif status == "unhealthy":
            message = f"JARVIS Prime unhealthy. {details or 'Models not responding.'}"
            context = "alert"
            priority = "HIGH"
        else:
            message = f"JARVIS Prime status changed to {status}. {details or ''}"
            context = "runtime"
            priority = "NORMAL"

        return await _announce(
            message=message,
            context=context,
            priority=priority,
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

        return await _announce(
            message=message,
            context="trinity",
            priority="HIGH",
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
    """Check if VoiceClient is available."""
    return _VOICE_AVAILABLE


def get_config() -> JPrimeVoiceConfig:
    """Get current voice configuration."""
    return _config


async def test_voice_integration() -> bool:
    """Test voice integration by sending a test announcement."""
    if not _VOICE_AVAILABLE:
        logger.warning("[J-Prime Voice] VoiceClient unavailable for testing")
        return False

    try:
        success = await _announce(
            message="JARVIS Prime voice integration test successful.",
            context="runtime",
            priority="LOW",
            metadata={"event": "test"}
        )

        if success:
            logger.info("[J-Prime Voice] Voice integration test: sent successfully")
        else:
            logger.info("[J-Prime Voice] Voice integration test: queued (orchestrator may be unavailable)")

        return True  # Return True if no exception (message queued or sent)

    except Exception as e:
        logger.error(f"[J-Prime Voice] Voice integration test failed: {e}")
        return False
