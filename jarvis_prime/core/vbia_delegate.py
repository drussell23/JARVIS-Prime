"""
JARVIS Prime VBIA (Voice Biometric Intelligent Authentication) Delegation
=========================================================================

Enables JARVIS Prime to delegate voice authentication tasks to main JARVIS.

Features:
- Voice authentication task delegation to main JARVIS
- Multi-factor security result handling
- Visual security integration awareness
- LangGraph reasoning chain access
- Cross-repo event synchronization

Architecture:
    JARVIS Prime (request) → ~/.jarvis/cross_repo/ → JARVIS (execution)
                                       ↓
                                   Results

Author: JARVIS Prime Team
Version: 6.2.0 - VBIA Delegation
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("jarvis-prime.vbia")


# ============================================================================
# Constants
# ============================================================================

VBIA_STATE_DIR = Path.home() / ".jarvis" / "cross_repo"
VBIA_REQUESTS_FILE = VBIA_STATE_DIR / "vbia_requests.json"
VBIA_RESULTS_FILE = VBIA_STATE_DIR / "vbia_results.json"
VBIA_STATE_FILE = VBIA_STATE_DIR / "vbia_state.json"
VBIA_EVENTS_FILE = VBIA_STATE_DIR / "vbia_events.json"

REQUEST_TIMEOUT = 30.0  # seconds
POLLING_INTERVAL = 0.3  # seconds


# ============================================================================
# Enums
# ============================================================================

class VBIARequestStatus(Enum):
    """VBIA request status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class VBIASecurityLevel(Enum):
    """VBIA security level."""
    STANDARD = "standard"  # Voice + behavioral only
    ENHANCED = "enhanced"  # + Physics analysis
    MAXIMUM = "maximum"  # + Visual security + All factors


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class VBIARequest:
    """A Voice Biometric Authentication request from JARVIS Prime."""
    request_id: str
    timestamp: str
    audio_data_b64: str  # Base64-encoded audio for voice verification
    context: Dict[str, Any]

    # Security settings
    security_level: VBIASecurityLevel = VBIASecurityLevel.MAXIMUM
    enable_visual_security: bool = True  # v6.2: Visual threat detection
    enable_langgraph_reasoning: bool = True  # Multi-step CoT reasoning
    enable_pattern_learning: bool = True  # Learn from this authentication

    # Requester info
    requester: str = "jarvis-prime"
    priority: int = 1  # 1=normal, 2=high, 3=urgent

    # Timeout
    timeout_seconds: float = 30.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["security_level"] = self.security_level.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VBIARequest":
        """Create from dictionary."""
        security_level_str = data.pop("security_level", "maximum")
        security_level = VBIASecurityLevel(security_level_str)

        return cls(
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__},
            security_level=security_level
        )


@dataclass
class VBIAResult:
    """Result of a Voice Biometric Authentication task."""
    request_id: str
    timestamp: str
    status: VBIARequestStatus

    # Authentication decision
    authenticated: bool = False
    speaker_name: str = ""
    decision_type: str = "reject"  # instant, confident, reasoned, borderline, reject

    # Multi-factor confidences
    ml_confidence: float = 0.0  # Voice ML confidence
    physics_confidence: float = 0.0  # Liveness, anti-spoofing
    behavioral_confidence: float = 0.0  # Time, location, patterns
    context_confidence: float = 0.0  # Environment quality
    visual_confidence: float = 0.0  # v6.2: Visual security confidence
    final_confidence: float = 0.0  # Bayesian fusion

    # Security analysis
    spoofing_detected: bool = False
    visual_threat_detected: bool = False
    liveness_passed: bool = True

    # Reasoning (if LangGraph was used)
    used_langgraph: bool = False
    reasoning_chain: List[str] = None  # Chain-of-thought steps
    hypotheses_evaluated: List[str] = None  # Hypotheses considered

    # Performance metrics
    execution_time_ms: float = 0.0
    analysis_mode_used: str = ""  # "fast_path", "standard", "deep_reasoning"

    # Messages
    success_message: str = ""
    error_message: str = ""
    warning_message: str = ""

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.reasoning_chain is None:
            self.reasoning_chain = []
        if self.hypotheses_evaluated is None:
            self.hypotheses_evaluated = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VBIAResult":
        """Create from dictionary."""
        status_str = data.pop("status", "pending")
        status = VBIARequestStatus(status_str)

        return cls(
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__},
            status=status
        )


# ============================================================================
# VBIA Delegate
# ============================================================================

class VBIADelegate:
    """
    Delegates Voice Biometric Authentication tasks from JARVIS Prime to main JARVIS.

    Features:
    - Async task delegation via shared state files
    - Result polling with timeout
    - Multi-factor security awareness
    - Visual security integration
    - LangGraph reasoning access
    """

    def __init__(
        self,
        default_security_level: VBIASecurityLevel = VBIASecurityLevel.MAXIMUM,
        enable_visual_security: bool = True,
        enable_langgraph_reasoning: bool = True,
    ):
        """
        Initialize VBIA delegate.

        Args:
            default_security_level: Default security level for requests
            enable_visual_security: Request visual security analysis
            enable_langgraph_reasoning: Request LangGraph multi-step reasoning
        """
        self.default_security_level = default_security_level
        self.enable_visual_security = enable_visual_security
        self.enable_langgraph_reasoning = enable_langgraph_reasoning

        self._requests: Dict[str, VBIARequest] = {}
        self._results: Dict[str, VBIAResult] = {}

        # Ensure state directory exists
        VBIA_STATE_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[VBIA DELEGATE] Initialized "
            f"(security_level={default_security_level.value}, "
            f"visual_security={enable_visual_security}, "
            f"langgraph_reasoning={enable_langgraph_reasoning})"
        )

    async def authenticate_speaker(
        self,
        audio_data_b64: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: float = REQUEST_TIMEOUT,
        security_level: Optional[VBIASecurityLevel] = None,
    ) -> VBIAResult:
        """
        Delegate voice authentication to main JARVIS.

        Args:
            audio_data_b64: Base64-encoded audio data
            context: Optional context dictionary
            timeout: Request timeout in seconds
            security_level: Security level (defaults to instance default)

        Returns:
            VBIAResult with authentication outcome
        """
        request_id = f"prime-vbia-{int(time.time() * 1000)}"

        request = VBIARequest(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            audio_data_b64=audio_data_b64,
            context=context or {},
            security_level=security_level or self.default_security_level,
            enable_visual_security=self.enable_visual_security,
            enable_langgraph_reasoning=self.enable_langgraph_reasoning,
            requester="jarvis-prime",
            timeout_seconds=timeout,
        )

        logger.info(
            f"[VBIA DELEGATE] Delegating authentication "
            f"(security_level={request.security_level.value})"
        )

        # Write request
        await self._write_request(request)
        self._requests[request_id] = request

        # Wait for result
        result = await self._wait_for_result(request_id, timeout)

        logger.info(
            f"[VBIA DELEGATE] Authentication completed: "
            f"authenticated={result.authenticated}, "
            f"confidence={result.final_confidence:.1%}, "
            f"time={result.execution_time_ms:.0f}ms"
        )

        return result

    async def check_jarvis_availability(self) -> bool:
        """
        Check if main JARVIS VBIA is available.

        Returns:
            True if JARVIS is running and has VBIA enabled
        """
        try:
            if not VBIA_STATE_FILE.exists():
                return False

            content = VBIA_STATE_FILE.read_text()
            state = json.loads(content)

            # Check if state is recent (not stale)
            last_update = state.get("last_update", "")
            if last_update:
                update_time = datetime.fromisoformat(last_update)
                age_seconds = (datetime.now() - update_time).total_seconds()

                if age_seconds < 120:  # Fresh within 2 minutes
                    return True

        except Exception as e:
            logger.debug(f"Failed to check JARVIS VBIA availability: {e}")

        return False

    async def get_jarvis_capabilities(self) -> Dict[str, bool]:
        """
        Get JARVIS VBIA capabilities.

        Returns:
            Dictionary of capability flags
        """
        try:
            if VBIA_STATE_FILE.exists():
                content = VBIA_STATE_FILE.read_text()
                state = json.loads(content)

                return {
                    "available": True,
                    "visual_security_enabled": state.get("visual_security_enabled", False),
                    "langgraph_reasoning_enabled": state.get("langgraph_reasoning_enabled", False),
                    "pattern_learning_enabled": state.get("pattern_learning_enabled", False),
                    "chromadb_enabled": state.get("chromadb_enabled", False),
                    "helicone_cost_tracking": state.get("helicone_cost_tracking", False),
                }
        except Exception:
            pass

        return {
            "available": False,
            "visual_security_enabled": False,
            "langgraph_reasoning_enabled": False,
            "pattern_learning_enabled": False,
            "chromadb_enabled": False,
            "helicone_cost_tracking": False,
        }

    async def get_recent_events(
        self,
        limit: int = 10,
        event_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get recent VBIA events from cross-repo bridge.

        Args:
            limit: Maximum number of events to return
            event_type: Filter by event type (e.g., "vbia_visual_security")

        Returns:
            List of event dictionaries
        """
        try:
            if not VBIA_EVENTS_FILE.exists():
                return []

            content = VBIA_EVENTS_FILE.read_text()
            events = json.loads(content)

            # Filter by type if specified
            if event_type:
                events = [e for e in events if e.get("event_type") == event_type]

            # Return latest N events
            return events[-limit:]

        except Exception as e:
            logger.debug(f"Failed to get VBIA events: {e}")
            return []

    async def _write_request(self, request: VBIARequest) -> None:
        """Write request to shared file."""
        try:
            # Load existing requests
            requests = []
            if VBIA_REQUESTS_FILE.exists():
                try:
                    content = VBIA_REQUESTS_FILE.read_text()
                    requests = json.loads(content)
                except json.JSONDecodeError:
                    requests = []

            # Add new request
            requests.append(request.to_dict())

            # Keep only last 100 requests
            requests = requests[-100:]

            # Write back
            VBIA_REQUESTS_FILE.write_text(json.dumps(requests, indent=2))

        except Exception as e:
            logger.error(f"Failed to write VBIA request: {e}")
            raise

    async def _wait_for_result(
        self,
        request_id: str,
        timeout: float,
    ) -> VBIAResult:
        """
        Wait for task result with timeout.

        Args:
            request_id: Request ID to wait for
            timeout: Timeout in seconds

        Returns:
            VBIAResult
        """
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            # Check for result
            result = await self._read_result(request_id)

            if result:
                self._results[request_id] = result
                return result

            # Poll interval
            await asyncio.sleep(POLLING_INTERVAL)

        # Timeout
        logger.warning(f"[VBIA DELEGATE] Authentication timeout after {timeout}s")
        return VBIAResult(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            status=VBIARequestStatus.TIMEOUT,
            authenticated=False,
            error_message=f"Authentication timeout after {timeout} seconds",
        )

    async def _read_result(self, request_id: str) -> Optional[VBIAResult]:
        """Read result from shared file."""
        try:
            if not VBIA_RESULTS_FILE.exists():
                return None

            content = VBIA_RESULTS_FILE.read_text()
            results = json.loads(content)

            # Find result for this request
            for result_data in results:
                if result_data.get("request_id") == request_id:
                    return VBIAResult.from_dict(result_data)

        except Exception as e:
            logger.debug(f"Failed to read VBIA result: {e}")

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get delegation statistics."""
        total_requests = len(self._requests)
        total_results = len(self._results)
        authenticated = sum(1 for r in self._results.values() if r.authenticated)

        return {
            "total_requests": total_requests,
            "total_results": total_results,
            "authenticated": authenticated,
            "authentication_rate": (
                authenticated / total_results * 100 if total_results > 0 else 0
            ),
            "default_security_level": self.default_security_level.value,
            "visual_security_enabled": self.enable_visual_security,
            "langgraph_reasoning_enabled": self.enable_langgraph_reasoning,
        }


# ============================================================================
# Global Instance
# ============================================================================

_delegate_instance: Optional[VBIADelegate] = None


def get_vbia_delegate(
    security_level: VBIASecurityLevel = VBIASecurityLevel.MAXIMUM,
    enable_visual_security: bool = True,
    enable_langgraph_reasoning: bool = True,
) -> VBIADelegate:
    """Get or create the global VBIA delegate."""
    global _delegate_instance

    if _delegate_instance is None:
        _delegate_instance = VBIADelegate(
            default_security_level=security_level,
            enable_visual_security=enable_visual_security,
            enable_langgraph_reasoning=enable_langgraph_reasoning,
        )

    return _delegate_instance


# ============================================================================
# Convenience Functions
# ============================================================================

async def delegate_voice_authentication(
    audio_data_b64: str,
    context: Optional[Dict[str, Any]] = None,
    timeout: float = REQUEST_TIMEOUT,
) -> VBIAResult:
    """Convenience function to delegate voice authentication."""
    delegate = get_vbia_delegate()
    return await delegate.authenticate_speaker(audio_data_b64, context, timeout)


async def check_vbia_available() -> bool:
    """Check if VBIA delegation is available."""
    delegate = get_vbia_delegate()
    return await delegate.check_jarvis_availability()
