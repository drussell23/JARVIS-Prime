"""
JARVIS Prime Computer Use Delegation
=====================================

Enables JARVIS Prime to delegate Computer Use tasks to main JARVIS.

Features:
- Computer Use task delegation to main JARVIS
- Vision analysis result caching
- Action Chaining optimization awareness
- OmniParser integration support

Architecture:
    JARVIS Prime (request) → ~/.jarvis/cross_repo/ → JARVIS (execution)
                                       ↓
                                   Results

Author: JARVIS Prime Team
Version: 3.1.0 - Computer Use Delegation
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

logger = logging.getLogger("jarvis-prime.computer-use")


# ============================================================================
# Constants
# ============================================================================

COMPUTER_USE_STATE_DIR = Path.home() / ".jarvis" / "cross_repo"
COMPUTER_USE_REQUESTS_FILE = COMPUTER_USE_STATE_DIR / "computer_use_requests.json"
COMPUTER_USE_RESULTS_FILE = COMPUTER_USE_STATE_DIR / "computer_use_results.json"
COMPUTER_USE_STATE_FILE = COMPUTER_USE_STATE_DIR / "computer_use_state.json"

REQUEST_TIMEOUT = 60.0  # seconds
POLLING_INTERVAL = 0.5  # seconds


# ============================================================================
# Enums
# ============================================================================

class RequestStatus(Enum):
    """Computer Use request status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class DelegationMode(Enum):
    """Computer Use delegation mode."""
    FULL_DELEGATION = "full_delegation"  # JARVIS Prime sends request, JARVIS executes
    VISION_ONLY = "vision_only"  # Only delegate vision analysis
    HYBRID = "hybrid"  # Use local inference + remote execution


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ComputerUseRequest:
    """A Computer Use task request from JARVIS Prime."""
    request_id: str
    timestamp: str
    goal: str
    context: Dict[str, Any]

    # Delegation settings
    mode: DelegationMode = DelegationMode.FULL_DELEGATION
    use_action_chaining: bool = True
    use_omniparser: bool = True  # v6.2: Enable by default with fallback

    # v6.2: Preferred parser mode (omniparser, claude_vision, ocr, auto)
    preferred_parser_mode: str = "auto"

    # Requester info
    requester: str = "jarvis-prime"
    priority: int = 1  # 1=normal, 2=high, 3=urgent

    # Timeout
    timeout_seconds: float = 60.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["mode"] = self.mode.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComputerUseRequest":
        """Create from dictionary."""
        mode_str = data.pop("mode", "full_delegation")
        mode = DelegationMode(mode_str)

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__}, mode=mode)


@dataclass
class ComputerUseResult:
    """Result of a Computer Use task."""
    request_id: str
    timestamp: str
    status: RequestStatus

    # Results
    success: bool = False
    actions_executed: int = 0
    execution_time_ms: float = 0.0

    # Outputs
    final_state: str = ""
    error_message: str = ""

    # Optimization metrics
    used_action_chaining: bool = False
    used_omniparser: bool = False
    time_saved_ms: float = 0.0
    tokens_saved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComputerUseResult":
        """Create from dictionary."""
        status_str = data.pop("status", "pending")
        status = RequestStatus(status_str)

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__}, status=status)


# ============================================================================
# Computer Use Delegate
# ============================================================================

class ComputerUseDelegate:
    """
    Delegates Computer Use tasks from JARVIS Prime to main JARVIS.

    Features:
    - Async task delegation via shared state files
    - Result polling with timeout
    - Request queueing and prioritization
    - Optimization metrics tracking
    """

    def __init__(
        self,
        default_mode: DelegationMode = DelegationMode.FULL_DELEGATION,
        enable_action_chaining: bool = True,
        enable_omniparser: bool = True,  # v6.2: Enable by default with fallback
    ):
        """
        Initialize Computer Use delegate.

        Args:
            default_mode: Default delegation mode
            enable_action_chaining: Request action chaining optimization
            enable_omniparser: Request OmniParser UI parsing (with intelligent fallback)
        """
        self.default_mode = default_mode
        self.enable_action_chaining = enable_action_chaining
        self.enable_omniparser = enable_omniparser

        self._requests: Dict[str, ComputerUseRequest] = {}
        self._results: Dict[str, ComputerUseResult] = {}

        # Ensure state directory exists
        COMPUTER_USE_STATE_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[COMPUTER USE DELEGATE] Initialized "
            f"(mode={default_mode.value}, "
            f"action_chaining={enable_action_chaining}, "
            f"omniparser={enable_omniparser})"
        )

    async def execute_task(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: float = REQUEST_TIMEOUT,
        mode: Optional[DelegationMode] = None,
    ) -> ComputerUseResult:
        """
        Delegate a Computer Use task to main JARVIS.

        Args:
            goal: The task goal
            context: Optional context dictionary
            timeout: Request timeout in seconds
            mode: Delegation mode (defaults to instance default)

        Returns:
            ComputerUseResult with task outcome
        """
        request_id = f"prime-{int(time.time() * 1000)}"

        request = ComputerUseRequest(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            goal=goal,
            context=context or {},
            mode=mode or self.default_mode,
            use_action_chaining=self.enable_action_chaining,
            use_omniparser=self.enable_omniparser,
            requester="jarvis-prime",
            timeout_seconds=timeout,
        )

        logger.info(f"[COMPUTER USE DELEGATE] Delegating task: {goal}")

        # Write request
        await self._write_request(request)
        self._requests[request_id] = request

        # Wait for result
        result = await self._wait_for_result(request_id, timeout)

        logger.info(
            f"[COMPUTER USE DELEGATE] Task completed: "
            f"success={result.success}, "
            f"time={result.execution_time_ms:.0f}ms"
        )

        return result

    async def check_jarvis_availability(self) -> bool:
        """
        Check if main JARVIS Computer Use is available.

        Returns:
            True if JARVIS is running and has Computer Use enabled
        """
        try:
            if not COMPUTER_USE_STATE_FILE.exists():
                return False

            content = COMPUTER_USE_STATE_FILE.read_text()
            state = json.loads(content)

            # Check if state is recent (not stale)
            last_update = state.get("last_update", "")
            if last_update:
                update_time = datetime.fromisoformat(last_update)
                age_seconds = (datetime.now() - update_time).total_seconds()

                if age_seconds < 120:  # Fresh within 2 minutes
                    return True

        except Exception as e:
            logger.debug(f"Failed to check JARVIS availability: {e}")

        return False

    async def get_jarvis_capabilities(self) -> Dict[str, bool]:
        """
        Get JARVIS Computer Use capabilities.

        Returns:
            Dictionary of capability flags
        """
        try:
            if COMPUTER_USE_STATE_FILE.exists():
                content = COMPUTER_USE_STATE_FILE.read_text()
                state = json.loads(content)

                return {
                    "available": True,
                    "action_chaining_enabled": state.get("action_chaining_enabled", False),
                    "omniparser_enabled": state.get("omniparser_enabled", False),
                    "omniparser_initialized": state.get("omniparser_initialized", False),
                }
        except Exception:
            pass

        return {
            "available": False,
            "action_chaining_enabled": False,
            "omniparser_enabled": False,
            "omniparser_initialized": False,
        }

    async def _write_request(self, request: ComputerUseRequest) -> None:
        """Write request to shared file."""
        try:
            # Load existing requests
            requests = []
            if COMPUTER_USE_REQUESTS_FILE.exists():
                try:
                    content = COMPUTER_USE_REQUESTS_FILE.read_text()
                    requests = json.loads(content)
                except json.JSONDecodeError:
                    requests = []

            # Add new request
            requests.append(request.to_dict())

            # Keep only last 100 requests
            requests = requests[-100:]

            # Write back
            COMPUTER_USE_REQUESTS_FILE.write_text(json.dumps(requests, indent=2))

        except Exception as e:
            logger.error(f"Failed to write Computer Use request: {e}")
            raise

    async def _wait_for_result(
        self,
        request_id: str,
        timeout: float,
    ) -> ComputerUseResult:
        """
        Wait for task result with timeout.

        Args:
            request_id: Request ID to wait for
            timeout: Timeout in seconds

        Returns:
            ComputerUseResult
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
        logger.warning(f"[COMPUTER USE DELEGATE] Task timeout after {timeout}s")
        return ComputerUseResult(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            status=RequestStatus.TIMEOUT,
            success=False,
            error_message=f"Task timeout after {timeout} seconds",
        )

    async def _read_result(self, request_id: str) -> Optional[ComputerUseResult]:
        """Read result from shared file."""
        try:
            if not COMPUTER_USE_RESULTS_FILE.exists():
                return None

            content = COMPUTER_USE_RESULTS_FILE.read_text()
            results = json.loads(content)

            # Find result for this request
            for result_data in results:
                if result_data.get("request_id") == request_id:
                    return ComputerUseResult.from_dict(result_data)

        except Exception as e:
            logger.debug(f"Failed to read Computer Use result: {e}")

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get delegation statistics."""
        total_requests = len(self._requests)
        total_results = len(self._results)
        successful = sum(1 for r in self._results.values() if r.success)

        return {
            "total_requests": total_requests,
            "total_results": total_results,
            "successful": successful,
            "success_rate": (successful / total_results * 100) if total_results > 0 else 0,
            "default_mode": self.default_mode.value,
            "action_chaining_enabled": self.enable_action_chaining,
            "omniparser_enabled": self.enable_omniparser,
        }


# ============================================================================
# Global Instance
# ============================================================================

_delegate_instance: Optional[ComputerUseDelegate] = None


def get_computer_use_delegate(
    mode: DelegationMode = DelegationMode.FULL_DELEGATION,
    enable_action_chaining: bool = True,
    enable_omniparser: bool = True,  # v6.2: Enable by default
) -> ComputerUseDelegate:
    """Get or create the global Computer Use delegate."""
    global _delegate_instance

    if _delegate_instance is None:
        _delegate_instance = ComputerUseDelegate(
            default_mode=mode,
            enable_action_chaining=enable_action_chaining,
            enable_omniparser=enable_omniparser,
        )

    return _delegate_instance


# ============================================================================
# Convenience Functions
# ============================================================================

async def delegate_computer_use_task(
    goal: str,
    context: Optional[Dict[str, Any]] = None,
    timeout: float = REQUEST_TIMEOUT,
) -> ComputerUseResult:
    """Convenience function to delegate a Computer Use task."""
    delegate = get_computer_use_delegate()
    return await delegate.execute_task(goal, context, timeout)


async def check_computer_use_available() -> bool:
    """Check if Computer Use delegation is available."""
    delegate = get_computer_use_delegate()
    return await delegate.check_jarvis_availability()
