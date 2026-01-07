"""
JARVIS Prime Bridge - Main JARVIS Integration
===============================================

v79.1 - Enhanced with circuit breakers and fallback chains

This bridge connects the main JARVIS action execution system with
JARVIS-Prime's AGI capabilities, replacing direct Claude API calls
with local cognitive processing.

CHANGES in v79.1:
    - Added circuit breaker for AGI component protection
    - Implemented fallback chain for graceful degradation
    - Added response caching for repeated failures
    - Enhanced error classification and recovery

INTEGRATION POINTS:
    1. Command Processing: Route commands through AGI orchestrator
    2. Action Planning: Use ActionModel for execution plans
    3. Safety Checks: Integrate with JARVIS safety manager
    4. Screen Understanding: Feed screen data to multimodal fusion
    5. Learning Feedback: Collect experiences for continuous learning

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        JARVIS (Body)                           │
    │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
    │  │ Command      │  │ Action       │  │  Safety               │ │
    │  │ Processor    │  │ Executor     │  │  Manager              │ │
    │  └──────┬───────┘  └──────┬───────┘  └───────────┬───────────┘ │
    └─────────┼─────────────────┼──────────────────────┼─────────────┘
              │                 │                      │
              ▼                 ▼                      ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    JARVIS PRIME BRIDGE                          │
    │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
    │  │ AGI          │  │ Action       │  │  Safety               │ │
    │  │ Orchestrator │  │ Model        │  │  Context              │ │
    │  └──────────────┘  └──────────────┘  └───────────────────────┘ │
    └─────────────────────────────────────────────────────────────────┘
              │                 │                      │
              ▼                 ▼                      ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    JARVIS-Prime (Mind)                          │
    │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
    │  │ Reasoning    │  │ Continuous   │  │  MultiModal           │ │
    │  │ Engine       │  │ Learning     │  │  Fusion               │ │
    │  └──────────────┘  └──────────────┘  └───────────────────────┘ │
    └─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# v79.1: CIRCUIT BREAKER AND FALLBACK CONFIGURATION
# =============================================================================

@dataclass
class BridgeResilienceConfig:
    """Configuration for bridge resilience features."""

    # Circuit breaker settings
    circuit_failure_threshold: int = int(os.getenv(
        "JARVIS_BRIDGE_CIRCUIT_FAILURES", "5"
    ))
    circuit_recovery_timeout_sec: float = float(os.getenv(
        "JARVIS_BRIDGE_CIRCUIT_RECOVERY_SEC", "30.0"
    ))
    circuit_half_open_requests: int = int(os.getenv(
        "JARVIS_BRIDGE_CIRCUIT_HALF_OPEN", "3"
    ))

    # Retry settings
    max_retries: int = int(os.getenv("JARVIS_BRIDGE_MAX_RETRIES", "3"))
    retry_base_delay_ms: int = int(os.getenv(
        "JARVIS_BRIDGE_RETRY_DELAY_MS", "500"
    ))
    retry_jitter: float = float(os.getenv("JARVIS_BRIDGE_RETRY_JITTER", "0.3"))

    # Fallback settings
    enable_simple_fallback: bool = os.getenv(
        "JARVIS_BRIDGE_SIMPLE_FALLBACK", "true"
    ).lower() == "true"
    enable_cache_fallback: bool = os.getenv(
        "JARVIS_BRIDGE_CACHE_FALLBACK", "true"
    ).lower() == "true"
    cache_ttl_seconds: int = int(os.getenv(
        "JARVIS_BRIDGE_CACHE_TTL", "300"
    ))

    # Timeout settings
    command_timeout_seconds: float = float(os.getenv(
        "JARVIS_BRIDGE_COMMAND_TIMEOUT", "60.0"
    ))


class ResponseCache:
    """
    v79.1: LRU cache for responses to support fallback on failures.

    Stores successful responses and can return cached versions
    when the AGI system is unavailable.
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 300):
        self._cache: Dict[str, Tuple[PrimeResponse, float]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._access_order: List[str] = []  # For LRU eviction

    def _make_key(self, command: "JARVISCommand") -> str:
        """Create a cache key from command content."""
        # Hash the command content for cache key
        content_hash = hashlib.sha256(
            f"{command.command_type.name}:{command.content}".encode()
        ).hexdigest()[:16]
        return content_hash

    def get(self, command: "JARVISCommand") -> Optional["PrimeResponse"]:
        """Get cached response if available and not expired."""
        key = self._make_key(command)

        if key in self._cache:
            response, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                # Update access order for LRU
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                logger.debug(f"Cache hit for command {command.id}")
                return response
            else:
                # Expired
                del self._cache[key]

        return None

    def put(self, command: "JARVISCommand", response: "PrimeResponse") -> None:
        """Store a response in the cache."""
        # Only cache successful responses
        if not response.success:
            return

        key = self._make_key(command)

        # Evict if at capacity
        while len(self._cache) >= self._max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._cache.pop(oldest_key, None)

        self._cache[key] = (response, time.time())
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================


class CommandType(Enum):
    """Types of commands from JARVIS."""

    NATURAL_LANGUAGE = auto()   # Free-form text commands
    STRUCTURED = auto()          # JSON-structured commands
    ACTION_SEQUENCE = auto()     # Pre-planned action sequences
    QUERY = auto()               # Information queries
    SCREEN_ANALYSIS = auto()     # Screen understanding requests
    CONFIRMATION = auto()        # User confirmation responses
    FEEDBACK = auto()            # User feedback on actions


class ProcessingMode(Enum):
    """How to process commands."""

    DIRECT = auto()             # Fast path, no reasoning
    REASONED = auto()           # Apply reasoning engine
    PLANNED = auto()            # Full action planning
    CAUTIOUS = auto()           # Extra safety checks


class ActionRisk(Enum):
    """Risk level of proposed actions."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class JARVISCommand:
    """Command from main JARVIS system."""

    id: str
    command_type: CommandType
    content: str
    context: Dict[str, Any] = field(default_factory=dict)

    # Screen context
    screen_data: Optional[bytes] = None
    screen_elements: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # Processing hints
    require_confirmation: bool = False
    max_actions: int = 10
    timeout_seconds: float = 30.0


@dataclass
class PrimeResponse:
    """Response from JARVIS-Prime."""

    command_id: str
    success: bool

    # Primary response
    response_text: str = ""
    actions: List[Dict[str, Any]] = field(default_factory=list)

    # Reasoning trace
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0

    # Safety assessment
    risk_level: ActionRisk = ActionRisk.SAFE
    safety_notes: List[str] = field(default_factory=list)
    requires_confirmation: bool = False

    # Models used
    models_used: List[str] = field(default_factory=list)

    # Metrics
    processing_time_ms: float = 0.0

    # Error info
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "success": self.success,
            "response_text": self.response_text,
            "actions": self.actions,
            "reasoning_trace": self.reasoning_trace,
            "confidence": self.confidence,
            "risk_level": self.risk_level.value,
            "safety_notes": self.safety_notes,
            "requires_confirmation": self.requires_confirmation,
            "models_used": self.models_used,
            "processing_time_ms": self.processing_time_ms,
            "error": self.error,
        }


@dataclass
class BridgeConfig:
    """Configuration for JARVIS Prime Bridge."""

    # Processing
    default_mode: ProcessingMode = ProcessingMode.REASONED
    enable_reasoning: bool = True
    enable_action_planning: bool = True
    enable_safety_checks: bool = True

    # Thresholds
    confidence_threshold: float = 0.7
    risk_auto_confirm: ActionRisk = ActionRisk.LOW
    max_retry_attempts: int = 3

    # Timeouts
    reasoning_timeout: float = 10.0
    planning_timeout: float = 15.0
    total_timeout: float = 30.0

    # Learning
    record_experiences: bool = True
    record_feedback: bool = True

    # Safety context path
    safety_context_file: Path = field(
        default_factory=lambda: Path.home() / ".jarvis" / "cross_repo" / "safety" / "context_for_prime.json"
    )


# =============================================================================
# SAFETY CONTEXT INTEGRATION
# =============================================================================


@dataclass
class SafetyContext:
    """Safety context from JARVIS."""

    kill_switch_active: bool = False
    current_risk_level: str = "low"
    user_trust_level: float = 1.0
    recent_blocks: int = 0
    recent_denials: int = 0
    pending_confirmation: bool = False

    def should_be_cautious(self) -> bool:
        """Check if we should be extra cautious."""
        return (
            self.kill_switch_active
            or self.current_risk_level in ("high", "critical")
            or self.user_trust_level < 0.7
            or self.recent_denials >= 2
        )

    def get_risk_multiplier(self) -> float:
        """Get risk multiplier for action assessment."""
        multiplier = self.user_trust_level
        if self.recent_denials > 0:
            multiplier *= 0.8
        if self.recent_blocks > 0:
            multiplier *= 0.9
        return max(0.5, multiplier)


class SafetyContextReader:
    """Reads safety context from shared file."""

    def __init__(self, filepath: Path) -> None:
        self._filepath = filepath
        self._cache: Optional[SafetyContext] = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 1.0

    def read(self) -> SafetyContext:
        """Read current safety context."""
        now = time.time()

        if self._cache and (now - self._cache_time) < self._cache_ttl:
            return self._cache

        try:
            if not self._filepath.exists():
                return SafetyContext()

            data = json.loads(self._filepath.read_text())
            context = SafetyContext(
                kill_switch_active=data.get("kill_switch_active", False),
                current_risk_level=data.get("current_risk_level", "low"),
                user_trust_level=data.get("user_trust_level", 1.0),
                recent_blocks=data.get("recent_blocks", 0),
                recent_denials=data.get("recent_denials", 0),
                pending_confirmation=data.get("pending_confirmation", False),
            )

            self._cache = context
            self._cache_time = now
            return context

        except Exception as e:
            logger.warning(f"Failed to read safety context: {e}")
            return SafetyContext()


# =============================================================================
# ACTION RISK ANALYZER
# =============================================================================


class ActionRiskAnalyzer:
    """Analyze risk level of proposed actions."""

    # Risk patterns
    HIGH_RISK_PATTERNS = [
        r"\bdelete\b", r"\bremove\b", r"\berase\b", r"\bwipe\b",
        r"\bformat\b", r"\bkill\b", r"\bterminate\b", r"\bshutdown\b",
        r"\bsudo\b", r"\badmin\b", r"\broot\b",
    ]

    MEDIUM_RISK_PATTERNS = [
        r"\binstall\b", r"\buninstall\b", r"\bmodify\b", r"\bchange\b",
        r"\bwrite\b", r"\bcreate\b", r"\bmove\b", r"\bcopy\b",
    ]

    CRITICAL_ACTIONS = {
        "system_shutdown", "format_disk", "delete_all", "rm_rf",
        "kill_process", "admin_execute", "privilege_escalation",
    }

    def analyze(
        self,
        actions: List[Dict[str, Any]],
        context: SafetyContext,
    ) -> Tuple[ActionRisk, List[str]]:
        """Analyze risk of action list."""
        import re

        max_risk = ActionRisk.SAFE
        notes = []

        for action in actions:
            action_type = action.get("type", "")
            action_desc = json.dumps(action).lower()

            # Check critical actions
            if action_type in self.CRITICAL_ACTIONS:
                max_risk = ActionRisk.CRITICAL
                notes.append(f"Critical action: {action_type}")
                continue

            # Check high-risk patterns
            for pattern in self.HIGH_RISK_PATTERNS:
                if re.search(pattern, action_desc):
                    if max_risk.value < ActionRisk.HIGH.value:
                        max_risk = ActionRisk.HIGH
                    notes.append(f"High-risk pattern: {pattern}")
                    break

            # Check medium-risk patterns
            for pattern in self.MEDIUM_RISK_PATTERNS:
                if re.search(pattern, action_desc):
                    if max_risk.value < ActionRisk.MEDIUM.value:
                        max_risk = ActionRisk.MEDIUM
                    notes.append(f"Medium-risk pattern: {pattern}")
                    break

        # Apply safety context multiplier
        if context.should_be_cautious():
            if max_risk == ActionRisk.MEDIUM:
                max_risk = ActionRisk.HIGH
            elif max_risk == ActionRisk.LOW:
                max_risk = ActionRisk.MEDIUM
            notes.append("Elevated risk due to safety context")

        return max_risk, notes


# =============================================================================
# JARVIS PRIME BRIDGE
# =============================================================================


class JARVISPrimeBridge:
    """
    Bridge connecting JARVIS (Body) to JARVIS-Prime (Mind).

    v79.1: Enhanced with circuit breaker and fallback chains for resilience.

    Replaces direct Claude API calls with local AGI processing,
    providing reasoning, planning, and safety-aware responses.

    Resilience Features:
        - Circuit breaker for AGI component protection
        - Fallback chain: AGI → Simple Model → Cache → Default
        - Response caching for graceful degradation
        - Retry with exponential backoff and jitter

    Usage:
        bridge = JARVISPrimeBridge()
        await bridge.initialize()

        # Process command (with automatic fallback on failures)
        response = await bridge.process_command(JARVISCommand(
            id="cmd-123",
            command_type=CommandType.NATURAL_LANGUAGE,
            content="Open Safari and navigate to google.com",
        ))

        # Execute actions if approved
        for action in response.actions:
            await execute_action(action)

        # Record feedback
        await bridge.record_feedback(response.command_id, success=True)
    """

    def __init__(
        self,
        config: Optional[BridgeConfig] = None,
        resilience_config: Optional[BridgeResilienceConfig] = None,
    ) -> None:
        self._config = config or BridgeConfig()
        self._resilience = resilience_config or BridgeResilienceConfig()

        # AGI components (lazy loaded)
        self._agi_hub = None
        self._reasoning_engine = None
        self._learning_engine = None
        self._multimodal_engine = None

        # Safety
        self._safety_reader = SafetyContextReader(self._config.safety_context_file)
        self._risk_analyzer = ActionRiskAnalyzer()

        # State
        self._initialized = False
        self._init_lock = asyncio.Lock()

        # v79.1: Circuit breaker for AGI protection
        from jarvis_prime.core.agi_error_handler import CircuitBreaker, CircuitBreakerConfig
        self._circuit_breaker = CircuitBreaker(
            name="jarvis_bridge_agi",
            config=CircuitBreakerConfig(
                failure_threshold=self._resilience.circuit_failure_threshold,
                timeout_seconds=self._resilience.circuit_recovery_timeout_sec,
                half_open_max_requests=self._resilience.circuit_half_open_requests,
            ),
        )

        # v79.1: Response cache for fallback
        self._response_cache = ResponseCache(
            max_size=500,
            ttl_seconds=self._resilience.cache_ttl_seconds,
        )

        # Metrics
        self._commands_processed = 0
        self._successful_commands = 0
        self._failed_commands = 0
        self._confirmations_requested = 0
        self._cache_hits = 0
        self._fallback_uses = 0
        self._circuit_opens = 0

        # Command history for learning
        self._command_history: Dict[str, JARVISCommand] = {}
        self._response_history: Dict[str, PrimeResponse] = {}
        self._max_history_size = 1000

    async def initialize(self) -> bool:
        """Initialize bridge and connect to AGI systems."""
        async with self._init_lock:
            if self._initialized:
                return True

            try:
                # Import and initialize AGI hub
                from jarvis_prime.core.agi_integration import get_agi_hub

                self._agi_hub = await get_agi_hub()

                if not self._agi_hub._initialized:
                    await self._agi_hub.initialize()

                # Get component references
                self._reasoning_engine = self._agi_hub.reasoning_engine
                self._learning_engine = self._agi_hub.learning_engine
                self._multimodal_engine = self._agi_hub.multimodal_engine

                self._initialized = True
                logger.info("JARVIS Prime Bridge initialized")
                return True

            except Exception as e:
                logger.error(f"Failed to initialize bridge: {e}")
                return False

    async def process_command(self, command: JARVISCommand) -> PrimeResponse:
        """
        Process a command from JARVIS.

        v79.1: Enhanced with circuit breaker and fallback chain.

        Fallback Chain:
            1. Primary: Full AGI processing (with circuit breaker)
            2. Fallback 1: Cached response (if available)
            3. Fallback 2: Simple direct processing (minimal reasoning)
            4. Fallback 3: Error response with recovery suggestions

        This is the main entry point for all JARVIS commands.
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        self._commands_processed += 1

        # Store command for potential learning
        self._command_history[command.id] = command

        try:
            # Get current safety context
            safety_context = self._safety_reader.read()

            # Check kill switch
            if safety_context.kill_switch_active:
                return self._create_kill_switch_response(command)

            # v79.1: Try with circuit breaker and fallback chain
            response = await self._process_with_resilience(command, safety_context)

            # Analyze action risk
            if response.actions:
                risk, notes = self._risk_analyzer.analyze(
                    response.actions, safety_context
                )
                response.risk_level = risk
                response.safety_notes.extend(notes)

                # Require confirmation for risky actions
                if risk.value > self._config.risk_auto_confirm.value:
                    response.requires_confirmation = True
                    self._confirmations_requested += 1

            # Record metrics
            response.processing_time_ms = (time.time() - start_time) * 1000

            # Store response for learning
            self._response_history[command.id] = response

            # v79.1: Cache successful responses for fallback
            if response.success:
                self._response_cache.put(command, response)

            # Trim history if needed
            self._trim_history()

            # Record experience if enabled
            if self._config.record_experiences and self._learning_engine:
                await self._record_experience(command, response)

            if response.success:
                self._successful_commands += 1
            else:
                self._failed_commands += 1

            return response

        except asyncio.TimeoutError:
            self._failed_commands += 1
            # Try cache fallback on timeout
            cached = self._response_cache.get(command)
            if cached:
                self._cache_hits += 1
                cached.warning = "Response from cache (processing timed out)"
                cached.processing_time_ms = (time.time() - start_time) * 1000
                return cached

            return PrimeResponse(
                command_id=command.id,
                success=False,
                error="Processing timed out",
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Error processing command: {e}", exc_info=True)
            self._failed_commands += 1
            return PrimeResponse(
                command_id=command.id,
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    async def _process_with_resilience(
        self,
        command: JARVISCommand,
        safety_context: SafetyContext,
    ) -> PrimeResponse:
        """
        v79.1: Process command with circuit breaker and fallback chain.

        Fallback Chain:
            1. Full AGI processing (with circuit breaker)
            2. Cached response (if available)
            3. Simple direct processing
            4. Default error response
        """
        # Acquire circuit breaker permit
        permit = await self._circuit_breaker.acquire_permit()

        if permit:
            # Circuit is allowing requests - try full AGI processing
            try:
                # Wrap in timeout
                response = await asyncio.wait_for(
                    self._process_primary(command, safety_context),
                    timeout=self._resilience.command_timeout_seconds,
                )

                # Success - release permit and return
                await self._circuit_breaker.release_permit(permit, success=True)
                return response

            except Exception as e:
                # Failed - release permit with failure
                await self._circuit_breaker.release_permit(permit, success=False)
                logger.warning(f"Primary processing failed: {e}")

                # Fall through to fallback chain

        else:
            # Circuit is open - log and fall through
            self._circuit_opens += 1
            logger.warning("Circuit breaker is open, using fallback chain")

        # Fallback 1: Try cache
        if self._resilience.enable_cache_fallback:
            cached = self._response_cache.get(command)
            if cached:
                self._cache_hits += 1
                self._fallback_uses += 1
                logger.info(f"Using cached response for command {command.id}")
                cached.warning = "Response from cache (AGI unavailable)"
                return cached

        # Fallback 2: Simple direct processing
        if self._resilience.enable_simple_fallback:
            try:
                self._fallback_uses += 1
                logger.info(f"Using simple fallback for command {command.id}")
                return await self._process_simple_fallback(command)
            except Exception as e:
                logger.warning(f"Simple fallback failed: {e}")

        # Fallback 3: Default error response
        self._fallback_uses += 1
        return PrimeResponse(
            command_id=command.id,
            success=False,
            error="AGI system is currently unavailable. Please try again later.",
            warning="All fallback options exhausted",
        )

    async def _process_primary(
        self,
        command: JARVISCommand,
        safety_context: SafetyContext,
    ) -> PrimeResponse:
        """Primary processing with full AGI capabilities."""
        # Determine processing mode
        mode = self._determine_mode(command, safety_context)

        # Process based on mode
        if mode == ProcessingMode.DIRECT:
            return await self._process_direct(command)
        elif mode == ProcessingMode.REASONED:
            return await self._process_with_reasoning(command, safety_context)
        elif mode == ProcessingMode.PLANNED:
            return await self._process_with_planning(command, safety_context)
        else:  # CAUTIOUS
            return await self._process_cautious(command, safety_context)

    async def _process_simple_fallback(
        self,
        command: JARVISCommand,
    ) -> PrimeResponse:
        """
        v79.1: Simple fallback processing without full AGI.

        Uses basic pattern matching and predefined responses
        for common command types.
        """
        # Simple query detection
        if command.command_type == CommandType.QUERY:
            return PrimeResponse(
                command_id=command.id,
                success=True,
                response_text=(
                    "I'm operating in limited mode. "
                    "Please try a simpler command or wait for full functionality."
                ),
                confidence=0.3,
                warning="Processed in fallback mode (limited capabilities)",
            )

        # Simple action detection patterns
        content_lower = command.content.lower()

        # Common simple actions that can be handled minimally
        if "open" in content_lower and any(app in content_lower for app in [
            "safari", "chrome", "finder", "terminal", "notes", "calendar"
        ]):
            # Extract app name
            apps = ["safari", "chrome", "finder", "terminal", "notes", "calendar"]
            app = next((a for a in apps if a in content_lower), "application")

            return PrimeResponse(
                command_id=command.id,
                success=True,
                response_text=f"Opening {app.title()}...",
                actions=[{
                    "type": "open_app",
                    "app_name": app.title(),
                    "fallback_mode": True,
                }],
                confidence=0.6,
                warning="Action suggested in fallback mode (verify before execution)",
            )

        # Default: can't handle in fallback mode
        return PrimeResponse(
            command_id=command.id,
            success=False,
            error="This command requires full AGI capabilities which are currently unavailable.",
            warning="Fallback mode cannot handle this command type",
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics including resilience metrics."""
        return {
            "commands_processed": self._commands_processed,
            "successful_commands": self._successful_commands,
            "failed_commands": self._failed_commands,
            "confirmations_requested": self._confirmations_requested,
            # v79.1: Resilience metrics
            "cache_hits": self._cache_hits,
            "fallback_uses": self._fallback_uses,
            "circuit_opens": self._circuit_opens,
            "circuit_state": self._circuit_breaker.state.name,
            "circuit_active_permits": self._circuit_breaker.active_permits_count,
        }

    def _determine_mode(
        self,
        command: JARVISCommand,
        safety: SafetyContext,
    ) -> ProcessingMode:
        """Determine processing mode based on command and context."""
        # Safety overrides
        if safety.should_be_cautious():
            return ProcessingMode.CAUTIOUS

        # Command type based
        if command.command_type == CommandType.QUERY:
            return ProcessingMode.DIRECT
        elif command.command_type == CommandType.ACTION_SEQUENCE:
            return ProcessingMode.PLANNED
        elif command.command_type == CommandType.SCREEN_ANALYSIS:
            return ProcessingMode.REASONED

        # Complexity heuristic
        if len(command.content) > 200:
            return ProcessingMode.PLANNED
        elif "?" in command.content:
            return ProcessingMode.REASONED

        return self._config.default_mode

    async def _process_direct(self, command: JARVISCommand) -> PrimeResponse:
        """Direct processing without reasoning."""
        result = await self._agi_hub.process(
            content=command.content,
            context=command.context,
        )

        return PrimeResponse(
            command_id=command.id,
            success=True,
            response_text=result.content,
            confidence=result.confidence,
            models_used=result.models_used,
        )

    async def _process_with_reasoning(
        self,
        command: JARVISCommand,
        safety: SafetyContext,
    ) -> PrimeResponse:
        """Process with reasoning engine."""
        # Apply reasoning
        reasoning_result = await asyncio.wait_for(
            self._agi_hub.reason(
                query=command.content,
                strategy="chain_of_thought",
                context=command.context,
            ),
            timeout=self._config.reasoning_timeout,
        )

        # Generate response
        result = await self._agi_hub.process(
            content=command.content,
            context={
                **command.context,
                "reasoning": reasoning_result.get("conclusion"),
            },
        )

        return PrimeResponse(
            command_id=command.id,
            success=True,
            response_text=result.content,
            reasoning_trace=reasoning_result.get("trace", []),
            confidence=reasoning_result.get("confidence", 0.5),
            models_used=result.models_used,
        )

    async def _process_with_planning(
        self,
        command: JARVISCommand,
        safety: SafetyContext,
    ) -> PrimeResponse:
        """Process with full action planning."""
        # Generate plan
        plan_result = await asyncio.wait_for(
            self._agi_hub.plan(
                goal=command.content,
                context=command.context,
            ),
            timeout=self._config.planning_timeout,
        )

        # Extract actions from plan
        actions = plan_result.get("actions", [])

        # Apply reasoning to validate plan
        if self._config.enable_reasoning:
            validation = await self._agi_hub.reason(
                query=f"Validate this action plan for '{command.content}': {json.dumps(actions)}",
                strategy="self_reflection",
            )
            reasoning_trace = validation.get("trace", [])
            confidence = validation.get("confidence", 0.5)
        else:
            reasoning_trace = []
            confidence = 0.7

        return PrimeResponse(
            command_id=command.id,
            success=True,
            response_text=plan_result.get("explanation", ""),
            actions=actions,
            reasoning_trace=reasoning_trace,
            confidence=confidence,
            models_used=["action_model", "goal_inference"],
        )

    async def _process_cautious(
        self,
        command: JARVISCommand,
        safety: SafetyContext,
    ) -> PrimeResponse:
        """Cautious processing with extra safety checks."""
        # First, analyze intent
        intent_result = await self._agi_hub.reason(
            query=f"Analyze the intent and potential risks of: {command.content}",
            strategy="hypothesis_test",
        )

        # Check if action seems safe
        intent_confidence = intent_result.get("confidence", 0.0)

        if intent_confidence < self._config.confidence_threshold:
            return PrimeResponse(
                command_id=command.id,
                success=False,
                response_text="I'm not confident enough to proceed safely. Please clarify your request.",
                reasoning_trace=intent_result.get("trace", []),
                confidence=intent_confidence,
                requires_confirmation=True,
                safety_notes=["Low confidence in intent understanding"],
            )

        # Proceed with planning
        response = await self._process_with_planning(command, safety)

        # Always require confirmation in cautious mode
        response.requires_confirmation = True
        response.safety_notes.append("Cautious mode active - confirmation required")

        return response

    def _create_kill_switch_response(self, command: JARVISCommand) -> PrimeResponse:
        """Create response when kill switch is active."""
        return PrimeResponse(
            command_id=command.id,
            success=False,
            response_text="Kill switch is active. All actions are paused. Please deactivate the kill switch to proceed.",
            risk_level=ActionRisk.CRITICAL,
            safety_notes=["KILL SWITCH ACTIVE"],
            requires_confirmation=True,
        )

    async def _record_experience(
        self,
        command: JARVISCommand,
        response: PrimeResponse,
    ) -> None:
        """Record experience for continuous learning."""
        try:
            self._learning_engine.record_experience(
                input_text=command.content,
                output_text=response.response_text,
                metadata={
                    "command_id": command.id,
                    "command_type": command.command_type.name,
                    "confidence": response.confidence,
                    "risk_level": response.risk_level.value,
                    "models_used": response.models_used,
                    "processing_time_ms": response.processing_time_ms,
                    "success": response.success,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to record experience: {e}")

    async def record_feedback(
        self,
        command_id: str,
        success: bool,
        score: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> bool:
        """Record user feedback on a command response."""
        if not self._config.record_feedback:
            return True

        try:
            # Calculate score if not provided
            if score is None:
                score = 1.0 if success else -1.0

            if self._learning_engine:
                await self._agi_hub.record_feedback(
                    experience_id=command_id,
                    score=score,
                    comment=comment,
                )

            return True

        except Exception as e:
            logger.warning(f"Failed to record feedback: {e}")
            return False

    async def process_screen(
        self,
        command: JARVISCommand,
    ) -> PrimeResponse:
        """Process command with screen context."""
        if not command.screen_data:
            return await self.process_command(command)

        try:
            # Analyze screen with multimodal fusion
            screen_understanding = await self._agi_hub.understand_screen(
                screen_data=command.screen_data,
                context=command.context,
            )

            # Enrich command context with screen understanding
            enriched_command = JARVISCommand(
                id=command.id,
                command_type=command.command_type,
                content=command.content,
                context={
                    **command.context,
                    "screen_understanding": screen_understanding,
                    "screen_elements": command.screen_elements,
                },
                screen_data=command.screen_data,
                screen_elements=command.screen_elements,
            )

            return await self.process_command(enriched_command)

        except Exception as e:
            logger.error(f"Screen processing failed: {e}")
            return await self.process_command(command)

    def _trim_history(self) -> None:
        """Trim command/response history to max size."""
        if len(self._command_history) > self._max_history_size:
            # Remove oldest half
            sorted_ids = sorted(
                self._command_history.keys(),
                key=lambda x: self._command_history[x].timestamp,
            )
            for cmd_id in sorted_ids[: len(sorted_ids) // 2]:
                self._command_history.pop(cmd_id, None)
                self._response_history.pop(cmd_id, None)

    def get_metrics(self) -> Dict[str, Any]:
        """Get bridge metrics."""
        return {
            "initialized": self._initialized,
            "commands_processed": self._commands_processed,
            "successful_commands": self._successful_commands,
            "failed_commands": self._failed_commands,
            "confirmations_requested": self._confirmations_requested,
            "success_rate": (
                self._successful_commands / max(self._commands_processed, 1)
            ),
            "history_size": len(self._command_history),
        }

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# =============================================================================
# SINGLETON AND CONVENIENCE FUNCTIONS
# =============================================================================


_bridge_instance: Optional[JARVISPrimeBridge] = None
_bridge_lock = asyncio.Lock()


async def get_jarvis_bridge(config: Optional[BridgeConfig] = None) -> JARVISPrimeBridge:
    """Get or create global bridge instance."""
    global _bridge_instance

    async with _bridge_lock:
        if _bridge_instance is None:
            _bridge_instance = JARVISPrimeBridge(config)
            await _bridge_instance.initialize()

        return _bridge_instance


async def process_jarvis_command(
    content: str,
    command_type: CommandType = CommandType.NATURAL_LANGUAGE,
    context: Optional[Dict[str, Any]] = None,
) -> PrimeResponse:
    """Convenience function to process a JARVIS command."""
    import uuid

    bridge = await get_jarvis_bridge()

    command = JARVISCommand(
        id=str(uuid.uuid4()),
        command_type=command_type,
        content=content,
        context=context or {},
    )

    return await bridge.process_command(command)
