"""
AGI Error Handler - Graceful Degradation and Recovery
=======================================================

v78.0 - Intelligent error handling for AGI systems

Provides multi-level fallback strategies when AGI components fail:
1. Retry with backoff
2. Fallback to simpler models
3. Fallback to cloud APIs
4. Graceful degradation

FEATURES:
    - Circuit breaker for failing components
    - Automatic retry with exponential backoff
    - Fallback chain with multiple strategies
    - Error classification and routing
    - Recovery suggestions
    - Telemetry and alerting

FALLBACK CHAIN:
    AGI Model → Simpler Model → Cloud API → Cached Response → Error Message
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# ERROR CLASSIFICATION
# =============================================================================


class ErrorCategory(Enum):
    """Categories of AGI errors."""

    # Transient errors (can retry)
    TIMEOUT = auto()
    OVERLOAD = auto()
    RATE_LIMIT = auto()
    NETWORK = auto()

    # Resource errors
    OUT_OF_MEMORY = auto()
    MODEL_NOT_LOADED = auto()
    RESOURCE_EXHAUSTED = auto()

    # Logic errors
    INVALID_INPUT = auto()
    UNSUPPORTED_OPERATION = auto()
    CONFIGURATION = auto()

    # Model errors
    MODEL_FAILURE = auto()
    INFERENCE_ERROR = auto()
    REASONING_FAILURE = auto()

    # Critical errors
    CORRUPTION = auto()
    SECURITY = auto()
    UNRECOVERABLE = auto()

    # Unknown
    UNKNOWN = auto()


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


@dataclass
class AGIError(Exception):
    """Structured error from AGI systems."""

    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    component: str = ""
    operation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    original_error: Optional[Exception] = None
    timestamp: float = field(default_factory=time.time)
    recoverable: bool = True
    retry_after: Optional[float] = None

    def __str__(self) -> str:
        return f"[{self.category.name}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.name,
            "severity": self.severity.name,
            "message": self.message,
            "component": self.component,
            "operation": self.operation,
            "details": self.details,
            "timestamp": self.timestamp,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
        }


# =============================================================================
# ERROR CLASSIFIER
# =============================================================================


class ErrorClassifier:
    """Classifies exceptions into AGIError categories."""

    # Exception to category mapping
    EXCEPTION_MAP: Dict[Type[Exception], ErrorCategory] = {
        asyncio.TimeoutError: ErrorCategory.TIMEOUT,
        MemoryError: ErrorCategory.OUT_OF_MEMORY,
        ValueError: ErrorCategory.INVALID_INPUT,
        TypeError: ErrorCategory.INVALID_INPUT,
        NotImplementedError: ErrorCategory.UNSUPPORTED_OPERATION,
        FileNotFoundError: ErrorCategory.MODEL_NOT_LOADED,
        PermissionError: ErrorCategory.SECURITY,
        ConnectionError: ErrorCategory.NETWORK,
        ConnectionRefusedError: ErrorCategory.NETWORK,
        ConnectionResetError: ErrorCategory.NETWORK,
    }

    # Message pattern to category
    MESSAGE_PATTERNS: Dict[str, ErrorCategory] = {
        "timeout": ErrorCategory.TIMEOUT,
        "timed out": ErrorCategory.TIMEOUT,
        "rate limit": ErrorCategory.RATE_LIMIT,
        "too many requests": ErrorCategory.RATE_LIMIT,
        "out of memory": ErrorCategory.OUT_OF_MEMORY,
        "cuda out of memory": ErrorCategory.OUT_OF_MEMORY,
        "model not found": ErrorCategory.MODEL_NOT_LOADED,
        "not loaded": ErrorCategory.MODEL_NOT_LOADED,
        "overload": ErrorCategory.OVERLOAD,
        "corrupted": ErrorCategory.CORRUPTION,
        "security": ErrorCategory.SECURITY,
        "unauthorized": ErrorCategory.SECURITY,
    }

    @classmethod
    def classify(
        cls,
        error: Exception,
        component: str = "",
        operation: str = "",
    ) -> AGIError:
        """Classify an exception into an AGIError."""
        # Check exception type
        error_type = type(error)
        category = cls.EXCEPTION_MAP.get(error_type, ErrorCategory.UNKNOWN)

        # Check message patterns
        error_msg = str(error).lower()
        for pattern, cat in cls.MESSAGE_PATTERNS.items():
            if pattern in error_msg:
                category = cat
                break

        # Determine severity
        severity = cls._determine_severity(category)

        # Determine recoverability
        recoverable = category not in (
            ErrorCategory.CORRUPTION,
            ErrorCategory.SECURITY,
            ErrorCategory.UNRECOVERABLE,
        )

        # Calculate retry delay for transient errors
        retry_after = None
        if category in (ErrorCategory.TIMEOUT, ErrorCategory.OVERLOAD, ErrorCategory.RATE_LIMIT):
            retry_after = 1.0
        elif category == ErrorCategory.NETWORK:
            retry_after = 5.0

        return AGIError(
            category=category,
            severity=severity,
            message=str(error),
            component=component,
            operation=operation,
            original_error=error,
            recoverable=recoverable,
            retry_after=retry_after,
            details={"traceback": traceback.format_exc()},
        )

    @classmethod
    def _determine_severity(cls, category: ErrorCategory) -> ErrorSeverity:
        """Determine severity based on category."""
        severity_map = {
            ErrorCategory.TIMEOUT: ErrorSeverity.WARNING,
            ErrorCategory.OVERLOAD: ErrorSeverity.WARNING,
            ErrorCategory.RATE_LIMIT: ErrorSeverity.WARNING,
            ErrorCategory.NETWORK: ErrorSeverity.WARNING,
            ErrorCategory.OUT_OF_MEMORY: ErrorSeverity.ERROR,
            ErrorCategory.MODEL_NOT_LOADED: ErrorSeverity.ERROR,
            ErrorCategory.RESOURCE_EXHAUSTED: ErrorSeverity.ERROR,
            ErrorCategory.INVALID_INPUT: ErrorSeverity.WARNING,
            ErrorCategory.UNSUPPORTED_OPERATION: ErrorSeverity.WARNING,
            ErrorCategory.CONFIGURATION: ErrorSeverity.ERROR,
            ErrorCategory.MODEL_FAILURE: ErrorSeverity.ERROR,
            ErrorCategory.INFERENCE_ERROR: ErrorSeverity.ERROR,
            ErrorCategory.REASONING_FAILURE: ErrorSeverity.ERROR,
            ErrorCategory.CORRUPTION: ErrorSeverity.CRITICAL,
            ErrorCategory.SECURITY: ErrorSeverity.CRITICAL,
            ErrorCategory.UNRECOVERABLE: ErrorSeverity.CRITICAL,
            ErrorCategory.UNKNOWN: ErrorSeverity.ERROR,
        }
        return severity_map.get(category, ErrorSeverity.ERROR)


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Blocking requests
    HALF_OPEN = auto()   # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 30.0
    half_open_max_requests: int = 3


class CircuitBreaker:
    """
    Circuit breaker for component protection.

    Prevents cascade failures by temporarily blocking requests
    to failing components.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        self._name = name
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_requests = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    async def can_execute(self) -> bool:
        """Check if execution is allowed."""
        async with self._lock:
            await self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_requests < self._config.half_open_max_requests:
                    self._half_open_requests += 1
                    return True
                return False
            else:  # OPEN
                return False

    async def record_success(self) -> None:
        """Record a successful execution."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)

    async def record_failure(self, error: Optional[AGIError] = None) -> None:
        """Record a failed execution."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    async def _check_state_transition(self) -> None:
        """Check if we should transition states."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self._config.timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_requests = 0

        logger.info(f"Circuit {self._name}: {old_state.name} -> {new_state.name}")


# =============================================================================
# FALLBACK CHAIN
# =============================================================================


class FallbackStrategy(Enum):
    """Fallback strategies."""

    RETRY = auto()           # Retry same operation
    SIMPLER_MODEL = auto()   # Use simpler model
    CLOUD_API = auto()       # Fallback to cloud
    CACHED = auto()          # Use cached response
    DEFAULT = auto()         # Return default value
    FAIL = auto()            # Propagate error


@dataclass
class FallbackResult(Generic[T]):
    """Result from fallback chain."""

    success: bool
    value: Optional[T] = None
    strategy_used: FallbackStrategy = FallbackStrategy.FAIL
    attempts: int = 0
    errors: List[AGIError] = field(default_factory=list)


class FallbackChain(Generic[T]):
    """
    Chain of fallback strategies for graceful degradation.

    Tries each strategy in order until one succeeds.
    """

    def __init__(
        self,
        primary: Callable[..., Awaitable[T]],
        component: str = "",
        operation: str = "",
    ) -> None:
        self._primary = primary
        self._component = component
        self._operation = operation
        self._fallbacks: List[Tuple[FallbackStrategy, Callable[..., Awaitable[T]]]] = []
        self._circuit_breaker: Optional[CircuitBreaker] = None
        self._max_retries = 3
        self._retry_delay = 1.0
        self._default_value: Optional[T] = None

    def with_retry(
        self,
        max_retries: int = 3,
        delay: float = 1.0,
    ) -> "FallbackChain[T]":
        """Configure retry behavior."""
        self._max_retries = max_retries
        self._retry_delay = delay
        return self

    def with_circuit_breaker(
        self,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> "FallbackChain[T]":
        """Add circuit breaker."""
        self._circuit_breaker = CircuitBreaker(
            f"{self._component}.{self._operation}",
            config,
        )
        return self

    def with_fallback(
        self,
        strategy: FallbackStrategy,
        handler: Callable[..., Awaitable[T]],
    ) -> "FallbackChain[T]":
        """Add a fallback handler."""
        self._fallbacks.append((strategy, handler))
        return self

    def with_default(self, value: T) -> "FallbackChain[T]":
        """Set default value for final fallback."""
        self._default_value = value
        return self

    async def execute(self, *args: Any, **kwargs: Any) -> FallbackResult[T]:
        """Execute with fallback chain."""
        result = FallbackResult[T](success=False)

        # Check circuit breaker
        if self._circuit_breaker and not await self._circuit_breaker.can_execute():
            result.errors.append(AGIError(
                category=ErrorCategory.OVERLOAD,
                severity=ErrorSeverity.WARNING,
                message=f"Circuit breaker open for {self._component}",
                component=self._component,
                operation=self._operation,
            ))
            # Skip to fallbacks
        else:
            # Try primary with retries
            for attempt in range(self._max_retries):
                result.attempts += 1
                try:
                    value = await self._primary(*args, **kwargs)
                    result.success = True
                    result.value = value
                    result.strategy_used = FallbackStrategy.RETRY if attempt > 0 else FallbackStrategy.FAIL

                    if self._circuit_breaker:
                        await self._circuit_breaker.record_success()

                    return result

                except Exception as e:
                    error = ErrorClassifier.classify(e, self._component, self._operation)
                    result.errors.append(error)

                    if self._circuit_breaker:
                        await self._circuit_breaker.record_failure(error)

                    if not error.recoverable or attempt >= self._max_retries - 1:
                        break

                    # Wait before retry
                    delay = self._retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

        # Try fallbacks
        for strategy, handler in self._fallbacks:
            result.attempts += 1
            try:
                value = await handler(*args, **kwargs)
                result.success = True
                result.value = value
                result.strategy_used = strategy
                return result

            except Exception as e:
                error = ErrorClassifier.classify(e, self._component, f"fallback_{strategy.name}")
                result.errors.append(error)

        # Use default if available
        if self._default_value is not None:
            result.success = True
            result.value = self._default_value
            result.strategy_used = FallbackStrategy.DEFAULT
            return result

        return result


# =============================================================================
# ERROR HANDLER
# =============================================================================


class AGIErrorHandler:
    """
    Central error handler for AGI systems.

    Provides unified error handling, logging, and recovery.

    Usage:
        handler = AGIErrorHandler()

        # With decorator
        @handler.wrap("orchestrator", "process")
        async def process_request(request):
            ...

        # With context manager
        async with handler.context("reasoning", "analyze"):
            result = await analyze()

        # With fallback chain
        result = await handler.with_fallbacks(
            primary=agi_model.generate,
            fallbacks=[
                (FallbackStrategy.SIMPLER_MODEL, simple_model.generate),
                (FallbackStrategy.CLOUD_API, claude_api.generate),
            ],
        ).execute(prompt)
    """

    def __init__(self) -> None:
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._error_counts: Dict[ErrorCategory, int] = {}
        self._recent_errors: List[AGIError] = []
        self._max_recent_errors = 100
        self._error_callbacks: List[Callable[[AGIError], Awaitable[None]]] = []

    def get_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(name, config)
        return self._circuit_breakers[name]

    def wrap(
        self,
        component: str,
        operation: str,
    ) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
        """Decorator to wrap async functions with error handling."""
        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                async with self.context(component, operation):
                    return await func(*args, **kwargs)
            return wrapper
        return decorator

    class ErrorContext:
        """Context manager for error handling."""

        def __init__(
            self,
            handler: "AGIErrorHandler",
            component: str,
            operation: str,
        ) -> None:
            self._handler = handler
            self._component = component
            self._operation = operation

        async def __aenter__(self) -> "AGIErrorHandler.ErrorContext":
            return self

        async def __aexit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Any,
        ) -> bool:
            if exc_val is not None and isinstance(exc_val, Exception):
                error = ErrorClassifier.classify(exc_val, self._component, self._operation)
                await self._handler.handle(error)
                # Don't suppress the exception
                return False
            return False

    def context(self, component: str, operation: str) -> ErrorContext:
        """Create error handling context."""
        return self.ErrorContext(self, component, operation)

    async def handle(self, error: AGIError) -> None:
        """Handle an AGI error."""
        # Record error
        self._error_counts[error.category] = self._error_counts.get(error.category, 0) + 1
        self._recent_errors.append(error)

        # Trim recent errors
        if len(self._recent_errors) > self._max_recent_errors:
            self._recent_errors = self._recent_errors[-self._max_recent_errors:]

        # Log error
        log_msg = f"[{error.component}:{error.operation}] {error.category.name}: {error.message}"

        if error.severity == ErrorSeverity.DEBUG:
            logger.debug(log_msg)
        elif error.severity == ErrorSeverity.INFO:
            logger.info(log_msg)
        elif error.severity == ErrorSeverity.WARNING:
            logger.warning(log_msg)
        elif error.severity == ErrorSeverity.ERROR:
            logger.error(log_msg)
        else:  # CRITICAL
            logger.critical(log_msg)

        # Notify callbacks
        for callback in self._error_callbacks:
            try:
                await callback(error)
            except Exception as e:
                logger.warning(f"Error callback failed: {e}")

    def with_fallbacks(
        self,
        primary: Callable[..., Awaitable[T]],
        component: str = "",
        operation: str = "",
    ) -> FallbackChain[T]:
        """Create a fallback chain."""
        return FallbackChain(primary, component, operation)

    def on_error(
        self,
        callback: Callable[[AGIError], Awaitable[None]],
    ) -> None:
        """Register error callback."""
        self._error_callbacks.append(callback)

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": {k.name: v for k, v in self._error_counts.items()},
            "total_errors": sum(self._error_counts.values()),
            "recent_errors": len(self._recent_errors),
            "circuit_breakers": {
                name: cb.state.name for name, cb in self._circuit_breakers.items()
            },
        }

    def get_recent_errors(
        self,
        limit: int = 10,
        category: Optional[ErrorCategory] = None,
    ) -> List[AGIError]:
        """Get recent errors."""
        errors = self._recent_errors
        if category:
            errors = [e for e in errors if e.category == category]
        return errors[-limit:]


# =============================================================================
# SINGLETON
# =============================================================================


_error_handler: Optional[AGIErrorHandler] = None


def get_error_handler() -> AGIErrorHandler:
    """Get global error handler."""
    global _error_handler
    if _error_handler is None:
        _error_handler = AGIErrorHandler()
    return _error_handler
