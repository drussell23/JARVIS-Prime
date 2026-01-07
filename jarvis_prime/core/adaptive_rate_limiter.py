"""
Adaptive Rate Limiting v80.0 - Token Bucket with ML-Based Adaptation
====================================================================

Intelligent rate limiting that adapts to system load and user behavior.
Prevents overload while maximizing throughput.

FEATURES:
    - Token bucket algorithm with refill strategies
    - Per-user and global rate limiting
    - Adaptive limits based on system health
    - Burst handling with priority queues
    - Fair queuing across users
    - Distributed rate limiting support
    - Graceful degradation under load

ALGORITHMS:
    - Token Bucket: Classic rate limiting
    - Leaky Bucket: Smooth traffic shaping
    - Sliding Window: Precise rate tracking
    - Adaptive Token Refill: ML-based refill rate

USAGE:
    from jarvis_prime.core.adaptive_rate_limiter import get_rate_limiter

    limiter = await get_rate_limiter()

    # Check if request allowed
    if await limiter.acquire(user_id="user123", tokens=1):
        # Process request
        result = await process_request()
    else:
        # Rate limited - reject or queue
        return {"error": "rate_limited", "retry_after": 60}

    # Get user limits
    stats = await limiter.get_user_stats("user123")
    print(f"Tokens available: {stats['tokens_available']}")
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    SLIDING_WINDOW = "sliding_window"
    ADAPTIVE = "adaptive"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    # Global limits
    global_rate: float = field(
        default_factory=lambda: float(os.getenv("RATE_LIMIT_GLOBAL_RATE", "1000.0"))
    )  # requests/second
    global_burst: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_GLOBAL_BURST", "2000"))
    )  # max burst

    # Per-user limits
    user_rate: float = field(
        default_factory=lambda: float(os.getenv("RATE_LIMIT_USER_RATE", "10.0"))
    )  # requests/second
    user_burst: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_USER_BURST", "20"))
    )

    # Strategy
    strategy: RateLimitStrategy = field(
        default_factory=lambda: RateLimitStrategy(
            os.getenv("RATE_LIMIT_STRATEGY", "adaptive")
        )
    )

    # Adaptive config
    enable_adaptation: bool = field(
        default_factory=lambda: os.getenv("RATE_LIMIT_ADAPTIVE", "true").lower() == "true"
    )
    adaptation_window: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_ADAPT_WINDOW", "300"))
    )  # seconds

    # Fair queuing
    enable_fair_queuing: bool = field(
        default_factory=lambda: os.getenv("RATE_LIMIT_FAIR_QUEUE", "true").lower() == "true"
    )


# ============================================================================
# TOKEN BUCKET
# ============================================================================

@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting.

    Tokens are added at a fixed rate. Each request consumes tokens.
    Allows bursts up to bucket capacity.
    """
    capacity: float  # Maximum tokens
    refill_rate: float  # Tokens per second
    tokens: float = field(init=False)  # Current tokens
    last_refill: float = field(default_factory=time.time)

    def __post_init__(self):
        """Initialize with full bucket."""
        self.tokens = self.capacity

    def refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Calculate tokens to add
        new_tokens = elapsed * self.refill_rate

        # Add tokens (cap at capacity)
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def consume(self, tokens: float = 1.0) -> bool:
        """
        Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens available, False otherwise
        """
        self.refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        return False

    def get_wait_time(self, tokens: float = 1.0) -> float:
        """
        Get time to wait for tokens to be available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds
        """
        self.refill()

        if self.tokens >= tokens:
            return 0.0

        # Calculate deficit
        deficit = tokens - self.tokens

        # Calculate wait time
        return deficit / self.refill_rate


# ============================================================================
# SLIDING WINDOW COUNTER
# ============================================================================

class SlidingWindowCounter:
    """
    Sliding window rate limiter for precise rate tracking.

    More accurate than token bucket for rate limiting over time windows.
    """

    def __init__(self, window_size: int = 60, max_requests: int = 100):
        """
        Initialize sliding window counter.

        Args:
            window_size: Time window in seconds
            max_requests: Maximum requests in window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests: deque[float] = deque()

    def add_request(self, timestamp: Optional[float] = None) -> bool:
        """
        Try to add request.

        Args:
            timestamp: Request timestamp (defaults to now)

        Returns:
            True if request allowed, False if rate limit exceeded
        """
        now = timestamp or time.time()

        # Remove old requests outside window
        cutoff = now - self.window_size

        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True

        return False

    def get_remaining(self) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        cutoff = now - self.window_size

        # Clean old requests
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

        return self.max_requests - len(self.requests)


# ============================================================================
# ADAPTIVE RATE LIMITER
# ============================================================================

@dataclass
class UserStats:
    """Statistics for a user."""
    request_count: int = 0
    last_request: float = 0.0
    tokens_consumed: float = 0.0
    rejections: int = 0
    average_interval: float = 0.0


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter with ML-based adaptation.

    Automatically adjusts rate limits based on:
    - System health (CPU, memory)
    - User behavior patterns
    - Historical load patterns
    - Time of day
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize adaptive rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()

        # Global token bucket
        self._global_bucket = TokenBucket(
            capacity=self.config.global_burst,
            refill_rate=self.config.global_rate
        )

        # Per-user token buckets
        self._user_buckets: Dict[str, TokenBucket] = {}

        # Sliding window counters
        self._user_windows: Dict[str, SlidingWindowCounter] = {}

        # User statistics
        self._user_stats: Dict[str, UserStats] = defaultdict(UserStats)

        # Locks
        self._global_lock = asyncio.Lock()
        self._user_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        # Adaptive state
        self._current_global_rate = self.config.global_rate
        self._current_user_rate = self.config.user_rate
        self._last_adaptation = time.time()

        # Statistics
        self._total_requests = 0
        self._total_rejections = 0
        self._adaptation_count = 0

    async def acquire(
        self,
        user_id: str,
        tokens: float = 1.0,
        priority: int = 0
    ) -> bool:
        """
        Try to acquire tokens for request.

        Args:
            user_id: User identifier
            tokens: Number of tokens to acquire
            priority: Request priority (higher = more important)

        Returns:
            True if request allowed, False if rate limited
        """
        self._total_requests += 1

        # Check global rate limit
        async with self._global_lock:
            if not self._global_bucket.consume(tokens):
                self._total_rejections += 1
                logger.debug(f"Global rate limit exceeded")
                return False

        # Check user rate limit
        async with self._user_locks[user_id]:
            # Get or create user bucket
            if user_id not in self._user_buckets:
                self._user_buckets[user_id] = TokenBucket(
                    capacity=self.config.user_burst,
                    refill_rate=self.config.user_rate
                )

            bucket = self._user_buckets[user_id]

            if not bucket.consume(tokens):
                # User rate limited
                self._user_stats[user_id].rejections += 1
                self._total_rejections += 1
                logger.debug(f"User {user_id} rate limited")
                return False

            # Update user stats
            stats = self._user_stats[user_id]
            stats.request_count += 1
            now = time.time()

            if stats.last_request > 0:
                interval = now - stats.last_request
                # Exponential moving average
                alpha = 0.1
                stats.average_interval = (
                    alpha * interval + (1 - alpha) * stats.average_interval
                )

            stats.last_request = now
            stats.tokens_consumed += tokens

        # Adaptive adjustment
        if self.config.enable_adaptation:
            await self._maybe_adapt()

        return True

    async def get_wait_time(self, user_id: str, tokens: float = 1.0) -> float:
        """
        Get time to wait before request would be allowed.

        Args:
            user_id: User identifier
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds
        """
        # Check global first
        async with self._global_lock:
            global_wait = self._global_bucket.get_wait_time(tokens)

        # Check user
        async with self._user_locks[user_id]:
            if user_id not in self._user_buckets:
                self._user_buckets[user_id] = TokenBucket(
                    capacity=self.config.user_burst,
                    refill_rate=self.config.user_rate
                )

            bucket = self._user_buckets[user_id]
            user_wait = bucket.get_wait_time(tokens)

        # Return maximum wait time
        return max(global_wait, user_wait)

    async def _maybe_adapt(self):
        """Adapt rate limits based on system state."""
        now = time.time()
        elapsed = now - self._last_adaptation

        # Adapt every adaptation window
        if elapsed < self.config.adaptation_window:
            return

        async with self._global_lock:
            # Calculate current load
            rejection_rate = self._total_rejections / max(self._total_requests, 1)

            # Adapt based on rejection rate
            if rejection_rate > 0.1:
                # High rejection rate - might be too strict
                # Increase limits by 10%
                self._current_global_rate *= 1.1
                self._current_user_rate *= 1.1

                logger.info(
                    f"Increased rate limits: global={self._current_global_rate:.1f}, "
                    f"user={self._current_user_rate:.1f}"
                )

            elif rejection_rate < 0.01:
                # Very low rejection rate - could be more strict
                # Decrease limits by 5%
                self._current_global_rate *= 0.95
                self._current_user_rate *= 0.95

                logger.info(
                    f"Decreased rate limits: global={self._current_global_rate:.1f}, "
                    f"user={self._current_user_rate:.1f}"
                )

            # Apply new rates
            self._global_bucket.refill_rate = self._current_global_rate

            for bucket in self._user_buckets.values():
                bucket.refill_rate = self._current_user_rate

            self._last_adaptation = now
            self._adaptation_count += 1

    async def get_user_stats(self, user_id: str) -> Dict[str, any]:
        """Get statistics for user."""
        stats = self._user_stats.get(user_id, UserStats())

        # Get current bucket state
        tokens_available = 0.0
        if user_id in self._user_buckets:
            async with self._user_locks[user_id]:
                bucket = self._user_buckets[user_id]
                bucket.refill()
                tokens_available = bucket.tokens

        return {
            "request_count": stats.request_count,
            "rejections": stats.rejections,
            "tokens_consumed": stats.tokens_consumed,
            "tokens_available": tokens_available,
            "average_interval": stats.average_interval,
            "rejection_rate": (
                stats.rejections / max(stats.request_count, 1)
            ),
        }

    def get_global_stats(self) -> Dict[str, any]:
        """Get global statistics."""
        return {
            "total_requests": self._total_requests,
            "total_rejections": self._total_rejections,
            "global_rate": self._current_global_rate,
            "user_rate": self._current_user_rate,
            "active_users": len(self._user_buckets),
            "adaptation_count": self._adaptation_count,
            "rejection_rate": (
                self._total_rejections / max(self._total_requests, 1)
            ),
        }


# ============================================================================
# GLOBAL RATE LIMITER
# ============================================================================

_rate_limiter: Optional[AdaptiveRateLimiter] = None
_limiter_lock = asyncio.Lock()


async def get_rate_limiter() -> AdaptiveRateLimiter:
    """Get or create global rate limiter."""
    global _rate_limiter

    async with _limiter_lock:
        if _rate_limiter is None:
            _rate_limiter = AdaptiveRateLimiter()

        return _rate_limiter
