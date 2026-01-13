"""
Unified Inference - Seamless Local/API Fallback System
=========================================================

v94.0 - Hybrid Tiered Inference with Intelligent Routing

This module provides a unified interface for model inference with:
- Intelligent complexity-based tier routing (v94.0)
- Automatic fallback from local models to Claude API
- Health-aware routing and circuit breakers
- Retry logic with exponential backoff
- Cost tracking and budget management
- Comprehensive metrics and observability

ARCHITECTURE (v94.0):
    Request → UnifiedClient → HybridTieredRouter → TierDecision
                                   ↓
    ┌──────────────────────────────────────────────────────────────┐
    │  TIER 0: Local Fast    │  TIER 1: Cloud   │  TIER 2: Deep   │
    │  (8B, FREE, ~20t/s)    │  (70B, $1.20/hr) │  (Opus, $0.015) │
    └──────────────────────────────────────────────────────────────┘
                                   ↓
                    Response ← CircuitBreaker ← RetryHandler

FALLBACK CHAIN (v94.0 Tiered):
    Tier 0: Local 8B Fast (Llama 3 8B, Qwen 7B)
    Tier 0.5: Local 14B Capable (Qwen 14B, Phi-3 Medium)
    Tier 1: Cloud 70B Intelligent (Llama 3.3 70B on GCP A100)
    Tier 2: Deep Reasoning (Claude Opus, DeepSeek V3)

USAGE:
    client = await get_unified_client()

    # Simple generation
    response = await client.generate("What is AI?")

    # Chat with fallback
    response = await client.chat([
        {"role": "user", "content": "Hello"}
    ])

    # Stream with automatic fallback
    async for chunk in client.stream("Tell me a story"):
        print(chunk, end="")

INTEGRATION:
    - PrimeModel: Uses local models
    - AutoModelSelector: Intelligent routing
    - Claude API: Fallback provider
    - Observability: Metrics and tracing
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# HYBRID TIERED ROUTER INTEGRATION (v94.0)
# =============================================================================

# Lazy import to avoid circular dependencies
_hybrid_router = None


async def _get_hybrid_router():
    """Get the HybridTieredRouter singleton lazily."""
    global _hybrid_router
    if _hybrid_router is None:
        try:
            from jarvis_prime.core.hybrid_tiered_router import get_hybrid_tiered_router
            _hybrid_router = await get_hybrid_tiered_router()
        except ImportError:
            logger.warning("HybridTieredRouter not available, using fallback routing")
            _hybrid_router = None
    return _hybrid_router


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class UnifiedInferenceConfig:
    """
    Configuration for unified inference.

    All values configurable via environment variables with UNIFIED_ prefix.
    """
    # Primary model
    primary_model: str = field(default_factory=lambda: os.getenv("UNIFIED_PRIMARY_MODEL", "prime-7b-chat-v1"))

    # Fallback chain
    fallback_chain: List[str] = field(default_factory=lambda: os.getenv(
        "UNIFIED_FALLBACK_CHAIN",
        "prime-13b-reasoning-v1,claude-3-haiku,claude-3-5-sonnet,claude-opus-4"
    ).split(","))

    # Timeouts
    request_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("UNIFIED_REQUEST_TIMEOUT", "60.0")))
    connect_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("UNIFIED_CONNECT_TIMEOUT", "10.0")))

    # Retries
    max_retries: int = field(default_factory=lambda: int(os.getenv("UNIFIED_MAX_RETRIES", "3")))
    retry_delay_seconds: float = field(default_factory=lambda: float(os.getenv("UNIFIED_RETRY_DELAY", "1.0")))
    retry_exponential_base: float = field(default_factory=lambda: float(os.getenv("UNIFIED_RETRY_EXP_BASE", "2.0")))

    # Circuit breaker
    circuit_breaker_threshold: int = field(default_factory=lambda: int(os.getenv("UNIFIED_CB_THRESHOLD", "5")))
    circuit_breaker_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("UNIFIED_CB_TIMEOUT", "60.0")))

    # Health check
    health_check_interval_seconds: float = field(default_factory=lambda: float(os.getenv("UNIFIED_HEALTH_INTERVAL", "30.0")))

    # Budget
    enable_budget_tracking: bool = field(default_factory=lambda: os.getenv("UNIFIED_BUDGET_TRACKING", "true").lower() == "true")
    daily_budget_usd: float = field(default_factory=lambda: float(os.getenv("UNIFIED_DAILY_BUDGET", "10.0")))

    # Performance
    prefer_local: bool = field(default_factory=lambda: os.getenv("UNIFIED_PREFER_LOCAL", "true").lower() == "true")
    min_local_quality: float = field(default_factory=lambda: float(os.getenv("UNIFIED_MIN_LOCAL_QUALITY", "0.7")))

    # Metrics
    enable_metrics: bool = field(default_factory=lambda: os.getenv("UNIFIED_METRICS", "true").lower() == "true")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_model": self.primary_model,
            "fallback_chain": self.fallback_chain,
            "max_retries": self.max_retries,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "daily_budget_usd": self.daily_budget_usd,
        }


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class InferenceProvider(Enum):
    """Inference provider types."""
    LOCAL = "local"
    GCP = "gcp"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class InferenceRequest:
    """A unified inference request."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    # Content
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None

    # Parameters
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    stream: bool = False

    # Routing hints
    preferred_provider: Optional[InferenceProvider] = None
    preferred_model: Optional[str] = None
    require_local: bool = False
    max_cost: Optional[float] = None
    max_latency_ms: Optional[float] = None

    # Metadata
    created_at: float = field(default_factory=time.time)
    trace_id: Optional[str] = None

    def to_chat_format(self) -> List[Dict[str, str]]:
        """Convert to chat message format."""
        if self.messages:
            return self.messages
        elif self.prompt:
            return [{"role": "user", "content": self.prompt}]
        return []


@dataclass
class InferenceResponse:
    """A unified inference response."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    request_id: str = ""

    # Content
    text: str = ""
    finish_reason: str = "stop"

    # Provider info
    provider: InferenceProvider = InferenceProvider.LOCAL
    model: str = ""
    was_fallback: bool = False
    fallback_reason: Optional[str] = None

    # Usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Performance
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    # Metadata
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "model": self.model,
            "provider": self.provider.value,
            "was_fallback": self.was_fallback,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "tokens": self.total_tokens,
        }


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, don't try
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.

    Prevents cascading failures by stopping requests to failing services.
    """
    name: str
    threshold: int = 5
    timeout_seconds: float = 60.0

    # State
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0

    # Statistics
    total_successes: int = 0
    total_failures: int = 0
    total_rejections: int = 0

    def record_success(self) -> None:
        """Record a successful call."""
        self.failure_count = 0
        self.last_success_time = time.time()
        self.total_successes += 1

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info(f"Circuit {self.name} closed after recovery")

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.total_failures += 1

        if self.failure_count >= self.threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name} opened after {self.failure_count} failures")

    def should_allow_request(self) -> bool:
        """Check if request should be allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if time.time() - self.last_failure_time > self.timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit {self.name} entering half-open state")
                return True
            self.total_rejections += 1
            return False

        # Half-open: allow one request to test
        return True

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
        }


# =============================================================================
# PROVIDER BACKENDS
# =============================================================================


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    @property
    @abstractmethod
    def provider(self) -> InferenceProvider:
        """Get provider type."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        ...

    @abstractmethod
    async def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """Generate a response."""
        ...

    @abstractmethod
    async def stream(
        self,
        request: InferenceRequest,
    ) -> AsyncIterator[str]:
        """Stream a response."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check backend health."""
        ...


class LocalModelBackend(InferenceBackend):
    """Backend for local PRIME models."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model: Optional[Any] = None
        self._is_available = False
        self._lock = asyncio.Lock()

    @property
    def provider(self) -> InferenceProvider:
        return InferenceProvider.LOCAL

    @property
    def is_available(self) -> bool:
        return self._is_available

    async def initialize(self) -> bool:
        """Initialize the local model."""
        async with self._lock:
            if self._model:
                return True

            try:
                from jarvis_prime.models.prime_model import PrimeModel
                self._model = await PrimeModel.from_pretrained_async(self._model_name)
                self._is_available = True
                logger.info(f"Initialized local model: {self._model_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize local model {self._model_name}: {e}")
                self._is_available = False
                return False

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate using local model."""
        if not self._model:
            await self.initialize()

        if not self._model:
            raise RuntimeError(f"Local model {self._model_name} not available")

        start_time = time.time()

        messages = request.to_chat_format()
        result = await self._model.chat_async(
            messages=messages,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        latency_ms = (time.time() - start_time) * 1000

        return InferenceResponse(
            request_id=request.id,
            text=result.text,
            provider=self.provider,
            model=self._model_name,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.tokens_generated,
            total_tokens=result.prompt_tokens + result.tokens_generated,
            latency_ms=latency_ms,
            cost_usd=0.0,  # Local models are free
        )

    async def stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Stream using local model."""
        if not self._model:
            await self.initialize()

        if not self._model:
            raise RuntimeError(f"Local model {self._model_name} not available")

        messages = request.to_chat_format()
        prompt = self._model._format_chat_messages(messages)

        async for chunk in self._model.stream_async(
            prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
        ):
            yield chunk

    async def health_check(self) -> bool:
        """Check if model is healthy."""
        return self._is_available and self._model is not None


class GCPModelBackend(InferenceBackend):
    """
    Backend for GCP-hosted models with auto-provisioning.

    v92.1 - Complete GCP integration with:
    - Auto-provisioning of GPU VMs when needed
    - Connection pooling for efficiency
    - Streaming support over SSE
    - Preemption handling with graceful migration
    - Cost tracking and optimization
    - Model warm-up and health probes
    - Platform-aware routing (routes 13B to GCP on M1)

    ARCHITECTURE:
        Request → GCPModelBackend
                      ↓
            Check if VM exists (GCPVMManager)
                      ↓
            Auto-provision if needed (with startup script)
                      ↓
            Wait for model server to be ready
                      ↓
            Execute inference with pooled connection
                      ↓
            Track costs and handle preemption
    """

    # Startup script to deploy model server on GCP VM
    MODEL_SERVER_STARTUP_SCRIPT = '''#!/bin/bash
set -e

# Log all output
exec > >(tee /var/log/model-server-startup.log) 2>&1

echo "=== JARVIS Prime Model Server Setup ==="
echo "Timestamp: $(date)"

# Install system dependencies
apt-get update && apt-get install -y python3-pip python3-venv git

# Create virtual environment
python3 -m venv /opt/jarvis-prime-env
source /opt/jarvis-prime-env/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install inference server dependencies
pip install transformers accelerate bitsandbytes
pip install fastapi uvicorn aiohttp pydantic
pip install sentencepiece protobuf

# Clone or download model (using HuggingFace)
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-13b-chat-hf}"
CACHE_DIR="/opt/models"
mkdir -p $CACHE_DIR

# Download model weights (this will be cached)
python3 << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-2-13b-chat-hf")
cache_dir = "/opt/models"

print(f"Downloading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    device_map="auto",
    load_in_4bit=True,  # Quantization for efficiency
)
print("Model downloaded successfully!")
EOF

# Create inference server
cat > /opt/inference_server.py << 'SERVEREOF'
import asyncio
import json
import logging
import os
import time
from typing import AsyncGenerator, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JARVIS Prime GCP Inference Server")

# Global model and tokenizer
model = None
tokenizer = None
model_loaded = False

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "llama-13b"
    messages: List[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.7
    stream: bool = False

class ChatResponse(BaseModel):
    id: str
    choices: List[Dict]
    usage: Dict

@app.on_event("startup")
async def load_model():
    global model, tokenizer, model_loaded

    model_name = os.environ.get("MODEL_NAME", "meta-llama/Llama-2-13b-chat-hf")
    cache_dir = "/opt/models"

    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
    )

    model_loaded = True
    logger.info("Model loaded successfully!")

@app.get("/health")
async def health():
    return {"status": "healthy" if model_loaded else "loading", "model_loaded": model_loaded}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Format prompt
    prompt = ""
    for msg in request.messages:
        if msg.role == "system":
            prompt += f"[INST] <<SYS>>\\n{msg.content}\\n<</SYS>>\\n\\n"
        elif msg.role == "user":
            prompt += f"{msg.content} [/INST]"
        elif msg.role == "assistant":
            prompt += f" {msg.content} </s><s>[INST] "

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs.input_ids.shape[1]

    if request.stream:
        return StreamingResponse(
            generate_stream(inputs, request, prompt_tokens),
            media_type="text/event-stream"
        )

    # Non-streaming generation
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][prompt_tokens:]
    completion_tokens = len(generated_tokens)
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    generation_time = time.time() - start_time

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "generation_time_seconds": generation_time,
    }

async def generate_stream(inputs, request, prompt_tokens) -> AsyncGenerator[str, None]:
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = {
        **inputs,
        "max_new_tokens": request.max_tokens,
        "temperature": request.temperature,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    completion_tokens = 0
    for text in streamer:
        completion_tokens += 1
        chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": text},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\\n\\n"

    # Final chunk
    final_chunk = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    }
    yield f"data: {json.dumps(final_chunk)}\\n\\n"
    yield "data: [DONE]\\n\\n"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
SERVEREOF

# Create systemd service
cat > /etc/systemd/system/model-server.service << 'SERVICEEOF'
[Unit]
Description=JARVIS Prime Model Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt
Environment="MODEL_NAME=meta-llama/Llama-2-13b-chat-hf"
ExecStart=/opt/jarvis-prime-env/bin/python /opt/inference_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICEEOF

# Enable and start service
systemctl daemon-reload
systemctl enable model-server
systemctl start model-server

echo "=== Model server setup complete ==="
echo "Server starting at http://0.0.0.0:8000"
'''

    def __init__(
        self,
        model_name: str = "prime-13b-reasoning-v1",
        auto_provision: bool = True,
    ):
        self._model_name = model_name
        self._auto_provision = auto_provision
        self._gcp_manager: Optional[Any] = None
        self._endpoint: Optional[str] = None
        self._session: Optional[Any] = None
        self._is_available = False
        self._lock = asyncio.Lock()

        # Connection pool settings
        self._pool_size = int(os.getenv("GCP_POOL_SIZE", "10"))
        self._connect_timeout = float(os.getenv("GCP_CONNECT_TIMEOUT", "30.0"))
        self._request_timeout = float(os.getenv("GCP_REQUEST_TIMEOUT", "120.0"))

        # Health check
        self._last_health_check = 0.0
        self._health_check_interval = 30.0
        self._consecutive_failures = 0
        self._max_failures = 3

        # Cost tracking
        self._total_cost = 0.0
        self._requests_count = 0

        # Platform detection
        self._is_m1_mac = self._detect_m1_mac()

    def _detect_m1_mac(self) -> bool:
        """Detect if running on Apple Silicon M1."""
        import platform
        return (
            platform.system() == "Darwin" and
            platform.machine() == "arm64"
        )

    @property
    def provider(self) -> InferenceProvider:
        return InferenceProvider.GCP

    @property
    def is_available(self) -> bool:
        return self._is_available

    async def _get_session(self) -> Any:
        """Get or create pooled HTTP session."""
        if self._session is None or self._session.closed:
            try:
                import aiohttp

                connector = aiohttp.TCPConnector(
                    limit=self._pool_size,
                    limit_per_host=self._pool_size,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    keepalive_timeout=30,
                )

                timeout = aiohttp.ClientTimeout(
                    total=self._request_timeout,
                    connect=self._connect_timeout,
                )

                self._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                )
            except ImportError:
                raise RuntimeError("aiohttp required for GCP backend")

        return self._session

    async def initialize(self) -> bool:
        """Initialize GCP backend with auto-provisioning."""
        async with self._lock:
            if self._endpoint and self._is_available:
                return True

            try:
                # Get GCP manager
                from jarvis_prime.core.gcp_vm_manager import get_gcp_manager
                self._gcp_manager = await get_gcp_manager()

                # Check for existing endpoint
                endpoint = await self._gcp_manager.get_inference_endpoint()

                if endpoint:
                    self._endpoint = endpoint
                    logger.info(f"Using existing GCP endpoint: {endpoint}")
                elif self._auto_provision:
                    # Auto-provision new instance
                    logger.info("No GCP endpoint available, auto-provisioning...")

                    from jarvis_prime.core.gcp_vm_manager import VMConfig

                    # Create VM config with startup script
                    vm_config = VMConfig(
                        name=f"jarvis-prime-13b-{int(time.time()) % 10000}",
                        machine_type=os.getenv("GCP_MACHINE_TYPE", "n1-standard-8"),
                        zone=os.getenv("GCP_ZONE", "us-central1-a"),
                        gpu_type=os.getenv("GCP_GPU_TYPE", "nvidia-tesla-t4"),
                        gpu_count=int(os.getenv("GCP_GPU_COUNT", "1")),
                        spot=True,  # Use spot for cost savings
                        disk_size_gb=200,  # Larger disk for model weights
                        startup_script=self.MODEL_SERVER_STARTUP_SCRIPT,
                        labels={
                            "purpose": "jarvis-prime-inference",
                            "model": self._model_name,
                        },
                    )

                    instance = await self._gcp_manager.provision_instance(vm_config=vm_config)

                    if instance and instance.inference_endpoint:
                        self._endpoint = instance.inference_endpoint
                        logger.info(f"Provisioned GCP instance: {instance.name}")

                        # Wait for model server to be ready
                        await self._wait_for_model_ready()
                    else:
                        logger.warning("Failed to provision GCP instance")
                        return False
                else:
                    logger.warning("GCP endpoint not available and auto-provision disabled")
                    return False

                # Verify endpoint is healthy
                if await self._check_endpoint_health():
                    self._is_available = True
                    logger.info(f"GCP backend initialized: {self._endpoint}")
                    return True
                else:
                    logger.warning("GCP endpoint not healthy after initialization")
                    return False

            except ImportError as e:
                logger.warning(f"GCP manager not available: {e}")
                return False
            except Exception as e:
                logger.error(f"Failed to initialize GCP backend: {e}")
                return False

    async def _wait_for_model_ready(self, timeout: float = 600.0) -> bool:
        """Wait for model server to be ready on GCP VM."""
        if not self._endpoint:
            return False

        logger.info(f"Waiting for model server at {self._endpoint}...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                session = await self._get_session()
                async with session.get(
                    f"{self._endpoint}/health",
                    timeout=10,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("model_loaded", False):
                            logger.info("Model server ready!")
                            return True
                        else:
                            logger.info("Model still loading...")
            except Exception as e:
                logger.debug(f"Health check failed (expected during startup): {e}")

            await asyncio.sleep(10)

        logger.warning(f"Model server not ready after {timeout}s")
        return False

    async def _check_endpoint_health(self) -> bool:
        """Check if endpoint is healthy."""
        if not self._endpoint:
            return False

        try:
            session = await self._get_session()
            async with session.get(
                f"{self._endpoint}/health",
                timeout=5,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("model_loaded", False) or data.get("status") == "healthy"
        except Exception:
            pass

        return False

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate using GCP-hosted model."""
        if not self._endpoint:
            if not await self.initialize():
                raise RuntimeError("GCP backend not available")

        start_time = time.time()

        messages = request.to_chat_format()
        payload = {
            "model": self._model_name,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False,
        }

        try:
            session = await self._get_session()
            async with session.post(
                f"{self._endpoint}/v1/chat/completions",
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self._consecutive_failures += 1
                    raise RuntimeError(f"GCP inference failed ({response.status}): {error_text}")

                data = await response.json()

                latency_ms = (time.time() - start_time) * 1000

                # Extract response
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                # Calculate cost (Spot T4 GPU pricing estimate)
                # ~$0.35/hr for T4, ~$0.38/hr for n1-standard-8
                # Total ~$0.73/hr = $0.0002/second
                seconds = latency_ms / 1000
                cost = seconds * 0.0002  # Rough estimate

                self._total_cost += cost
                self._requests_count += 1
                self._consecutive_failures = 0

                return InferenceResponse(
                    request_id=request.id,
                    text=text,
                    provider=self.provider,
                    model=self._model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    latency_ms=latency_ms,
                    cost_usd=cost,
                )

        except Exception as e:
            self._consecutive_failures += 1

            # Check if we need to reprovision
            if self._consecutive_failures >= self._max_failures:
                logger.warning("Too many failures, marking GCP backend unavailable")
                self._is_available = False
                self._endpoint = None

            raise

    async def stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Stream response from GCP model."""
        if not self._endpoint:
            if not await self.initialize():
                raise RuntimeError("GCP backend not available")

        messages = request.to_chat_format()
        payload = {
            "model": self._model_name,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
        }

        try:
            session = await self._get_session()
            async with session.post(
                f"{self._endpoint}/v1/chat/completions",
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"GCP streaming failed: {error_text}")

                # Parse SSE stream
                async for line in response.content:
                    line = line.decode().strip()

                    if not line or line == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            self._consecutive_failures += 1
            raise

    async def health_check(self) -> bool:
        """Check GCP backend health."""
        now = time.time()

        if now - self._last_health_check < self._health_check_interval:
            return self._is_available

        self._last_health_check = now

        if await self._check_endpoint_health():
            self._is_available = True
            self._consecutive_failures = 0
            return True
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._max_failures:
                self._is_available = False
            return False

    async def shutdown(self) -> None:
        """Shutdown GCP backend and optionally release resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        logger.info(f"GCP backend shutdown. Total cost: ${self._total_cost:.4f}, Requests: {self._requests_count}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get backend statistics."""
        return {
            "provider": self.provider.value,
            "model": self._model_name,
            "endpoint": self._endpoint,
            "is_available": self._is_available,
            "is_m1_mac": self._is_m1_mac,
            "total_cost_usd": self._total_cost,
            "requests_count": self._requests_count,
            "consecutive_failures": self._consecutive_failures,
        }


class AnthropicBackend(InferenceBackend):
    """Backend for Claude API."""

    # Model mapping
    MODEL_MAP = {
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-opus-4": "claude-opus-4-20250514",
    }

    # Pricing per 1K tokens
    PRICING = {
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-opus-4": {"input": 0.015, "output": 0.075},
    }

    def __init__(self, model_name: str = "claude-3-haiku"):
        self._model_name = model_name
        self._client: Optional[Any] = None
        self._is_available = False

    @property
    def provider(self) -> InferenceProvider:
        return InferenceProvider.ANTHROPIC

    @property
    def is_available(self) -> bool:
        return self._is_available

    async def initialize(self) -> bool:
        """Initialize the Anthropic client."""
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic()
            self._is_available = True
            logger.info(f"Initialized Anthropic backend: {self._model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic backend: {e}")
            self._is_available = False
            return False

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate using Claude API."""
        if not self._client:
            await self.initialize()

        if not self._client:
            raise RuntimeError("Anthropic client not available")

        start_time = time.time()

        messages = request.to_chat_format()
        api_model = self.MODEL_MAP.get(self._model_name, self._model_name)

        response = await self._client.messages.create(
            model=api_model,
            max_tokens=request.max_tokens,
            messages=messages,
            temperature=request.temperature,
        )

        latency_ms = (time.time() - start_time) * 1000
        text = response.content[0].text

        # Calculate cost
        pricing = self.PRICING.get(self._model_name, {"input": 0.01, "output": 0.03})
        cost = (
            (response.usage.input_tokens / 1000) * pricing["input"] +
            (response.usage.output_tokens / 1000) * pricing["output"]
        )

        return InferenceResponse(
            request_id=request.id,
            text=text,
            provider=self.provider,
            model=self._model_name,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    async def stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Stream using Claude API."""
        if not self._client:
            await self.initialize()

        if not self._client:
            raise RuntimeError("Anthropic client not available")

        messages = request.to_chat_format()
        api_model = self.MODEL_MAP.get(self._model_name, self._model_name)

        async with self._client.messages.stream(
            model=api_model,
            max_tokens=request.max_tokens,
            messages=messages,
            temperature=request.temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def health_check(self) -> bool:
        """Check if API is accessible."""
        try:
            if not self._client:
                await self.initialize()
            # Could do a lightweight API call here
            return self._is_available
        except Exception:
            return False


# =============================================================================
# UNIFIED CLIENT
# =============================================================================


class BudgetTracker:
    """Tracks API spending against budget."""

    def __init__(self, daily_budget: float):
        self._daily_budget = daily_budget
        self._daily_spend: Dict[str, float] = {}  # date -> spend
        self._lock = asyncio.Lock()

    async def record_spend(self, amount: float) -> None:
        """Record spending."""
        async with self._lock:
            today = datetime.now().strftime("%Y-%m-%d")
            self._daily_spend[today] = self._daily_spend.get(today, 0.0) + amount

    async def get_remaining_budget(self) -> float:
        """Get remaining daily budget."""
        today = datetime.now().strftime("%Y-%m-%d")
        spent = self._daily_spend.get(today, 0.0)
        return max(0, self._daily_budget - spent)

    async def can_spend(self, amount: float) -> bool:
        """Check if spending is within budget."""
        remaining = await self.get_remaining_budget()
        return amount <= remaining

    def get_statistics(self) -> Dict[str, Any]:
        today = datetime.now().strftime("%Y-%m-%d")
        return {
            "daily_budget": self._daily_budget,
            "today_spend": self._daily_spend.get(today, 0.0),
            "remaining": max(0, self._daily_budget - self._daily_spend.get(today, 0.0)),
        }


class UnifiedInferenceClient:
    """
    Unified client for seamless local/API inference with fallback.

    Provides a single interface that automatically handles:
    - Primary model selection
    - Fallback to alternative models
    - Retries with exponential backoff
    - Circuit breaker protection
    - Budget tracking
    """

    def __init__(self, config: Optional[UnifiedInferenceConfig] = None):
        self._config = config or UnifiedInferenceConfig()

        # Backends
        self._backends: Dict[str, InferenceBackend] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Budget tracking
        self._budget_tracker: Optional[BudgetTracker] = None
        if self._config.enable_budget_tracking:
            self._budget_tracker = BudgetTracker(self._config.daily_budget_usd)

        # Health monitoring
        self._health_task: Optional[asyncio.Task] = None

        # Statistics
        self._total_requests = 0
        self._total_fallbacks = 0
        self._total_failures = 0
        self._request_history: Deque[Dict[str, Any]] = deque(maxlen=100)

    async def initialize(self) -> None:
        """Initialize the client and backends."""
        # Initialize primary backend
        await self._get_or_create_backend(self._config.primary_model)

        # Initialize fallback backends (lazy - on first use)
        # Start health monitoring
        self._health_task = asyncio.create_task(self._health_monitor())

        logger.info("UnifiedInferenceClient initialized")

    async def shutdown(self) -> None:
        """Shutdown the client."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        logger.info("UnifiedInferenceClient shutdown")

    async def _get_or_create_backend(self, model_name: str) -> InferenceBackend:
        """
        Get or create a backend for a model with platform-aware routing.

        v92.1 - Enhanced with:
        - Platform detection (M1 Mac routes 13B to GCP)
        - GCP backend integration with auto-provisioning
        - Intelligent model size routing
        """
        if model_name in self._backends:
            return self._backends[model_name]

        # Determine backend type with platform-aware routing
        backend: InferenceBackend

        # Platform detection for intelligent routing
        is_m1_mac = self._detect_platform() == "m1_mac"
        is_large_model = self._is_large_model(model_name)

        if model_name.startswith("claude-"):
            # Claude API backend
            backend = AnthropicBackend(model_name)
        elif model_name.startswith("gcp-") or model_name.startswith("remote-"):
            # Explicit GCP routing
            backend = GCPModelBackend(model_name)
        elif is_large_model and is_m1_mac:
            # Route large models to GCP on M1 Mac
            logger.info(f"M1 Mac detected: routing {model_name} to GCP for better performance")
            backend = GCPModelBackend(model_name)
        elif model_name.startswith("prime-"):
            # Local PRIME model
            backend = LocalModelBackend(model_name)
        else:
            # Default to local
            backend = LocalModelBackend(model_name)

        # Initialize
        await backend.initialize()

        # Create circuit breaker
        self._circuit_breakers[model_name] = CircuitBreaker(
            name=model_name,
            threshold=self._config.circuit_breaker_threshold,
            timeout_seconds=self._config.circuit_breaker_timeout_seconds,
        )

        self._backends[model_name] = backend
        return backend

    def _detect_platform(self) -> str:
        """Detect current platform for intelligent routing."""
        import platform as plat
        system = plat.system()
        machine = plat.machine()

        if system == "Darwin" and machine == "arm64":
            return "m1_mac"
        elif system == "Darwin":
            return "intel_mac"
        elif system == "Linux":
            # Check for GPU
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    return "linux_gpu"
            except Exception:
                pass
            return "linux_cpu"
        else:
            return "unknown"

    def _is_large_model(self, model_name: str) -> bool:
        """Check if model is too large for efficient local inference on M1."""
        # Models with 13B+ parameters should go to GCP on M1
        large_model_indicators = [
            "13b", "13B", "14b", "14B",
            "30b", "30B", "33b", "33B",
            "65b", "65B", "70b", "70B",
            "reasoning", "code-expert",
        ]
        return any(indicator in model_name for indicator in large_model_indicators)

    async def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any,
    ) -> InferenceResponse:
        """
        Generate a response with automatic fallback.

        Args:
            prompt: Text prompt (alternative to messages)
            messages: Chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream (returns full response)
            **kwargs: Additional parameters

        Returns:
            InferenceResponse with generation result
        """
        request = InferenceRequest(
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **{k: v for k, v in kwargs.items() if k in InferenceRequest.__dataclass_fields__}
        )

        return await self._execute_with_fallback(request)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> InferenceResponse:
        """Chat interface."""
        return await self.generate(messages=messages, **kwargs)

    async def generate_with_tier_routing(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> InferenceResponse:
        """
        Generate with intelligent tier-based routing (v94.0).

        Uses the HybridTieredRouter to analyze prompt complexity and
        select the optimal model tier automatically.

        Args:
            prompt: Text prompt
            messages: Chat messages
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            context: Additional context for routing:
                - max_latency_ms: Maximum acceptable latency
                - max_cost_usd: Maximum cost for this request
                - prefer_local: Prefer local models (default: True)
                - require_capability: Required capability

        Returns:
            InferenceResponse with tier routing metadata

        Example:
            response = await client.generate_with_tier_routing(
                "Explain quantum computing in detail",
                context={"max_latency_ms": 5000}
            )
            print(f"Tier used: {response.model}")  # "Llama-3.3-70B-Instruct"
        """
        # Get the hybrid router
        router = await _get_hybrid_router()

        if router is None:
            # Fallback to standard generation
            logger.warning("Tier routing unavailable, using standard fallback")
            return await self.generate(
                prompt=prompt,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

        # Format prompt for complexity analysis
        analysis_prompt = prompt or ""
        if messages:
            analysis_prompt = " ".join(m.get("content", "") for m in messages)

        # Route to optimal tier
        routing_result = await router.route(analysis_prompt, context)

        logger.info(
            f"Tier routing: {routing_result.tier_name} "
            f"(complexity={routing_result.complexity_score:.2f})"
        )

        # Map tier to model chain
        model_chain = self._build_tier_model_chain(routing_result)

        # Create request
        request = InferenceRequest(
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **{k: v for k, v in kwargs.items() if k in InferenceRequest.__dataclass_fields__}
        )

        # Execute with tier-aware fallback
        response = await self._execute_with_tier_chain(request, model_chain, routing_result)

        # Record cost if applicable
        if response.cost_usd > 0:
            await router.record_cost(routing_result.tier_id, response.cost_usd)

        return response

    def _build_tier_model_chain(self, routing_result) -> List[str]:
        """Build model chain from tier routing result."""
        chain = []

        # Primary model from selected tier
        tier_to_model = {
            "tier_0_local_fast": "prime-7b-chat-v1",
            "tier_05_local_capable": "prime-13b-reasoning-v1",
            "tier_1_cloud_intelligent": "gcp-llama-70b",
            "tier_2_deep_reasoning": "claude-opus-4",
        }

        # Add primary model
        primary = tier_to_model.get(routing_result.tier_id, self._config.primary_model)
        chain.append(primary)

        # Add fallback chain
        for fallback_tier_id in routing_result.fallback_chain:
            fallback_model = tier_to_model.get(fallback_tier_id)
            if fallback_model and fallback_model not in chain:
                chain.append(fallback_model)

        # Ensure Claude fallbacks at the end
        if "claude-3-5-sonnet" not in chain:
            chain.append("claude-3-5-sonnet")
        if "claude-opus-4" not in chain:
            chain.append("claude-opus-4")

        return chain

    async def _execute_with_tier_chain(
        self,
        request: InferenceRequest,
        model_chain: List[str],
        routing_result,
    ) -> InferenceResponse:
        """Execute request with tier-based model chain."""
        self._total_requests += 1

        last_error: Optional[Exception] = None
        was_fallback = False
        fallback_reason = None

        for model_name in model_chain:
            # Check circuit breaker
            breaker = self._circuit_breakers.get(model_name)
            if breaker and not breaker.should_allow_request():
                was_fallback = True
                fallback_reason = f"Circuit breaker open for {model_name}"
                continue

            # Check budget for API models
            if model_name.startswith("claude-") and self._budget_tracker:
                estimated_cost = 0.01
                if not await self._budget_tracker.can_spend(estimated_cost):
                    was_fallback = True
                    fallback_reason = "Budget exceeded"
                    continue

            try:
                response = await self._execute_single(request, model_name)

                # Record success
                if breaker:
                    breaker.record_success()

                # Record cost
                if self._budget_tracker and response.cost_usd > 0:
                    await self._budget_tracker.record_spend(response.cost_usd)

                response.was_fallback = was_fallback
                response.fallback_reason = fallback_reason

                if was_fallback:
                    self._total_fallbacks += 1

                # Add tier routing metadata
                self._request_history.append({
                    "request_id": request.id,
                    "model": model_name,
                    "tier_id": routing_result.tier_id,
                    "tier_name": routing_result.tier_name,
                    "complexity_score": routing_result.complexity_score,
                    "was_fallback": was_fallback,
                    "latency_ms": response.latency_ms,
                    "cost": response.cost_usd,
                    "timestamp": time.time(),
                })

                return response

            except Exception as e:
                logger.warning(f"Tier chain: {model_name} failed: {e}")
                last_error = e
                was_fallback = True
                fallback_reason = str(e)

                if breaker:
                    breaker.record_failure()

        # All models failed
        self._total_failures += 1
        raise RuntimeError(f"All models in tier chain failed. Last error: {last_error}")

    async def stream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a response with automatic fallback.

        Yields text chunks as they're generated.
        """
        request = InferenceRequest(
            prompt=prompt,
            messages=messages,
            stream=True,
            **{k: v for k, v in kwargs.items() if k in InferenceRequest.__dataclass_fields__}
        )

        # Get model chain
        model_chain = [self._config.primary_model] + self._config.fallback_chain

        for model_name in model_chain:
            # Check circuit breaker
            breaker = self._circuit_breakers.get(model_name)
            if breaker and not breaker.should_allow_request():
                continue

            try:
                backend = await self._get_or_create_backend(model_name)
                if not backend.is_available:
                    continue

                async for chunk in backend.stream(request):
                    yield chunk

                if breaker:
                    breaker.record_success()
                return

            except Exception as e:
                logger.warning(f"Streaming from {model_name} failed: {e}")
                if breaker:
                    breaker.record_failure()
                continue

        raise RuntimeError("All models failed for streaming")

    async def _execute_with_fallback(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """Execute request with fallback chain."""
        self._total_requests += 1

        # Build model chain
        model_chain = []

        # Check if specific model requested
        if request.preferred_model:
            model_chain.append(request.preferred_model)
        else:
            model_chain.append(self._config.primary_model)

        # Add fallbacks (unless local required)
        if not request.require_local:
            model_chain.extend(self._config.fallback_chain)

        # Remove duplicates while preserving order
        seen = set()
        model_chain = [m for m in model_chain if not (m in seen or seen.add(m))]

        last_error: Optional[Exception] = None
        was_fallback = False
        fallback_reason = None

        for model_name in model_chain:
            # Check circuit breaker
            breaker = self._circuit_breakers.get(model_name)
            if breaker and not breaker.should_allow_request():
                was_fallback = True
                fallback_reason = f"Circuit breaker open for {model_name}"
                continue

            # Check budget for API models
            if model_name.startswith("claude-") and self._budget_tracker:
                # Estimate cost
                estimated_cost = 0.01  # Simple estimate
                if not await self._budget_tracker.can_spend(estimated_cost):
                    was_fallback = True
                    fallback_reason = "Budget exceeded"
                    continue

            # Try to generate
            try:
                response = await self._execute_single(request, model_name)

                # Record success
                if breaker:
                    breaker.record_success()

                # Record cost
                if self._budget_tracker and response.cost_usd > 0:
                    await self._budget_tracker.record_spend(response.cost_usd)

                response.was_fallback = was_fallback
                response.fallback_reason = fallback_reason

                if was_fallback:
                    self._total_fallbacks += 1

                # Record to history
                self._request_history.append({
                    "request_id": request.id,
                    "model": model_name,
                    "was_fallback": was_fallback,
                    "latency_ms": response.latency_ms,
                    "cost": response.cost_usd,
                    "timestamp": time.time(),
                })

                return response

            except Exception as e:
                logger.warning(f"Generation from {model_name} failed: {e}")
                last_error = e
                was_fallback = True
                fallback_reason = str(e)

                if breaker:
                    breaker.record_failure()

        # All models failed
        self._total_failures += 1
        raise RuntimeError(f"All models in fallback chain failed. Last error: {last_error}")

    async def _execute_single(
        self,
        request: InferenceRequest,
        model_name: str,
    ) -> InferenceResponse:
        """Execute on a single model with retry."""
        backend = await self._get_or_create_backend(model_name)

        if not backend.is_available:
            raise RuntimeError(f"Backend {model_name} not available")

        # Retry loop
        last_error: Optional[Exception] = None

        for attempt in range(self._config.max_retries):
            try:
                return await asyncio.wait_for(
                    backend.generate(request),
                    timeout=self._config.request_timeout_seconds,
                )
            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError(f"Request to {model_name} timed out")
            except Exception as e:
                last_error = e

            # Exponential backoff
            if attempt < self._config.max_retries - 1:
                delay = self._config.retry_delay_seconds * (self._config.retry_exponential_base ** attempt)
                await asyncio.sleep(delay)

        raise last_error or RuntimeError(f"Failed to execute on {model_name}")

    async def _health_monitor(self) -> None:
        """Background health monitoring."""
        while True:
            try:
                await asyncio.sleep(self._config.health_check_interval_seconds)

                for model_name, backend in self._backends.items():
                    try:
                        is_healthy = await backend.health_check()
                        if not is_healthy:
                            logger.warning(f"Backend {model_name} unhealthy")
                    except Exception as e:
                        logger.error(f"Health check failed for {model_name}: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics with GCP backend details."""
        backend_stats = {}
        for name, backend in self._backends.items():
            stats = {
                "available": backend.is_available,
                "provider": backend.provider.value,
            }
            # Add GCP-specific stats
            if hasattr(backend, 'get_statistics'):
                stats.update(backend.get_statistics())
            backend_stats[name] = stats

        return {
            "total_requests": self._total_requests,
            "total_fallbacks": self._total_fallbacks,
            "total_failures": self._total_failures,
            "fallback_rate": self._total_fallbacks / max(self._total_requests, 1),
            "failure_rate": self._total_failures / max(self._total_requests, 1),
            "platform": self._detect_platform(),
            "backends": backend_stats,
            "circuit_breakers": {
                name: breaker.get_statistics()
                for name, breaker in self._circuit_breakers.items()
            },
            "budget": self._budget_tracker.get_statistics() if self._budget_tracker else None,
            "config": self._config.to_dict(),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get unified inference client status (compatibility method)."""
        stats = self.get_statistics()
        return {
            "backends": [
                {
                    "name": name,
                    "healthy": backend.is_available,
                    "provider": backend.provider.value,
                }
                for name, backend in self._backends.items()
            ],
            "circuit_breakers": len(self._circuit_breakers),
            "fallback_order": self._config.fallback_chain,
            "total_requests": self._total_requests,
            "fallback_count": self._total_fallbacks,
            "platform": self._detect_platform(),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


_unified_client: Optional[UnifiedInferenceClient] = None
_client_lock = asyncio.Lock()


async def get_unified_client(
    config: Optional[UnifiedInferenceConfig] = None,
) -> UnifiedInferenceClient:
    """Get or create the global unified inference client."""
    global _unified_client

    async with _client_lock:
        if _unified_client is None:
            _unified_client = UnifiedInferenceClient(config)
            await _unified_client.initialize()

        return _unified_client


async def shutdown_unified_client() -> None:
    """Shutdown the global unified client."""
    global _unified_client

    async with _client_lock:
        if _unified_client:
            await _unified_client.shutdown()
            _unified_client = None


# Aliases for compatibility
get_unified_inference_client = get_unified_client
shutdown_unified_inference_client = shutdown_unified_client


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Config
    "UnifiedInferenceConfig",
    # Data structures
    "InferenceProvider",
    "InferenceRequest",
    "InferenceResponse",
    # Components
    "CircuitBreaker",
    "CircuitState",
    "BudgetTracker",
    # Backends
    "InferenceBackend",
    "LocalModelBackend",
    "GCPModelBackend",
    "AnthropicBackend",
    # Client
    "UnifiedInferenceClient",
    "get_unified_client",
    "shutdown_unified_client",
    # Aliases
    "get_unified_inference_client",
    "shutdown_unified_inference_client",
]


# =============================================================================
# HYBRID TIERED ROUTING CONVENIENCE FUNCTIONS (v94.0)
# =============================================================================


async def generate_with_routing(
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> InferenceResponse:
    """
    Convenience function to generate with intelligent tier routing.

    v94.0 - Automatically routes to optimal tier based on complexity.

    Args:
        prompt: The prompt to generate from
        context: Optional routing context:
            - max_latency_ms: Maximum acceptable latency
            - max_cost_usd: Maximum cost for this request
            - prefer_local: Prefer local models
        **kwargs: Additional generation parameters

    Returns:
        InferenceResponse with tier routing metadata

    Example:
        # Simple usage - automatic routing
        response = await generate_with_routing("What is AI?")

        # With constraints
        response = await generate_with_routing(
            "Design a microservice architecture",
            context={"max_latency_ms": 10000}
        )
        print(f"Used tier: {response.model}")  # Shows which model handled it
    """
    client = await get_unified_client()
    return await client.generate_with_tier_routing(prompt=prompt, context=context, **kwargs)


async def analyze_and_route(
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze prompt complexity and return routing decision without executing.

    Useful for debugging or UI display.

    Args:
        prompt: The prompt to analyze
        context: Optional routing context

    Returns:
        Dict with complexity analysis and routing decision
    """
    router = await _get_hybrid_router()

    if router is None:
        return {
            "error": "HybridTieredRouter not available",
            "fallback_mode": True,
        }

    routing_result = await router.route(prompt, context)

    return {
        "complexity_score": routing_result.complexity_score,
        "confidence": routing_result.confidence,
        "selected_tier": routing_result.tier_id,
        "selected_tier_name": routing_result.tier_name,
        "model_name": routing_result.model_name,
        "reasoning": routing_result.reasoning,
        "estimated_latency_ms": routing_result.estimated_latency_ms,
        "estimated_cost_usd": routing_result.estimated_cost_usd,
        "available_tiers": routing_result.available_tiers,
        "unavailable_tiers": routing_result.unavailable_tiers,
    }


# Add to exports
__all__.extend([
    "generate_with_routing",
    "analyze_and_route",
])
