"""
JARVIS Prime Serving Module
===========================

Production-grade model serving infrastructure.

Components:
- InferenceServer: FastAPI-based inference server
- RequestBatcher: Intelligent request batching
- RateLimiter: Token bucket rate limiter
- ModelPool: Managed pool of loaded models

Usage:
    from jarvis_prime.serving import InferenceServer, ServingConfig

    config = ServingConfig(port=8000)
    server = InferenceServer(config)
    await server.start()
"""

from jarvis_prime.serving.model_serving import (
    InferenceServer,
    ServingConfig,
    RequestBatcher,
    RateLimiter,
    ModelPool,
    ChatCompletionRequest,
    ChatCompletionResponse,
    get_inference_server,
    shutdown_inference_server,
)

__all__ = [
    "InferenceServer",
    "ServingConfig",
    "RequestBatcher",
    "RateLimiter",
    "ModelPool",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "get_inference_server",
    "shutdown_inference_server",
]
