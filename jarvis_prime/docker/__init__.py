"""
JARVIS-Prime Docker Support
===========================

Docker-specific components for running JARVIS-Prime in containers:
- LlamaServerExecutor: Communicates with llama.cpp server
- ModelDownloader: Downloads GGUF models from HuggingFace
- ReactorCoreWatcher: Auto-deploys models from reactor-core
- DockerEntrypoint: Container entry point with command dispatch

Usage (Docker Compose):
    docker-compose up -d jarvis-prime

Usage (Manual):
    docker run -p 8000:8000 -v ./models:/app/models jarvis-prime:latest

Model Download:
    docker-compose run model-downloader download --catalog mistral-7b-instruct
"""

from jarvis_prime.docker.llama_server_executor import LlamaServerExecutor, LlamaServerConfig
from jarvis_prime.docker.model_downloader import (
    ModelDownloader,
    download_model,
    list_available_models,
    recommend_model,
    MODEL_CATALOG,
    ModelSpec,
    ModelMetadata,
)
from jarvis_prime.docker.reactor_core_watcher import (
    ReactorCoreWatcher,
    ReactorCoreModelManifest,
    DeploymentResult,
    push_model_to_jarvis_prime,
)
from jarvis_prime.docker.entrypoint import main as entrypoint_main

__all__ = [
    # Llama Server
    "LlamaServerExecutor",
    "LlamaServerConfig",
    # Model Download
    "ModelDownloader",
    "download_model",
    "list_available_models",
    "recommend_model",
    "MODEL_CATALOG",
    "ModelSpec",
    "ModelMetadata",
    # Reactor-Core Integration
    "ReactorCoreWatcher",
    "ReactorCoreModelManifest",
    "DeploymentResult",
    "push_model_to_jarvis_prime",
    # Entrypoint
    "entrypoint_main",
]
