"""Shared fixtures for adaptive quantization engine tests."""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def tmp_models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory with fake GGUF files."""
    models = tmp_path / "models"
    models.mkdir()
    return models


@pytest.fixture
def fake_gguf_files(tmp_models_dir: Path) -> Dict[str, Path]:
    """Create fake GGUF files of known sizes for testing."""
    files = {}
    specs = {
        "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf": 4_400_000_000,
        "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf": 8_400_000_000,
        "Qwen2.5-Coder-32B-Instruct-IQ2_M.gguf": 11_000_000_000,
        "Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf": 19_000_000_000,
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf": 771_000_000,
    }
    for name, size in specs.items():
        p = tmp_models_dir / name
        # Create sparse file (doesn't use disk space)
        with open(p, "wb") as f:
            f.seek(size - 1)
            f.write(b"\0")
        files[name] = p
    return files


@pytest.fixture
def l4_vram_bytes() -> int:
    """NVIDIA L4 total VRAM in bytes."""
    return 23_034 * 1024 * 1024  # 23,034 MiB


@pytest.fixture
def mock_executor() -> MagicMock:
    """Mock LlamaCppExecutor for transition tests."""
    executor = MagicMock()
    executor.load = AsyncMock()
    executor.unload = AsyncMock()
    executor.validate = AsyncMock(return_value=True)
    executor.generate = AsyncMock(return_value="test response")
    executor.is_loaded = MagicMock(return_value=True)
    executor._model_path = None
    executor.config = MagicMock()
    executor.config.n_gpu_layers = -1
    executor.config.cache_type_k = "f16"
    executor.config.cache_type_v = "f16"
    return executor
