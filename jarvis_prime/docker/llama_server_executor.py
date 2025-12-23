"""
Llama Server Executor - llama.cpp Server Communication
======================================================

ModelExecutor implementation that communicates with a running llama.cpp server.
This provides efficient inference with memory-mapped model files and
OpenAI-compatible API access.

Features:
- Async HTTP communication with llama-server
- Streaming support for token-by-token generation
- Health monitoring and auto-reconnection
- Memory-efficient (model stays in llama-server process)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LlamaServerConfig:
    """Configuration for llama.cpp server."""
    model_path: Path
    host: str = field(default_factory=lambda: os.getenv("LLAMA_SERVER_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("LLAMA_SERVER_PORT", "8080")))
    n_ctx: int = field(default_factory=lambda: int(os.getenv("LLAMA_N_CTX", "4096")))
    n_batch: int = field(default_factory=lambda: int(os.getenv("LLAMA_N_BATCH", "512")))
    n_gpu_layers: int = field(default_factory=lambda: int(os.getenv("LLAMA_N_GPU_LAYERS", "0")))
    threads: int = field(default_factory=lambda: int(os.getenv("LLAMA_THREADS", "4")))
    llama_server_path: Path = field(
        default_factory=lambda: Path(os.getenv("LLAMA_CPP_PATH", "/app/llama.cpp")) / "llama-server"
    )
    startup_timeout: float = 120.0
    request_timeout: float = 300.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class LlamaServerExecutor:
    """
    Model executor using llama.cpp server backend.

    Manages a llama-server subprocess and communicates via HTTP.
    Optimized for Docker deployment with memory-efficient model serving.

    Usage:
        executor = LlamaServerExecutor()
        await executor.load(Path("./models/model.gguf"))

        # Generate
        response = await executor.generate("Hello, world!")

        # Stream
        async for token in executor.generate_stream("Tell me a story"):
            print(token, end="", flush=True)

        # Cleanup
        await executor.unload()
    """

    def __init__(self, config: Optional[LlamaServerConfig] = None):
        self._config: Optional[LlamaServerConfig] = config
        self._process: Optional[subprocess.Popen] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._is_loaded = False
        self._model_path: Optional[Path] = None
        self._health_check_task: Optional[asyncio.Task] = None

    async def load(self, model_path: Path, **kwargs) -> None:
        """
        Load model by starting llama-server subprocess.

        Args:
            model_path: Path to GGUF model file
            **kwargs: Additional config overrides
        """
        if self._is_loaded:
            await self.unload()

        self._model_path = model_path

        # Create config if not provided
        if self._config is None:
            self._config = LlamaServerConfig(model_path=model_path)
        else:
            self._config.model_path = model_path

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        # Verify model exists
        if not self._config.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self._config.model_path}")

        # Start llama-server
        await self._start_server()

        # Create HTTP client
        self._client = httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=httpx.Timeout(self._config.request_timeout),
        )

        # Wait for server to be ready
        await self._wait_for_ready()

        # Start health monitoring
        self._health_check_task = asyncio.create_task(self._health_monitor())

        self._is_loaded = True
        logger.info(f"LlamaServerExecutor loaded: {model_path}")

    async def _start_server(self) -> None:
        """Start the llama-server subprocess."""
        cmd = [
            str(self._config.llama_server_path),
            "--model", str(self._config.model_path),
            "--host", self._config.host,
            "--port", str(self._config.port),
            "--ctx-size", str(self._config.n_ctx),
            "--batch-size", str(self._config.n_batch),
            "--threads", str(self._config.threads),
        ]

        if self._config.n_gpu_layers > 0:
            cmd.extend(["--n-gpu-layers", str(self._config.n_gpu_layers)])

        logger.info(f"Starting llama-server: {' '.join(cmd)}")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )

        # Start log reader
        asyncio.create_task(self._read_server_logs())

    async def _read_server_logs(self) -> None:
        """Read and log server output."""
        if not self._process or not self._process.stdout:
            return

        try:
            while self._process.poll() is None:
                line = self._process.stdout.readline()
                if line:
                    logger.debug(f"llama-server: {line.strip()}")
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.debug(f"Log reader stopped: {e}")

    async def _wait_for_ready(self) -> None:
        """Wait for llama-server to be ready."""
        start_time = time.time()

        while time.time() - start_time < self._config.startup_timeout:
            try:
                response = await self._client.get("/health")
                if response.status_code == 200:
                    logger.info("llama-server is ready")
                    return
            except (httpx.ConnectError, httpx.ConnectTimeout):
                pass

            # Check if process died
            if self._process and self._process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self._process.returncode}")

            await asyncio.sleep(0.5)

        raise TimeoutError(f"llama-server did not start within {self._config.startup_timeout}s")

    async def _health_monitor(self) -> None:
        """Background health monitoring."""
        while self._is_loaded:
            try:
                response = await self._client.get("/health")
                if response.status_code != 200:
                    logger.warning(f"llama-server health check failed: {response.status_code}")
            except Exception as e:
                logger.error(f"llama-server health check error: {e}")
            await asyncio.sleep(30)

    async def unload(self) -> None:
        """Stop llama-server and cleanup."""
        self._is_loaded = False

        # Cancel health monitor
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close HTTP client
        if self._client:
            await self._client.aclose()
            self._client = None

        # Stop server process
        if self._process:
            try:
                # Try graceful shutdown
                if os.name != "nt":
                    os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                else:
                    self._process.terminate()

                # Wait for exit
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill
                    if os.name != "nt":
                        os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                    else:
                        self._process.kill()
                    self._process.wait()

            except ProcessLookupError:
                pass  # Already dead
            except Exception as e:
                logger.error(f"Error stopping llama-server: {e}")

            self._process = None

        logger.info("LlamaServerExecutor unloaded")

    async def validate(self) -> bool:
        """Validate loaded model by running a test generation."""
        if not self._is_loaded or not self._client:
            return False

        try:
            response = await self.generate("Hello", max_tokens=10)
            return len(response) > 0
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Generate completion from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if not self._is_loaded or not self._client:
            raise RuntimeError("Model not loaded")

        # Build request payload (OpenAI-compatible)
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.9),
            "stop": kwargs.get("stop", []),
            "stream": False,
        }

        try:
            response = await self._client.post("/completion", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("content", "")

        except httpx.HTTPStatusError as e:
            logger.error(f"Generation failed: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream generated tokens.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Yields:
            Generated tokens one at a time
        """
        if not self._is_loaded or not self._client:
            raise RuntimeError("Model not loaded")

        # Build request payload
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.9),
            "stop": kwargs.get("stop", []),
            "stream": True,
        }

        try:
            async with self._client.stream("POST", "/completion", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    try:
                        data = json.loads(line[6:])
                        content = data.get("content", "")
                        if content:
                            yield content

                        # Check for stop condition
                        if data.get("stop", False):
                            break

                    except json.JSONDecodeError:
                        continue

        except httpx.HTTPStatusError as e:
            logger.error(f"Stream generation failed: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            raise

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        OpenAI-compatible chat completion.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            OpenAI-compatible response dict
        """
        if not self._is_loaded or not self._client:
            raise RuntimeError("Model not loaded")

        # Build chat format prompt
        prompt = self._format_chat_messages(messages)

        # Generate
        content = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        # Build OpenAI-compatible response
        return {
            "id": f"chatcmpl-{hash(prompt) % 1000000:06d}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(self._model_path.name) if self._model_path else "unknown",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len(prompt.split()) + len(content.split()),
            },
        }

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into prompt string."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted.append(f"<|system|>\n{content}</s>")
            elif role == "user":
                formatted.append(f"<|user|>\n{content}</s>")
            elif role == "assistant":
                formatted.append(f"<|assistant|>\n{content}</s>")
        formatted.append("<|assistant|>\n")
        return "\n".join(formatted)

    def get_status(self) -> Dict[str, Any]:
        """Get executor status."""
        return {
            "loaded": self._is_loaded,
            "model_path": str(self._model_path) if self._model_path else None,
            "server_url": self._config.base_url if self._config else None,
            "process_running": self._process is not None and self._process.poll() is None,
        }
