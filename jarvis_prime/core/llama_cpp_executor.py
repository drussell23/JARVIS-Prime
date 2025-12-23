"""
LlamaCpp Model Executor - GGUF Model Support for JARVIS-Prime
==============================================================

Uses llama-cpp-python for efficient GGUF model inference.
Supports CPU, Metal (M1/M2/M3), and CUDA backends.

Features:
- Zero-downtime hot-swap via reference counting
- Async-safe thread pool execution
- Memory-efficient Q4_K_M quantization support
- Chat template formatting
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from jarvis_prime.core.model_manager import ChatMessage, ModelExecutor

logger = logging.getLogger(__name__)


@dataclass
class LlamaCppConfig:
    """Configuration for LlamaCpp model loading."""
    n_ctx: int = 2048              # Context window size
    n_threads: int = 4             # CPU threads (0 = auto)
    n_gpu_layers: int = 0          # GPU layers (0 = CPU, -1 = all, >0 = specific)
    n_batch: int = 512             # Batch size for prompt processing
    use_mmap: bool = True          # Memory-map model file
    use_mlock: bool = False        # Lock memory (root required)
    verbose: bool = False          # Verbose logging
    seed: int = -1                 # RNG seed (-1 = random)

    # Chat template (TinyLlama default)
    chat_template: str = "tinyllama"

    # Generation defaults
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9

    # Stop tokens
    stop_tokens: List[str] = field(default_factory=lambda: ["</s>", "<|user|>", "\n\n\n"])


CHAT_TEMPLATES = {
    "tinyllama": {
        "system": "<|system|>\n{content}\n</s>\n",
        "user": "<|user|>\n{content}\n</s>\n",
        "assistant": "<|assistant|>\n{content}",
        "generation_prefix": "<|assistant|>\n",
    },
    "llama2": {
        "system": "<<SYS>>\n{content}\n<</SYS>>\n\n",
        "user": "[INST] {content} [/INST]",
        "assistant": "{content}",
        "generation_prefix": "",
    },
    "chatml": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>",
        "generation_prefix": "<|im_start|>assistant\n",
    },
    "plain": {
        "system": "System: {content}\n\n",
        "user": "User: {content}\n",
        "assistant": "Assistant: {content}\n",
        "generation_prefix": "Assistant: ",
    },
}


class LlamaCppExecutor(ModelExecutor):
    """
    Model executor using llama-cpp-python for GGUF models.

    Thread-safe and async-compatible via ThreadPoolExecutor.

    Usage:
        executor = LlamaCppExecutor(config=LlamaCppConfig(n_gpu_layers=-1))
        await executor.load(Path("models/tinyllama.gguf"))
        response = await executor.generate("Hello!")
        await executor.unload()
    """

    def __init__(self, config: Optional[LlamaCppConfig] = None):
        self.config = config or LlamaCppConfig()
        self._model = None
        self._model_path: Optional[Path] = None
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llama-cpp")

        # Chat template
        self._chat_template = CHAT_TEMPLATES.get(
            self.config.chat_template,
            CHAT_TEMPLATES["plain"]
        )

        # Statistics
        self._generation_count = 0
        self._total_tokens = 0

    async def load(self, model_path: Path, **kwargs) -> None:
        """Load a GGUF model."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._load_sync, model_path, kwargs)

    def _load_sync(self, model_path: Path, kwargs: Dict[str, Any]) -> None:
        """Synchronous model loading."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python required: pip install llama-cpp-python")

        with self._lock:
            if self._model:
                logger.warning("Model already loaded, unloading first")
                self._unload_sync()

            logger.info(f"Loading GGUF model: {model_path}")

            # Merge kwargs with config
            n_gpu_layers = kwargs.get("n_gpu_layers", self.config.n_gpu_layers)

            self._model = Llama(
                model_path=str(model_path),
                n_ctx=kwargs.get("n_ctx", self.config.n_ctx),
                n_threads=kwargs.get("n_threads", self.config.n_threads),
                n_gpu_layers=n_gpu_layers,
                n_batch=kwargs.get("n_batch", self.config.n_batch),
                use_mmap=kwargs.get("use_mmap", self.config.use_mmap),
                use_mlock=kwargs.get("use_mlock", self.config.use_mlock),
                verbose=kwargs.get("verbose", self.config.verbose),
                seed=kwargs.get("seed", self.config.seed),
            )

            self._model_path = model_path
            logger.info(f"Model loaded successfully: {model_path.name}")

    async def unload(self) -> None:
        """Unload the model from memory."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._unload_sync)

    def _unload_sync(self) -> None:
        """Synchronous model unloading."""
        with self._lock:
            if self._model:
                del self._model
                self._model = None
                self._model_path = None

                import gc
                gc.collect()

                logger.info("Model unloaded")

    async def validate(self) -> bool:
        """Validate the model by running a simple generation."""
        if not self._model:
            return False

        try:
            result = await self.generate("Hello", max_tokens=5)
            return len(result) > 0
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Generate text from a prompt."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._generate_sync,
            prompt,
            max_tokens,
            temperature,
            top_p,
            stop,
            kwargs,
        )

    def _generate_sync(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]],
        kwargs: Dict[str, Any],
    ) -> str:
        """Synchronous generation."""
        with self._lock:
            if not self._model:
                raise RuntimeError("Model not loaded")

            # Merge stop tokens
            stop_tokens = list(stop or []) + self.config.stop_tokens

            output = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_tokens,
                echo=False,
            )

            self._generation_count += 1
            self._total_tokens += output["usage"]["completion_tokens"]

            return output["choices"][0]["text"].strip()

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generated text token by token."""
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()

        def stream_sync():
            with self._lock:
                if not self._model:
                    queue.put_nowait(StopIteration)
                    return

                stop_tokens = list(stop or []) + self.config.stop_tokens

                for output in self._model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop_tokens,
                    echo=False,
                    stream=True,
                ):
                    token = output["choices"][0]["text"]
                    asyncio.run_coroutine_threadsafe(
                        queue.put(token),
                        loop
                    )

                asyncio.run_coroutine_threadsafe(
                    queue.put(StopIteration),
                    loop
                )

        # Start streaming in background
        self._executor.submit(stream_sync)

        # Yield tokens as they arrive
        while True:
            token = await queue.get()
            if token is StopIteration:
                break
            yield token

    async def chat(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate a chat completion from messages."""
        prompt = self.format_messages(messages)
        return await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def format_messages(self, messages: List[ChatMessage]) -> str:
        """Format chat messages using the configured template."""
        formatted_parts = []

        for msg in messages:
            template = self._chat_template.get(msg.role)
            if template:
                formatted_parts.append(template.format(content=msg.content))
            else:
                # Fallback for unknown roles
                formatted_parts.append(f"{msg.role.title()}: {msg.content}\n")

        # Add generation prefix
        formatted_parts.append(self._chat_template.get("generation_prefix", ""))

        return "".join(formatted_parts)

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            "loaded": self.is_loaded(),
            "model_path": str(self._model_path) if self._model_path else None,
            "generation_count": self._generation_count,
            "total_tokens_generated": self._total_tokens,
            "avg_tokens_per_generation": (
                self._total_tokens / self._generation_count
                if self._generation_count > 0 else 0
            ),
            "config": {
                "n_ctx": self.config.n_ctx,
                "n_threads": self.config.n_threads,
                "n_gpu_layers": self.config.n_gpu_layers,
                "chat_template": self.config.chat_template,
            },
        }

    async def close(self) -> None:
        """Clean up resources."""
        await self.unload()
        self._executor.shutdown(wait=True)


class LlamaCppModelLoader:
    """
    Model loader for HotSwapManager integration.

    Creates and manages LlamaCppExecutor instances for hot-swapping.
    """

    def __init__(self, config: Optional[LlamaCppConfig] = None):
        self.config = config or LlamaCppConfig()

    async def load(self, model_path: Path, **kwargs) -> LlamaCppExecutor:
        """Load a model and return the executor."""
        executor = LlamaCppExecutor(self.config)
        await executor.load(model_path, **kwargs)
        return executor

    async def unload(self, executor: LlamaCppExecutor) -> None:
        """Unload a model."""
        await executor.unload()

    async def validate(self, executor: LlamaCppExecutor) -> bool:
        """Validate a loaded model."""
        return await executor.validate()
