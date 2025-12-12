"""Tiny Prime semantic security guard.

Fast path (default): next-token scoring for intent labels after `<intent>`.

This is designed for low latency classification on short commands.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from jarvis_prime.tiny_prime.config import TinyPrimeConfig
from jarvis_prime.tiny_prime.presets import TinyPrimePresets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntentResult:
    label: str
    confidence: float
    scores: Dict[str, float]
    latency_ms: float


def _detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TinyPrimeGuard:
    """Semantic intent classifier with optional async auto-batching."""

    def __init__(
        self,
        *,
        model: LlamaForCausalLM,
        tokenizer,
        intent_token_ids: Dict[str, List[int]],
        fmt: Dict[str, str],
        device: str,
        context_window: int,
        async_enabled: bool,
        max_batch_size: int,
        microbatch_wait_ms: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.intent_token_ids = intent_token_ids
        self.fmt = fmt
        self.device = device
        self.context_window = context_window

        self.async_enabled = async_enabled
        self.max_batch_size = max_batch_size
        self.microbatch_wait_ms = microbatch_wait_ms

        self._queue: "asyncio.Queue[Tuple[str, asyncio.Future]]" = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    @classmethod
    def from_config(
        cls,
        cfg: TinyPrimeConfig,
        *,
        strict_single_token_labels: bool = True,
    ) -> "TinyPrimeGuard":
        model_dir = cfg.resolve_path_for("training.output_dir")
        tok_dir = cfg.resolve_path_for("tokenizer.output_dir")

        device = _detect_device(str(cfg.get("inference.device", "auto")))

        tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
        if tokenizer.pad_token is None:
            # keep things robust even if pad was not set
            tokenizer.pad_token = tokenizer.eos_token or "</s>"

        model = LlamaForCausalLM.from_pretrained(model_dir)
        model.eval()
        model = model.to(device)

        labels = cfg.intent_labels()
        intent_token_ids: Dict[str, List[int]] = {}
        for label in labels:
            ids = tokenizer.encode(label, add_special_tokens=False)
            intent_token_ids[label] = ids

        if strict_single_token_labels:
            multi = {k: v for k, v in intent_token_ids.items() if len(v) != 1}
            if multi:
                raise ValueError(
                    "Intent labels must tokenize to a single id for fast scoring. "
                    f"Fix tokenizer.special_tokens or labels. Offenders: {multi}"
                )

        fmt = cfg.format_tokens()
        context_window = int(cfg.get("model.context_window", 512))

        async_enabled = bool(cfg.get("inference.async_enabled", True))
        max_batch_size = int(cfg.get("inference.max_batch_size", 128))
        microbatch_wait_ms = int(cfg.get("inference.microbatch_wait_ms", 5))

        return cls(
            model=model,
            tokenizer=tokenizer,
            intent_token_ids=intent_token_ids,
            fmt=fmt,
            device=device,
            context_window=context_window,
            async_enabled=async_enabled,
            max_batch_size=max_batch_size,
            microbatch_wait_ms=microbatch_wait_ms,
        )

    def _build_prompt(self, text: str) -> str:
        return (
            f"{self.fmt['user_open']} {text} {self.fmt['user_close']} "
            f"{self.fmt['intent_open']} "
        )

    @torch.no_grad()
    def _score_batch_next_token(self, prompts: Sequence[str]) -> List[IntentResult]:
        t0 = time.perf_counter()

        enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.context_window,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.model(**enc)
        logits = out.logits  # [B, T, V]

        # For each sequence, find last non-pad token index
        attn = enc.get("attention_mask")
        if attn is None:
            last_idx = torch.full((logits.shape[0],), logits.shape[1] - 1, device=logits.device)
        else:
            last_idx = attn.sum(dim=1) - 1

        results: List[IntentResult] = []
        labels = list(self.intent_token_ids.keys())

        # Fast single-token path
        label_ids = [self.intent_token_ids[l][0] for l in labels]
        label_ids_t = torch.tensor(label_ids, device=logits.device, dtype=torch.long)

        for i in range(logits.shape[0]):
            li = int(last_idx[i].item())
            next_logits = logits[i, li, :]
            sel = next_logits.index_select(0, label_ids_t)
            probs = torch.softmax(sel, dim=0)

            best_j = int(torch.argmax(probs).item())
            best_label = labels[best_j]

            scores = {labels[j]: float(probs[j].item()) for j in range(len(labels))}
            latency_ms = (time.perf_counter() - t0) * 1000.0
            results.append(
                IntentResult(
                    label=best_label,
                    confidence=float(scores[best_label]),
                    scores=scores,
                    latency_ms=latency_ms,
                )
            )

        return results

    def check(self, text: str) -> IntentResult:
        prompt = self._build_prompt(text)
        return self._score_batch_next_token([prompt])[0]

    async def _ensure_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def check_async(self, text: str) -> IntentResult:
        if not self.async_enabled:
            # Run sync scoring in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.check(text))

        await self._ensure_worker()

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put((text, fut))
        return await fut

    async def _worker_loop(self) -> None:
        while True:
            # First item blocks
            text, fut = await self._queue.get()
            batch: List[Tuple[str, asyncio.Future]] = [(text, fut)]

            # Micro-batch window
            deadline = time.perf_counter() + (self.microbatch_wait_ms / 1000.0)
            while len(batch) < self.max_batch_size:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                batch.append(item)

            prompts = [self._build_prompt(t) for (t, _) in batch]

            try:
                # Do the scoring in a thread to avoid blocking the loop on torch work
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, lambda: self._score_batch_next_token(prompts))

                for (_, bfut), res in zip(batch, results):
                    if not bfut.cancelled():
                        bfut.set_result(res)
            except Exception as e:
                for _, bfut in batch:
                    if not bfut.cancelled():
                        bfut.set_exception(e)


_GLOBAL_GUARD: Optional[TinyPrimeGuard] = None


def get_semantic_guard(cfg: Optional[TinyPrimeConfig] = None, *, cfg_path: Optional[str] = None) -> TinyPrimeGuard:
    """Singleton-style guard accessor for easy pipeline injection."""
    global _GLOBAL_GUARD
    if _GLOBAL_GUARD is not None:
        return _GLOBAL_GUARD

    if cfg is None:
        if cfg_path is not None:
            p = Path(cfg_path)
            if p.suffix in {".yaml", ".yml"}:
                cfg = TinyPrimeConfig.from_yaml(str(p))
            elif p.suffix == ".json":
                cfg = TinyPrimeConfig.from_json(str(p))
            else:
                raise ValueError("cfg_path must be .yaml/.yml/.json")

            cfg = TinyPrimeConfig.from_env(base=cfg)
        else:
            cfg = TinyPrimePresets.tiny_prime_v0_1_0()

    _GLOBAL_GUARD = TinyPrimeGuard.from_config(cfg)
    return _GLOBAL_GUARD


def check_semantic_security(text: str, *, cfg_path: Optional[str] = None) -> IntentResult:
    """Drop-in sync API for the VBI pipeline."""
    guard = get_semantic_guard(cfg_path=cfg_path)
    return guard.check(text)


async def check_semantic_security_async(text: str, *, cfg_path: Optional[str] = None) -> IntentResult:
    """Drop-in async API for the VBI pipeline (auto-batched)."""
    guard = get_semantic_guard(cfg_path=cfg_path)
    return await guard.check_async(text)
