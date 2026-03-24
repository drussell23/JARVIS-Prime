"""
DoublewordModelProvider — async batch inference via Doubleword API.

Implements the ModelProvider protocol for the Qwen3.5-397B-A17B-FP8 model
via Doubleword's OpenAI-compatible batch API (https://api.doubleword.ai/v1).

Two modes:
  - **Sync-ish (default):** Submit batch, poll until complete, return result.
    Suitable for governance pipeline tasks where we need the answer before continuing.
  - **Fire-and-forget:** Submit batch, return batch_id. Caller retrieves later.
    Suitable for DPO scoring in Reactor-Core.

The batch API is async by nature (1h SLA default), so this provider is best
for latency-insensitive tasks: code review, complex reasoning, judge scoring.
For real-time interactive tasks, use Claude or local models.

Environment variables
---------------------
DOUBLEWORD_API_KEY        str   (required) — Bearer token
DOUBLEWORD_BASE_URL       str   (default https://api.doubleword.ai/v1)
DOUBLEWORD_MODEL          str   (default Qwen/Qwen3.5-397B-A17B-FP8)
DOUBLEWORD_WINDOW         str   (default 1h) — batch completion window
DOUBLEWORD_POLL_INTERVAL  int   (default 10) — seconds between status polls
DOUBLEWORD_MAX_WAIT       int   (default 3600) — max seconds to wait for batch
DOUBLEWORD_MAX_TOKENS     int   (default 4096) — default max output tokens
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (all env-var driven)
# ---------------------------------------------------------------------------
_API_KEY = os.environ.get("DOUBLEWORD_API_KEY", "")
_BASE_URL = os.environ.get("DOUBLEWORD_BASE_URL", "https://api.doubleword.ai/v1")
_MODEL = os.environ.get("DOUBLEWORD_MODEL", "Qwen/Qwen3.5-397B-A17B-FP8")
_WINDOW = os.environ.get("DOUBLEWORD_WINDOW", "1h")
_POLL_INTERVAL = int(os.environ.get("DOUBLEWORD_POLL_INTERVAL", "10"))
_MAX_WAIT = int(os.environ.get("DOUBLEWORD_MAX_WAIT", "3600"))
_DEFAULT_MAX_TOKENS = int(os.environ.get("DOUBLEWORD_MAX_TOKENS", "4096"))

# Pricing (March 2026)
_INPUT_COST_PER_M = float(os.environ.get("DOUBLEWORD_INPUT_COST_PER_M", "0.10"))
_OUTPUT_COST_PER_M = float(os.environ.get("DOUBLEWORD_OUTPUT_COST_PER_M", "0.40"))


class DoublewordModelProvider:
    """ModelProvider implementation for Doubleword batch API.

    Satisfies the ModelProvider protocol used by J-Prime's reasoning graph.
    Submits a single-request batch, polls to completion, returns the result.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or _API_KEY
        self._model = model or _MODEL
        self._base_url = base_url or _BASE_URL
        self._session = None  # lazy aiohttp session
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost = 0.0
        self._request_count = 0

    # ------------------------------------------------------------------
    # ModelProvider protocol
    # ------------------------------------------------------------------

    async def infer(
        self,
        messages: List[Dict],
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = 0.2,
        timeout_s: float = 3600.0,
    ) -> Dict:
        """Submit a batch request and wait for completion.

        Returns a dict with at least 'content' (the model's response text).
        Also includes 'reasoning_content' if the model produced chain-of-thought.
        """
        if not self._api_key:
            raise RuntimeError("DOUBLEWORD_API_KEY not set")

        request_id = f"trinity-{uuid.uuid4().hex[:12]}"

        # Build JSONL batch file (single request)
        batch_request = {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        }

        try:
            return await asyncio.wait_for(
                self._submit_and_wait(batch_request, request_id),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            logger.error("[Doubleword] Batch timed out after %.0fs", timeout_s)
            return {"content": "", "error": "Doubleword batch timed out"}

    def is_model_loaded(self) -> bool:
        """Doubleword is always 'loaded' — it's a cloud API."""
        return bool(self._api_key)

    def loaded_model_name(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # Batch API flow
    # ------------------------------------------------------------------

    async def _submit_and_wait(
        self, batch_request: Dict, request_id: str,
    ) -> Dict:
        """Upload JSONL → create batch → poll → retrieve result."""
        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # 1. Upload JSONL file
        jsonl_content = json.dumps(batch_request)

        # Use multipart upload for the file
        import aiohttp
        form = aiohttp.FormData()
        form.add_field(
            "file",
            jsonl_content.encode("utf-8"),
            filename=f"{request_id}.jsonl",
            content_type="application/jsonl",
        )
        form.add_field("purpose", "batch")

        async with session.post(
            f"{self._base_url}/files",
            headers={"Authorization": f"Bearer {self._api_key}"},
            data=form,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"File upload failed ({resp.status}): {body}")
            file_data = await resp.json()
            file_id = file_data["id"]
            logger.info("[Doubleword] Uploaded batch file: %s", file_id)

        # 2. Create batch
        batch_payload = {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": _WINDOW,
        }
        async with session.post(
            f"{self._base_url}/batches",
            headers=headers,
            json=batch_payload,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Batch creation failed ({resp.status}): {body}")
            batch_data = await resp.json()
            batch_id = batch_data["id"]
            logger.info("[Doubleword] Batch created: %s (window=%s)", batch_id, _WINDOW)

        # 3. Poll until complete
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            if elapsed > _MAX_WAIT:
                raise asyncio.TimeoutError(f"Batch {batch_id} exceeded max wait {_MAX_WAIT}s")

            async with session.get(
                f"{self._base_url}/batches/{batch_id}",
                headers=headers,
            ) as resp:
                status_data = await resp.json()
                status = status_data.get("status", "unknown")

            if status == "completed":
                output_file_id = status_data.get("output_file_id")
                logger.info(
                    "[Doubleword] Batch %s completed in %.1fs",
                    batch_id, time.monotonic() - start,
                )
                break
            elif status in ("failed", "expired", "cancelled"):
                raise RuntimeError(f"Batch {batch_id} {status}: {status_data}")
            else:
                logger.debug("[Doubleword] Batch %s status: %s (%.0fs)", batch_id, status, elapsed)
                await asyncio.sleep(_POLL_INTERVAL)

        # 4. Retrieve output
        async with session.get(
            f"{self._base_url}/files/{output_file_id}/content",
            headers=headers,
        ) as resp:
            output_text = await resp.text()

        # Parse JSONL output (single line for single request)
        for line in output_text.strip().split("\n"):
            if not line.strip():
                continue
            result = json.loads(line)
            if result.get("custom_id") == request_id:
                return self._extract_response(result)

        raise RuntimeError(f"Request {request_id} not found in batch output")

    def _extract_response(self, batch_result: Dict) -> Dict:
        """Extract content from a batch API response line."""
        response = batch_result.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])

        if not choices:
            return {"content": "", "error": "No choices in response"}

        message = choices[0].get("message", {})
        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")

        # Track usage
        usage = body.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

        cost = (input_tokens * _INPUT_COST_PER_M / 1_000_000 +
                output_tokens * _OUTPUT_COST_PER_M / 1_000_000)
        self._total_cost += cost
        self._request_count += 1

        logger.info(
            "[Doubleword] Response: %d input + %d output tokens ($%.6f). "
            "Session total: $%.6f across %d requests",
            input_tokens, output_tokens, cost,
            self._total_cost, self._request_count,
        )

        return {
            "content": content,
            "reasoning_content": reasoning,
            "model": self._model,
            "source": "doubleword",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "finish_reason": choices[0].get("finish_reason", "unknown"),
        }

    # ------------------------------------------------------------------
    # Fire-and-forget mode (for DPO scoring)
    # ------------------------------------------------------------------

    async def submit_batch(
        self,
        requests: List[Dict],
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = 0.2,
    ) -> str:
        """Submit multiple requests as a batch, return batch_id.

        Caller retrieves results later via retrieve_batch(batch_id).
        Use this for DPO scoring and other async workloads.
        """
        if not self._api_key:
            raise RuntimeError("DOUBLEWORD_API_KEY not set")

        session = await self._get_session()
        headers = {"Authorization": f"Bearer {self._api_key}"}

        # Build JSONL
        lines = []
        for i, req in enumerate(requests):
            lines.append(json.dumps({
                "custom_id": f"dpo-{uuid.uuid4().hex[:8]}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self._model,
                    "messages": req.get("messages", []),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            }))

        jsonl_content = "\n".join(lines)

        import aiohttp
        form = aiohttp.FormData()
        form.add_field(
            "file",
            jsonl_content.encode("utf-8"),
            filename=f"batch-{uuid.uuid4().hex[:8]}.jsonl",
            content_type="application/jsonl",
        )
        form.add_field("purpose", "batch")

        async with session.post(f"{self._base_url}/files", headers=headers, data=form) as resp:
            file_data = await resp.json()
            file_id = file_data["id"]

        async with session.post(
            f"{self._base_url}/batches",
            headers={**headers, "Content-Type": "application/json"},
            json={"input_file_id": file_id, "endpoint": "/v1/chat/completions", "completion_window": _WINDOW},
        ) as resp:
            batch_data = await resp.json()

        batch_id = batch_data["id"]
        logger.info("[Doubleword] Batch submitted: %s (%d requests)", batch_id, len(requests))
        return batch_id

    async def retrieve_batch(self, batch_id: str) -> List[Dict]:
        """Retrieve results for a previously submitted batch."""
        session = await self._get_session()
        headers = {"Authorization": f"Bearer {self._api_key}"}

        async with session.get(f"{self._base_url}/batches/{batch_id}", headers=headers) as resp:
            status_data = await resp.json()

        if status_data.get("status") != "completed":
            return [{"status": status_data.get("status"), "batch_id": batch_id}]

        output_file_id = status_data.get("output_file_id")
        async with session.get(f"{self._base_url}/files/{output_file_id}/content", headers=headers) as resp:
            output_text = await resp.text()

        results = []
        for line in output_text.strip().split("\n"):
            if line.strip():
                results.append(self._extract_response(json.loads(line)))
        return results

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _get_session(self):
        """Lazy-init aiohttp session."""
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cost_usd": round(self._total_cost, 6),
            "request_count": self._request_count,
            "model": self._model,
        }
