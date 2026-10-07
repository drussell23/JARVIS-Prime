"""J-Prime Engine HTTP surface.

Two faces over one engine pool:

* **OpenAI** -- ``/v1/chat/completions``, ``/v1/completions``, ``/v1/models``:
  passed through to the resident ``llama-server`` (after lifting Ollama-style
  ``options``/``keep_alive`` and applying the model's published defaults).
* **Model management** (the Ollama-compatible routes O+V probes) --
  ``/api/tags``, ``/api/ps``, ``/api/show``, ``/api/chat``, ``/api/generate``
  (``keep_alive: 0`` unloads), ``/api/version``.

Every generation holds an in-flight reference for the whole stream, so a
model can never be evicted or keep-alive-expired mid-response, and a client
disconnect closes the upstream request (llama-server then stops decoding).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from . import __version__, gguf_meta, gpu, protocol
from .engine_pool import (EnginePool, EngineStartError, InsufficientVram, ModelNotFound,
                          find_engine_binary)
from .model_store import ModelSpec, canonical_name

logger = logging.getLogger(__name__)

_FOREVER = "2318-01-01T00:00:00Z"  # how Ollama reports keep_alive -1


def _err(status: int, msg: str) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


def _details(spec: ModelSpec) -> Dict[str, Any]:
    meta = gguf_meta.read_metadata(spec.model_path)
    arch = meta.get("general.architecture", "")
    return {"format": "gguf", "family": arch, "families": [arch] if arch else [],
            "parameter_size": meta.get("general.size_label", ""),
            "quantization_level": str(meta.get("general.file_type", "")),
            "adapters": len(spec.adapter_paths)}


def create_app(pool: Optional[EnginePool] = None) -> FastAPI:
    pool = pool or EnginePool()
    default_model = os.environ.get("JPRIME_ENGINE_DEFAULT_MODEL", "").strip()
    preload = os.environ.get("JPRIME_ENGINE_PRELOAD", "").strip()
    state: Dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        pool.start()
        state["client"] = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))
        if preload:
            async def _warm() -> None:
                try:
                    async with pool.acquire(preload, keep_alive=-1):
                        pass
                    logger.info("[Engine] preloaded %s (resident until evicted)", preload)
                except Exception as exc:  # noqa: BLE001 -- a failed preload must not stop the server
                    logger.error("[Engine] preload of %s failed: %s", preload, exc)
            state["warm"] = asyncio.create_task(_warm())
        try:
            yield
        finally:
            await state["client"].aclose()
            await pool.shutdown()

    app = FastAPI(title="J-Prime Engine", version=__version__, lifespan=lifespan)

    def _model_of(body: Dict[str, Any]) -> str:
        name = str(body.get("model") or body.get("name") or "").strip()
        if not name or name == "jarvis-prime":
            name = default_model
        return name

    def _spec_or_404(name: str) -> ModelSpec:
        spec = pool.store.resolve(name) if name else None
        if spec is None:
            raise ModelNotFound(f"model '{name}' not found")
        return spec

    @app.exception_handler(ModelNotFound)
    async def _nf(_r: Request, exc: ModelNotFound) -> JSONResponse:
        return _err(404, str(exc))

    @app.exception_handler(InsufficientVram)
    async def _iv(_r: Request, exc: InsufficientVram) -> JSONResponse:
        return _err(503, str(exc))

    @app.exception_handler(EngineStartError)
    async def _es(_r: Request, exc: EngineStartError) -> JSONResponse:
        return _err(500, str(exc))

    # ------------------------------------------------------------ discovery
    @app.get("/")
    async def root() -> PlainTextResponse:
        return PlainTextResponse("J-Prime Engine is running")

    @app.get("/api/version")
    async def version() -> Dict[str, str]:
        return {"version": f"jprime-engine-{__version__}"}

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        mem = await asyncio.to_thread(gpu.query)
        binary = find_engine_binary()
        return {
            "service": "jarvis_prime", "component": "engine", "version": __version__,
            "status": "healthy" if binary else "degraded",
            "engine": "llama.cpp", "engine_binary": str(binary) if binary else None,
            "default_model": default_model or None,
            "resident": [{"name": e.spec.name, "ctx": e.ctx, "size_vram": e.size_vram_bytes,
                          "inflight": e.inflight} for e in pool.resident()],
            "gpu": {"name": mem.name, "total_mib": mem.total_mib, "free_mib": mem.free_mib} if mem else None,
        }

    @app.get("/api/tags")
    async def tags() -> Dict[str, Any]:
        specs = await asyncio.to_thread(pool.store.all_specs)
        return {"models": [{"name": s.name, "model": s.name, "modified_at": s.modified_at,
                            "size": s.size_bytes, "digest": s.digest, "details": _details(s)}
                           for s in specs]}

    @app.get("/v1/models")
    async def v1_models() -> Dict[str, Any]:
        specs = await asyncio.to_thread(pool.store.all_specs)
        resident = {e.spec.name for e in pool.resident()}
        return {"object": "list", "data": [
            {"id": s.name, "object": "model", "owned_by": "jarvis-prime", "size": s.size_bytes,
             "loaded": s.name in resident, "vision": s.has_vision} for s in specs]}

    @app.get("/api/ps")
    async def ps() -> Dict[str, Any]:
        out = []
        for e in pool.resident():
            exp = (datetime.fromtimestamp(e.expires_wall, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                   if e.expires_wall else _FOREVER)
            out.append({"name": e.spec.name, "model": e.spec.name, "size": e.spec.size_bytes,
                        "size_vram": e.size_vram_bytes, "digest": e.spec.digest,
                        "details": _details(e.spec), "expires_at": exp, "context_length": e.ctx})
        return {"models": out}

    @app.post("/api/show")
    async def show(request: Request) -> Dict[str, Any]:
        body = await request.json()
        spec = _spec_or_404(_model_of(body))
        meta = gguf_meta.read_metadata(spec.model_path)
        caps = ["completion", "tools"] + (["vision"] if spec.has_vision else [])
        params = "\n".join(f"{k} {v}" for k, v in spec.defaults.items() if not isinstance(v, list))
        return {"modelfile": "", "parameters": params, "template": "", "details": _details(spec),
                "model_info": meta, "capabilities": caps, "modified_at": spec.modified_at}

    @app.post("/v1/models/{name:path}/unload")
    async def unload(name: str) -> Dict[str, Any]:
        return {"model": canonical_name(name), "unloaded": await pool.unload(name)}

    # ------------------------------------------------------------ generation
    async def _upstream_json(name: str, num_ctx: Optional[int], keep_alive: Any,
                             path: str, body: Dict[str, Any]) -> httpx.Response:
        async with pool.acquire(name, num_ctx, keep_alive) as eng:
            return await state["client"].post(eng.base_url + path, json=body)

    async def _upstream_stream(name: str, num_ctx: Optional[int], keep_alive: Any, path: str,
                               body: Dict[str, Any],
                               translator: Optional[protocol.OllamaStreamTranslator]) -> AsyncIterator[bytes]:
        async with pool.acquire(name, num_ctx, keep_alive) as eng:
            async with state["client"].stream("POST", eng.base_url + path, json=body) as r:
                if r.status_code != 200:
                    detail = (await r.aread()).decode(errors="replace")[:2000]
                    if translator is None:
                        yield f"data: {{\"error\": {detail!r}}}\n\n".encode()
                    else:
                        yield (protocol.json.dumps({"error": detail}) + "\n").encode()
                    return
                if translator is None:
                    async for chunk in r.aiter_raw():
                        yield chunk
                    return
                buf = b""
                async for chunk in r.aiter_raw():
                    buf += chunk
                    events, buf = protocol.parse_sse(buf)
                    for ev in events:
                        if ev is None:
                            continue
                        for line in translator.feed(ev):
                            yield line
                yield translator.final()

    def _passthrough(r: httpx.Response) -> Response:
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        raw = await request.json()
        spec = _spec_or_404(_model_of(raw))
        body, num_ctx, keep_alive = protocol.prepare_openai_chat(raw, spec.defaults)
        body["model"] = spec.name
        await pool.ensure(spec.name, num_ctx)  # surface 404/503/500 before streaming starts
        if body.get("stream"):
            return StreamingResponse(
                _upstream_stream(spec.name, num_ctx, keep_alive, "/v1/chat/completions", body, None),
                media_type="text/event-stream")
        return _passthrough(await _upstream_json(spec.name, num_ctx, keep_alive, "/v1/chat/completions", body))

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        raw = await request.json()
        spec = _spec_or_404(_model_of(raw))
        body, num_ctx, keep_alive = protocol.prepare_openai_chat(raw, spec.defaults)
        body["model"] = spec.name
        await pool.ensure(spec.name, num_ctx)
        if body.get("stream"):
            return StreamingResponse(
                _upstream_stream(spec.name, num_ctx, keep_alive, "/v1/completions", body, None),
                media_type="text/event-stream")
        return _passthrough(await _upstream_json(spec.name, num_ctx, keep_alive, "/v1/completions", body))

    @app.post("/api/chat")
    async def api_chat(request: Request) -> Response:
        raw = await request.json()
        spec = _spec_or_404(_model_of(raw))
        body, num_ctx, keep_alive, stream = protocol.prepare_ollama_chat(raw, spec.defaults)
        body["model"] = spec.name
        if not body["messages"]:  # Ollama: an empty chat just loads the model
            async with pool.acquire(spec.name, num_ctx, keep_alive):
                pass
            return JSONResponse({"model": spec.name, "created_at": protocol._now_iso(),
                                 "message": {"role": "assistant", "content": ""},
                                 "done": True, "done_reason": "load"})
        await pool.ensure(spec.name, num_ctx)
        if stream:
            tr = protocol.OllamaStreamTranslator(spec.name, "chat")
            return StreamingResponse(
                _upstream_stream(spec.name, num_ctx, keep_alive, "/v1/chat/completions", body, tr),
                media_type="application/x-ndjson")
        t0 = time.monotonic()
        r = await _upstream_json(spec.name, num_ctx, keep_alive, "/v1/chat/completions", body)
        if r.status_code != 200:
            return _err(r.status_code, r.text[:2000])
        return JSONResponse(protocol.ollama_chat_response(spec.name, r.json(), time.monotonic() - t0))

    @app.post("/api/generate")
    async def api_generate(request: Request) -> Response:
        raw = await request.json()
        name = _model_of(raw)
        ttl = raw.get("keep_alive")
        if not raw.get("prompt"):
            # Ollama: an empty generate loads the model, or unloads it with keep_alive 0.
            if ttl is not None and str(ttl).strip() in ("0", "0s", "0m"):
                await pool.unload(name)
                return JSONResponse({"model": canonical_name(name), "created_at": protocol._now_iso(),
                                     "response": "", "done": True, "done_reason": "unload"})
            spec = _spec_or_404(name)
            async with pool.acquire(spec.name, None, ttl):
                pass
            return JSONResponse({"model": spec.name, "created_at": protocol._now_iso(), "response": "",
                                 "done": True, "done_reason": "load"})
        spec = _spec_or_404(name)
        body, num_ctx, keep_alive, stream = protocol.prepare_ollama_generate(raw, spec.defaults)
        body["model"] = spec.name
        await pool.ensure(spec.name, num_ctx)
        if stream:
            body["stream_options"] = {"include_usage": True}
            tr = protocol.OllamaStreamTranslator(spec.name, "generate")
            return StreamingResponse(
                _upstream_stream(spec.name, num_ctx, keep_alive, "/v1/completions", body, tr),
                media_type="application/x-ndjson")
        t0 = time.monotonic()
        r = await _upstream_json(spec.name, num_ctx, keep_alive, "/v1/completions", body)
        if r.status_code != 200:
            return _err(r.status_code, r.text[:2000])
        return JSONResponse(protocol.ollama_generate_response(spec.name, r.json(), time.monotonic() - t0))

    return app
