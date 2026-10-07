"""Protocol translation -- the wire shapes O+V speaks, mapped onto llama-server.

``llama-server`` speaks OpenAI (``/v1/chat/completions``, ``/v1/completions``)
natively, including ``response_format`` json_schema, ``tools``, seeded
sampling, ``stream_options.include_usage`` and LoRA. O+V's local lane also
speaks the Ollama shape (``/api/chat`` NDJSON with ``options``/``format``/
``keep_alive``) -- its default "native" transport. This module is the only
place that knows both, so the server stays a thin router.

Rules that matter:

* A model's published defaults (its params layer: temperature, top_k,
  top_p, repeat_penalty, stop) apply when the request does not set them --
  exactly what Ollama does with a Modelfile. Dropping them would silently
  change sampling for every O+V request.
* ``num_ctx`` and ``keep_alive`` are lifecycle, not sampling: they are
  returned to the caller (the engine pool) and never forwarded.
* Unknown fields are not invented into the request; known-dead ones
  (``think``, ``draft_num_predict``) are dropped with intent, not by accident.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Ollama option name -> OpenAI/llama-server field name.
_OPTION_MAP = {
    "temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
    "typical_p": "typical_p", "repeat_penalty": "repeat_penalty", "repeat_last_n": "repeat_last_n",
    "presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty",
    "seed": "seed", "stop": "stop", "mirostat": "mirostat", "mirostat_tau": "mirostat_tau",
    "mirostat_eta": "mirostat_eta",
}
_LIFECYCLE_OPTIONS = {"num_ctx", "num_gpu", "num_thread", "num_batch", "use_mmap", "use_mlock",
                      "numa", "low_vram", "main_gpu", "draft_num_predict", "num_keep"}
_DEAD_FIELDS = ("think", "keep_alive", "options", "format")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _apply_options(out: Dict[str, Any], options: Dict[str, Any]) -> Optional[int]:
    """Lift Ollama ``options`` into top-level fields; return requested num_ctx."""
    num_ctx = None
    for k, v in (options or {}).items():
        if k == "num_ctx":
            try:
                num_ctx = int(v)
            except (TypeError, ValueError):
                pass
        elif k == "num_predict":
            if isinstance(v, (int, float)) and v > 0 and "max_tokens" not in out:
                out["max_tokens"] = int(v)
        elif k in _OPTION_MAP and _OPTION_MAP[k] not in out:
            out[_OPTION_MAP[k]] = v
        # lifecycle / unknown options are deliberately not forwarded
    return num_ctx


def _apply_defaults(out: Dict[str, Any], defaults: Dict[str, Any]) -> None:
    for k, v in (defaults or {}).items():
        field = _OPTION_MAP.get(k)
        if field and field not in out:
            out[field] = v


def _response_format(fmt: Any) -> Optional[Dict[str, Any]]:
    if not fmt:
        return None
    if fmt == "json":
        return {"type": "json_object"}
    if isinstance(fmt, dict):
        return {"type": "json_schema", "json_schema": {"name": "response", "schema": fmt, "strict": True}}
    return None


def prepare_openai_chat(body: Dict[str, Any], defaults: Dict[str, Any]
                        ) -> Tuple[Dict[str, Any], Optional[int], Any]:
    """An OpenAI-shaped request as O+V sends it (may carry ``options``/``keep_alive``)."""
    out = {k: v for k, v in body.items() if k not in _DEAD_FIELDS}
    num_ctx = _apply_options(out, body.get("options") or {})
    if "response_format" not in out and body.get("format"):
        rf = _response_format(body.get("format"))
        if rf:
            out["response_format"] = rf
    _apply_defaults(out, defaults)
    if out.get("stream") and "stream_options" not in out:
        out["stream_options"] = {"include_usage": True}
    return out, num_ctx, body.get("keep_alive")


def _convert_message(m: Dict[str, Any]) -> Dict[str, Any]:
    msg: Dict[str, Any] = {"role": m.get("role", "user")}
    content = m.get("content", "")
    images = m.get("images") or []
    if images:
        parts: List[Dict[str, Any]] = [{"type": "text", "text": content or ""}]
        for b64 in images:
            url = b64 if str(b64).startswith("data:") else f"data:image/png;base64,{b64}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        msg["content"] = parts
    else:
        msg["content"] = content
    if m.get("tool_calls"):
        calls = []
        for i, tc in enumerate(m["tool_calls"]):
            fn = tc.get("function") or {}
            args = fn.get("arguments", {})
            calls.append({"id": tc.get("id") or f"call_{i}", "type": "function",
                          "function": {"name": fn.get("name", ""),
                                       "arguments": args if isinstance(args, str) else json.dumps(args)}})
        msg["tool_calls"] = calls
    if m.get("role") == "tool":
        if m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        if m.get("tool_name"):
            msg["name"] = m["tool_name"]
    return msg


def prepare_ollama_chat(body: Dict[str, Any], defaults: Dict[str, Any]
                        ) -> Tuple[Dict[str, Any], Optional[int], Any, bool]:
    """``/api/chat`` request -> OpenAI chat body. Ollama streams by default."""
    stream = bool(body.get("stream", True))
    out: Dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": [_convert_message(m) for m in body.get("messages") or []],
        "stream": stream,
    }
    if body.get("tools"):
        out["tools"] = body["tools"]
    rf = _response_format(body.get("format"))
    if rf:
        out["response_format"] = rf
    num_ctx = _apply_options(out, body.get("options") or {})
    _apply_defaults(out, defaults)
    if stream:
        out["stream_options"] = {"include_usage": True}
    return out, num_ctx, body.get("keep_alive"), stream


def prepare_ollama_generate(body: Dict[str, Any], defaults: Dict[str, Any]
                            ) -> Tuple[Dict[str, Any], Optional[int], Any, bool]:
    """``/api/generate`` with a prompt -> OpenAI completion body."""
    stream = bool(body.get("stream", True))
    prompt = body.get("prompt", "")
    if body.get("system"):
        prompt = f"{body['system']}\n\n{prompt}"
    out: Dict[str, Any] = {"model": body.get("model", ""), "prompt": prompt, "stream": stream}
    rf = _response_format(body.get("format"))
    if rf:
        out["response_format"] = rf
    num_ctx = _apply_options(out, body.get("options") or {})
    _apply_defaults(out, defaults)
    return out, num_ctx, body.get("keep_alive"), stream


# ----------------------------------------------------------------- responses
def _ollama_tool_calls(calls: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for c in calls or []:
        fn = c.get("function") or {}
        args = fn.get("arguments", "{}")
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except ValueError:
            args = {"_raw": args}
        out.append({"function": {"name": fn.get("name", ""), "arguments": args}})
    return out


def _durations(usage: Dict[str, Any], timings: Dict[str, Any], wall_s: float) -> Dict[str, Any]:
    ns = 1_000_000
    return {
        "total_duration": int(wall_s * 1e9),
        "load_duration": 0,
        "prompt_eval_count": int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0),
        "prompt_eval_duration": int(float(timings.get("prompt_ms") or 0) * ns),
        "eval_count": int(usage.get("completion_tokens") or timings.get("predicted_n") or 0),
        "eval_duration": int(float(timings.get("predicted_ms") or 0) * ns),
    }


def ollama_chat_response(model: str, resp: Dict[str, Any], wall_s: float) -> Dict[str, Any]:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    message: Dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        message["tool_calls"] = _ollama_tool_calls(msg["tool_calls"])
    return {"model": model, "created_at": _now_iso(), "message": message, "done": True,
            "done_reason": choice.get("finish_reason") or "stop",
            **_durations(resp.get("usage") or {}, resp.get("timings") or {}, wall_s)}


def ollama_generate_response(model: str, resp: Dict[str, Any], wall_s: float) -> Dict[str, Any]:
    choice = (resp.get("choices") or [{}])[0]
    return {"model": model, "created_at": _now_iso(), "response": choice.get("text") or "",
            "done": True, "done_reason": choice.get("finish_reason") or "stop",
            **_durations(resp.get("usage") or {}, resp.get("timings") or {}, wall_s)}


class OllamaStreamTranslator:
    """OpenAI SSE frames in, Ollama NDJSON lines out (chat or generate)."""

    def __init__(self, model: str, kind: str = "chat") -> None:
        self.model, self.kind = model, kind
        self.t0 = time.monotonic()
        self.usage: Dict[str, Any] = {}
        self.timings: Dict[str, Any] = {}
        self.finish: Optional[str] = None
        self._tool_acc: Dict[int, Dict[str, Any]] = {}

    def _line(self, obj: Dict[str, Any]) -> bytes:
        return (json.dumps(obj, separators=(",", ":")) + "\n").encode()

    def feed(self, frame: Dict[str, Any]) -> List[bytes]:
        out: List[bytes] = []
        if frame.get("usage"):
            self.usage = frame["usage"]
        if frame.get("timings"):
            self.timings = frame["timings"]
        for ch in frame.get("choices") or []:
            if ch.get("finish_reason"):
                self.finish = ch["finish_reason"]
            if self.kind == "generate":
                piece = ch.get("text")
                if piece:
                    out.append(self._line({"model": self.model, "created_at": _now_iso(),
                                           "response": piece, "done": False}))
                continue
            delta = ch.get("delta") or {}
            for tc in delta.get("tool_calls") or []:
                acc = self._tool_acc.setdefault(int(tc.get("index", 0)), {"function": {"name": "", "arguments": ""}})
                fn = tc.get("function") or {}
                acc["function"]["name"] += fn.get("name") or ""
                acc["function"]["arguments"] += fn.get("arguments") or ""
            piece = delta.get("content")
            if piece:
                out.append(self._line({"model": self.model, "created_at": _now_iso(),
                                       "message": {"role": "assistant", "content": piece}, "done": False}))
        return out

    def final(self) -> bytes:
        wall = time.monotonic() - self.t0
        body: Dict[str, Any] = {"model": self.model, "created_at": _now_iso(), "done": True,
                                "done_reason": self.finish or "stop",
                                **_durations(self.usage, self.timings, wall)}
        if self.kind == "generate":
            body["response"] = ""
        else:
            msg: Dict[str, Any] = {"role": "assistant", "content": ""}
            if self._tool_acc:
                msg["tool_calls"] = _ollama_tool_calls(self._tool_acc[i] for i in sorted(self._tool_acc))
            body["message"] = msg
        return self._line(body)


def parse_sse(buffer: bytes) -> Tuple[List[Optional[Dict[str, Any]]], bytes]:
    """Split complete SSE events out of ``buffer``. ``None`` marks ``[DONE]``."""
    events: List[Optional[Dict[str, Any]]] = []
    while b"\n\n" in buffer:
        raw, buffer = buffer.split(b"\n\n", 1)
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                events.append(None)
            elif data:
                try:
                    events.append(json.loads(data))
                except ValueError:
                    continue
    return events, buffer
