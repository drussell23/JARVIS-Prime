"""J-Prime Engine -- store, GGUF physics, protocol translation, pool lifecycle.

No GPU and no llama-server needed: the pool is driven through its launcher,
health and GPU-probe seams with fakes that behave like the real things
(a process that is alive until terminated, VRAM that a load consumes).
"""
from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path

import pytest

from jarvis_prime.engine import gguf_meta, protocol
from jarvis_prime.engine.engine_pool import (EngineConfig, EnginePool, InsufficientVram,
                                             ModelNotFound, parse_keep_alive)
from jarvis_prime.engine.gpu import GpuMemory
from jarvis_prime.engine.model_store import (DeclaredModels, ModelStore, OllamaManifestStore,
                                             canonical_name)


# ------------------------------------------------------------------ fixtures
def _gguf(path: Path, kv: dict) -> Path:
    """Write a minimal GGUF v3 header carrying ``kv`` (plus one skipped array)."""
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    body = b""
    n = 0
    for k, v in kv.items():
        n += 1
        if isinstance(v, str):
            body += s(k) + struct.pack("<I", 8) + s(v)
        else:
            body += s(k) + struct.pack("<I", 4) + struct.pack("<I", v)
    # a string array, as tokenizers carry -- must be skipped, not parsed
    body += s("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", 3)
    body += s("a") + s("bb") + s("ccc")
    n += 1
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", n) + body)
    return path


QWEN_KV = {"general.architecture": "qwen3moe", "qwen3moe.context_length": 262144,
           "qwen3moe.block_count": 48, "qwen3moe.attention.head_count_kv": 4,
           "qwen3moe.attention.key_length": 128, "qwen3moe.attention.value_length": 128,
           "general.size_label": "30B-A3B"}


def _ollama_store(root: Path) -> Path:
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    _gguf(blobs / "sha256-base", QWEN_KV)
    (blobs / "sha256-lora").write_bytes(b"GGUF-lora")
    (blobs / "sha256-params").write_text(json.dumps(
        {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "repeat_penalty": 1.05,
         "stop": ["<|im_end|>"]}))
    for name, layers in {
        "qwen3-coder-ov/30b": [("model", "base"), ("adapter", "lora"), ("params", "params")],
        "qwen3-coder/30b": [("model", "base"), ("params", "params")],
    }.items():
        d = root / "manifests" / "registry.ollama.ai" / "library" / name
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(json.dumps({"config": {"digest": "sha256:cfg"}, "layers": [
            {"mediaType": f"application/vnd.ollama.image.{mt}", "digest": f"sha256:{b}"} for mt, b in layers]}))
    return root


# ---------------------------------------------------------------- model store
def test_canonical_name():
    assert canonical_name("qwen3-coder") == "qwen3-coder:latest"
    assert canonical_name("qwen3-coder-ov:30b") == "qwen3-coder-ov:30b"
    assert canonical_name("hf.co/org/model") == "hf.co/org/model:latest"


def test_ollama_store_resolves_base_adapter_and_defaults(tmp_path):
    store = OllamaManifestStore(_ollama_store(tmp_path / "models"))
    assert store.names() == ["qwen3-coder-ov:30b", "qwen3-coder:30b"]
    spec = store.resolve("qwen3-coder-ov:30b")
    assert spec.model_path.name == "sha256-base"
    assert [p.name for p in spec.adapter_paths] == ["sha256-lora"]
    assert spec.defaults["top_k"] == 20
    assert spec.size_bytes == spec.model_path.stat().st_size + len(b"GGUF-lora")
    assert store.resolve("qwen3-coder:30b").adapter_paths == ()
    assert store.resolve("nope:1b") is None


def test_declared_models_win_over_ollama(tmp_path):
    root = _ollama_store(tmp_path / "models")
    other = _gguf(tmp_path / "declared.gguf", QWEN_KV)
    y = tmp_path / "models.yaml"
    y.write_text(f"models:\n  - name: qwen3-coder-ov:30b\n    model: {other.as_posix()}\n"
                 f"    defaults: {{temperature: 0.2}}\n")
    store = ModelStore(DeclaredModels(y), OllamaManifestStore(root))
    spec = store.resolve("qwen3-coder-ov:30b")
    assert spec.model_path == other and spec.source == "declared"
    assert spec.defaults == {"temperature": 0.2}
    assert store.names().count("qwen3-coder-ov:30b") == 1


# ------------------------------------------------------------------- GGUF
def test_gguf_reads_scalars_and_skips_arrays(tmp_path):
    meta = gguf_meta.read_metadata(_gguf(tmp_path / "m.gguf", QWEN_KV))
    assert meta["qwen3moe.context_length"] == 262144
    assert meta["general.size_label"] == "30B-A3B"
    assert "tokenizer.ggml.tokens" not in meta


def test_kv_bytes_from_the_models_own_geometry(tmp_path):
    meta = gguf_meta.read_metadata(_gguf(tmp_path / "m.gguf", QWEN_KV))
    # 48 blocks * 4 kv heads * (128+128) * 2 bytes
    assert gguf_meta.kv_bytes_per_token(meta) == 48 * 4 * 256 * 2


def test_gguf_unreadable_is_empty_not_raised(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"NOPE")
    assert gguf_meta.read_metadata(p) == {}
    assert gguf_meta.read_metadata(tmp_path / "missing.gguf") == {}


# --------------------------------------------------------------- protocol
DEFAULTS = {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "repeat_penalty": 1.05, "stop": ["<|im_end|>"]}


def test_ollama_chat_maps_format_options_and_defaults():
    body, num_ctx, ka, stream = protocol.prepare_ollama_chat({
        "model": "m", "messages": [{"role": "user", "content": "hi"}],
        "format": {"type": "object", "properties": {}},
        "options": {"num_ctx": 32768, "num_predict": 900, "temperature": 0.3, "seed": 5,
                    "draft_num_predict": 4},
        "keep_alive": 1800, "think": False}, DEFAULTS)
    assert num_ctx == 32768 and ka == 1800 and stream is True
    assert body["response_format"]["type"] == "json_schema"
    assert body["max_tokens"] == 900 and body["temperature"] == 0.3 and body["seed"] == 5
    assert body["top_k"] == 20 and body["repeat_penalty"] == 1.05  # model defaults fill the rest
    assert body["stream_options"] == {"include_usage": True}
    for dead in ("think", "keep_alive", "options", "format", "num_ctx", "draft_num_predict"):
        assert dead not in body


def test_openai_chat_lifts_options_and_keeps_request_values():
    body, num_ctx, ka = protocol.prepare_openai_chat({
        "model": "m", "messages": [], "temperature": 0.1, "keep_alive": "30m",
        "options": {"top_k": 40, "num_ctx": 16384, "temperature": 0.9}}, DEFAULTS)
    assert body["temperature"] == 0.1          # explicit field beats options
    assert body["top_k"] == 40                 # options beat model defaults
    assert body["top_p"] == 0.8                # defaults fill the gap
    assert num_ctx == 16384 and ka == "30m"
    assert "options" not in body and "keep_alive" not in body


def test_json_format_and_images():
    body, *_ = protocol.prepare_ollama_chat({
        "model": "m", "format": "json", "stream": False,
        "messages": [{"role": "user", "content": "look", "images": ["QUJD"]}]}, {})
    assert body["response_format"] == {"type": "json_object"}
    parts = body["messages"][0]["content"]
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,QUJD"
    assert "stream_options" not in body


def test_stream_translation_text_usage_and_tool_calls():
    tr = protocol.OllamaStreamTranslator("m", "chat")
    lines = tr.feed({"choices": [{"delta": {"content": "Hel"}}]})
    lines += tr.feed({"choices": [{"delta": {"content": "lo"}}]})
    lines += tr.feed({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"name": "read_file", "arguments": "{\"path\":"}}]}}]})
    lines += tr.feed({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": "\"a.py\"}"}}]}, "finish_reason": "tool_calls"}]})
    lines += tr.feed({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                      "timings": {"predicted_ms": 70.0, "prompt_ms": 5.0}})
    out = [json.loads(x) for x in lines]
    assert "".join(o["message"]["content"] for o in out) == "Hello"
    final = json.loads(tr.final())
    assert final["done"] is True and final["done_reason"] == "tool_calls"
    assert final["eval_count"] == 7 and final["prompt_eval_count"] == 11
    assert final["eval_duration"] == 70_000_000
    assert final["message"]["tool_calls"] == [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}]


def test_parse_sse_handles_partial_frames_and_done():
    events, rest = protocol.parse_sse(b'data: {"a":1}\n\ndata: [DONE]\n\ndata: {"b"')
    assert events == [{"a": 1}, None] and rest == b'data: {"b"'


@pytest.mark.parametrize("v,expect", [(None, 300.0), (0, 0.0), (-1, None), ("5m", 300.0),
                                      ("30s", 30.0), ("1h", 3600.0), ("-1", None), (1800, 1800.0)])
def test_keep_alive(v, expect):
    assert parse_keep_alive(v, 300.0) == expect


# ------------------------------------------------------------------- pool
class FakeProc:
    pid = 4242

    def __init__(self, gpu_state, mib):
        self.gpu, self.mib, self.returncode = gpu_state, mib, None
        gpu_state["used"] += mib

    def poll(self):
        return self.returncode

    def terminate(self):
        if self.returncode is None:
            self.returncode = 0
            self.gpu["used"] -= self.mib

    kill = terminate

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def pool_factory(tmp_path):
    root = _ollama_store(tmp_path / "models")

    def make(total_mib=32000, model_mib=20000):
        gpu_state = {"used": 4000}
        launched = []

        def launcher(argv, log):
            launched.append(argv)
            return FakeProc(gpu_state, model_mib)

        async def healthy(_eng):
            return True

        cfg = EngineConfig(binary=Path("llama-server"), state_dir=tmp_path / "state",
                           default_keep_alive_s=300.0, vram_headroom_mib=512)
        pool = EnginePool(ModelStore(DeclaredModels(tmp_path / "none.yaml"), OllamaManifestStore(root)), cfg,
                          gpu_probe=lambda: GpuMemory("RTX 5090", total_mib, gpu_state["used"]),
                          launcher=launcher, health_check=healthy)
        return pool, launched, gpu_state
    return make


def test_load_reuse_and_measured_vram(pool_factory):
    pool, launched, _ = pool_factory()

    async def go():
        async with pool.acquire("qwen3-coder-ov:30b", 32768, 1800) as eng:
            assert eng.inflight == 1
        async with pool.acquire("qwen3-coder-ov:30b", 8192):
            pass
        return pool.engines["qwen3-coder-ov:30b"]

    eng = asyncio.run(go())
    assert len(launched) == 1                       # second request reused the engine
    argv = launched[0]
    assert argv[argv.index("-c") + 1] == "32768"
    assert "--lora" in argv and "--jinja" in argv and argv[argv.index("--alias") + 1] == "qwen3-coder-ov:30b"
    assert eng.size_vram_bytes == 20000 * 1024 * 1024  # the measured delta, not an estimate


def test_larger_ctx_reloads(pool_factory):
    pool, launched, _ = pool_factory()

    async def go():
        async with pool.acquire("qwen3-coder-ov:30b", 8192):
            pass
        async with pool.acquire("qwen3-coder-ov:30b", 65536):
            pass
    asyncio.run(go())
    assert [a[a.index("-c") + 1] for a in launched] == ["8192", "65536"]


# The fixture GGUFs are bytes, not gigabytes, so their file-size estimate is
# ~1.3 GiB (KV for 8k ctx + compute). A 6000 MiB card with 4000 used leaves
# room for exactly one such model plus headroom -- the second must evict.
def test_lru_idle_engine_is_evicted_to_admit_another(pool_factory):
    pool, launched, gpu_state = pool_factory(total_mib=6000, model_mib=1500)

    async def go():
        async with pool.acquire("qwen3-coder-ov:30b", 8192):
            pass
        async with pool.acquire("qwen3-coder:30b", 8192):
            pass
    asyncio.run(go())
    assert list(pool.engines) == ["qwen3-coder:30b"]
    assert gpu_state["used"] == 4000 + 1500


def test_measured_footprint_replaces_the_estimate(pool_factory):
    pool, _, _ = pool_factory(total_mib=32000, model_mib=20000)

    async def go():
        async with pool.acquire("qwen3-coder-ov:30b", 8192):
            pass
    asyncio.run(go())
    spec = pool.store.resolve("qwen3-coder-ov:30b")
    assert pool.estimate_mib(spec, 8192) == 20000        # learned from the real load
    assert pool.estimate_mib(spec, 16384) < 20000        # unseen ctx: still the geometry estimate


def test_busy_engines_are_never_evicted(pool_factory):
    pool, _, _ = pool_factory(total_mib=6000, model_mib=1500)

    async def go():
        async with pool.acquire("qwen3-coder-ov:30b", 8192):
            with pytest.raises(InsufficientVram):
                async with pool.acquire("qwen3-coder:30b", 8192):
                    pass
    asyncio.run(go())


def test_keep_alive_zero_unloads_when_idle_and_unknown_model_404(pool_factory):
    pool, _, gpu_state = pool_factory()

    async def go():
        async with pool.acquire("qwen3-coder-ov:30b", 8192):
            pass
        assert await pool.unload("qwen3-coder-ov:30b") is True
        with pytest.raises(ModelNotFound):
            await pool.ensure("ghost:7b")
    asyncio.run(go())
    assert pool.engines == {} and gpu_state["used"] == 4000
