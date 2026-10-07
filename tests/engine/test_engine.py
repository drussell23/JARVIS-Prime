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


# ------------------------------------------------------------ training lease
from jarvis_prime.engine.engine_pool import AdmissionClosed  # noqa: E402
from jarvis_prime.engine.lease import LeaseConflict, LeaseState, LeaseTokenMismatch, TrainingLease  # noqa: E402


def _leased(pool_factory, **kw):
    pool, launched, gpu_state = pool_factory(**kw)
    lease = TrainingLease(pool, gpu_probe=pool._gpu)
    pool.admission_gate = lambda: (("leased", 5) if lease.state in (LeaseState.DRAINING, LeaseState.RELEASED)
                                   else None)
    return pool, lease, launched, gpu_state


def test_lease_unloads_verifies_release_refuses_loads_and_restores(pool_factory):
    pool, lease, launched, gpu_state = _leased(pool_factory)

    async def go():
        async with pool.acquire("qwen3-coder-ov:30b", 8192):
            pass
        got = await lease.acquire(holder="handoff", purpose="grpo", ttl_s=60)
        assert got["freed_mib"] == 20000 and pool.engines == {}
        assert gpu_state["used"] == 4000                       # the card is measurably free
        with pytest.raises(AdmissionClosed):                    # nobody reloads during training
            await pool.ensure("qwen3-coder-ov:30b")
        with pytest.raises(LeaseConflict):
            await lease.acquire(holder="other", purpose="x", ttl_s=60)
        with pytest.raises(LeaseTokenMismatch):
            await lease.release("wrong-token")
        out = await lease.release(got["token"])
        assert out["reloaded"] == ["qwen3-coder-ov:30b"] and lease.state == LeaseState.SERVING
    asyncio.run(go())
    assert len(launched) == 2                                   # loaded, then restored


def test_lease_waits_for_inflight_generation_to_finish(pool_factory):
    pool, lease, _, _ = _leased(pool_factory)

    async def go():
        async with pool.acquire("qwen3-coder-ov:30b", 8192):
            task = asyncio.ensure_future(lease.acquire(holder="h", purpose="p", ttl_s=60))
            await asyncio.sleep(0.2)
            assert lease.state == LeaseState.DRAINING and not task.done()
        got = await task
        assert got["freed_mib"] == 20000
    asyncio.run(go())


def test_unverifiable_release_aborts_and_restores_service(pool_factory):
    pool, lease, _, gpu_state = _leased(pool_factory)

    async def go():
        async with pool.acquire("qwen3-coder-ov:30b", 8192):
            pass
        eng = pool.engines["qwen3-coder-ov:30b"]
        eng.proc.terminate = lambda: None                       # a process that will not die
        eng.proc.kill = lambda: None
        with pytest.raises(TimeoutError):
            await lease.acquire(holder="h", purpose="p", ttl_s=60, release_timeout_s=1.5)
        assert lease.state == LeaseState.SERVING and lease.lease is None
    asyncio.run(go())


def test_a_lease_error_belongs_to_its_own_cycle(pool_factory):
    # 2026-10-07: a restore that failed in an EARLIER cycle (and was since
    # recovered) was still reported beside the next live lease, so the boot
    # alert described a failure that was not happening.
    pool, lease, _, _ = _leased(pool_factory)

    async def go():
        async with pool.acquire("qwen3-coder-ov:30b", 8192):
            pass
        eng = pool.engines["qwen3-coder-ov:30b"]
        eng.proc.terminate = lambda: None
        eng.proc.kill = lambda: None
        with pytest.raises(TimeoutError):
            await lease.acquire(holder="h", purpose="p", ttl_s=60, release_timeout_s=1.5)
        failed = lease.status()
        assert failed["last_error"].startswith("acquire failed") and failed["last_error_at"] is not None
        for name in list(pool.engines):                         # the operator clears the wedge
            pool.engines.pop(name)
        got = await lease.acquire(holder="next", purpose="grpo", ttl_s=60)
        st = lease.status()
        assert st["last_error"] == "" and st["last_error_at"] is None   # a new cycle starts clean
        await lease.release(got["token"])
    asyncio.run(go())


def test_concurrent_allocator_delta_is_not_attributed(pool_factory, monkeypatch):
    pool, launched, gpu_state = pool_factory(model_mib=20000)
    import jarvis_prime.engine.engine_pool as ep
    from jarvis_prime.engine.model_store import ModelSpec
    # The real 30B's mapped files; the fixture GGUFs are bytes, not gigabytes.
    monkeypatch.setattr(ModelSpec, "size_bytes", property(lambda self: 18_583_454_368))

    real = pool._launch
    def freeing_launcher(argv, log):                            # another process frees 19.9 GB mid-load
        p = real(argv, log)
        gpu_state["used"] -= 19900
        return p
    pool._launch = freeing_launcher

    async def go():
        async with pool.acquire("qwen3-coder-ov:30b", 8192):
            pass
    asyncio.run(go())
    eng = pool.engines["qwen3-coder-ov:30b"]
    assert eng.size_vram_bytes == pool.estimate_mib(eng.spec, 8192) * ep._MIB
    assert ("qwen3-coder-ov:30b", 8192) not in pool._measured


# ------------------------------------------------------------ adapter registry
from jarvis_prime.engine.adapter_registry import AdapterRegistry, AdapterRejected  # noqa: E402


def _adapter_bytes(tmp_path, arch="qwen3moe", gtype="adapter"):
    p = _gguf(tmp_path / "lora.gguf", {"general.architecture": arch, "general.type": gtype,
                                         "adapter.type": "lora"})
    return p.read_bytes()


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("JPRIME_ENGINE_ADAPTER_DIR", str(tmp_path / "adapters"))
    monkeypatch.setenv("JPRIME_ENGINE_MODELS", str(tmp_path / "models.yaml"))
    store = ModelStore(DeclaredModels(tmp_path / "models.yaml"), OllamaManifestStore(_ollama_store(tmp_path / "om")))
    return AdapterRegistry(store), store


def test_publish_activates_new_adapter_and_rollback_restores_origin(registry, tmp_path):
    import hashlib as _h
    reg, store = registry
    origin = store.resolve("qwen3-coder-ov:30b")
    data = _adapter_bytes(tmp_path)
    out = reg.publish("qwen3-coder-ov:30b", data, sha256=_h.sha256(data).hexdigest(), source={"run": "r1"})
    now = store.resolve("qwen3-coder-ov:30b")
    assert now.source == "declared" and now.model_path == origin.model_path
    assert now.adapter_paths != origin.adapter_paths and now.defaults == origin.defaults
    assert out["previous"] == "origin"
    reg.rollback("qwen3-coder-ov:30b")
    assert store.resolve("qwen3-coder-ov:30b").adapter_paths == origin.adapter_paths


@pytest.mark.parametrize("kw,why", [({"arch": "llama"}, "architecture"), ({"gtype": "model"}, "LoRA")])
def test_wrong_adapters_are_refused_and_leave_nothing_behind(registry, tmp_path, kw, why):
    import hashlib as _h
    reg, store = registry
    data = _adapter_bytes(tmp_path, **kw)
    with pytest.raises(AdapterRejected, match=why):
        reg.publish("qwen3-coder-ov:30b", data, sha256=_h.sha256(data).hexdigest())
    assert store.resolve("qwen3-coder-ov:30b").source == "ollama-store"
    assert not list((tmp_path / "adapters").rglob("*.gguf"))


def test_sha_mismatch_is_refused(registry, tmp_path):
    reg, _ = registry
    with pytest.raises(AdapterRejected, match="sha256"):
        reg.publish("qwen3-coder-ov:30b", _adapter_bytes(tmp_path), sha256="0" * 64)


def test_changed_definition_reloads_resident_engine(pool_factory, registry, tmp_path):
    import hashlib as _h
    reg, store = registry
    pool, launched, _ = pool_factory()
    pool.store = store

    async def go():
        async with pool.acquire("qwen3-coder-ov:30b", 8192):
            pass
        data = _adapter_bytes(tmp_path)
        reg.publish("qwen3-coder-ov:30b", data, sha256=_h.sha256(data).hexdigest())
        async with pool.acquire("qwen3-coder-ov:30b", 8192):
            pass
    asyncio.run(go())
    assert len(launched) == 2 and "adapters" in launched[1][launched[1].index("--lora") + 1]


def test_reject_deletes_the_bad_weights_and_restores_the_last_good(registry, tmp_path):
    import hashlib as _h
    reg, store = registry
    origin = store.resolve("qwen3-coder-ov:30b")
    good = _adapter_bytes(tmp_path)
    reg.publish("qwen3-coder-ov:30b", good, sha256=_h.sha256(good).hexdigest())
    good_version = reg.versions("qwen3-coder-ov:30b")["active"]
    bad = good + b"\0" * 64                                   # distinct bytes, same valid header
    out = reg.publish("qwen3-coder-ov:30b", bad, sha256=_h.sha256(bad).hexdigest())
    bad_version = out["active"]
    bad_file = Path(reg.versions("qwen3-coder-ov:30b")["versions"][-1]["adapters"][0])
    assert bad_file.is_file()

    rej = reg.reject("qwen3-coder-ov:30b", bad_version, "smoke failed: 0/3 parsed")
    assert rej["active"] == good_version and not bad_file.exists()
    state = reg.versions("qwen3-coder-ov:30b")
    entry = next(v for v in state["versions"] if v["version"] == bad_version)
    assert entry["status"] == "rejected" and "smoke failed" in entry["reason"]
    assert store.resolve("qwen3-coder-ov:30b").adapter_paths != origin.adapter_paths   # the GOOD fine-tune
    with pytest.raises(AdapterRejected, match="rejected"):
        reg.activate("qwen3-coder-ov:30b", bad_version)


def test_rejecting_the_only_fine_tune_falls_back_to_origin_and_origin_is_never_deleted(registry, tmp_path):
    import hashlib as _h
    reg, store = registry
    origin = store.resolve("qwen3-coder-ov:30b")
    data = _adapter_bytes(tmp_path)
    v = reg.publish("qwen3-coder-ov:30b", data, sha256=_h.sha256(data).hexdigest())["active"]
    reg.reject("qwen3-coder-ov:30b", v, "corrupt")
    assert store.resolve("qwen3-coder-ov:30b").adapter_paths == origin.adapter_paths
    assert all(p.exists() for p in origin.adapter_paths)
    with pytest.raises(AdapterRejected, match="never deleted"):
        reg.reject("qwen3-coder-ov:30b", "origin", "x")


# ------------------------------------------------------------ evaluation scope
def test_evaluation_scope_disables_prompt_cache_only_while_open(pool_factory, monkeypatch):
    import httpx
    from fastapi.testclient import TestClient
    from jarvis_prime.engine.app import create_app

    pool, _, _ = pool_factory()
    sent = []

    async def fake_post(self, url, json=None, **kw):
        sent.append(dict(json or {}))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    body = {"model": "qwen3-coder-ov:30b", "messages": [{"role": "user", "content": "x"}]}
    with TestClient(create_app(pool)) as c:
        c.post("/v1/chat/completions", json=body)
        assert c.post("/v1/models/qwen3-coder-ov:30b/evaluation", json={"on": True, "ttl_s": 60}).json()["evaluation"]
        c.post("/v1/chat/completions", json=body)
        assert c.get("/health").json()["evaluation_scopes"] == ["qwen3-coder-ov:30b"]
        c.post("/v1/models/qwen3-coder-ov:30b/evaluation", json={"on": False})
        c.post("/v1/chat/completions", json=body)
        c.post("/v1/models/qwen3-coder-ov:30b/evaluation", json={"on": True, "ttl_s": 1})
        import time as _t
        _t.sleep(1.1)                                         # an abandoned scope closes itself
        c.post("/v1/chat/completions", json=body)
    assert ["cache_prompt" in s for s in sent] == [False, True, False, False]
    assert sent[1]["cache_prompt"] is False


# ---------------------------------------------------------------- pinning
# The preloaded model is the operator's declaration of the primary: a
# best-effort model (a vision read) must never evict it, or every screenshot
# between two generations costs the generation lane a cold reload.

def test_a_pinned_primary_is_never_evicted_for_an_unpinned_model(pool_factory):
    pool, _, _ = pool_factory(total_mib=6000, model_mib=1500)
    pool.config.pinned = frozenset({"qwen3-coder-ov:30b"})

    async def go():
        async with pool.acquire("qwen3-coder-ov:30b", 8192):
            pass                                            # idle now, but pinned
        with pytest.raises(InsufficientVram, match="pinned"):
            async with pool.acquire("qwen3-coder:30b", 8192):
                pass
    asyncio.run(go())
    assert list(pool.engines) == ["qwen3-coder-ov:30b"]


def test_a_pinned_requester_may_evict_an_idle_unpinned_model(pool_factory):
    pool, _, _ = pool_factory(total_mib=6000, model_mib=1500)
    pool.config.pinned = frozenset({"qwen3-coder-ov:30b"})

    async def go():
        async with pool.acquire("qwen3-coder:30b", 8192):
            pass
        async with pool.acquire("qwen3-coder-ov:30b", 8192):
            pass
    asyncio.run(go())
    assert list(pool.engines) == ["qwen3-coder-ov:30b"]


def test_the_preload_is_pinned_by_declaration(monkeypatch):
    monkeypatch.setenv("JPRIME_ENGINE_PRELOAD", "qwen3-coder-ov:30b")
    monkeypatch.setenv("JPRIME_ENGINE_PINNED", "other:7b, ")
    assert EngineConfig(binary=Path("x")).pinned == frozenset({"qwen3-coder-ov:30b", "other:7b"})


# ---------------------------------------------------------------- declaring a model

VL_KV = {"general.architecture": "qwen3vl", "qwen3vl.context_length": 262144, "qwen3vl.block_count": 36,
         "qwen3vl.attention.head_count_kv": 8, "qwen3vl.attention.key_length": 128,
         "qwen3vl.attention.value_length": 128}
CLIP_KV = {"general.architecture": "clip"}


@pytest.fixture
def declared_env(tmp_path, monkeypatch):
    path = tmp_path / "models.yaml"
    monkeypatch.setenv("JPRIME_ENGINE_MODELS", str(path))
    path.write_text("models:\n- name: qwen3-coder-ov:30b\n  model: /x.gguf\n", encoding="utf-8")
    return tmp_path


def _registry(tmp_path):
    from jarvis_prime.engine.adapter_registry import AdapterRegistry
    store = ModelStore(DeclaredModels(tmp_path / "models.yaml"), OllamaManifestStore(_ollama_store(tmp_path / "o")))
    return AdapterRegistry(store), store


def test_a_vision_model_is_declared_verified_and_served_by_name(declared_env):
    import hashlib
    from jarvis_prime.engine.adapter_registry import AdapterRejected
    reg, store = _registry(declared_env)
    model = _gguf(declared_env / "vl.gguf", VL_KV)
    proj = _gguf(declared_env / "mmproj.gguf", CLIP_KV)
    sha = {str(model): hashlib.sha256(model.read_bytes()).hexdigest()}
    reg.declare(name="jarvis-vision:8b", model=model, projector=proj, defaults={"temperature": 0.1}, sha256=sha)
    spec = store.resolve("jarvis-vision:8b")
    assert spec.projector_path == proj and spec.defaults["temperature"] == 0.1
    assert store.resolve("qwen3-coder-ov:30b") is not None or "qwen3-coder-ov:30b" in store.declared.names()
    with pytest.raises(AdapterRejected, match="projector"):
        reg.declare(name="v", model=model, projector=model)            # a language model is not a projector
    with pytest.raises(AdapterRejected, match="must not be"):
        reg.declare(name="v", model=proj)                              # a projector is not a model
    with pytest.raises(AdapterRejected, match="sha256"):
        reg.declare(name="v", model=model, sha256={str(model): "0" * 64})
    with pytest.raises(AdapterRejected, match="readable GGUF"):
        (declared_env / "junk.gguf").write_bytes(b"not a gguf")
        reg.declare(name="v", model=declared_env / "junk.gguf")


def test_the_cli_inherits_the_names_existing_defaults(declared_env, monkeypatch, capsys):
    from jarvis_prime.engine import declare as dc
    root = _ollama_store(declared_env / "o2")
    monkeypatch.setattr(dc, "ModelStore", lambda: ModelStore(DeclaredModels(declared_env / "models.yaml"),
                                                             OllamaManifestStore(root)))
    model = _gguf(declared_env / "vl.gguf", VL_KV)
    proj = _gguf(declared_env / "mmproj.gguf", CLIP_KV)
    # qwen3-coder:30b already resolves (from the Ollama manifest) with params.
    rc = dc.main(["--name", "qwen3-coder:30b", "--model", str(model), "--projector", str(proj),
                  "--inherit-defaults", "--default", "num_ctx=8192"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["defaults"]["temperature"] == 0.7 and out["defaults"]["num_ctx"] == 8192
