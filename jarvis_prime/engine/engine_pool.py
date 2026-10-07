"""Engine pool -- one ``llama-server`` per resident model, admitted by measured VRAM.

## Lifecycle

* ``acquire(name, num_ctx, keep_alive)`` makes the model resident (loading it
  if needed) and holds an in-flight reference for the duration of a request.
* A model stays resident until its ``keep_alive`` expires with nothing in
  flight, it is explicitly unloaded (``keep_alive: 0``), or VRAM is needed
  for another model (least-recently-used idle engine is evicted first).
* A request needing MORE context than the resident engine was started with
  reloads it larger (bounded by ``JPRIME_ENGINE_MAX_CTX``), the same contract
  O+V's ``num_ctx`` negotiation already expects. A smaller request reuses it.

## Why admission is measured

Loads are serialized under one lock so the VRAM delta across a load is
attributable to that engine -- that delta is what ``/api/ps`` reports as
``size_vram``. Admission compares ``nvidia-smi`` free memory against the
model's file size plus its KV cache computed from its own GGUF geometry.

## Orphans

A J-Prime that dies must not strand 20+ GiB on the card. Children are bound
to the parent (a kill-on-close Job Object on Windows, ``PR_SET_PDEATHSIG`` on
Linux), and every engine PID is recorded so a restarted pool reaps leftovers
from a hard kill before it loads anything.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import httpx

from . import gguf_meta, gpu
from .model_store import ModelSpec, ModelStore, canonical_name

logger = logging.getLogger(__name__)

_MIB = 1024 * 1024
_IS_WINDOWS = sys.platform == "win32"


class ModelNotFound(LookupError):
    pass


class InsufficientVram(RuntimeError):
    pass


class EngineStartError(RuntimeError):
    pass


class AdmissionClosed(RuntimeError):
    """The pool is not loading models right now (e.g. a training lease holds
    the card). Carries the seconds after which a retry is sensible."""

    def __init__(self, reason: str, retry_after_s: int) -> None:
        super().__init__(reason)
        self.retry_after_s = retry_after_s


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def default_engine_home() -> Path:
    if _IS_WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "JARVIS" / "jprime" / "engines"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "jarvis" / "jprime" / "engines"


def find_engine_binary(home: Optional[Path] = None) -> Optional[Path]:
    """``JPRIME_LLAMA_SERVER_BIN``, else the newest build under the engine home."""
    explicit = os.environ.get("JPRIME_LLAMA_SERVER_BIN", "").strip()
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    exe = "llama-server.exe" if _IS_WINDOWS else "llama-server"
    root = Path(home or os.environ.get("JPRIME_ENGINE_HOME", "") or default_engine_home()) / "llama.cpp"
    if not root.is_dir():
        return None

    def build_no(p: Path) -> int:
        m = re.search(r"(\d+)", p.parent.name)
        return int(m.group(1)) if m else -1

    found = [p for p in root.glob(f"*/{exe}") if p.is_file()]
    return max(found, key=build_no) if found else None


def parse_keep_alive(value: Any, default_s: Optional[float]) -> Optional[float]:
    """Ollama keep_alive semantics. Seconds to stay resident after the last
    request; ``None`` = forever; ``0`` = unload as soon as idle."""
    if value is None or value == "":
        return default_s
    if isinstance(value, (int, float)):
        return None if value < 0 else float(value)
    s = str(value).strip().lower()
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)(ms|s|m|h)?", s)
    if not m:
        return default_s
    n = float(m.group(1))
    if n < 0:
        return None
    return n * {"ms": 0.001, "s": 1, "m": 60, "h": 3600, None: 1}[m.group(2)]


@dataclass
class EngineConfig:
    binary: Optional[Path] = None
    gpu_layers: int = field(default_factory=lambda: _env_int("JPRIME_ENGINE_GPU_LAYERS", 999))
    default_ctx: int = field(default_factory=lambda: _env_int("JPRIME_ENGINE_DEFAULT_CTX", 8192))
    max_ctx: int = field(default_factory=lambda: _env_int("JPRIME_ENGINE_MAX_CTX", 131072))
    parallel: int = field(default_factory=lambda: _env_int("JPRIME_ENGINE_PARALLEL", 1))
    flash_attn: str = field(default_factory=lambda: os.environ.get("JPRIME_ENGINE_FLASH_ATTN", "on"))
    load_timeout_s: float = field(default_factory=lambda: float(_env_int("JPRIME_ENGINE_LOAD_TIMEOUT_S", 300)))
    default_keep_alive_s: Optional[float] = field(
        default_factory=lambda: parse_keep_alive(os.environ.get("JPRIME_ENGINE_KEEP_ALIVE", "5m"), 300.0))
    vram_headroom_mib: int = field(default_factory=lambda: _env_int("JPRIME_ENGINE_VRAM_HEADROOM_MIB", 1024))
    #: A load's VRAM delta is attributed to it only if it covers at least this
    #: fraction of the files it mapped (weights are fully offloaded).
    min_attributable_fraction: float = field(default_factory=lambda: float(
        os.environ.get("JPRIME_ENGINE_MIN_ATTRIBUTABLE_FRACTION", "") or 0.9))
    extra_args: List[str] = field(
        default_factory=lambda: shlex.split(os.environ.get("JPRIME_ENGINE_EXTRA_ARGS", ""), posix=not _IS_WINDOWS))
    state_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("JPRIME_ENGINE_STATE_DIR", "") or default_engine_home().parent / "state"))

    def resolve_binary(self) -> Path:
        b = self.binary or find_engine_binary()
        if b is None:
            raise EngineStartError(
                "no llama-server binary: set JPRIME_LLAMA_SERVER_BIN or install a build under "
                f"{default_engine_home() / 'llama.cpp' / '<build>'}")
        return b


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------
# Child binding: the engine must die with J-Prime.
# --------------------------------------------------------------------------
_job_handle = None


def _windows_job():
    """A kill-on-close Job Object; every engine is assigned to it."""
    global _job_handle
    if _job_handle is not None:
        return _job_handle
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32.CreateJobObjectW.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    _job_handle = (k32, job)
    return _job_handle


def _bind_to_parent(proc: subprocess.Popen) -> None:
    if not _IS_WINDOWS:
        return  # handled by preexec PR_SET_PDEATHSIG
    try:
        import ctypes
        k32, job = _windows_job()
        PROCESS_ALL_ACCESS = 0x1F0FFF
        h = k32.OpenProcess(PROCESS_ALL_ACCESS, False, proc.pid)
        if not k32.AssignProcessToJobObject(job, h):
            logger.warning("[Engine] could not bind pid %s to the job object (err %s)",
                           proc.pid, ctypes.get_last_error())
        k32.CloseHandle(h)
    except Exception as exc:  # noqa: BLE001 -- binding is protection, never a load blocker
        logger.warning("[Engine] job-object binding failed: %s", exc)


def _linux_preexec() -> None:  # pragma: no cover -- runs in the child
    os.setsid()
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------
@dataclass
class Engine:
    spec: ModelSpec
    ctx: int
    port: int
    proc: subprocess.Popen
    log_path: Path
    started_at: float = field(default_factory=time.time)
    size_vram_bytes: int = 0
    inflight: int = 0
    last_used: float = field(default_factory=time.monotonic)
    expires_at: Optional[float] = None       # monotonic; None = resident until evicted
    expires_wall: Optional[float] = None     # wall clock mirror for /api/ps

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def alive(self) -> bool:
        return self.proc.poll() is None


Launcher = Callable[[List[str], Path], subprocess.Popen]


def _default_launcher(argv: List[str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")  # noqa: SIM115 -- owned by the child for its lifetime
    kw: Dict[str, Any] = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
    if _IS_WINDOWS:
        kw["creationflags"] = 0x08000000 | 0x00000200  # CREATE_NO_WINDOW | NEW_PROCESS_GROUP
    else:
        kw["preexec_fn"] = _linux_preexec
    proc = subprocess.Popen(argv, **kw)
    log.close()
    _bind_to_parent(proc)
    return proc


class EnginePool:
    def __init__(self, store: Optional[ModelStore] = None, config: Optional[EngineConfig] = None,
                 gpu_probe: Callable[[], Optional[gpu.GpuMemory]] = gpu.query,
                 launcher: Launcher = _default_launcher,
                 health_check: Optional[Callable[[Engine], Any]] = None) -> None:
        self.store = store or ModelStore()
        self.config = config or EngineConfig()
        self._gpu = gpu_probe
        self._launch = launcher
        self._health = health_check or self._http_health
        self.engines: Dict[str, Engine] = {}
        self._measured: Dict[Any, int] = {}   # (name, ctx) -> MiB a load actually took
        self._load_lock = asyncio.Lock()
        self._reaper: Optional[asyncio.Task] = None
        #: Set by the owner of exclusivity (the training lease). Returns None
        #: when generations may proceed, else (reason, retry_after_s).
        self.admission_gate: Optional[Callable[[], Optional[Tuple[str, int]]]] = None

    # ------------------------------------------------------------------ state
    @property
    def _state_file(self) -> Path:
        return self.config.state_dir / "engines.json"

    def _record_pids(self) -> None:
        try:
            self.config.state_dir.mkdir(parents=True, exist_ok=True)
            data = {n: e.proc.pid for n, e in self.engines.items()}
            self._state_file.write_text(json.dumps(data), encoding="utf-8")
        except OSError as exc:
            logger.debug("[Engine] could not record pids: %s", exc)

    def reap_orphans(self) -> int:
        """Kill engines a previous (hard-killed) J-Prime left behind."""
        try:
            pids = json.loads(self._state_file.read_text(encoding="utf-8")).values()
        except (OSError, ValueError):
            return 0
        killed = 0
        try:
            import psutil
        except ImportError:
            return 0
        for pid in pids:
            try:
                p = psutil.Process(int(pid))
                if "llama-server" in p.name().lower():
                    p.kill()
                    killed += 1
                    logger.warning("[Engine] reaped orphan llama-server pid=%s", pid)
            except (psutil.Error, ValueError):
                continue
        try:
            self._state_file.unlink()
        except OSError:
            pass
        return killed

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.reap_orphans()
        if self._reaper is None:
            self._reaper = asyncio.get_running_loop().create_task(self._reap_loop())

    async def shutdown(self) -> None:
        if self._reaper:
            self._reaper.cancel()
            self._reaper = None
        for name in list(self.engines):
            await self.unload(name)

    async def _reap_loop(self) -> None:
        while True:
            await asyncio.sleep(2.0)
            now = time.monotonic()
            for name, eng in list(self.engines.items()):
                if not eng.alive():
                    logger.error("[Engine] %s exited unexpectedly (rc=%s) -- see %s",
                                 name, eng.proc.returncode, eng.log_path)
                    self.engines.pop(name, None)
                    self._record_pids()
                elif eng.inflight == 0 and eng.expires_at is not None and now >= eng.expires_at:
                    logger.info("[Engine] %s keep_alive expired -- unloading", name)
                    await self.unload(name)

    # ------------------------------------------------------------- admission
    def estimate_mib(self, spec: ModelSpec, ctx: int) -> int:
        """VRAM a load will need: what it measurably took last time at this
        context if we have seen it, else file size + KV from its geometry."""
        measured = self._measured.get((spec.name, ctx))
        if measured:
            return measured
        kv = gguf_meta.kv_bytes_per_token(gguf_meta.read_metadata(spec.model_path)) * ctx
        compute = 512 * _MIB
        return int((spec.size_bytes + kv + compute) / _MIB)

    async def _make_room(self, need_mib: int, keep: str) -> None:
        while True:
            mem = await asyncio.to_thread(self._gpu)
            if mem is None or mem.free_mib >= need_mib + self.config.vram_headroom_mib:
                return
            idle = [e for n, e in self.engines.items() if n != keep and e.inflight == 0]
            if not idle:
                if any(n != keep for n in self.engines):
                    raise InsufficientVram(
                        f"need ~{need_mib} MiB, {mem.free_mib} MiB free, and every resident model is busy")
                return  # nothing of ours to evict; let the engine try (another process owns the VRAM)
            victim = min(idle, key=lambda e: e.last_used)
            logger.info("[Engine] evicting %s to admit ~%d MiB (free %d MiB)",
                        victim.spec.name, need_mib, mem.free_mib)
            await self._unload_locked(victim.spec.name)  # caller holds _load_lock

    # ---------------------------------------------------------------- loading
    def _argv(self, spec: ModelSpec, ctx: int, port: int) -> List[str]:
        c = self.config
        argv = [str(c.resolve_binary()), "--model", str(spec.model_path)]
        for a in spec.adapter_paths:
            argv += ["--lora", str(a)]
        if spec.projector_path:
            argv += ["--mmproj", str(spec.projector_path)]
        argv += ["--alias", spec.name, "--host", "127.0.0.1", "--port", str(port),
                 "-ngl", str(c.gpu_layers), "-c", str(ctx * max(c.parallel, 1)),
                 "--parallel", str(max(c.parallel, 1)), "--jinja", "-fa", c.flash_attn, "--metrics"]
        return argv + list(c.extra_args)

    def _wanted_ctx(self, spec: ModelSpec, num_ctx: Optional[int]) -> int:
        want = num_ctx or int(spec.defaults.get("num_ctx") or 0) or self.config.default_ctx
        meta = gguf_meta.read_metadata(spec.model_path)
        native = int(meta.get(f"{meta.get('general.architecture', '')}.context_length", 0) or 0)
        cap = min(self.config.max_ctx, native) if native else self.config.max_ctx
        return max(512, min(int(want), cap))

    async def _http_health(self, eng: Engine) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get(eng.base_url + "/health")
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def _load(self, spec: ModelSpec, ctx: int) -> Engine:
        await self._make_room(self.estimate_mib(spec, ctx), keep=spec.name)
        before = await asyncio.to_thread(self._gpu)
        port = _free_port()
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.name)
        log_path = self.config.state_dir / "logs" / f"{safe}.log"
        argv = self._argv(spec, ctx, port)
        logger.info("[Engine] loading %s ctx=%d port=%d adapters=%d", spec.name, ctx, port,
                    len(spec.adapter_paths))
        proc = await asyncio.to_thread(self._launch, argv, log_path)
        eng = Engine(spec=spec, ctx=ctx, port=port, proc=proc, log_path=log_path)
        deadline = time.monotonic() + self.config.load_timeout_s
        while True:
            if not eng.alive():
                raise EngineStartError(f"llama-server for {spec.name} exited rc={proc.returncode}; see {log_path}")
            if await self._health(eng):
                break
            if time.monotonic() > deadline:
                await asyncio.to_thread(self._terminate, eng)
                raise EngineStartError(f"{spec.name} not ready after {self.config.load_timeout_s:.0f}s; see {log_path}")
            await asyncio.sleep(0.5)
        after = await asyncio.to_thread(self._gpu)
        delta = (after.used_mib - before.used_mib) if (before and after) else 0
        # A fully offloaded model cannot occupy less VRAM than its own weights.
        # A smaller delta means another process freed memory during the load
        # (measured 2026-10-07: 2226 MiB "for" an 18.6 GB model while a trainer
        # was releasing the card), so the reading is NOT attributable and is
        # neither reported nor learned.
        if delta * _MIB >= spec.size_bytes * self.config.min_attributable_fraction:
            eng.size_vram_bytes = delta * _MIB
            self._measured[(spec.name, ctx)] = delta
        else:
            if delta:
                logger.warning("[Engine] %s VRAM delta %d MiB is below its %d MiB of weights -- a "
                               "concurrent allocator moved; using the estimate", spec.name, delta,
                               spec.size_bytes // _MIB)
            eng.size_vram_bytes = self.estimate_mib(spec, ctx) * _MIB
        logger.info("[Engine] %s ready in %.1fs, vram=%d MiB", spec.name,
                    time.time() - eng.started_at, eng.size_vram_bytes // _MIB)
        return eng

    async def ensure(self, name: str, num_ctx: Optional[int] = None) -> Engine:
        closed = self.admission_gate() if self.admission_gate else None
        if closed:
            raise AdmissionClosed(*closed)
        spec = self.store.resolve(name)
        if spec is None:
            raise ModelNotFound(f"model '{name}' not found")
        key = spec.name
        async with self._load_lock:
            want = self._wanted_ctx(spec, num_ctx)
            eng = self.engines.get(key)
            if eng is not None and eng.alive():
                if eng.spec != spec and eng.inflight == 0:
                    # The model's files changed under its name (a new adapter
                    # was published): serve the new one, never the stale one.
                    logger.info("[Engine] %s definition changed -- reloading", key)
                    await self._unload_locked(key)
                elif eng.ctx >= want or eng.inflight > 0:
                    return eng
                else:
                    logger.info("[Engine] %s ctx %d < requested %d -- reloading larger", key, eng.ctx, want)
                    await self._unload_locked(key)
            elif eng is not None:
                self.engines.pop(key, None)
            eng = await self._load(spec, want)
            self.engines[key] = eng
            self._record_pids()
            return eng

    @asynccontextmanager
    async def acquire(self, name: str, num_ctx: Optional[int] = None,
                      keep_alive: Any = None) -> AsyncIterator[Engine]:
        eng = await self.ensure(name, num_ctx)
        eng.inflight += 1
        eng.expires_at = None
        try:
            yield eng
        finally:
            eng.inflight -= 1
            eng.last_used = time.monotonic()
            ttl = parse_keep_alive(keep_alive, self.config.default_keep_alive_s)
            if ttl is None:
                eng.expires_at = eng.expires_wall = None
            else:
                eng.expires_at = eng.last_used + ttl
                eng.expires_wall = time.time() + ttl

    # -------------------------------------------------------------- unloading
    @staticmethod
    def _terminate(eng: Engine) -> None:
        if not eng.alive():
            return
        eng.proc.terminate()
        try:
            eng.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            eng.proc.kill()
            eng.proc.wait(timeout=10)

    async def _unload_locked(self, name: str) -> bool:
        eng = self.engines.pop(canonical_name(name), None) or self.engines.pop(name, None)
        if eng is None:
            return False
        await asyncio.to_thread(self._terminate, eng)
        self._record_pids()
        logger.info("[Engine] unloaded %s", eng.spec.name)
        return True

    async def unload(self, name: str) -> bool:
        spec = self.store.resolve(name)
        key = spec.name if spec else canonical_name(name)
        eng = self.engines.get(key)
        if eng is not None and eng.inflight > 0:
            # Ollama semantics: keep_alive 0 on a busy model unloads when it goes idle.
            eng.expires_at, eng.expires_wall = time.monotonic(), time.time()
            return True
        async with self._load_lock:
            return await self._unload_locked(key)

    def resident(self) -> List[Engine]:
        return [e for e in self.engines.values() if e.alive()]
