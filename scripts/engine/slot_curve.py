"""Measure the J-Prime VRAM-to-throughput curve across generation slot counts.

For each slot count the engine is relaunched through ``jprime_engine.ps1``
with ``-Parallel N`` (nothing else changes), then the resident footprint is
read from ``/api/ps`` and ``nvidia-smi``, and ``1..N+1`` concurrent
generations are fired at it. Every request carries a unique nonce ahead of
the source text so no slot can reuse another request's KV prefix: each one
pays its own prefill, as O+V's distinct ops do.

    .venv-engine\\Scripts\\python.exe scripts\\engine\\slot_curve.py ^
        --prompt-file <source file> --slots 1 2 3 --restore auto

Writes ``%LOCALAPPDATA%\\JARVIS\\jprime\\state\\slot_curve-<ts>.json`` and
prints one row per (slots, concurrency). The engine is left on ``--restore``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import httpx

_LAUNCHER = Path(__file__).with_name("jprime_engine.ps1")
_STATE = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "JARVIS" / "jprime" / "state"


def _engine(action: str, **kw: Any) -> str:
    argv = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(_LAUNCHER),
            "-Action", action]
    for k, v in kw.items():
        argv += [f"-{k}", str(v)]
    return subprocess.run(argv, capture_output=True, text=True, timeout=600).stdout.strip()


def _gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout
    return int(out.split()[0])


async def _resident(base: str, model: str, timeout_s: float) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=10) as c:
        while time.monotonic() < deadline:
            try:
                for m in (await c.get(f"{base}/api/ps")).json().get("models", []):
                    if m.get("name") == model:
                        return m
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"{model} not resident within {timeout_s:.0f}s")


async def _one(c: httpx.AsyncClient, base: str, model: str, source: str, task: str,
               max_tokens: int) -> Dict[str, Any]:
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0.2, "stream": False,
            "messages": [{"role": "user",
                          "content": f"[request {uuid.uuid4().hex}]\n\n{source}\n\n{task}"}]}
    t0 = time.monotonic()
    r = await c.post(f"{base}/v1/chat/completions", json=body)
    wall = time.monotonic() - t0
    r.raise_for_status()
    usage = r.json().get("usage") or {}
    return {"wall_s": wall, "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0)}


async def _burst(base: str, model: str, source: str, task: str, n: int, max_tokens: int) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800)) as c:
        t0 = time.monotonic()
        rows = await asyncio.gather(*(_one(c, base, model, source, task, max_tokens) for _ in range(n)))
        span = time.monotonic() - t0
    out_tok = sum(r["completion_tokens"] for r in rows)
    return {"concurrency": n, "span_s": round(span, 2),
            "aggregate_tok_s": round(out_tok / span, 2) if span else 0.0,
            "per_stream_tok_s": round(statistics.mean(r["completion_tokens"] / r["wall_s"] for r in rows), 2),
            "mean_wall_s": round(statistics.mean(r["wall_s"] for r in rows), 2),
            "prompt_tokens": rows[0]["prompt_tokens"], "completion_tokens": out_tok,
            "zero_token_streams": sum(1 for r in rows if not r["completion_tokens"])}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file", required=True, type=Path)
    ap.add_argument("--task", default=("Rewrite the module above in full, adding a precise docstring to every "
                                       "function and class that lacks one. Output only the code."))
    ap.add_argument("--model", default="qwen3-coder-ov:30b")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--slots", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--restore", default="auto", help="-Parallel to leave the engine on afterwards")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    source = a.prompt_file.read_text(encoding="utf-8", errors="replace")
    report: Dict[str, Any] = {"model": a.model, "ctx": a.ctx, "prompt_file": str(a.prompt_file),
                              "max_tokens": a.max_tokens, "rounds": a.rounds, "arms": []}
    try:
        for slots in a.slots:
            _engine("stop", Port=a.port)
            await asyncio.sleep(3)
            idle = _gpu_used_mib()
            _engine("start", Port=a.port, Preload=a.model, Ctx=a.ctx, Parallel=slots)
            m = await _resident(base, a.model, 600)
            arm: Dict[str, Any] = {"slots": m.get("parallel", slots), "requested": slots,
                                   "engine_vram_mib": int(m.get("size_vram", 0)) // (1 << 20),
                                   "gpu_used_mib": _gpu_used_mib(), "gpu_idle_mib": idle, "bursts": []}
            await _burst(base, a.model, source, a.task, 1, 16)        # warm the graph, not measured
            for n in range(1, slots + 2):
                runs = [await _burst(base, a.model, source, a.task, n, a.max_tokens) for _ in range(a.rounds)]
                best = max(runs, key=lambda r: r["aggregate_tok_s"])
                best["aggregate_tok_s_runs"] = [r["aggregate_tok_s"] for r in runs]
                arm["bursts"].append(best)
                print(f"slots={arm['slots']} vram={arm['engine_vram_mib']}MiB conc={n} "
                      f"agg={best['aggregate_tok_s']} tok/s per_stream={best['per_stream_tok_s']} "
                      f"wall={best['mean_wall_s']}s prompt={best['prompt_tokens']} "
                      f"zero={best['zero_token_streams']}", flush=True)
            report["arms"].append(arm)
    finally:
        _engine("stop", Port=a.port)
        await asyncio.sleep(3)
        print(_engine("start", Port=a.port, Preload=a.model, Ctx=a.ctx, Parallel=a.restore), flush=True)
    path = _STATE / f"slot_curve-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
