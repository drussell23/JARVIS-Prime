# J-Prime benchmarks

Measured, not estimated. Every figure here comes from llama-server's own per-request timing lines and can be regenerated with:

```bash
python scripts/engine/timing_report.py "%LOCALAPPDATA%\JARVIS\jprime\state\logs\qwen3-coder-ov_30b.log"
```

## Configuration

| | |
|---|---|
| Date | 2026-10-07 |
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB |
| Engine | J-Prime engine 0.1.0 → llama.cpp **b11462** (win-cuda-13.4-x64), `-ngl 999 -fa on --parallel 1` |
| Model | `qwen3-coder-ov:30b`: Qwen3-Coder 30.5B MoE (128 experts, 8 active), Q4_K_M, 17.3 GiB weights, plus a 27 MB GRPO LoRA |
| Context | 32,768 |
| Workload | Real JARVIS O+V autonomous-coding traffic: schema-constrained generation and tool rounds, not a synthetic prompt set |

## Results

| Metric | p50 | Spread | n |
|---|---|---|---|
| Decode throughput (generations ≥ 32 tokens) | **172 tok/s** | p10 152 · p90 184 · max 210 | 337 |
| Prefill, 1k–4k uncached tokens | **470 ms** (5.7k tok/s) | | 47 |
| Prefill, 4k–8k uncached tokens | **885 ms** (6.5k tok/s) | | 88 |
| Prefill, 8k–16k uncached tokens | **1.41 s** (6.2k tok/s) | | 43 |

Prefill counts only tokens **not** served from llama.cpp's prompt cache, so the latency shown is time to first token for that much new context.

## Lifecycle figures

| Event | Measured | Source |
|---|---|---|
| Cold load, 30B + LoRA, to resident | 15 s, 18.9 GiB | commit `fb85d48` |
| Training lease: VRAM released and verified by `nvidia-smi` | 21,236 MiB | JARVIS `eecadb334f` |
| Resident footprint, serving vs GRPO training | ~21 GiB vs ~28 GiB | `jarvis_prime/engine/lease.py` |
| Prompt cache on vs off, greedy decode | output differs vs **3/3 byte-identical** | commit `71b0ca4` (why evaluation scopes disable `cache_prompt`) |

## Caveats

- One host, one model, one day of traffic. Re-measure before trusting these figures on other hardware.
- Decode rate falls as context grows. The figures above are dominated by the 4k–16k-token prompts the O+V pipeline sends.
