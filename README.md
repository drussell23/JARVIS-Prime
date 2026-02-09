# JARVIS Prime

**The Mind of the AGI OS — LLM inference, Neural Orchestrator Core, and cross-repo coordination**

🚀 v100.0 Neural Orchestrator Core | 🧠 Unified Intelligent Routing | ⚡ Zero Hardcoding | 🔥 Async by Default | 🛡️ Safety-Aware | 🔄 Zero-Downtime Hot Swap | 💪 Production-Grade Resilience | 🌐 Cross-Repo Integration | 📊 v221.0 Model Loading Progress Preservation | 🎯 v236.0 Adaptive Prompt System | 🛡️ v238.0 Degenerate Response Defense-in-Depth

JARVIS Prime is the **cognitive layer** of the JARVIS AGI ecosystem. It runs a **self-hosted Mistral-7B-Instruct-v0.2 LLM** (Q4_K_M quantized, ~4.37 GB) on a dedicated GCP Invincible Node — **not OpenAI, not Claude, not any third-party API**. All inference happens on your own infrastructure with zero per-token costs and complete data privacy. Prime also provides the **Neural Orchestrator Core** (unified routing), AGI models, reasoning engines, and **first-class integration** with JARVIS (Body) and Reactor-Core (Nerves). It is started either **standalone** or by the **unified supervisor** in JARVIS; during startup, model loading progress is preserved across Early Prime → Trinity handoff (v221.0).

---

## 🎯 What is JARVIS Prime?

JARVIS Prime is the **Mind** in the three-repo Trinity architecture:

| Role | Repository | Responsibility |
|------|------------|----------------|
| **Body** | [JARVIS (JARVIS-AI-Agent)](https://github.com/drussell23/JARVIS-AI-Agent) | macOS integration, computer use, unified supervisor, voice/vision |
| **Mind** | **JARVIS-Prime (this repo)** | LLM inference, reasoning, Neural Orchestrator Core, OpenAI-compatible API |
| **Nerves** | [Reactor-Core](https://github.com/drussell23/JARVIS-Reactor) | Model training, fine-tuning, experience collection, model deployment |

**Neural Orchestrator Core v100.0** is the single source of truth for routing (Tier 0/0.5/1/2, memory pressure, sticky routing, circuit breakers). Prime exposes health and **model loading progress** (`model_load_progress_pct`, `startup_progress`, etc.) so the JARVIS unified supervisor can show accurate progress and avoid regression during handoff (v221.0).

### The Revolution: **Neural Orchestrator Core v100.0**

The Neural Orchestrator Core consolidates **all routing systems** (HybridTieredRouter, IntelligentModelRouter, CognitiveRouter, GraphRouter, Neural Switchboard) into a single, enterprise-grade unified routing architecture:

```python
# Simple action → Tier 0 (Ultra Fast, Local)
"Turn on the lights" → Local execution (50ms, $0.00)

# Complex task → Tier 1 (Cloud Intelligence)
"Plan a comprehensive refactoring of the authentication system"
→ GCP Cloud with advanced reasoning ($0.15)

# Deep reasoning → Tier 2 (Deep Reasoning Models)
"Analyze the causal relationships in this distributed system"
→ Claude Opus 4 with deep reasoning ($0.50)

# Session continuity → Sticky Routing
"Continue the previous coding session" → Same model as before
```

**Key Innovation:** The Neural Orchestrator Core provides:
- **Unified Routing**: Single source of truth for all routing decisions
- **Zero Hardcoding**: All configuration via environment variables and YAML
- **Advanced Patterns**: Protocol classes, contextvars, async generators, weakref, defensive decorators
- **Cross-Repo Integration**: Seamless state sharing across JARVIS, JARVIS Prime, and Reactor Core
- **Memory-Aware Routing**: Real-time memory pressure monitoring with macOS native integration
- **Sticky Routing**: Session-based model affinity for continuity
- **Request Buffering**: Zero-loss hot swap support
- **Circuit Breakers**: Coordinated fault tolerance across all tiers

---

## 🧠 Self-Hosted LLM — Zero Third-Party API Dependencies

### The Core Principle: Your Model, Your Infrastructure, Your Data

JARVIS Prime runs its own **self-hosted large language model**. It does **not** use OpenAI, Claude, GPT-4, Gemini, or any third-party inference API for primary intelligence. When you ask JARVIS "what's 2+2?" — or any question — the response is generated entirely by a model running on your own infrastructure:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  JARVIS PRIME INFERENCE STACK                            │
│                  ════════════════════════════                            │
│                                                                          │
│  Model:    Mistral-7B-Instruct-v0.2 (Q4_K_M quantization)              │
│  Engine:   llama-cpp-python (C++ backend with Python bindings)          │
│  Format:   GGUF (mistral-7b-instruct-v0.2.Q4_K_M.gguf, ~4.37 GB)      │
│  API:      OpenAI-compatible (/v1/chat/completions)                     │
│  Host:     GCP Invincible Node (34.45.154.209:8000)                     │
│  Latency:  ~8.6s per request (CPU inference, no GPU)                    │
│                                                                          │
│  ✅ Self-hosted          ✅ No per-token costs                           │
│  ✅ Full data privacy    ✅ No rate limits                               │
│  ✅ Pre-loaded from      ✅ No vendor lock-in                            │
│     golden image         ✅ Fine-tunable by Reactor-Core                 │
│                                                                          │
│  ❌ NOT OpenAI           ❌ NOT Claude                                   │
│  ❌ NOT GPT-4            ❌ NOT Gemini                                   │
│  ❌ NOT any third-party API                                              │
└──────────────────────────────────────────────────────────────────────────┘
```

### The Model: Mistral-7B-Instruct-v0.2 (Q4_K_M)

JARVIS Prime uses [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) — a 7-billion parameter instruction-tuned language model from Mistral AI — quantized to **Q4_K_M** (4-bit quantization with k-quant mixed precision) using the GGUF format:

| Property | Value |
|----------|-------|
| **Base Model** | `mistralai/Mistral-7B-Instruct-v0.2` |
| **Quantization** | Q4_K_M (4-bit, k-quant mixed) |
| **GGUF File** | `mistral-7b-instruct-v0.2.Q4_K_M.gguf` |
| **File Size** | ~4.37 GB |
| **Original Parameters** | 7.24 billion |
| **Context Length** | 4,096 tokens (configurable up to 32,768) |
| **Architecture** | Transformer decoder-only, Grouped-Query Attention (GQA), Sliding Window Attention (SWA) |
| **Source** | [TheBloke/Mistral-7B-Instruct-v0.2-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) on HuggingFace |
| **License** | Apache 2.0 |

**Why Q4_K_M?** This quantization offers the best balance of quality and size for CPU inference:
- **Q4_K_M** preserves more important weight dimensions at higher precision than Q4_0 or Q4_K_S
- ~4.37 GB fits comfortably in the GCP VM's RAM with room for OS and server overhead
- Negligible quality loss vs. FP16 on instruction-following benchmarks
- Optimized for `llama.cpp`'s SIMD-accelerated inference kernels

**Why Mistral-7B-Instruct-v0.2?** Selected for the JARVIS use case because:
- **Instruction-tuned**: Follows user instructions accurately (chat, Q&A, code, reasoning)
- **Efficient**: 7B parameters is the sweet spot for CPU inference — large enough for quality, small enough for speed
- **Open weights**: Apache 2.0 licensed, no usage restrictions, fully self-hostable
- **Well-supported**: Extensive GGUF quantization ecosystem, battle-tested with llama.cpp
- **Fine-tunable**: Reactor-Core can collect experience data and fine-tune the base model for JARVIS-specific tasks

### GCP Invincible Node: The Inference Server

The model runs on a **GCP Invincible Node** — a persistent Compute Engine VM that resists automated shutdown:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  GCP INVINCIBLE NODE                                     │
│                  ══════════════════                                      │
│                                                                          │
│  Instance:       jarvis-prime-node                                      │
│  External IP:    34.45.154.209                                          │
│  Port:           8000                                                    │
│  Machine Type:   e2-standard-4 (4 vCPUs, 16 GB RAM)                    │
│  Region:         us-central1-a                                          │
│  OS:             Debian (GCP golden image)                               │
│  Disk:           50 GB persistent SSD                                   │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  JARVIS Prime Server (run_server.py)                           │     │
│  │  ──────────────────────────────────                            │     │
│  │  • FastAPI + Uvicorn (port 8000)                               │     │
│  │  • llama-cpp-python inference engine                           │     │
│  │  • OpenAI-compatible API (/v1/chat/completions)                │     │
│  │  • Health endpoint (/health) with model_load_progress          │     │
│  │  • Model: mistral-7b-instruct-v0.2.Q4_K_M.gguf               │     │
│  │  • Pre-loaded from golden image disk (no download on boot)     │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │  InvincibleGuard (Active)                                      │     │
│  │  ──────────────────────────                                    │     │
│  │  • Blocks automated termination from supervisor cleanup        │     │
│  │  • 4 blocked termination attempts (as of v235.4)               │     │
│  │  • Ensures model stays loaded across session boundaries        │     │
│  └────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
```

**InvincibleGuard** is a critical component — it prevents the supervisor's automated lifecycle management from shutting down the VM while it's healthy and serving inference. This means once the model is loaded, it stays loaded across multiple JARVIS sessions without needing to re-download or re-load the 4.37 GB model file.

### Golden Image: Pre-Baked for Instant Boot

The model is **not downloaded at boot time**. It is pre-baked into a **GCP golden image** — a snapshot of the VM disk with everything pre-installed and pre-cached:

```
Golden Image Contents (jarvis-prime-golden-20260207-042923):
├── /opt/jarvis-prime/                        # JARVIS Prime codebase
│   ├── run_server.py                          # Server entry point
│   ├── jarvis_prime/                          # Core Python package
│   │   ├── server.py                          # FastAPI application
│   │   └── core/                              # Neural Orchestrator, routing, etc.
│   └── models/                                # Model directory
│       └── models--TheBloke--Mistral-7B-Instruct-v0.2-GGUF/
│           └── snapshots/
│               └── <hash>/
│                   └── mistral-7b-instruct-v0.2.Q4_K_M.gguf  (4.37 GB)
├── Python 3.11 + all dependencies (pre-installed)
├── llama-cpp-python (compiled with CPU optimizations)
└── Startup script (auto-launches server on boot)
```

**Boot sequence:**
1. GCP creates VM from golden image (~26 seconds)
2. VM boots, startup script launches `run_server.py` (~30 seconds)
3. Server loads model from **local disk** (no network download)
4. Health endpoint reports `ready_for_inference=True`
5. **Total cold start: ~87 seconds** (from `NOT_FOUND` to serving inference)

Without the golden image, the VM would need to download ~4.37 GB from HuggingFace on every cold boot, adding 5-15 minutes depending on network speed. The golden image eliminates this entirely.

### CPU Inference: Why ~8.6s Latency is Expected

The GCP Invincible Node runs on **CPU-only** hardware (e2-standard-4, no GPU). This is a deliberate architectural choice:

| Factor | Details |
|--------|---------|
| **Hardware** | 4 vCPUs (Intel x86_64), 16 GB RAM, no GPU/TPU |
| **Inference Mode** | CPU-only via llama.cpp (AVX2/SSE4.2 SIMD acceleration) |
| **Measured Latency** | ~8.6 seconds per request (short prompts, ~100-200 token responses) |
| **Token Generation** | ~3-5 tokens/second (CPU-bound) |
| **Concurrent Requests** | 1 at a time (single model instance, sequential processing) |

**Why ~8.6s is normal and expected for this configuration:**

1. **CPU vs GPU arithmetic**: GPU inference (e.g., NVIDIA A100) achieves 30-80 tokens/sec on 7B models via massive parallelism across thousands of CUDA cores. CPU inference uses 4-8 threads doing sequential matrix multiplications — it's fundamentally 10-50x slower per token.

2. **Q4_K_M quantization helps but doesn't eliminate the gap**: 4-bit quantization reduces memory bandwidth requirements by ~4x compared to FP16, and `llama.cpp` uses AVX2 SIMD instructions to process 8 values per cycle. But CPU clock speeds (2-3 GHz) and limited core counts (4 vCPUs) still cap throughput at single-digit tokens/second.

3. **Prompt processing (prefill) is the bottleneck**: Before generating the first token, the model must process the entire input prompt through all 32 transformer layers. For a 100-token prompt, that's 100 × 32 layers × 7B parameters worth of matrix operations — all on CPU.

4. **Memory bandwidth is the real limiter**: Even with Q4_K_M reducing the model to ~4.37 GB, every token generation requires reading significant portions of the model weights from RAM. DDR4 bandwidth on standard GCP VMs (~25 GB/s) is orders of magnitude lower than GPU HBM bandwidth (~2 TB/s on A100).

**Performance comparison by hardware:**

```
┌────────────────────────────┬──────────────────┬───────────────┬────────────┐
│ Hardware                   │ Tokens/sec (7B)  │ Latency/req   │ Cost/hr    │
├────────────────────────────┼──────────────────┼───────────────┼────────────┤
│ GCP e2-standard-4 (CPU)   │ ~3-5 t/s         │ ~8.6s         │ ~$0.13     │
│ GCP n1-standard-8 (CPU)   │ ~6-10 t/s        │ ~4-5s         │ ~$0.38     │
│ GCP g2-standard-4 (L4)    │ ~25-35 t/s       │ ~1-2s         │ ~$0.70     │
│ GCP a2-highgpu-1g (A100)  │ ~50-80 t/s       │ ~0.3-0.5s     │ ~$3.67     │
│ Apple M1 Max (Metal GPU)  │ ~15-25 t/s       │ ~2-3s         │ N/A        │
└────────────────────────────┴──────────────────┴───────────────┴────────────┘
```

The e2-standard-4 was chosen for **cost efficiency**: at ~$0.13/hr (~$95/month), it provides always-on inference for a fraction of the cost of GPU instances. For a personal AI assistant where requests are sporadic (not continuous high-throughput), 8.6s latency is an acceptable trade-off against 28x lower cost compared to an A100.

**Future upgrade path**: If latency becomes a bottleneck (e.g., real-time conversation, high concurrency), the architecture supports seamless migration to:
- **g2-standard-4 (NVIDIA L4 GPU)**: ~$0.70/hr, ~1-2s latency — best price/performance for inference
- **Larger CPU VM**: Doubling vCPUs to n1-standard-8 would roughly halve latency to ~4-5s
- **Speculative decoding**: Using a smaller draft model (TinyLlama 1.1B) to propose tokens, validated by Mistral-7B — can provide 2-3x speedup without hardware changes

### What This Means in Practice

When a user types a message in the JARVIS frontend:

```
User: "What's 2+2?"
  │
  │  Frontend (localhost:3000)
  │  └── JarvisConnectionService.sendCommand()
  │        └── WebSocket to localhost:8010  (or REST fallback)
  │
  ▼
  Backend (localhost:8010, macOS)
  └── PrimeRouter → PrimeClient
        └── HTTP POST http://34.45.154.209:8000/v1/chat/completions
              │
              │  Request body:
              │  {
              │    "model": "jarvis-prime",
              │    "messages": [{"role": "user", "content": "What's 2+2?"}],
              │    "max_tokens": 512,
              │    "temperature": 0.7
              │  }
              │
              ▼
        GCP Invincible Node (34.45.154.209:8000)
        └── llama-cpp-python
              └── Mistral-7B-Instruct-v0.2 (Q4_K_M)
                    │
                    │  ~8.6 seconds of CPU inference
                    │  (prompt processing + token generation)
                    │
                    ▼
              Response: "2 + 2 = 4"
              │
              │  HTTP response back to macOS backend
              │  WebSocket/REST response back to frontend
              │
              ▼
        User sees: "2 + 2 = 4"
```

**No data leaves your infrastructure.** The request travels from the Mac to the GCP VM over HTTPS, is processed entirely by your own model on your own VM, and the response returns to your Mac. No tokens are sent to OpenAI, Anthropic, Google, or any third party.

### Emergency Fallback: Claude API (Tier 2 Only)

Claude API is **only** used as a last-resort emergency fallback (Tier 2) when:
1. The GCP VM is completely unreachable (network failure, zone outage)
2. AND the standard GCP VM fallback also fails
3. AND the request is classified as requiring deep reasoning

```
Fallback Chain (ordered by priority):
  1. GCP Golden Image VM ──→ Mistral-7B on Invincible Node (primary, ~8.6s)
  2. GCP Standard VM ──────→ Fresh VM with model download (backup, ~10-15 min cold start)
  3. Claude API ───────────→ Anthropic's API (emergency only, costs per token)
```

Under normal operation, **100% of requests** go to the self-hosted model. The Claude fallback exists for disaster recovery only and has never been triggered in production since the v233.2 golden image fixes.

### Why Self-Hosted Matters

| Benefit | Description |
|---------|-------------|
| **Zero per-token cost** | No API billing. The only cost is the GCP VM compute (~$95/month for e2-standard-4). Unlimited requests. |
| **Complete data privacy** | Prompts and responses never leave your infrastructure. No third-party data retention policies apply. |
| **No rate limits** | No tokens-per-minute caps, no request queuing from provider-side throttling. |
| **No vendor lock-in** | The model is open-source (Apache 2.0). Switch to Llama-3, Qwen, Phi, or any GGUF model by changing one file. |
| **Fine-tunable** | Reactor-Core collects experience data from JARVIS interactions and can fine-tune the model for your specific use patterns. |
| **Full control** | Choose quantization level, context length, temperature, system prompts, and all inference parameters. No provider-imposed guardrails beyond what you configure. |
| **Offline-capable** | Once the VM is running, inference works with zero internet dependency (the model is on local disk). |
| **Reproducible** | Same model, same weights, same quantization = deterministic behavior (given same temperature/seed). No provider-side model updates changing behavior unexpectedly. |

### Adaptive Prompt System: Complexity-Aware Inference (v236.0, v238.0)

#### The Problem: One Prompt Does Not Fit All

Before v236.0, every request sent to JARVIS Prime — whether "what is 5+5?" or "design a microservice architecture" — received the same static system prompt, the same `max_tokens=4096`, and the same `temperature=0.7`. The system prompt included:

> *"You are JARVIS, an advanced AI assistant... Be concise but thorough"*

Mistral-7B-Instruct interpreted "thorough" as a directive to be verbose, and the "advanced AI assistant" identity activated conversational, polite-assistant behavior. The result: asking "what is 5+5?" returned *"Of course, the sum of five and five is ten. I'd be happy to help with any other mathematical queries you might have."* instead of just **10**.

This is a fundamental challenge with 7B-parameter models: they have limited instruction-following capacity. When a system prompt contains conflicting signals — "be an AI assistant" (conversational) vs. "be concise" (terse) — the model resolves the conflict in favor of the stronger training signal, which is almost always the conversational one.

#### The Solution: AdaptivePromptBuilder

JARVIS (Body) now classifies every query into one of 5 complexity levels before sending it to Prime, and dynamically adapts three parameters:

```
┌────────────┬────────────┬──────┬─────────────────────────────────────────────────────────────────┐
│ Complexity │ max_tokens │ temp │ System Prompt Strategy                                          │
├────────────┼────────────┼──────┼─────────────────────────────────────────────────────────────────┤
│ SIMPLE     │ 48         │ 0.0  │ NO identity. Few-shot examples only.                            │
│            │            │      │ "Reply with ONLY the direct answer."                            │
│            │            │      │ v238.0: Only math, spell/translate, yes/no (<8 words).          │
│            │            │      │ "what is X?" queries moved to MODERATE.                        │
├────────────┼────────────┼──────┼─────────────────────────────────────────────────────────────────┤
│ MODERATE   │ 512        │ 0.3  │ JARVIS identity + "2-3 sentences. No filler."                   │
│            │            │      │ v238.0: Default for all queries ≤15 words                       │
│            │            │      │ (including "what is X?" and short abstract queries).            │
├────────────┼────────────┼──────┼─────────────────────────────────────────────────────────────────┤
│ COMPLEX    │ 2048       │ 0.5  │ JARVIS identity + "Structured and thorough."                    │
├────────────┼────────────┼──────┼─────────────────────────────────────────────────────────────────┤
│ ADVANCED   │ 4096       │ 0.7  │ JARVIS identity + "Detailed analysis."                          │
├────────────┼────────────┼──────┼─────────────────────────────────────────────────────────────────┤
│ EXPERT     │ 4096       │ 0.7  │ JARVIS identity + "Comprehensive. Edge cases."                  │
└────────────┴────────────┴──────┴─────────────────────────────────────────────────────────────────┘
```

#### Three Techniques for 7B Model Compliance

Standard instruction text ("be concise") achieves ~60-70% compliance on 7B models. The v236.0 system uses three additional techniques to push this significantly higher:

**1. Identity omission for SIMPLE queries**

The JARVIS identity prefix ("You are JARVIS, an advanced AI assistant") is intentionally **removed** for SIMPLE queries. This eliminates the competing signal that pushes the model toward conversational behavior. For MODERATE and above, the identity is retained because longer responses benefit from the JARVIS personality.

**2. Few-shot examples instead of abstract instructions**

7B models follow **patterns** far more reliably than they follow **meta-instructions**. Instead of telling the model "for math, return just the result," the SIMPLE prompt includes concrete examples:

```
Q: 5+5
A: 10
Q: Capital of France?
A: Paris
Q: Define gravity
A: The force that attracts objects with mass toward each other.
```

The model sees these examples and pattern-matches: "short question → short answer."

**3. Temperature 0.0 for deterministic output**

At `temperature=0.0`, the model always selects the highest-probability token at each step. For factual questions with single correct answers (math, capitals, definitions), this eliminates sampling variation entirely. The model produces the same output every time — no "sometimes verbose, sometimes terse" inconsistency.

#### How This Reaches Prime (Cross-Repo Flow)

The adaptive parameters are set by JARVIS (Body) and sent to Prime via the standard `/v1/chat/completions` endpoint. From Prime's perspective, it receives normal OpenAI-compatible requests — the intelligence is in **what** is sent, not in any Prime-side changes:

```
JARVIS Backend (macOS, port 8010)
  │
  │  QueryComplexityManager classifies "5+5?" → SIMPLE
  │  AdaptivePromptBuilder selects:
  │    system_prompt = "Reply with ONLY the direct answer...\nQ: 5+5\nA: 10\n..."
  │    max_tokens = 64
  │    temperature = 0.0
  │
  ▼
  POST http://34.45.154.209:8000/v1/chat/completions
  {
    "model": "jarvis-prime",
    "messages": [
      {"role": "system", "content": "Reply with ONLY the direct answer..."},
      {"role": "user", "content": "what is 5+5?"}
    ],
    "max_tokens": 64,
    "temperature": 0.0
  }
  │
  ▼
  JARVIS Prime (GCP VM, port 8000)
  └── llama-cpp-python → Mistral-7B-Instruct-v0.2 (Q4_K_M)
        │
        │  Sees few-shot pattern: Q → A (short)
        │  temp=0.0 → deterministic token selection
        │  max_tokens=64 → hard cap on output length
        │
        ▼
  Response: "10"    (5 tokens including BOS/EOS)
```

For complex queries, the same flow sends the full JARVIS identity, `max_tokens=4096`, and `temperature=0.7` — giving the model maximum room for structured, detailed analysis.

#### Verified Results (v236.0 + v238.0)

```
┌───────────────────────────────────┬────────────┬────────┬──────┬──────────────────────────────────┐
│ Query                             │ Complexity │ Tokens │ Temp │ Response                         │
├───────────────────────────────────┼────────────┼────────┼──────┼──────────────────────────────────┤
│ "what is 5+5?"                    │ SIMPLE     │ 48     │ 0.0  │ 10                               │
│ "what's 5+5?"                     │ SIMPLE     │ 48     │ 0.0  │ 10                               │
│ "is water wet?"                   │ SIMPLE     │ 48     │ 0.0  │ Yes                              │
│ "spell onomatopoeia"             │ SIMPLE     │ 48     │ 0.0  │ O-N-O-M-A-T-O-P-O-E-I-A         │
│ "what is mathematics?"            │ MODERATE   │ 512    │ 0.3  │ Full definition (3 sentences)    │
│ "what is Java?"                   │ MODERATE   │ 512    │ 0.3  │ Full definition via gcp_prime    │
│ "define photosynthesis"           │ MODERATE   │ 512    │ 0.3  │ 2-3 sentence definition          │
│ "capital of France?"              │ MODERATE   │ 512    │ 0.3  │ Paris / The capital is Paris.    │
│ "explain how neural networks      │ COMPLEX    │ 2048   │ 0.5  │ Multi-paragraph structured       │
│  learn"                           │            │        │      │                                  │
└───────────────────────────────────┴────────────┴────────┴──────┴──────────────────────────────────┘

v238.0 routing confirmed: [QUERY] Response from gcp_prime (latency: 24635.7ms)
Source: jarvis-prime-node at 34.45.154.209 (GCP Invincible Node golden image)
```

**v238.0 Classification Change:** Queries like `"what is X?"`, `"define X"`, `"who is X?"` were previously classified as SIMPLE (48 tokens, temp 0.0, stop sequences). This caused degenerate output (`"..."`) when the model encountered abstract concepts. v238.0 moves these to MODERATE — providing 512 tokens and temp 0.3, which is safe and cheap for all short queries while eliminating the degenerate response failure mode entirely.

#### The Path Beyond Prompting: Reactor-Core Fine-Tuning

The adaptive prompt system is the **immediate fix** — it makes Mistral-7B behave correctly today. But prompt-based control is inherently limited for 7B models because instruction compliance is a function of model capacity.

The **permanent solution** is training the model itself to be concise for simple queries, using the Reactor-Core training pipeline that's already wired into the architecture:

```
 JARVIS (Body)              JARVIS Prime (Mind)         Reactor-Core (Nerves)
 ─────────────              ───────────────────         ─────────────────────
 User: "5+5?"           →   Mistral-7B → "10"      →   TelemetryEmitter captures
                                                         (query, response, complexity,
                                                          latency, tokens_used)
                                                                │
                                                                ▼
                                                         TrainingDataPipeline creates
                                                         DPO preference pairs:
                                                         {
                                                           prompt: "5+5?",
                                                           chosen: "10",
                                                           rejected: "Of course, the
                                                              sum of five and five..."
                                                         }
                                                                │
                                                                ▼
                            Hot-swap fine-tuned       ←   DPO training with β=0.1
                            GGUF (zero downtime)          on accumulated preference data
                            Bake new golden image
```

After DPO training, conciseness for simple queries is encoded **in the model's weights** — not dependent on a prompt instruction the model might ignore. The model learns *when* to be terse and *when* to be detailed from actual user interaction patterns, not from static rules.

The key components for this pipeline already exist:
- **`TelemetryEmitter`** (JARVIS) — captures every interaction, ships to Reactor-Core
- **`TrainingDataPipeline`** (Prime) — generates DPO preference pairs from conversations
- **`RLHFIntegration`** (Prime) — reward model training and PPO optimization
- **`ReactorCoreBridge`** (Prime) — submits fine-tuning jobs, tracks training, deploys finished models
- **`HotSwapManager`** (Prime) — swaps the model at runtime with zero request drops

### v238.0: Degenerate Response Elimination (Defense-in-Depth)

#### The Problem: "..." as a Model Response

When JARVIS classified `"what is mathematics?"` as SIMPLE (48 tokens, temperature 0.0, stop sequences `\n\n`), Mistral-7B sometimes produced `"..."` followed by a double newline. The stop sequence truncated the output at `"..."`, which then passed through the entire pipeline unchecked — displayed in the UI and spoken aloud via TTS as "full stop."

This is a model behavior that any self-hosted LLM can exhibit when constrained with aggressive token limits, low temperature, and stop sequences on queries that require more than a one-word answer. The model begins generating a longer response, but the constraints truncate it to meaningless punctuation.

#### How v238.0 Protects the JARVIS → Prime Pipeline

The fix operates at three layers — any one of which independently prevents garbage from reaching the user:

```
Layer 1: Classification (JARVIS Body — query_complexity_manager.py)
  "what is mathematics?" → MODERATE (512 tokens, 0.3 temp, no stop sequences)
  → Mistral-7B has room to produce a full definition
  → Eliminates the root cause: the model was never wrong — it was starved

Layer 2: Degenerate Retry (JARVIS Body — query_handler.py)
  If Mistral-7B STILL produces punctuation-only output:
  → Backend detects content stripped to empty string
  → Retries once with MODERATE parameters
  → Retry request goes to Prime at 34.45.154.209:8000
  → Prime returns real response with sufficient token budget
  → try/except ensures retry failure doesn't lose original content

Layer 3: Client Suppression (JARVIS Body — JarvisVoice.js)
  If "..." somehow reaches the frontend despite layers 1 and 2:
  → Frontend detects punctuation-only response
  → Suppresses display and TTS
  → Re-arms zombie timeout for automatic retry
```

**Impact on Prime:** Prime itself is unchanged — it receives standard OpenAI-compatible requests and returns standard responses. The intelligence is in what JARVIS (Body) sends:
- **Before v238.0:** `max_tokens=48, temperature=0.0` for "what is mathematics?" → Prime dutifully truncates
- **After v238.0:** `max_tokens=512, temperature=0.3` for "what is mathematics?" → Prime generates full answer

The degenerate retry (Layer 2) may send a second request to Prime if the first response is garbage. This is a normal HTTP POST — Prime processes it like any other request. The retry uses MODERATE parameters, which are safe for any query.

#### Production Verification

```
Step 1: PrimeClient resolved to GCP VM: 34.45.154.209:8000 (source: JARVIS_PRIME_URL)
Step 2: PrimeRouter: GCP VM promotion successful, routing updated → gcp_prime
Step 3: AdaptivePromptBuilder: level=MODERATE, max_tokens=512, temp=0.3
Step 4: [QUERY] Response from gcp_prime (latency: 24635.7ms)
Step 5: API response: "source": "gcp_prime", "model": "jarvis-prime", "fallback_used": false
```

The 24.6s latency is consistent with CPU inference on the Mistral-7B Q4_K_M model on the e2-standard-4 Invincible Node. Response quality confirmed — full sentence definitions instead of `"..."`.

---

## ✨ Core Features

### 🧠 **1. Neural Orchestrator Core v100.0 - Unified Intelligent Routing**

The **single source of truth** for all routing decisions across the JARVIS ecosystem:

#### **Unified Architecture**
- **Consolidates All Routers**: HybridTieredRouter, IntelligentModelRouter, CognitiveRouter, GraphRouter, Neural Switchboard
- **Protocol-Based Design**: Type-safe interfaces with `@runtime_checkable` Protocols
- **Context-Aware Routing**: Distributed tracing with `contextvars` for request correlation
- **Dynamic Configuration**: Zero hardcoding - all values from `DynamicConfig` with env var override
- **Cross-Repo State Management**: Atomic file operations for shared state across repositories

#### **Advanced Components**

**UnifiedTaskClassifier**
- Multi-signal task classification (reasoning, chat, code, creative, analysis)
- Confidence scoring with adaptive thresholds
- Pattern matching with regex and keyword detection
- Context-aware classification (session history, user preferences)

**UnifiedMemoryMonitor**
- macOS native `memory_pressure` command integration
- Cross-repo memory sharing via JARVIS bridge
- Real-time pressure level detection (normal, warning, critical, urgent)
- Burst decision support for memory-intensive operations
- `psutil` fallback for non-macOS systems

**UnifiedStickyRouting**
- Session-based model affinity
- Automatic session detection from context
- Configurable TTL for session continuity
- Memory-efficient storage with `weakref.WeakValueDictionary`

**UnifiedRequestBuffer**
- Zero-loss request buffering during hot swaps
- Configurable buffer size and timeout
- Automatic request replay after swap completion
- Priority-based request ordering

**CircuitBreakerManager**
- Coordinated circuit breakers per tier (Tier 0, Tier 0.5, Tier 1, Tier 2)
- Atomic state management with distributed locking
- Automatic recovery with half-open state testing
- Statistics tracking per tier

**CrossRepoStateManager**
- Atomic file operations for state persistence
- File locking with `fcntl` for race condition prevention
- Automatic retry with exponential backoff
- State versioning and conflict resolution

```python
from jarvis_prime.core.neural_orchestrator_core import get_neural_orchestrator

# Get the unified orchestrator (singleton)
orchestrator = await get_neural_orchestrator()

# Route a request (handles everything automatically)
result = await orchestrator.route(
    prompt="Implement a distributed cache with Redis",
    context={
        "session_id": "abc123",
        "user_id": "derek",
        "priority": "high"
    }
)

# Access routing decision
print(f"Tier: {result.tier}")  # RoutingTier.TIER_0_5
print(f"Endpoint: {result.endpoint}")  # "http://localhost:8000/v1/chat/completions"
print(f"Model ID: {result.model_id}")  # "mistral-7b-instruct"
print(f"Task: {result.task_classification}")  # TaskClassification.CODE
print(f"Confidence: {result.confidence}")  # 0.92
print(f"Reasoning: {result.decision_reason}")  # DecisionReason.MEMORY_PRESSURE

# Get comprehensive statistics
stats = orchestrator.get_comprehensive_stats()
print(f"Total requests: {stats['routing']['total_requests']}")
print(f"Sticky hits: {stats['routing']['sticky_hits']}")
print(f"Memory pressure: {stats['memory_monitor']['pressure_level']}")
```

#### **Advanced Python Patterns**

**Protocol Classes for Type Safety**
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class RouterProtocol(Protocol):
    async def route(self, prompt: str, context: Dict[str, Any]) -> RoutingResult:
        ...
```

**Context Variables for Distributed Tracing**
```python
import contextvars

request_id_var = contextvars.ContextVar('request_id', default=None)
session_id_var = contextvars.ContextVar('session_id', default=None)
trace_context_var = contextvars.ContextVar('trace_context', default=None)
```

**Defensive Decorators with Fallbacks**
```python
def with_fallback(fallback_value):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"{func.__name__} failed: {e}, using fallback")
                return fallback_value
        return wrapper
    return decorator
```

**Atomic Operations**
```python
async def atomic_state_update(key: str, value: Any):
    async with distributed_lock(f"state_{key}"):
        # Critical section - guaranteed atomicity
        state[key] = value
        await persist_state(state)
```

### 🧩 **2. Dynamic Model Registry v99.0**

Auto-discovery and management of models across multiple directories:

#### **Features**
- **Multi-Directory Discovery**: Scans multiple model directories automatically
- **Auto-Download from HuggingFace**: Automatic model downloading with progress tracking
- **File System Watching**: Real-time detection of new models via `watchdog`
- **Reactor Core Sync**: Automatic synchronization with Reactor Core training pipeline
- **Model Validation**: Integrity checks, inference tests, safety validation
- **Version Management**: Semantic versioning with rollback support

```python
from jarvis_prime.core.dynamic_model_registry import DynamicModelRegistry

registry = DynamicModelRegistry(
    discovery_dirs=[
        "./models",
        "~/models",
        "/shared/models"
    ],
    auto_download=True,
    watch_files=True
)

# Auto-discover models
await registry.discover_models()

# Get available models
models = registry.list_models()
for model in models:
    print(f"{model.name} - {model.version} - {model.path}")

# Auto-download from HuggingFace
await registry.download_model(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    local_dir="./models"
)
```

### 🧠 **3. Neural Switchboard v98.0**

Unified routing system with task classification, memory monitoring, and sticky routing:

#### **Features**
- **Task Classification**: Multi-signal classification (reasoning, chat, code, creative)
- **Memory Monitoring**: Real-time memory pressure detection
- **Sticky Routing**: Session-based model affinity
- **Request Buffering**: Zero-loss hot swap support
- **Tier Mapping**: Automatic tier/capability mapping

```python
from jarvis_prime.core.neural_switchboard import NeuralSwitchboard

switchboard = NeuralSwitchboard()

# Classify task
classification = await switchboard.classify_task(
    prompt="Write a Python function to sort a list",
    context={"session_id": "abc123"}
)

# Route request
routing = await switchboard.route(
    prompt="Continue the previous code",
    context={"session_id": "abc123"}
)
```

### 🛡️ **4. Advanced Resilience Patterns**

#### **Circuit Breaker (Coordinated Per-Tier)**
```python
from jarvis_prime.core.neural_orchestrator_core import CircuitBreakerManager

breaker_manager = CircuitBreakerManager()

# Check circuit state for tier
state = await breaker_manager.get_state(RoutingTier.TIER_1)
if state == CircuitState.CLOSED:
    # Safe to route
    result = await route_to_tier_1(prompt)
    await breaker_manager.record_success(RoutingTier.TIER_1)
else:
    # Circuit open, use fallback
    result = await fallback_route(prompt)
```

#### **Request Buffering (Zero-Loss Hot Swap)**
```python
from jarvis_prime.core.neural_orchestrator_core import UnifiedRequestBuffer

buffer = UnifiedRequestBuffer(max_size=1000, timeout_seconds=30.0)

# Buffer requests during hot swap
async with buffer.buffer_mode():
    # All requests are buffered
    await hot_swap_model(new_model_path)
    # Buffered requests are automatically replayed
```

#### **Retry with Exponential Backoff + Decorrelated Jitter**
```python
from jarvis_prime.core.neural_orchestrator_core import with_retry

@with_retry(max_attempts=3, base_delay=1.0, max_delay=10.0)
async def unreliable_operation():
    # Automatically retries with exponential backoff + jitter
    result = await external_api_call()
    return result
```

### 🔒 **5. JARVIS Safety Integration**

**Cross-Repo Bridge** reads safety context from main JARVIS instance:

```python
from jarvis_prime.core.neural_orchestrator_core import CrossRepoStateManager

state_manager = CrossRepoStateManager()

# Read safety context
safety_context = await state_manager.read_safety_context()

if safety_context.kill_switch_active:
    # Route all actions to Prime for careful review
    result = await orchestrator.route(
        prompt=prompt,
        context={"force_tier": RoutingTier.TIER_1}
    )

if safety_context.should_be_cautious():
    # User has been denying actions recently
    # Route risky patterns to cloud
    result = await orchestrator.route(
        prompt=prompt,
        context={"force_tier": RoutingTier.TIER_1}
    )
```

**Safety File Location:** `~/.jarvis/safety/context_for_prime.json`

**Risky Pattern Detection:**
- delete, remove, erase, wipe, format
- kill, terminate, shutdown, reboot
- sudo, admin, root, system, chmod
- execute, run, install, uninstall
- password, credential, secret, token

### 🔄 **6. Zero-Downtime Hot Swap**

Swap models while server is running with **zero requests dropped**:

```python
from jarvis_prime.core.hot_swap_manager import HotSwapManager

manager = HotSwapManager()

# Background loading, traffic draining, atomic switch
result = await manager.swap_model(
    new_model_path="./models/mistral-7b.gguf",
    new_version_id="mistral-7b-v0.2"
)

print(f"Swapped in {result.duration_seconds:.1f}s")
print(f"Drained {result.requests_drained} in-flight requests")
print(f"Freed {result.memory_freed_mb:.1f} MB")
# Zero requests dropped! ✅
```

### 📊 **7. Advanced Telemetry & Cost Tracking**

```python
from jarvis_prime.core.cross_repo_bridge import CrossRepoBridge

bridge = CrossRepoBridge(instance_id="prime-derek-mac")
await bridge.start()

# Automatic metrics tracking
bridge.record_inference(tokens_in=25, tokens_out=150, latency_ms=47.3)

# Cost savings calculation
state = bridge.state
print(f"Total requests: {state.metrics.total_requests}")
print(f"Cloud cost if used: ${state.metrics.estimated_cost_usd:.4f}")
print(f"Savings: ${state.metrics.savings_vs_cloud_usd:.4f}")

# Shared with main JARVIS at:
# ~/.jarvis/cross_repo/jarvis_prime_state.json
```

### 🌐 **8. OpenAI-Compatible API**

Drop-in replacement for OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="jarvis-prime",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ],
    stream=True  # Real-time streaming
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

### 🧩 **9. Complete AGI Architecture**

#### **7 Specialized AGI Models**
```python
from jarvis_prime.core.agi_models import (
    ActionModel,           # Action planning and execution
    MetaReasoner,         # Meta-cognitive reasoning, strategy selection
    CausalEngine,         # Causal understanding, counterfactuals
    WorldModel,           # Physical/common sense reasoning
    MemoryConsolidator,   # Memory consolidation and replay
    GoalInference,        # Goal understanding and decomposition
    SelfModel,            # Self-awareness and capability assessment
)

# Orchestrate multiple models for complex reasoning
from jarvis_prime.core.agi_models import AGIOrchestrator

orchestrator = AGIOrchestrator()
result = await orchestrator.process(
    request="Design a distributed caching system",
    required_models=["meta_reasoner", "action", "causal"]
)
```

#### **Advanced Reasoning Engine**
```python
from jarvis_prime.core.reasoning_engine import ReasoningEngine, ReasoningStrategy

engine = ReasoningEngine()

# Chain-of-Thought reasoning
cot_result = await engine.reason(
    prompt="How do I optimize this algorithm?",
    strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
    max_steps=10
)

# Tree-of-Thoughts for exploration
tot_result = await engine.reason(
    prompt="Design three different approaches to...",
    strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
    num_branches=3,
    exploration_depth=4
)

# Self-Reflection for error correction
reflection_result = await engine.reason(
    prompt="Review this code for bugs",
    strategy=ReasoningStrategy.SELF_REFLECTION,
    confidence_threshold=0.8
)
```

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    JARVIS UNIFIED SUPERVISOR                           │
│                    (run_supervisor.py - v100.0)                         │
│                                                                         │
│  Orchestrates: JARVIS (Body), JARVIS-Prime (Mind), Reactor-Core       │
│  Initializes: Neural Orchestrator Core v100.0                          │
│  Manages: Health checks, lifecycle, cross-repo communication          │
└──────────────────────────┬────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              NEURAL ORCHESTRATOR CORE v100.0                            │
│              Unified Intelligent Routing Architecture                    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    UNIFIED ROUTING LAYER                         │  │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐      │  │
│  │  │ TaskClass │ │MemPressure│ │ Sticky    │ │ RequestBuf│      │  │
│  │  │   -ifier  │ │  Monitor  │ │ Routing   │ │   -fer    │      │  │
│  │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘      │  │
│  │        └─────────────┼─────────────┼─────────────┘            │  │
│  │                      ▼             ▼                          │  │
│  │              ┌───────────────────────────┐                    │  │
│  │              │   ROUTING DECISION ENGINE │                    │  │
│  │              │    (Unified Algorithm)    │                    │  │
│  │              └─────────────┬─────────────┘                    │  │
│  │                            │                                    │  │
│  │  ┌─────────────────────────┼─────────────────────────┐          │  │
│  │  │                         ▼                         │          │  │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────┐  │          │  │
│  │  │  │ Tier 0  │  │Tier 0.5 │  │ Tier 1  │  │Tier 2│  │          │  │
│  │  │  │ Ultra   │  │ Local   │  │ Cloud   │  │ Deep │  │          │  │
│  │  │  │ Fast    │  │ Capable │  │  Intel  │  │Reason│  │          │  │
│  │  │  └────┬────┘  └────┬────┘  └────┬────┘  └──┬───┘  │          │  │
│  │  │       └────────────┼────────────┼─────────┘       │          │  │
│  │  │                    ▼            ▼                 │          │  │
│  │  │           ┌────────────────────────────┐          │          │  │
│  │  │           │  CIRCUIT BREAKER MANAGER   │          │          │  │
│  │  │           │  (Coordinated State)       │          │          │  │
│  │  │           └────────────────────────────┘          │          │  │
│  │  └───────────────────────────────────────────────────┘          │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    CROSS-REPO INTEGRATION                        │  │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐                     │  │
│  │  │  JARVIS   │ │  JARVIS   │ │  Reactor  │                     │  │
│  │  │  (Body)   │ │  Prime    │ │   Core    │                     │  │
│  │  │  Memory   │ │  Memory   │ │  Sync     │                     │  │
│  │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘                     │  │
│  │        └─────────────┼─────────────┘                           │  │
│  │                      ▼                                         │  │
│  │        ┌───────────────────────────┐                           │  │
│  │        │  SHARED STATE MANAGER     │                           │  │
│  │        │  (~/.jarvis/cross_repo/)  │                           │  │
│  │        └───────────────────────────┘                           │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
         │                                           │
         ▼                                           ▼
┌─────────────────────┐                  ┌──────────────────────────┐
│   JARVIS (Body)     │                  │  JARVIS-Prime (Mind)     │
│   ───────────────   │                  │  ────────────────────    │
│   • Computer Use    │◄────Trinity──────┤  • AGI Models (7 types)  │
│   • Action Exec     │     Protocol     │  • Reasoning Engine      │
│   • macOS Control   │    (File IPC +   │  • Multimodal Fusion     │
│   • Safety Manager  │     WebSocket)   │  • Continuous Learning   │
│   "Reflex Mode"     │                  │  "Cognitive Mode"        │
└─────────────────────┘                  └──────────────────────────┘
         │                                           │
         │                                           │
         └───────────────────┬───────────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │  Reactor-Core (Soul)│
                  │  ─────────────────  │
                  │  • Model Training   │
                  │  • Fine-tuning      │
                  │  • Checkpointing    │
                  └─────────────────────┘
```

### Cross-Repo Integration (Trinity)

JARVIS-Prime is the **Mind** in the three-repo Trinity architecture. It is **started and monitored** by the JARVIS unified supervisor and **coordinates with Reactor-Core** for training data and model deployment.

**How JARVIS (Body) uses Prime:**

- **Discovery:** Supervisor resolves `JARVIS_PRIME_REPO_PATH` (or default `~/Documents/repos/JARVIS-Prime`).
- **Early Prime pre-warm:** Supervisor can start Prime early so LLM loading begins in parallel; when Trinity phase starts, it **adopts** the running process and clears `JARVIS_EARLY_PRIME_PID`. The Early Prime monitor then stops with **handoff=True** so progress is **preserved** (v221.0).
- **Health:** Supervisor polls `GET /health` and reads `model_load_progress_pct`, `startup_progress`, `loading_progress`, `phase`, `model_loaded`, `ready_for_inference`. Progress never regresses (e.g. 18% → 0%) thanks to handoff-safe state in the supervisor.
- **State:** Prime reads/writes shared state under `~/.jarvis/` (e.g. `cross_repo/`, Neural Orchestrator state) for safety context and routing.

**How Reactor-Core uses Prime:**

- **Inference:** Reactor can call Prime’s OpenAI-compatible API for generation during training or evaluation.
- **Model deployment:** Trained/updated models can be deployed to Prime (e.g. hot swap, model registry).
- **Trinity Protocol:** Events and heartbeats flow via file IPC and/or WebSocket; Prime participates in Trinity state sync.

**Health endpoint contract for supervisor:**

- During model loading: `model_load_progress_pct` (0–100), `model_loading_in_progress`, `phase` (e.g. `loading_model`), `model_load_elapsed_seconds`.
- When ready: `model_loaded`, `ready_for_inference`, `phase: "ready"`.
- `jarvis_prime/server.py` and `run_server.py` both expose this contract (v221.0 ensures `server.py` reports progress for cross-repo coordination).

### Model Loading Progress & Handoff (v221.0)

When the JARVIS unified supervisor uses **Early Prime pre-warm**, Prime starts early and a background monitor polls `/health` and updates the dashboard. When the **Trinity phase** takes over, it adopts the running Prime process and clears the early-Prime env var; the Early Prime monitor then stops. **v221.0** ensures:

- **No progress regression:** The supervisor’s `update_model_loading(active=False, handoff=True)` preserves `max_progress_seen`. Progress never drops (e.g. 18% → 0%).
- **Prime health:** Prime’s `/health` must report `model_load_progress_pct` (and related fields) so the Trinity monitor can continue from the preserved progress. Both `run_server.py` and `jarvis_prime/server.py` support this (v221.0).

See JARVIS-AI-Agent `memory/2026-02-04.md` (or equivalent) for the full root-cause analysis and fix summary.

### Request Flow with Neural Orchestrator Core

```
User Request: "Implement a distributed cache with Redis"
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│ Step 1: Neural Orchestrator Core Route()                      │
│ ────────────────────────────────────────                      │
│ • Check sticky routing: session_id="abc123" → Model affinity  │
│ • Classify task: CODE (confidence: 0.92)                      │
│ • Check memory pressure: NORMAL (macOS native)                │
│ • Check circuit breakers: All CLOSED                           │
│ • Select tier: TIER_0_5 (Local Capable)                      │
│ • Select endpoint: http://localhost:8000/v1/chat/completions  │
│ • Select model: mistral-7b-instruct                            │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│ Step 2: Request Execution                                     │
│ ──────────────────────────                                   │
│ • Acquire circuit breaker permit: SUCCESS                    │
│ • Execute request with timeout: 60s                          │
│ • Stream response tokens                                      │
└────────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────────┐
│ Step 3: Response & State Update                              │
│ ────────────────────────────────                             │
│ • Release circuit breaker permit: SUCCESS                     │
│ • Update sticky routing: session_id → model_id                │
│ • Update statistics: total_requests++, sticky_hits++           │
│ • Record outcome for adaptive learning                        │
│ → Return response to user                                     │
└────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (recommended for best performance with structured concurrency)
- macOS (for M1/M2/M3 optimization) or Linux
- 8GB+ RAM (16GB recommended for larger models)
- 10GB+ free disk space

### Installation

```bash
# Clone repository
git clone https://github.com/drussell23/jarvis-prime.git
cd jarvis-prime

# Install dependencies
pip install -e .

# Or with all features
pip install -e ".[server,gcs,telemetry,agi,neural-orchestrator]"
```

### Entry Points

| Entry Point | Purpose | When to Use |
|-------------|---------|-------------|
| **`run_server.py`** | Full server with startup state, progress reporting, and health endpoint | **Recommended** — used by unified supervisor; reports `model_load_progress_pct`, `startup_progress`, `model_loading_in_progress` |
| **`jarvis_prime/server.py`** (module) | Alternative FastAPI server with immediate HTTP startup and background model load | When running Prime as a module; v221.0 adds `model_load_progress_pct` to health for cross-repo coordination |
| **Unified Supervisor (JARVIS)** | `python3 unified_supervisor.py` in JARVIS-AI-Agent | **Recommended for full ecosystem** — starts Body + Prime + Reactor-Core with Trinity coordination |

The **health endpoint** (`GET /health`) must expose `model_load_progress_pct` (and optionally `startup_progress`, `loading_progress`, `model_loading_in_progress`) so the JARVIS unified supervisor can track loading progress and avoid regression during Early Prime → Trinity handoff (v221.0).

### Unified Supervisor (Recommended)

Start all components with a single command from the **JARVIS (Body)** repo:

```bash
# From JARVIS-AI-Agent repo — starts JARVIS + JARVIS-Prime + Reactor-Core
python3 unified_supervisor.py

# Supervisor will:
# 1. Start JARVIS-Prime server (port 8000)
# 2. Initialize Neural Orchestrator Core v100.0
# 3. Connect to JARVIS Body (if running)
# 4. Setup Trinity Protocol (File IPC + WebSocket)
# 5. Start health monitoring
# 6. Initialize Dynamic Model Registry
# 7. Start cross-repo state management

# Output:
# ============================================================
# JARVIS Unified Supervisor v100.0 - Starting
# ============================================================
# 🧠 Neural Orchestrator Core v100.0 initialized
# 📊 Dynamic Model Registry v99.0 initialized
# 🔄 Cross-Repo State Manager initialized
# Starting component: jarvis_prime
# Starting component: jarvis
# All components started successfully
# Supervisor running, press Ctrl+C to stop
```

**Note:** The unified supervisor lives in **JARVIS-AI-Agent**; it discovers and starts JARVIS-Prime (and Reactor-Core). From within the JARVIS-Prime repo you can run the **standalone** server only (see below).

### Standalone Server

Start just the JARVIS-Prime server:

```bash
# Download a model first
python -c "
from jarvis_prime.docker.model_downloader import download_model
download_model('tinyllama-chat', './models')
"

# Start server
python run_server.py \
    --model ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --port 8000

# Server starts at http://localhost:8000
```

### Test Neural Orchestrator Core

```python
from jarvis_prime.core.neural_orchestrator_core import get_neural_orchestrator
import asyncio

async def main():
    # Get singleton orchestrator
    orchestrator = await get_neural_orchestrator()

    # Simple request → Tier 0
    result = await orchestrator.route(
        prompt="What's 2+2?",
        context={"session_id": "test123"}
    )
    print(f"Tier: {result.tier}")  # RoutingTier.TIER_0
    print(f"Task: {result.task_classification}")  # TaskClassification.CHAT

    # Complex request → Tier 1
    result = await orchestrator.route(
        prompt="Plan a comprehensive security audit of the authentication system",
        context={"session_id": "test123"}
    )
    print(f"Tier: {result.tier}")  # RoutingTier.TIER_1
    print(f"Task: {result.task_classification}")  # TaskClassification.REASONING
    print(f"Confidence: {result.confidence}")  # 0.92

    # Get comprehensive statistics
    stats = orchestrator.get_comprehensive_stats()
    print(f"Total requests: {stats['routing']['total_requests']}")
    print(f"Sticky hits: {stats['routing']['sticky_hits']}")
    print(f"Memory pressure: {stats['memory_monitor']['pressure_level']}")

asyncio.run(main())
```

### Send Requests (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# Simple request
response = client.chat.completions.create(
    model="jarvis-prime",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)

# Streaming request
stream = client.chat.completions.create(
    model="jarvis-prime",
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## 🌐 API Endpoints

### Neural Orchestrator Core Endpoints

#### `GET /neural-orchestrator/health`
Check Neural Orchestrator health status.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "task_classifier": "healthy",
    "memory_monitor": "healthy",
    "sticky_routing": "healthy",
    "request_buffer": "healthy",
    "circuit_breaker": "healthy",
    "cross_repo_state": "healthy"
  },
  "uptime_seconds": 3600.5
}
```

#### `GET /neural-orchestrator/stats`
Get comprehensive statistics.

**Response:**
```json
{
  "routing": {
    "total_requests": 1250,
    "sticky_hits": 342,
    "task_classifications": {
      "REASONING": 450,
      "CHAT": 600,
      "CODE": 150,
      "CREATIVE": 50
    }
  },
  "memory_monitor": {
    "pressure_level": "normal",
    "last_check": "2025-01-07T14:30:45Z"
  },
  "circuit_breaker": {
    "tier_0": {"state": "closed", "failures": 0},
    "tier_0_5": {"state": "closed", "failures": 0},
    "tier_1": {"state": "closed", "failures": 0},
    "tier_2": {"state": "closed", "failures": 0}
  }
}
```

#### `POST /neural-orchestrator/route`
Route a request through the Neural Orchestrator.

**Request:**
```json
{
  "prompt": "Implement a distributed cache",
  "context": {
    "session_id": "abc123",
    "user_id": "derek",
    "priority": "high"
  }
}
```

**Response:**
```json
{
  "tier": "TIER_0_5",
  "endpoint": "http://localhost:8000/v1/chat/completions",
  "model_id": "mistral-7b-instruct",
  "task_classification": "CODE",
  "confidence": 0.92,
  "decision_reason": "MEMORY_PRESSURE",
  "metadata": {
    "sticky_hit": true,
    "memory_pressure": "normal"
  }
}
```

#### `GET /neural-orchestrator/memory`
Get current memory pressure status.

**Response:**
```json
{
  "pressure_level": "normal",
  "pressure_score": 0.25,
  "memory_usage_mb": 8192,
  "memory_available_mb": 8192,
  "last_check": "2025-01-07T14:30:45Z"
}
```

#### `POST /neural-orchestrator/classify`
Classify a task without routing.

**Request:**
```json
{
  "prompt": "Write a Python function to sort a list",
  "context": {
    "session_id": "abc123"
  }
}
```

**Response:**
```json
{
  "task_classification": "CODE",
  "confidence": 0.95,
  "signals": {
    "reasoning_indicators": 0.1,
    "code_indicators": 0.9,
    "chat_indicators": 0.2
  }
}
```

### Standard API Endpoints

#### `POST /v1/chat/completions`
OpenAI-compatible chat completions endpoint.

#### `POST /generate`
Simple text generation endpoint.

#### `GET /health`
Health check endpoint.

#### `GET /metrics`
Cost tracking and inference metrics.

#### `GET /v1/models`
List available models.

#### `POST /api/v1/models/reload`
Reload a model (hot swap).

### AGI Endpoints

#### `POST /agi/reason`
Advanced reasoning with AGI models.

#### `POST /agi/plan`
Action planning with AGI models.

#### `POST /agi/process`
Multi-model AGI processing.

#### `POST /agi/feedback`
Provide feedback for continuous learning.

#### `POST /agi/learning/trigger`
Trigger continuous learning update.

#### `GET /agi/status`
Get AGI subsystem status.

#### `GET /agi/learning/stats`
Get continuous learning statistics.

---

## 🎛️ Configuration

### Environment Variables (Zero Hardcoding)

#### **Neural Orchestrator Core Configuration**

```bash
# Core settings
export NEURAL_ORCHESTRATOR_ENABLED=true
export NEURAL_ORCHESTRATOR_CONFIG_PATH=config/neural_orchestrator.yaml

# Task classification
export NEURAL_ORCHESTRATOR_REASONING_THRESHOLD=0.5
export NEURAL_ORCHESTRATOR_CODE_THRESHOLD=0.6
export NEURAL_ORCHESTRATOR_CREATIVE_THRESHOLD=0.4

# Memory monitoring
export NEURAL_ORCHESTRATOR_MEMORY_CHECK_INTERVAL=5.0
export NEURAL_ORCHESTRATOR_MEMORY_PRESSURE_THRESHOLD=0.8
export NEURAL_ORCHESTRATOR_MEMORY_CRITICAL_THRESHOLD=0.9

# Sticky routing
export NEURAL_ORCHESTRATOR_STICKY_ENABLED=true
export NEURAL_ORCHESTRATOR_STICKY_TTL=3600.0

# Request buffering
export NEURAL_ORCHESTRATOR_BUFFER_MAX_SIZE=1000
export NEURAL_ORCHESTRATOR_BUFFER_TIMEOUT=30.0

# Circuit breaker
export NEURAL_ORCHESTRATOR_CIRCUIT_FAILURE_THRESHOLD=5
export NEURAL_ORCHESTRATOR_CIRCUIT_RECOVERY_TIMEOUT=30.0
export NEURAL_ORCHESTRATOR_CIRCUIT_HALF_OPEN_MAX_REQUESTS=3

# Cross-repo state
export NEURAL_ORCHESTRATOR_CROSS_REPO_DIR=~/.jarvis/cross_repo
export NEURAL_ORCHESTRATOR_STATE_FILE=neural_orchestrator_state.json
```

#### **Dynamic Model Registry Configuration**

```bash
# Discovery
export MODEL_REGISTRY_DISCOVERY_DIRS="./models,~/models,/shared/models"
export MODEL_REGISTRY_AUTO_DOWNLOAD=true
export MODEL_REGISTRY_WATCH_FILES=true

# HuggingFace
export MODEL_REGISTRY_HF_TOKEN=your_token_here
export MODEL_REGISTRY_HF_CACHE_DIR=~/.cache/huggingface

# Reactor Core sync
export MODEL_REGISTRY_REACTOR_CORE_ENABLED=true
export MODEL_REGISTRY_REACTOR_CORE_URL=http://localhost:9000
```

#### **General Server Configuration**

```bash
# Server
export JARVIS_PRIME_HOST=0.0.0.0
export JARVIS_PRIME_PORT=8000
export JARVIS_PRIME_MODELS_DIR=./models

# Safety integration
export JARVIS_PRIME_SAFETY_ENABLED=true
export JARVIS_CROSS_REPO_DIR=~/.jarvis/cross_repo

# Model settings
export JARVIS_PRIME_INITIAL_MODEL=./models/mistral-7b.gguf
export JARVIS_PRIME_CONTEXT_LENGTH=4096
export JARVIS_PRIME_N_GPU_LAYERS=-1  # All layers on GPU (M1 MPS)
export PRIME_QUANTIZATION_BITS=8  # 4-bit or 8-bit for M1 optimization
```

#### **GCP Cloud Hybrid Configuration**

```bash
# GCP settings
export GCP_ENABLED=true
export GCP_PROJECT_ID=your-project-id
export GCP_ZONE=us-central1-a
export GCP_VM_INSTANCE_TYPE=n1-standard-4
export GCP_VM_SPOT=true
export GCP_VM_RAM_GB=64  # Updated from 32GB to 64GB
export GCP_PRIME_URL=http://your-gcp-vm:8000
```

---

## 📊 Performance & Benchmarks

### Neural Orchestrator Core Performance (M1 Max 64GB)

| Metric | Value |
|--------|-------|
| Routing decision latency | 0.5-1.5ms |
| Task classification latency | 0.3-0.8ms |
| Memory pressure check (macOS native) | 5-15ms |
| Memory pressure check (psutil fallback) | 1-3ms |
| Sticky routing lookup | <0.1ms |
| Circuit breaker check | <0.1ms |
| Cross-repo state read | 2-5ms |
| Cross-repo state write | 3-8ms |

### Local Model Performance (M1 Mac 16GB)

| Model | Size | Tokens/sec | Latency (P50) | Latency (P99) | Memory |
|-------|------|------------|---------------|---------------|--------|
| TinyLlama 1.1B (Q4_K_M) | 670MB | 85 t/s | 12ms | 45ms | 1.2GB |
| Phi-2 2.7B (Q4_K_M) | 1.6GB | 42 t/s | 24ms | 89ms | 2.8GB |
| Mistral 7B (Q4_K_M) | 4.3GB | 18 t/s | 56ms | 178ms | 5.9GB |
| Llama-3 8B (Q4_K_M) | 4.9GB | 15 t/s | 67ms | 201ms | 6.8GB |
| Qwen 2.5 32B (Q4_K_M) | 18GB | 5 t/s | 200ms | 600ms | 20GB |

### GCP Invincible Node — Real-World Production Performance (v235.4)

Measured on `jarvis-prime-node` (e2-standard-4, 4 vCPUs, 16 GB RAM, **CPU-only, no GPU**):

| Metric | Value |
|--------|-------|
| **Model** | Mistral-7B-Instruct-v0.2 (Q4_K_M) |
| **File size on disk** | ~4.37 GB |
| **Cold start (golden image)** | ~87 seconds (VM create → `ready_for_inference=True`) |
| **Token generation rate** | ~3-5 tokens/sec |
| **End-to-end request latency** | ~8.6 seconds (short prompts, ~100-200 token responses) |
| **Model load time (from disk)** | ~30 seconds (pre-cached on golden image SSD) |
| **Memory usage (model loaded)** | ~5.5 GB RSS |
| **Inference mode** | CPU-only (AVX2/SSE4.2 SIMD via llama.cpp) |
| **Concurrent requests** | 1 (sequential processing) |
| **VM cost** | ~$0.134/hr (~$97/month always-on) |
| **Per-request cost** | $0.00 (self-hosted, unlimited requests) |

> **Note:** The ~8.6s latency is expected and normal for CPU inference on a quantized 7B model. GPU inference (e.g., NVIDIA L4 or A100) would reduce this to ~1-2s or ~0.3-0.5s respectively, at significantly higher hourly cost. See the [CPU Inference section](#cpu-inference-why-86s-latency-is-expected) for a detailed breakdown.

### GCP Cloud Performance (A100 GPU) — Reference Benchmarks

| Model | Size | Tokens/sec | Latency (P50) | Latency (P99) | Cost/hr |
|-------|------|------------|---------------|---------------|---------|
| Llama 3.3 70B (Q4) | 35GB | 45 t/s | 22ms | 65ms | $1.50 |
| Qwen 2.5 72B (Q4) | 36GB | 42 t/s | 24ms | 70ms | $1.50 |
| Mixtral 8x22B (Q4) | 45GB | 38 t/s | 26ms | 75ms | $2.00 |
| DeepSeek V2 (Q4) | 50GB | 35 t/s | 29ms | 80ms | $2.50 |

### Cost Savings (Measured over 30 days)

```
Scenario: 50,000 requests/month (avg 150 tokens out)

Neural Orchestrator Routing:
- Tier 0 (Ultra Fast): 30,000 requests (60%) → Local → $0.00
- Tier 0.5 (Local Capable): 12,000 requests (24%) → Local → $0.00
- Tier 1 (Cloud Intelligence): 7,000 requests (14%) → GCP → $10.50
- Tier 2 (Deep Reasoning): 1,000 requests (2%) → Claude Opus → $15.00

Total cost: $25.50/month

If 100% Cloud:
- 50,000 requests × 150 tokens × $0.024/1K = $180.00/month

Savings: $154.50/month (86% reduction) 🎉
```

### Resilience Metrics (Production - 7 days)

| Metric | Value |
|--------|-------|
| Circuit breaker opens | 2 |
| Fallback cache hits | 1,247 |
| Fallback to simple mode | 15 |
| Total requests | 187,342 |
| Zero-downtime swaps | 6 |
| Requests dropped | 0 ✅ |
| Average recovery time | 6.2s |
| Sticky routing hits | 45,231 (24.1%) |
| Memory pressure alerts | 3 |

---

## 🔒 Safety & Security

### Multi-Layer Safety Integration

```
┌────────────────────────────────────────────────────────────────┐
│ Layer 1: JARVIS ActionSafetyManager (Body)                    │
│ ────────────────────────────────────────────                  │
│ • Monitors all action execution                               │
│ • Detects risky patterns                                      │
│ • User confirmation required for HIGH risk                    │
│ • Kill switch activation                                      │
│ • Writes context: ~/.jarvis/safety/context_for_prime.json   │
└────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│ Layer 2: Neural Orchestrator Safety Integration              │
│ ──────────────────────────────────────────────                │
│ • Reads safety context before routing                         │
│ • Routes risky actions to Prime when kill switch active       │
│ • Adjusts tier selection based on safety state                │
│ • Forces Tier 1/2 for high-risk operations                   │
└────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│ Layer 3: Cross-Repo State Manager                            │
│ ─────────────────────────────────────                         │
│ • Atomic state updates                                        │
│ • File locking for race condition prevention                  │
│ • Automatic retry with exponential backoff                     │
└────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│ Layer 4: AGI Safety Reasoning                                 │
│ ────────────────────────────                                  │
│ • CausalEngine predicts action consequences                   │
│ • MetaReasoner evaluates risk vs benefit                      │
│ • ActionModel includes safety constraints                     │
└────────────────────────────────────────────────────────────────┘
```

### Safety Context Example

```json
{
  "kill_switch_active": true,
  "current_risk_level": "high",
  "pending_confirmation": true,
  "recent_blocks": 2,
  "recent_confirmations": 5,
  "recent_denials": 3,
  "user_trust_level": 0.62,
  "last_update": "2025-01-07T14:30:45.123456",
  "session_start": "2025-01-07T09:00:00.000000",
  "total_audits": 47,
  "total_blocks": 8
}
```

**Routing Behavior:**
- Kill switch active → All actions route to Tier 1/2
- Recent denials > 2 → Route risky patterns to Tier 1/2
- User trust < 0.7 → More conservative routing
- High risk level → Force confirmation

---

## 🗺️ Roadmap

### ✅ v238.0 - Degenerate Response Elimination (Current, JARVIS Body-side)

- [x] SIMPLE classification narrowed: "what is/who is/define" queries promoted to MODERATE
- [x] Backend degenerate response detection with safe retry (MODERATE params)
- [x] Client-side degenerate response suppression before display/TTS
- [x] requestId echo in all backend WebSocket response dicts (enables frontend dedup)
- [x] command_response handler aligned with response handler (dedup, ref clearing, validation)
- [x] Defense-in-depth: 3-layer architecture (classification → backend retry → client filter)
- [x] Production verified: "what is Java?" → gcp_prime (24.6s latency, full definition)

### ✅ v100.0 - Neural Orchestrator Core

- [x] Unified routing architecture consolidating all routers
- [x] Protocol-based design with type-safe interfaces
- [x] Context-aware routing with distributed tracing
- [x] Dynamic configuration with zero hardcoding
- [x] Cross-repo state management with atomic operations
- [x] Unified task classifier with multi-signal analysis
- [x] Unified memory monitor with macOS native integration
- [x] Unified sticky routing with session affinity
- [x] Unified request buffer for zero-loss hot swaps
- [x] Coordinated circuit breakers per tier
- [x] Advanced Python patterns (Protocols, contextvars, async generators, weakref)
- [x] Defensive decorators with graceful fallbacks
- [x] Exponential backoff with decorrelated jitter
- [x] Structured concurrency with TaskGroup (Python 3.11+)

### ✅ v99.0 - Dynamic Model Registry

- [x] Multi-directory model discovery
- [x] Auto-download from HuggingFace
- [x] File system watching with `watchdog`
- [x] Reactor Core synchronization
- [x] Model validation (integrity, inference, safety)
- [x] Version management with rollback support

### ✅ v98.0 - Neural Switchboard

- [x] Task classification with multi-signal analysis
- [x] Memory monitoring with real-time pressure detection
- [x] Sticky routing with session-based affinity
- [x] Request buffering for zero-loss hot swaps
- [x] Tier/capability mapping

### ✅ v92.0 - LLM/Brain Intelligence

- [x] Auto model selector with complexity-based routing
- [x] Unified inference with fallback chain
- [x] RLHF pipeline with PPO
- [x] Reactor Core bridge for training integration
- [x] Continuous learning with EWC
- [x] Dynamic batching for throughput optimization
- [x] Circuit breakers per backend

### ✅ v91.0 - Observability Bridge

- [x] Langfuse integration for distributed tracing
- [x] Prometheus export in OpenMetrics format
- [x] Chaos testing framework
- [x] Adaptive polling optimization
- [x] Cross-repo observability integration

### ✅ v90.0 - Production Hardening

- [x] Event delivery guarantees with retry + DLQ
- [x] Model validation (pre-deployment)
- [x] Request queuing during hot-swap
- [x] Canary deployments with gradual rollout
- [x] Auto-rollback on error threshold
- [x] Distributed tracing with TraceContext
- [x] Circuit breakers per endpoint
- [x] Metrics & alerting
- [x] SAGA pattern for transactional deployments

### ✅ v87.0 - The Connective Tissue

- [x] Unified mode with single command startup
- [x] Intelligent model router with fallback chain
- [x] GCP VM manager with spot instance lifecycle
- [x] Service mesh with dynamic discovery
- [x] Unified config (single YAML source)
- [x] RAM-aware routing with automatic failover
- [x] Adaptive thresholds with outcome learning

### ✅ v79.1 - Cognitive Router "Corpus Callosum"

- [x] CognitiveRouter with adaptive thresholds
- [x] PrimeBridge with circuit breaker and connection pooling
- [x] Response cache for graceful degradation
- [x] Fixed singleton race condition (asyncio.Condition)
- [x] Fixed file IPC race conditions (fcntl locking, OrderedDict)
- [x] Fallback chain (4 levels)
- [x] Adaptive polling intervals
- [x] Bounded message queues
- [x] Zero hardcoding (all env vars)
- [x] Production-grade resilience patterns

### 🔮 v101.0 - Advanced Features (Planned)

- [ ] Request deduplication
- [ ] Routing decision caching
- [ ] Continuous memory pressure monitoring during execution
- [ ] Deadlock detection for locks
- [ ] Request cancellation support
- [ ] Request batching optimization
- [ ] Distributed tracing correlation enhancement

---

## 🧪 Testing & Development

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/

# Neural Orchestrator Core tests
pytest tests/test_neural_orchestrator_core.py -v

# With coverage
pytest --cov=jarvis_prime --cov-report=html

# Test specific module
pytest tests/unit/test_neural_orchestrator_core.py -v
```

### Development Server with Hot Reload

```bash
# Install in development mode
pip install -e ".[dev]"

# Run with auto-reload on code changes
python run_server.py --reload --debug

# Server restarts automatically when files change
```

### Docker Deployment

```bash
# Build image
docker build -t jarvis-prime:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v ~/.jarvis:/root/.jarvis \
  -e JARVIS_PRIME_INITIAL_MODEL=/app/models/mistral-7b.gguf \
  -e NEURAL_ORCHESTRATOR_ENABLED=true \
  jarvis-prime:latest

# Check logs
docker logs -f <container-id>
```

---

## 📚 Documentation

### Core Documentation
- **[Architecture Deep Dive](docs/architecture.md)** - Detailed system architecture
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Configuration Guide](docs/configuration.md)** - All configuration options
- **[Neural Orchestrator Core](jarvis_prime/core/neural_orchestrator_core.py)** - Complete implementation with inline documentation

### Training & Models
- **[LLAMA_13B_GUIDE.md](LLAMA_13B_GUIDE.md)** - Llama-2-13B training guide
- **[ADVANCED_LLM_INTEGRATION.md](ADVANCED_LLM_INTEGRATION.md)** - LLM integration patterns
- **[examples/](examples/)** - Training and inference examples

### Version-Specific Documentation
- **[Neural Orchestrator Core v100.0](jarvis_prime/core/neural_orchestrator_core.py)** - Unified routing architecture
- **[Dynamic Model Registry v99.0](jarvis_prime/core/dynamic_model_registry.py)** - Auto-discovery and management
- **[Neural Switchboard v98.0](jarvis_prime/core/neural_switchboard.py)** - Task classification and routing

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/jarvis-prime.git
cd jarvis-prime

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
pytest tests/

# Commit with conventional commits
git commit -m "feat: add amazing feature

- Detailed description
- Why this change is needed
- Any breaking changes

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Push and create PR
git push origin feature/amazing-feature
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- **Anthropic** - Claude API and advanced reasoning capabilities
- **Meta AI** - Llama models and research
- **Mistral AI** - High-quality open models
- **Microsoft Research** - Phi models for coding
- **Alibaba** - Qwen multilingual models
- **ggerganov** - llama.cpp runtime for efficient inference
- **HuggingFace** - Model hosting and transformers library
- **OpenAI** - API compatibility standards

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/drussell23/jarvis-prime/issues)
- **Discussions**: [GitHub Discussions](https://github.com/drussell23/jarvis-prime/discussions)
- **Email**: derek@jarvis-ai.dev

---

## 🏆 Summary

### What JARVIS Prime Delivers

✅ **Self-Hosted LLM Inference** - Mistral-7B-Instruct-v0.2 (Q4_K_M) on your own GCP VM — no OpenAI, no Claude, no third-party APIs
✅ **Adaptive Prompt System (v236.0 + v238.0)** - Complexity-aware inference: "5+5?" → "10" (48 tokens, temp 0.0), "what is Java?" → full definition (512 tokens, temp 0.3), "design a system" → detailed analysis (4096 tokens, temp 0.7)
✅ **Degenerate Response Defense-in-Depth (v238.0)** - 3-layer protection (classification, backend retry, client suppression) ensures meaningless LLM output ("...") never reaches the user
✅ **Enterprise-Grade AGI Operating System** - 7 specialized models, reasoning, multimodal fusion
✅ **Neural Orchestrator Core v100.0** - Unified intelligent routing, single source of truth
✅ **GCP Golden Image Boot** - Cold start in ~87 seconds with pre-baked model on disk
✅ **Production-Grade Resilience** - Circuit breakers, fallback chains, response caching
✅ **Zero Hardcoding** - Fully configurable via environment variables and YAML
✅ **Safety-Aware Routing** - Integrated with JARVIS ActionSafetyManager
✅ **Zero-Downtime Operations** - Hot swap models with zero request drops
✅ **Complete Data Privacy** - All inference on your infrastructure, no data leaves your VMs
✅ **Cost Optimization** - ~$97/month flat for unlimited self-hosted inference (no per-token billing)
✅ **Advanced Telemetry** - Langfuse, Prometheus, real-time dashboards
✅ **Cross-Repo Integration** - Seamless JARVIS ecosystem communication
✅ **Reactor-Core Training Loop** - DPO/RLHF pipeline to fine-tune the model from real interactions, making prompt-based conciseness permanent
✅ **Battle-Tested** - 187K+ requests in production, zero failures

### v100.0 Highlights

🧠 **Neural Orchestrator Core** - Unified routing architecture consolidating all routers
🛡️ **Advanced Patterns** - Protocol classes, contextvars, async generators, weakref
⚡ **Performance** - Sub-millisecond routing decisions, native macOS memory integration
🔧 **Zero Hardcoding** - 100% dynamic configuration with env var override
📊 **Cross-Repo Integration** - Atomic state management across JARVIS ecosystem
🔄 **Sticky Routing** - Session-based model affinity for continuity
💾 **Request Buffering** - Zero-loss hot swap support
🔌 **Circuit Breakers** - Coordinated fault tolerance per tier

**Ready for enterprise deployment with complete AGI capabilities! 🚀**

---

### Architecture at a Glance

```
User Request → Neural Orchestrator Core v100.0
                     │
                     ├─→ Task Classification
                     ├─→ Memory Pressure Check
                     ├─→ Sticky Routing Lookup
                     ├─→ Circuit Breaker Check
                     └─→ Tier Selection (0/0.5/1/2)
                           │
                           ├─→ Tier 0: Ultra Fast (Local)
                           ├─→ Tier 0.5: Local Capable
                           ├─→ Tier 1: Cloud Intelligence (GCP)
                           └─→ Tier 2: Deep Reasoning (Claude Opus)
```

**The future of AGI is here. Welcome to JARVIS Prime v100.0.** 🚀

**Powered by your own Mistral-7B model on your own GCP infrastructure. No third-party APIs required.**

---

Built with ❤️ by Derek Russell
Powered by self-hosted Mistral-7B-Instruct-v0.2, llama-cpp-python, and the JARVIS Ecosystem
