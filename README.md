# JARVIS Prime

**Advanced Hybrid LLM Inference Platform with Tier-0 Local Brain**

🚀 Hybrid Cloud-Local Architecture | ⚡ Zero Hardcoding | 🔥 Async by Default | 🧠 Safety-Aware Routing | 🔄 Zero-Downtime Hot Swap

JARVIS Prime provides a **production-ready, intelligent, and adaptive** platform for hybrid LLM deployment. Seamlessly route between **local Tier-0 inference** (M1 Mac) and **cloud Tier-1 models** (Claude API) with safety-aware complexity analysis, zero-downtime model swapping, and cross-repository integration with JARVIS.

---

## 🎯 What is JARVIS Prime?

JARVIS Prime is a **hybrid inference orchestration platform** that intelligently routes requests between:

- **Tier-0 (Local)**: Fast, cost-free inference on M1 Mac using quantized GGUF models (Llama, Mistral, Phi, Qwen)
- **Tier-1 (Cloud)**: Advanced reasoning using Claude API for complex, multimodal, or high-stakes tasks

### Key Innovation: **Intelligent Hybrid Router**

The HybridRouter dynamically analyzes each prompt's complexity, task type, and safety context to route requests optimally:

```python
# Simple greeting → Local Tier-0 (free, fast)
"Hello JARVIS" → TinyLlama (50ms, $0.00)

# Complex reasoning → Cloud Tier-1 (powerful, accurate)
"Analyze this codebase for security vulnerabilities" → Claude Opus 4.5 ($0.15)

# Safety-critical → Cloud Tier-1 (with JARVIS safety context)
"Delete all files in /tmp" → Claude with safety warnings
```

---

## ✨ Core Features

### 🧠 **1. Hybrid Intelligence Router**
- **Complexity Analysis**: Token count, technical depth, task type detection
- **Dynamic Thresholds**: Learns from outcomes, adjusts routing over time
- **Safety-Aware Routing**: Integrates JARVIS ActionSafetyManager state
- **Force Tier Options**: Override for testing or user preference

### 🔒 **2. JARVIS Safety Integration**
- **Cross-Repo Bridge**: Reads safety context from main JARVIS instance
- **Kill Switch Detection**: Routes to cloud when safety controls active
- **Risk Pattern Analysis**: Detects risky actions (delete, format, sudo, etc.)
- **Prompt Context Injection**: Informs models of current safety state
- **User Trust Scoring**: Adapts routing based on recent user confirmations/denials

### 🔄 **3. Zero-Downtime Hot Swap**
- **Background Loading**: New models load in parallel with zero request drops
- **Traffic Draining**: In-flight requests complete on old model
- **Atomic Switch**: Instant pointer swap to new model
- **Rollback on Failure**: Automatic revert if new model fails validation
- **Memory Management**: MPS optimization for M1 Mac, automatic cleanup

### 📊 **4. Advanced Telemetry & Cost Tracking**
- **Unified Metrics**: Combined cost tracking across Tier-0 and Tier-1
- **Real-Time Dashboards**: Langfuse/Helicone integration for observability
- **Savings Calculator**: Track cost savings from local inference
- **Performance Analytics**: Latency, throughput, cache hit rates
- **Cross-Repo Reporting**: Share metrics with main JARVIS orchestrator

### 🚀 **5. Model Management & Versioning**
- **Semantic Versioning**: v1.0-base, v1.1-weekly-2025-05-12
- **Model Registry**: Track lineage, rollback capability, metadata
- **Automatic Downloads**: GCS and HuggingFace integration
- **Reactor-Core Watcher**: Auto-detect new trained models
- **Multi-Model Support**: TinyLlama, Phi-2, Mistral, Llama-3, Qwen2

### 🌐 **6. OpenAI-Compatible API**
- **Drop-in Replacement**: Compatible with OpenAI Python SDK
- **FastAPI Server**: Async, production-ready with uvicorn
- **Streaming Support**: Real-time token streaming
- **Model Swap Endpoint**: `POST /v1/admin/swap-model` for hot reloads
- **Health Checks**: Liveness and readiness probes

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         JARVIS ECOSYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐         ┌──────────────────┐              │
│  │  JARVIS Main    │◄───────►│  JARVIS Prime    │              │
│  │  (Orchestrator) │         │  (Tier-0 Brain)  │              │
│  └────────┬────────┘         └────────┬─────────┘              │
│           │                           │                         │
│           │  Cross-Repo Bridge        │                         │
│           │  ~/.jarvis/cross_repo/    │                         │
│           │                           │                         │
│           ├─ safety_context.json      │                         │
│           ├─ bridge_state.json        │                         │
│           └─ inference_metrics.json   │                         │
│                                       │                         │
└───────────────────────────────────────┼─────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
            ┌───────▼──────┐                      ┌────────▼────────┐
            │   Tier-0     │                      │    Tier-1       │
            │   (Local)    │                      │    (Cloud)      │
            ├──────────────┤                      ├─────────────────┤
            │ HybridRouter │───Complexity─────────► Claude Opus 4.5 │
            │   Analysis   │   > 0.7              │   (Advanced)    │
            │              │                      ├─────────────────┤
            │              │                      │ Claude Sonnet   │
            │              │◄──Safety Context──   │   (Balanced)    │
            │              │                      ├─────────────────┤
            │              │                      │ Claude Haiku    │
            └──────┬───────┘                      │   (Fast)        │
                   │                              └─────────────────┘
                   │ Simple prompts
                   │ Complexity < 0.7
                   │
         ┌─────────▼─────────┐
         │  Local Models     │
         ├───────────────────┤
         │ TinyLlama 1.1B    │ ← Fast, simple chat
         │ Phi-2 2.7B        │ ← Coding, reasoning
         │ Mistral 7B        │ ← Production quality
         │ Llama-3 8B        │ ← Latest, powerful
         │ Qwen2 7B          │ ← Multilingual
         └───────────────────┘
              GGUF Models
           Quantized (Q4_K_M)
            Llama.cpp Runtime
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/drussell23/JARVIS-Prime.git
cd JARVIS-Prime

# Install dependencies
pip install -e .

# Or with all features
pip install -e ".[server,gcs,telemetry]"
```

### 1. Download a Model

```bash
# Quick start with TinyLlama (670MB, fast)
python -c "
from jarvis_prime.docker.model_downloader import download_model
download_model('tinyllama-chat', './models')
"

# Or production-ready Mistral 7B (4.3GB)
python -c "
from jarvis_prime.docker.model_downloader import download_model
download_model('mistral-7b-instruct', './models')
"

# Available models:
# - tinyllama-chat: 670MB, simple chat
# - phi-2: 1.6GB, coding/reasoning
# - mistral-7b-instruct: 4.3GB, production
# - llama-3-8b-instruct: 4.9GB, latest
# - qwen2-7b-instruct: 4.6GB, multilingual
```

### 2. Start the Server

```bash
# Start with downloaded model
python -m jarvis_prime.server \
    --initial-model ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --port 8000

# Server starts at http://localhost:8000
# OpenAI-compatible endpoint: http://localhost:8000/v1/chat/completions
```

### 3. Use Hybrid Router

```python
from jarvis_prime.core.hybrid_router import HybridRouter, DefaultComplexityAnalyzer

# Initialize router
router = HybridRouter(
    analyzer=DefaultComplexityAnalyzer(),
    enable_safety_context=True,  # Read JARVIS safety state
    enable_learning=True,         # Adapt thresholds over time
)

# Classify a prompt
decision = router.classify_prompt(
    prompt="What's the weather like?",
    context={"user_id": "derek"}
)

print(f"Route to: {decision.tier.value}")
print(f"Reasoning: {decision.reasoning}")
print(f"Confidence: {decision.confidence:.2%}")

# Output:
# Route to: tier-0-preferred
# Reasoning: Simple conversational task, low complexity
# Confidence: 92%

# Complex prompt → Tier-1
decision = router.classify_prompt(
    prompt="Analyze this Python codebase for security vulnerabilities, "
           "provide a detailed report with CVE references and remediation strategies."
)
print(f"Route to: {decision.tier.value}")  # tier-1-required
```

### 4. Send Requests (OpenAI SDK)

```python
from openai import OpenAI

# Connect to JARVIS Prime (Tier-0)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not required for local
)

# Simple request → Routed to local model
response = client.chat.completions.create(
    model="jarvis-prime",
    messages=[
        {"role": "user", "content": "What is artificial intelligence?"}
    ],
    max_tokens=200
)

print(response.choices[0].message.content)
# Response from TinyLlama in ~50ms, cost: $0.00
```

### 5. Hot Swap Models (Zero Downtime)

```python
import requests

# Swap to a different model while server is running
response = requests.post(
    "http://localhost:8000/v1/admin/swap-model",
    json={
        "model_path": "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "timeout_seconds": 60
    }
)

print(response.json())
# {
#   "success": true,
#   "old_version": "tinyllama-1.1b-v1.0",
#   "new_version": "mistral-7b-v0.2",
#   "duration_seconds": 12.4,
#   "requests_drained": 3,
#   "memory_freed_mb": 1240.5
# }

# All in-flight requests completed on old model
# New requests instantly use new model
# Zero requests dropped!
```

---

## 🔒 Safety-Aware Routing

JARVIS Prime integrates with JARVIS ActionSafetyManager for safety-aware routing decisions.

### How It Works

1. **JARVIS writes safety context** to `~/.jarvis/cross_repo/safety/context_for_prime.json`
2. **Prime reads safety state** before routing each request
3. **Router adjusts complexity scoring** based on safety signals
4. **Risky actions route to cloud** when kill switch active or user cautious

### Example: Kill Switch Active

```python
# JARVIS detects risky action and activates kill switch
# Safety state file written:
{
  "kill_switch_active": true,
  "current_risk_level": "high",
  "recent_denials": 2,
  "user_trust_level": 0.6
}

# User asks Prime to delete files
decision = router.classify_prompt("Delete all log files older than 30 days")

# Router reads safety context and forces Tier-1
print(decision.tier)  # tier-1-required
print(decision.reasoning)  # "Safety: kill switch active"

# Request routed to Claude with safety context prepended:
# [JARVIS SAFETY CONTEXT]
# - KILL SWITCH ACTIVE: All actions paused
# - Risk Level: HIGH
# - User denied 2 action(s) recently
# [/JARVIS SAFETY CONTEXT]
#
# User request: Delete all log files older than 30 days
```

### Safety Context Snapshot

```python
from jarvis_prime.core.hybrid_router import SafetyContextReader

reader = SafetyContextReader()
context = reader.read_context()

print(f"Kill switch: {context.kill_switch_active}")
print(f"Risk level: {context.current_risk_level}")
print(f"User trust: {context.user_trust_level:.2f}")
print(f"Recent blocks: {context.recent_blocks}")
print(f"Risky actions? {context.should_avoid_risky_actions()}")
```

### Risky Pattern Detection

The router automatically detects patterns that suggest risky actions:

```python
# Patterns flagged as risky:
- delete, remove, erase, wipe
- format, kill, terminate, shutdown
- execute, run, install, uninstall
- sudo, admin, root, system
- password, credential, secret
- file write operations

# When detected + safety caution active → Force cloud routing
```

---

## 📊 Telemetry & Cost Tracking

### Cross-Repo Metrics Bridge

```python
from jarvis_prime.core.cross_repo_bridge import CrossRepoBridge, InferenceMetrics

# Initialize bridge
bridge = CrossRepoBridge(instance_id="prime-derek-mac", port=8000)
await bridge.start()

# Record inference
bridge.record_inference(
    tokens_in=25,
    tokens_out=150,
    latency_ms=47.3
)

# Get cost savings
state = bridge.state
print(f"Total requests: {state.metrics.total_requests}")
print(f"Cloud cost if used: ${state.metrics.estimated_cost_usd:.4f}")
print(f"Savings: ${state.metrics.savings_vs_cloud_usd:.4f}")

# Metrics automatically written to:
# ~/.jarvis/cross_repo/jarvis_prime_state.json
# Main JARVIS reads this for unified dashboard
```

### Router Statistics

```python
stats = router.get_statistics()

print(stats)
# {
#   "total_classifications": 1847,
#   "tier_0_count": 1523,
#   "tier_1_count": 324,
#   "tier_0_ratio": 0.824,
#   "tier_1_ratio": 0.176,
#   "safety_influenced_count": 47,
#   "current_safety_context": {
#     "kill_switch_active": false,
#     "risk_level": "low",
#     "user_trust_level": 0.95,
#     "is_stale": false
#   }
# }

# 82.4% of requests handled locally → massive cost savings!
```

---

## 🔄 Hot Swap Manager

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Hot Swap Process                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. LOADING_BACKGROUND                                  │
│     ├─ Load new model in parallel thread               │
│     ├─ Old model continues serving requests             │
│     └─ Progress: [████████░░] 80%                       │
│                                                          │
│  2. VALIDATING                                          │
│     ├─ Test inference on new model                     │
│     ├─ Check memory consumption                         │
│     └─ Verify output quality                            │
│                                                          │
│  3. DRAINING                                            │
│     ├─ Stop accepting new requests on old model        │
│     ├─ Wait for in-flight requests (3 active)          │
│     └─ Timeout: 30s                                     │
│                                                          │
│  4. SWAPPING (Atomic)                                   │
│     ├─ Update model pointer                             │
│     └─ Duration: <10ms                                  │
│                                                          │
│  5. CLEANUP                                             │
│     ├─ Unload old model from memory                    │
│     ├─ Free MPS cache (M1 Mac)                         │
│     └─ Garbage collection                               │
│                                                          │
│  Total time: ~12-15 seconds                             │
│  Requests dropped: 0 ✅                                  │
└─────────────────────────────────────────────────────────┘
```

### Usage

```python
from jarvis_prime.core.hot_swap_manager import HotSwapManager, ModelLoader
from pathlib import Path

# Initialize manager
manager = HotSwapManager(
    model_loader=YourModelLoader(),  # Implement ModelLoader protocol
    drain_timeout_seconds=30,
    validation_required=True
)

# Perform hot swap
result = await manager.swap_model(
    new_model_path=Path("./models/mistral-7b.gguf"),
    new_version_id="mistral-7b-v0.2"
)

if result.success:
    print(f"✅ Swapped from {result.old_version} to {result.new_version}")
    print(f"   Duration: {result.duration_seconds:.1f}s")
    print(f"   Drained: {result.requests_drained} requests")
    print(f"   Memory freed: {result.memory_freed_mb:.1f} MB")
else:
    print(f"❌ Swap failed: {result.error_message}")
    print(f"   Rolled back to {result.old_version}")
```

### Rollback on Failure

```python
# Automatic rollback if new model fails validation
result = await manager.swap_model(
    new_model_path=Path("./models/corrupted-model.gguf")
)

print(result.state)  # SwapState.FAILED
print(result.error_message)  # "Validation failed: Model output corrupted"
# Old model still serving requests ✅
```

---

## 🧩 Model Registry & Versioning

### Semantic Versioning

```python
from jarvis_prime.core.model_registry import ModelRegistry, ModelVersion, ModelMetadata
from datetime import datetime

# Initialize registry
registry = ModelRegistry(registry_dir=Path("./model_registry"))

# Register a model version
version = ModelVersion(
    version_id="v1.1-weekly-2025-05-12",
    model_path=Path("./models/mistral-7b.gguf"),
    metadata=ModelMetadata(
        created_at=datetime.now(),
        source="reactor-core",
        training_config={"epochs": 3, "lr": 2e-5},
        performance_metrics={"perplexity": 3.42, "accuracy": 0.87},
        capabilities=["chat", "reasoning", "coding"],
        checksum="sha256:abc123..."
    )
)

await registry.register_version(version)

# List all versions
versions = registry.list_versions()
for v in versions:
    print(f"{v.version_id}: {v.state.value}")

# Activate a version
await registry.activate_version("v1.1-weekly-2025-05-12")

# Rollback to previous
await registry.rollback()
```

### Reactor-Core Integration

```python
# Watch for new models from training pipeline
from jarvis_prime.docker.reactor_core_watcher import ReactorCoreWatcher

watcher = ReactorCoreWatcher(
    reactor_core_dir=Path("../reactor-core/outputs"),
    models_dir=Path("./models"),
    auto_swap=True  # Automatically hot-swap when new model ready
)

await watcher.start()

# When reactor-core finishes training:
# 1. Watcher detects new checkpoint
# 2. Converts to GGUF if needed
# 3. Validates model
# 4. Triggers hot swap
# 5. Registers in model registry
# All automatic! 🎉
```

---

## 🌐 API Reference

### Chat Completions (OpenAI-Compatible)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jarvis-prime",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Explain quantum computing."}
    ],
    "max_tokens": 500,
    "temperature": 0.7,
    "stream": false
  }'
```

### Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1703001234,
  "model": "jarvis-prime-tinyllama-1.1b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum computing is a revolutionary approach..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 150,
    "total_tokens": 175
  }
}
```

### Model Swap (Admin)

```bash
curl -X POST http://localhost:8000/v1/admin/swap-model \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "timeout_seconds": 60
  }'
```

### Health Check

```bash
curl http://localhost:8000/health

# Response
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
  "active_requests": 2,
  "total_requests": 1847,
  "uptime_seconds": 86400,
  "safety_context": {
    "kill_switch_active": false,
    "risk_level": "low"
  }
}
```

---

## 🎛️ Configuration

### Environment Variables

```bash
# Server configuration
export JARVIS_PRIME_HOST=0.0.0.0
export JARVIS_PRIME_PORT=8000
export JARVIS_PRIME_MODELS_DIR=./models
export JARVIS_PRIME_TELEMETRY_DIR=./telemetry

# Routing thresholds
export JARVIS_PRIME_TIER0_THRESHOLD=0.7
export JARVIS_PRIME_TIER1_THRESHOLD=0.85

# Safety integration
export JARVIS_PRIME_SAFETY_ENABLED=true
export JARVIS_CROSS_REPO_DIR=~/.jarvis/cross_repo

# Model settings
export JARVIS_PRIME_INITIAL_MODEL=./models/mistral-7b.gguf
export JARVIS_PRIME_CONTEXT_LENGTH=4096
export JARVIS_PRIME_N_GPU_LAYERS=-1  # All layers on GPU (M1 MPS)
```

### YAML Configuration

```yaml
# config/prime_server.yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 1
  reload: false

models:
  directory: ./models
  initial_model: mistral-7b-instruct-v0.2.Q4_K_M.gguf
  context_length: 4096
  n_gpu_layers: -1

router:
  tier_0_threshold: 0.7
  tier_1_threshold: 0.85
  enable_learning: true
  enable_safety_context: true

telemetry:
  enabled: true
  directory: ./telemetry
  langfuse_public_key: ${LANGFUSE_PUBLIC_KEY}
  langfuse_secret_key: ${LANGFUSE_SECRET_KEY}
  helicone_api_key: ${HELICONE_API_KEY}

bridge:
  enabled: true
  instance_id: prime-derek-mac
  auto_heartbeat: true
  cross_repo_dir: ~/.jarvis/cross_repo
```

Load configuration:

```python
from jarvis_prime.core.config_loader import load_config

config = load_config("config/prime_server.yaml")
```

---

## 📈 Performance

### Benchmarks (M1 Mac 16GB)

| Model | Size | Tokens/sec | Latency (P50) | Latency (P99) | Memory |
|-------|------|------------|---------------|---------------|--------|
| TinyLlama 1.1B (Q4_K_M) | 670MB | 85 t/s | 12ms | 45ms | 1.2GB |
| Phi-2 2.7B (Q4_K_M) | 1.6GB | 42 t/s | 24ms | 89ms | 2.8GB |
| Mistral 7B (Q4_K_M) | 4.3GB | 18 t/s | 56ms | 178ms | 5.9GB |
| Llama-3 8B (Q4_K_M) | 4.9GB | 15 t/s | 67ms | 201ms | 6.8GB |
| Qwen2 7B (Q4_K_M) | 4.6GB | 17 t/s | 59ms | 185ms | 6.2GB |

### Cost Savings

```
Scenario: 10,000 requests/month (avg 100 tokens out)

Tier-0 Local (82% of requests):
- 8,200 requests × $0.00 = $0.00

Tier-1 Cloud (18% of requests):
- 1,800 requests × 100 tokens × $0.024/1K = $4.32

Total cost: $4.32/month

If 100% Cloud:
- 10,000 requests × 100 tokens × $0.024/1K = $24.00/month

Savings: $19.68/month (82% reduction) 🎉
```

---

## 🔗 Cross-Repository Integration

### JARVIS Main ↔ JARVIS Prime Bridge

#### Shared State Directory Structure

```
~/.jarvis/
└── cross_repo/
    ├── bridge_state.json          # Main JARVIS orchestrator state
    ├── jarvis_prime_state.json    # Prime instance metrics
    └── safety/
        └── context_for_prime.json # Safety context from ActionSafetyManager
```

#### Safety Context File (Written by JARVIS)

```json
{
  "kill_switch_active": false,
  "current_risk_level": "low",
  "pending_confirmation": false,
  "recent_blocks": 0,
  "recent_confirmations": 12,
  "recent_denials": 0,
  "user_trust_level": 0.95,
  "last_update": "2025-05-12T14:30:45.123456",
  "session_start": "2025-05-12T09:00:00.000000",
  "total_audits": 847,
  "total_blocks": 3
}
```

#### Prime Metrics File (Written by Prime)

```json
{
  "instance_id": "prime-derek-mac",
  "started_at": "2025-05-12T09:00:00.000000",
  "last_heartbeat": "2025-05-12T14:30:45.123456",
  "status": "ready",
  "model_loaded": true,
  "model_path": "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
  "endpoint": "http://localhost:8000",
  "port": 8000,
  "metrics": {
    "total_requests": 1847,
    "total_tokens_in": 46234,
    "total_tokens_out": 189472,
    "total_latency_ms": 87234.5,
    "avg_latency_ms": 47.2,
    "model_name": "mistral-7b-instruct-v0.2",
    "estimated_cost_usd": 6.47,
    "savings_vs_cloud_usd": 6.47
  },
  "connected_to_jarvis": true,
  "jarvis_session_id": "session-abc123"
}
```

### Integration Example

```python
# In main JARVIS - write safety context
from jarvis.safety import ActionSafetyManager

safety_manager = ActionSafetyManager()
safety_manager.activate_kill_switch()  # User triggered safety pause

# Automatically writes to:
# ~/.jarvis/cross_repo/safety/context_for_prime.json

# In JARVIS Prime - read and adapt routing
from jarvis_prime.core.hybrid_router import HybridRouter

router = HybridRouter(enable_safety_context=True)

# Next request routed with safety awareness
decision = router.classify_prompt("Delete old backups")
# → Forced to Tier-1 due to kill switch

# Prime writes metrics back for JARVIS dashboard
bridge.record_inference(tokens_in=15, tokens_out=50, latency_ms=45.2)
# Main JARVIS reads from:
# ~/.jarvis/cross_repo/jarvis_prime_state.json
```

---

## 🛠️ Development

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/

# With coverage
pytest --cov=jarvis_prime --cov-report=html
```

### Local Development Server

```bash
# Install in development mode
pip install -e ".[dev]"

# Run with hot reload
python -m jarvis_prime.server --reload --debug

# Test endpoint
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
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
  jarvis-prime:latest

# Check logs
docker logs -f <container-id>
```

---

## 📚 Documentation

- **[LLAMA_13B_GUIDE.md](LLAMA_13B_GUIDE.md)** - Complete Llama-2-13B training guide
- **[ADVANCED_LLM_INTEGRATION.md](ADVANCED_LLM_INTEGRATION.md)** - LLM library integration
- **[examples/](examples/)** - Training and inference examples
- **[docs/api.md](docs/api.md)** - Full API reference
- **[docs/architecture.md](docs/architecture.md)** - System architecture details

---

## 🗺️ Roadmap

### ✅ Completed (v1.0)

- [x] Hybrid router with complexity analysis
- [x] JARVIS safety integration
- [x] Zero-downtime hot swap
- [x] Cross-repo bridge
- [x] Model registry & versioning
- [x] OpenAI-compatible API
- [x] GCS and HuggingFace model downloads
- [x] Telemetry and cost tracking
- [x] M1 Mac MPS optimization

### 🚧 In Progress (v1.1)

- [ ] LangGraph integration for multi-step authentication reasoning
- [ ] LangChain multi-factor authentication orchestration
- [ ] ChromaDB voice pattern recognition
- [ ] Enhanced behavioral biometrics
- [ ] Playwright remote authentication workflows
- [ ] Advanced voice context memory

### 🔮 Planned (v2.0)

- [ ] Multi-user support with per-user models
- [ ] Fine-tuning pipeline integration
- [ ] Advanced caching strategies (semantic caching)
- [ ] Model ensemble routing
- [ ] A/B testing framework
- [ ] Enhanced security (API keys, rate limiting)
- [ ] Kubernetes deployment manifests
- [ ] Grafana dashboards

---

## 🤝 Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/JARVIS-Prime.git
cd JARVIS-Prime

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
pytest tests/

# Commit with conventional commits
git commit -m "feat: add amazing feature"

# Push and create PR
git push origin feature/amazing-feature
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- **Meta AI** - Llama models
- **Mistral AI** - Mistral models
- **Microsoft Research** - Phi models
- **Alibaba** - Qwen models
- **ggerganov** - llama.cpp runtime
- **Anthropic** - Claude API
- **HuggingFace** - Model hosting and transformers library

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/drussell23/JARVIS-Prime/issues)
- **Discussions**: [GitHub Discussions](https://github.com/drussell23/JARVIS-Prime/discussions)
- **Email**: derek@jarvis-ai.dev

---

## 🏆 Summary

JARVIS Prime delivers:

✅ **Hybrid Cloud-Local Intelligence** - Seamless Tier-0/Tier-1 routing
✅ **Safety-Aware Routing** - Integrates JARVIS ActionSafetyManager
✅ **Zero-Downtime Hot Swap** - Swap models with zero request drops
✅ **Cost Optimization** - 80%+ savings with local Tier-0 inference
✅ **Production-Ready** - Async, OpenAI-compatible, Docker-ready
✅ **Cross-Repo Integration** - Unified metrics and safety context
✅ **Zero Hardcoding** - Fully configurable via YAML/JSON/env
✅ **Advanced Telemetry** - Langfuse, Helicone, real-time dashboards

**Ready to deploy hybrid intelligence with safety guarantees! 🚀**

---

Built with ❤️ for JARVIS by Derek Russell
