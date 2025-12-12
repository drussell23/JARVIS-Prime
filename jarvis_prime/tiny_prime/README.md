# Tiny Prime: Semantic Security Guard (v0.1.0)

Tiny Prime is a lightweight, **decoder-only (Llama-style)** model for **semantic intent classification** of short voice commands, with special focus on:

- **Negation** ("don’t unlock", "wait, don’t open")
- **Duress** ("unlock… please", "open it or else")
- **Ambiguity** ("maybe open", "I think not")

It is designed to plug into JARVIS VBI as `check_semantic_security(text)`.

## Quick start (local)

### 1) Generate training data

```bash
python -m jarvis_prime.tiny_prime.scripts.generate_security_intents \
  --config jarvis_prime/tiny_prime/config/model_config.yaml
```

### 2) Train tokenizer

```bash
python -m jarvis_prime.tiny_prime.scripts.train_tokenizer \
  --config jarvis_prime/tiny_prime/config/model_config.yaml
```

### 3) Train Tiny Prime (from scratch)

```bash
python -m jarvis_prime.tiny_prime.scripts.train_tiny_prime \
  --config jarvis_prime/tiny_prime/config/model_config.yaml
```

### 4) Inference (semantic security guard)

```python
from jarvis_prime.tiny_prime.semantic_guard import TinyPrimeGuard
from jarvis_prime.tiny_prime.config import TinyPrimeConfig

cfg = TinyPrimeConfig.from_yaml("jarvis_prime/tiny_prime/config/model_config.yaml")
guard = TinyPrimeGuard.from_config(cfg)

result = guard.check("Wait, don’t unlock the door")
print(result.label, result.confidence)
```

## Notes
- **Zero hardcoding**: all paths, labels, formats, and hyperparameters are YAML/JSON/env driven.
- **Async by default**: `check_async()` supports concurrent calls with auto-batching.
- **Spot resilient**: training installs a SIGTERM handler and checkpoints frequently.
