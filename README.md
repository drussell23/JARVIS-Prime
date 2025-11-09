# JARVIS Prime

**Specialized PRIME Models for JARVIS AI Assistant**

JARVIS Prime provides trained, optimized language models specifically designed for the JARVIS ecosystem. Built on [Reactor Core](https://github.com/drussell23/reactor-core), it delivers production-ready models for reasoning, chat, and multimodal interactions.

## What is JARVIS Prime?

JARVIS Prime bridges the gap between raw ML training (Reactor Core) and production deployment (JARVIS). It provides:

- **Pre-trained PRIME models** (7B, 13B parameter sizes)
- **Fine-tuned for JARVIS** use cases
- **Quantized versions** for M1 Mac (4-bit/8-bit)
- **Easy integration** via `PrimeModel.from_pretrained()`

## Architecture

```
JARVIS Prime
├── Reactor Core (training engine)
└── PRIME Models
    ├── prime-7b-chat-v1 (chat/reasoning)
    ├── prime-7b-vision-v1 (multimodal)
    └── prime-13b-reasoning-v1 (advanced reasoning)
```

## Installation

```bash
# For JARVIS runtime (inference only)
pip install jarvis-prime

# For model training (requires Reactor Core)
pip install jarvis-prime[training]
```

## Quick Start

### Using Pre-trained Models

```python
from jarvis_prime import PrimeModel

# Load quantized model for M1 Mac
model = PrimeModel.from_pretrained(
    "prime-7b-chat-v1",
    quantization="8bit",  # 4bit, 8bit, or None
    device="mps"  # auto-detected
)

# Generate response
response = model.generate("What is machine learning?")
print(response)
```

### Training Custom PRIME Models

```python
from jarvis_prime import PrimeTrainer
from jarvis_prime.configs import Prime7BChatConfig

# Configure training
config = Prime7BChatConfig(
    base_model="meta-llama/Llama-2-7b-hf",
    use_lora=True,
    lora_rank=16,
    num_epochs=3,
)

# Train on GCP
trainer = PrimeTrainer(config)
trainer.train("./data/jarvis_conversations.jsonl")
```

## Available Models

| Model | Size | Use Case | Quantization | M1 Compatible |
|-------|------|----------|--------------|---------------|
| prime-7b-chat-v1 | 7B | Chat, Q&A | 4bit/8bit | ✅ |
| prime-7b-vision-v1 | 7B | Vision + Text | 8bit | ✅ |
| prime-13b-reasoning-v1 | 13B | Complex reasoning | 8bit | ⚠️ (slow) |

## Environment Support

| Environment | Mode | Models Supported |
|-------------|------|------------------|
| M1 Mac 16GB | Inference | 7B (quantized) |
| GCP 32GB VM | Training + Inference | All models |

## Integration with JARVIS

In your JARVIS backend:

```python
# Before: Direct model imports
# from transformers import AutoModelForCausalLM

# After: Use JARVIS Prime
from jarvis_prime import PrimeModel

class ClaudeChatbot:
    def __init__(self):
        self.model = PrimeModel.from_pretrained("prime-7b-chat-v1")

    def generate_response(self, prompt: str) -> str:
        return self.model.generate(prompt, max_length=512)
```

## Configuration Files

Example model configs in `jarvis_prime/configs/`:

- `prime_7b_chat_v1.yaml` - Chat/Q&A model
- `prime_7b_vision_v1.yaml` - Multimodal model
- `prime_13b_reasoning_v1.yaml` - Advanced reasoning

## Training Your Own PRIME Model

```bash
# 1. Prepare training data
python -m jarvis_prime.data.prepare_dataset \
    --input ./jarvis_logs/ \
    --output ./data/jarvis_train.jsonl

# 2. Train on GCP Spot VM (auto-resume enabled)
python -m jarvis_prime.train \
    --config configs/prime_7b_chat_v1.yaml \
    --data ./data/jarvis_train.jsonl \
    --output ./models/prime-7b-jarvis-custom

# 3. Export quantized version for M1
python -m jarvis_prime.quantize \
    --model ./models/prime-7b-jarvis-custom \
    --bits 8 \
    --output ./models/prime-7b-jarvis-custom-8bit
```

## Version Compatibility

| JARVIS Prime | Reactor Core | JARVIS |
|--------------|--------------|--------|
| v0.6.x | ≥ v1.0.0 | ≥ v2.0.0 |

## Dependencies

- **Reactor Core** v1.0+ (training engine)
- **PyTorch** 2.0+
- **Transformers** 4.30+
- **PEFT** 0.5+ (LoRA support)

## Performance

| Model | M1 Mac (8-bit) | GCP VM (Full) |
|-------|----------------|---------------|
| prime-7b-chat-v1 | 15 tokens/sec | 50 tokens/sec |
| prime-7b-vision-v1 | 10 tokens/sec | 40 tokens/sec |
| prime-13b-reasoning-v1 | 5 tokens/sec | 30 tokens/sec |

## License

MIT License

## Links

- **Reactor Core**: https://github.com/drussell23/reactor-core
- **JARVIS**: https://github.com/drussell23/JARVIS-AI-Agent
- **Model Hub**: Coming soon

---

Built with ❤️ for JARVIS
