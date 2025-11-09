# JARVIS Prime Quick Start

## ⚡ 30 Second Setup

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Set environment (IMPORTANT!)
export TRANSFORMERS_NO_TF=1

# 3. Test
python test_imports.py
```

## 🎓 Training on GCP (3 lines)

```python
from jarvis_prime.configs import LlamaPresets
from jarvis_prime.trainer import LlamaTrainer

config = LlamaPresets.llama_13b_gcp_training()
config.dataset_path = "./data/train.jsonl"

trainer = LlamaTrainer(config)
trainer.train()  # Auto-checkpoints every 30 min!
```

**Memory: 6.5GB** (4-bit QLoRA)

## 💬 Inference on M1 Mac (2 lines)

```python
from jarvis_prime.models import load_llama_13b_m1

model = load_llama_13b_m1()
print(model.generate("What is AI?"))
```

**Memory: ~13GB** (8-bit quantization)

## 📖 More Examples

- Full guide: [LLAMA_13B_GUIDE.md](LLAMA_13B_GUIDE.md)
- Examples: `examples/train_llama_13b.py`, `examples/inference_llama_13b.py`
- Configs: `examples/config_examples/*.yaml`

That's it! 🚀
