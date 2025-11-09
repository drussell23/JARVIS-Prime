# Llama-2-13B Advanced Implementation Guide

## 🚀 Overview

This guide covers the **advanced, robust, async, and dynamic** Llama-2-13B implementation for JARVIS Prime. Zero hardcoding, fully configurable, production-ready.

## ✨ Features

✅ **Zero Hardcoding** - All configuration from YAML/JSON/environment variables
✅ **Async Inference** - Concurrent requests with automatic batching
✅ **Dynamic Configuration** - Load settings from files or environment
✅ **QLoRA Training** - 4-bit quantization for 32GB GCP VMs
✅ **DPO/RLHF Support** - Advanced preference learning
✅ **GCP Spot VM Recovery** - Automatic checkpointing and resume
✅ **Monitoring** - W&B, TensorBoard, memory tracking
✅ **M1 Mac Optimized** - 8-bit inference on 16GB Mac

---

## 📁 File Structure

```
jarvis_prime/
├── configs/
│   └── llama_config.py          # Dynamic configuration system
├── models/
│   ├── llama_model.py           # Advanced Llama model implementation
│   └── prime_model.py           # Legacy PRIME model
└── trainer/
    ├── llama_trainer.py         # QLoRA, DPO, RLHF trainer
    └── prime_trainer.py         # Legacy trainer

examples/
├── train_llama_13b.py           # QLoRA training example
├── inference_llama_13b.py       # Inference examples (sync/async)
├── train_dpo.py                 # DPO training example
└── config_examples/
    ├── llama_13b_gcp.yaml       # GCP 32GB training config
    └── llama_13b_m1.yaml        # M1 Mac inference config
```

---

## 🎯 Quick Start

### 1. Training on GCP 32GB Spot VM

```python
from jarvis_prime.configs import LlamaPresets
from jarvis_prime.trainer import LlamaTrainer

# Load preset configuration
config = LlamaPresets.llama_13b_gcp_training()

# Customize
config.dataset_path = "./data/jarvis_conversations.jsonl"
config.output_dir = "./outputs/llama-13b-jarvis"

# Train
trainer = LlamaTrainer(config)
trainer.train()
```

**Memory Usage:** ~6.5GB with 4-bit quantization ✅

### 2. Inference on M1 Mac 16GB

```python
from jarvis_prime.models import load_llama_13b_m1

# Load optimized model
model = load_llama_13b_m1()

# Generate
response = model.generate(
    "What is artificial intelligence?",
    max_length=512,
    temperature=0.7
)
print(response)
```

**Memory Usage:** ~13GB with 8-bit quantization ✅

### 3. Async Inference (Multiple Requests)

```python
import asyncio
from jarvis_prime.models import load_llama_13b_m1

model = load_llama_13b_m1()

# Concurrent requests
prompts = [
    "Explain quantum computing",
    "What is machine learning?",
    "How does AI work?"
]

async def generate_all():
    tasks = [model.generate_async(p) for p in prompts]
    return await asyncio.gather(*tasks)

responses = asyncio.run(generate_all())
```

---

## ⚙️ Configuration System

### Three Ways to Configure

#### 1. **Presets** (Recommended)

```python
from jarvis_prime.configs import LlamaPresets

# GCP 32GB training
config = LlamaPresets.llama_13b_gcp_training()

# M1 Mac inference
config = LlamaPresets.llama_13b_m1_inference()

# Fast iteration (7B model)
config = LlamaPresets.llama_7b_fast_training()
```

#### 2. **YAML/JSON Files**

```python
from jarvis_prime.configs import LlamaModelConfig

# Load from YAML
config = LlamaModelConfig.from_yaml("configs/llama_13b_gcp.yaml")

# Load from JSON
config = LlamaModelConfig.from_json("configs/llama_13b.json")
```

#### 3. **Environment Variables**

```bash
export JARVIS_MODEL_NAME="meta-llama/Llama-2-13b-hf"
export JARVIS_QUANT_BITS=4
export JARVIS_BATCH_SIZE=4
export JARVIS_LEARNING_RATE=0.0002
export JARVIS_DATASET_PATH="./data/train.jsonl"
```

```python
from jarvis_prime.configs import LlamaModelConfig

config = LlamaModelConfig.from_env()
```

### Save Configuration

```python
# Save to YAML
config.save_yaml("my_config.yaml")

# Save to JSON
config.save_json("my_config.json")
```

---

## 🧠 Model Features

### Dynamic Model Loading

```python
from jarvis_prime.models import LlamaModel
from jarvis_prime.configs import LlamaModelConfig

# Create config
config = LlamaModelConfig(
    model_name="meta-llama/Llama-2-13b-hf",
    device="auto",  # Auto-detect CUDA/MPS/CPU
)

# Customize quantization
config.quantization.enabled = True
config.quantization.bits = 4
config.quantization.compute_dtype = "bfloat16"

# Customize LoRA
config.lora.enabled = True
config.lora.rank = 64
config.lora.alpha = 128

# Load model
model = LlamaModel(config)
model.load()
```

### Quantization Options

| Bits | Memory (13B) | Quality | Use Case |
|------|-------------|---------|----------|
| 16   | 26 GB       | Best    | Not practical |
| 8    | 13 GB       | Excellent | M1 Mac inference |
| 4    | 6.5 GB      | Good    | GCP training |

### LoRA Configuration

```python
config.lora.rank = 64              # Higher = more parameters, better quality
config.lora.alpha = 128            # Scaling factor (typically 2x rank)
config.lora.dropout = 0.05         # Regularization
config.lora.target_modules = [     # Which layers to adapt
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

---

## 🎓 Training Features

### QLoRA Training

```python
from jarvis_prime.trainer import LlamaTrainer
from jarvis_prime.configs import LlamaPresets

config = LlamaPresets.llama_13b_gcp_training()
config.dataset_path = "./data/train.jsonl"

trainer = LlamaTrainer(config)
trainer.train()
```

### DPO (Direct Preference Optimization)

```python
trainer = LlamaTrainer(config)
trainer.train_dpo(
    preference_dataset_path="Anthropic/hh-rlhf",
    beta=0.1,  # KL penalty
)
```

**Dataset Format:**
```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "France's capital is London."
}
```

### RLHF (Reinforcement Learning from Human Feedback)

```python
trainer = LlamaTrainer(config)
trainer.train_rlhf(
    reward_model_path="./models/reward_model",
    dataset_path="./data/prompts.jsonl"
)
```

---

## 💾 GCP Spot VM Recovery

### Automatic Checkpointing

```python
config.checkpoint.enabled = True
config.checkpoint.save_frequency_minutes = 30  # Save every 30 min
config.checkpoint.save_frequency_steps = 500   # Or every 500 steps
config.checkpoint.auto_resume = True           # Resume automatically
config.checkpoint.max_checkpoints = 5          # Keep last 5
```

### Preemption Handling

The trainer automatically:
1. **Detects SIGTERM** from GCP
2. **Saves checkpoint** immediately
3. **Writes signal file** for monitoring
4. **Auto-resumes** on next startup

```python
# Training will automatically resume from latest checkpoint
trainer.train()  # Handles everything!
```

### Manual Resume

```python
trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-epoch2-step500")
```

---

## 📊 Monitoring & Logging

### Weights & Biases

```python
config.monitoring.wandb_enabled = True
config.monitoring.wandb_project = "jarvis-prime"
config.monitoring.wandb_run_name = "llama-13b-qlora-v1"
```

### TensorBoard

```python
config.monitoring.tensorboard_enabled = True
config.monitoring.tensorboard_dir = "./runs"
```

```bash
# View in browser
tensorboard --logdir=./runs
```

### Memory Tracking

```python
memory = model.get_memory_usage()
print(memory)
# {
#   'gpu_allocated_gb': 6.5,
#   'gpu_reserved_gb': 7.2,
#   'gpu_max_allocated_gb': 6.8
# }
```

---

## 🔧 Advanced Usage

### Chat Interface

```python
messages = [
    {"role": "system", "content": "You are JARVIS, an AI assistant."},
    {"role": "user", "content": "Hello!"},
]

response = model.chat(messages)
print(response)

# Continue conversation
messages.append({"role": "assistant", "content": response})
messages.append({"role": "user", "content": "What can you do?"})

response = model.chat(messages)
```

### Load Fine-tuned Adapter

```python
model = load_llama_13b_m1()

# Load your trained LoRA adapter
model.load_adapter("./outputs/llama-13b-jarvis")

# Now use fine-tuned model
response = model.generate("Hello JARVIS!")
```

### Batch Generation

```python
prompts = [
    "What is AI?",
    "Explain machine learning",
    "How do neural networks work?"
]

responses = model.generate(prompts, max_length=256)

for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

### Custom Generation Parameters

```python
response = model.generate(
    "Tell me a story",
    max_length=1024,
    temperature=0.9,        # Higher = more creative
    top_p=0.95,             # Nucleus sampling
    top_k=50,               # Top-k sampling
    repetition_penalty=1.2, # Reduce repetition
    do_sample=True
)
```

---

## 📋 Training Workflow

### Complete Training Pipeline

```python
from jarvis_prime.configs import LlamaPresets
from jarvis_prime.trainer import LlamaTrainer

# 1. Configure
config = LlamaPresets.llama_13b_gcp_training()
config.dataset_path = "./data/jarvis_conversations.jsonl"
config.output_dir = "./outputs/llama-13b-jarvis-v1"

# 2. Enable monitoring
config.monitoring.wandb_enabled = True
config.monitoring.wandb_project = "jarvis-prime"

# 3. Enable checkpointing
config.checkpoint.enabled = True
config.checkpoint.save_frequency_minutes = 30

# 4. Train
trainer = LlamaTrainer(config)
trainer.train()

# 5. Model is saved to config.output_dir
print(f"Model saved to: {config.output_dir}")
```

### Deploy to M1 Mac

```python
from jarvis_prime.models import load_from_config

# Load your fine-tuned model
model = load_from_config("./outputs/llama-13b-jarvis-v1/jarvis_config.yaml")

# Or load with adapter
from jarvis_prime.models import load_llama_13b_m1
model = load_llama_13b_m1()
model.load_adapter("./outputs/llama-13b-jarvis-v1")

# Use for inference
response = model.generate("Hello JARVIS, what can you do?")
```

---

## 🎯 Best Practices

### For Training (GCP 32GB)

✅ Use **4-bit quantization** to fit 13B model
✅ **LoRA rank 64-128** for good quality
✅ **Gradient accumulation** to simulate larger batches
✅ **Enable checkpointing** for Spot VM recovery
✅ **Monitor with W&B** to track experiments
✅ **Save every 30 minutes** for preemption safety

### For Inference (M1 Mac 16GB)

✅ Use **8-bit quantization** for best quality/memory balance
✅ **Disable LoRA** for inference (or load adapters)
✅ **Enable async** for responsive UI
✅ **Batch requests** when possible
✅ **Load adapters** for specialized tasks

### Configuration Management

✅ **Use YAML files** for reproducibility
✅ **Save configs** with trained models
✅ **Use presets** for quick iteration
✅ **Environment variables** for CI/CD

---

## 🐛 Troubleshooting

### Out of Memory

```python
# Reduce batch size
config.training.batch_size = 2

# Increase gradient accumulation
config.training.gradient_accumulation_steps = 8

# Use more aggressive quantization
config.quantization.bits = 4

# Enable gradient checkpointing
config.training.gradient_checkpointing = True
```

### Slow Training

```python
# Increase batch size
config.training.batch_size = 8

# Reduce gradient accumulation
config.training.gradient_accumulation_steps = 2

# Use mixed precision
config.training.mixed_precision = "bf16"

# Reduce LoRA rank
config.lora.rank = 32
```

### Model Not Loading

```python
# Check device
print(f"Device: {model.device}")

# Check memory
import torch
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

# Verify model name
config.model_name = "meta-llama/Llama-2-13b-hf"  # Correct format
```

---

## 📚 Example Scripts

All examples are in the `examples/` directory:

- **`train_llama_13b.py`** - QLoRA training on GCP
- **`inference_llama_13b.py`** - Sync/async inference on M1
- **`train_dpo.py`** - DPO preference learning
- **`config_examples/`** - YAML configuration templates

Run examples:

```bash
# Training
python examples/train_llama_13b.py

# Inference
python examples/inference_llama_13b.py

# DPO training
python examples/train_dpo.py
```

---

## 🏆 Summary

The advanced Llama-2-13B implementation provides:

✅ **Production-ready** - Robust, tested, documented
✅ **Zero hardcoding** - Fully configurable
✅ **Async by default** - High performance
✅ **GCP Spot VM safe** - Auto-recovery
✅ **M1 Mac optimized** - Efficient inference
✅ **Advanced techniques** - QLoRA, DPO, RLHF

**Ready to train on your 32GB GCP Spot VMs and deploy to your 16GB M1 Mac! 🚀**
