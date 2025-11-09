# Advanced LLM Libraries for JARVIS Prime

## Overview

JARVIS Prime serves as the **LLM playground and training ground** for JARVIS, leveraging **GCP 32GB Spot VMs** for training and **M1 Mac 16GB** for inference.

---

## 🎯 Architecture Strategy

### **Training Environment: GCP 32GB Spot VM**
- Heavy lifting: Fine-tuning, training, experimentation
- Advanced techniques: LoRA, QLoRA, DPO, RLHF
- Large models: 7B-13B parameters (quantized)

### **Inference Environment: M1 Mac 16GB**
- Lightweight: Quantized models (4-bit, 8-bit)
- Fast inference: MPS acceleration
- Production-ready: Optimized for JARVIS runtime

---

## 📚 Essential LLM Libraries

### **1. Core Training & Fine-tuning**

#### **Transformers** (Hugging Face)
```toml
# pyproject.toml
dependencies = [
    "transformers>=4.35.0",
]
```

**Purpose:**
- Load pre-trained models (Llama, Mistral, Phi-3, etc.)
- Tokenization and model inference
- Model conversion and export

**Example:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
```

---

#### **PEFT** (Parameter-Efficient Fine-Tuning)
```toml
dependencies = [
    "peft>=0.7.0",
]
```

**Purpose:**
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Adapter layers
- Prefix tuning

**Example:**
```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # LoRA rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4M || all params: 7B || trainable%: 0.057%
```

---

#### **BitsAndBytes** (Quantization)
```toml
dependencies = [
    "bitsandbytes>=0.41.0",
]
```

**Purpose:**
- 8-bit and 4-bit quantization
- Reduce memory usage by 4-8x
- Enable larger models on 32GB RAM

**Example:**
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
# 13B model runs on 32GB RAM!
```

---

#### **Accelerate** (Distributed Training)
```toml
dependencies = [
    "accelerate>=0.25.0",
]
```

**Purpose:**
- Multi-GPU training
- Mixed precision (FP16, BF16)
- Gradient accumulation
- Device management

**Example:**
```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=4,
)

model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)
```

---

### **2. Advanced Training Techniques**

#### **TRL** (Transformer Reinforcement Learning)
```toml
dependencies = [
    "trl>=0.7.0",
]
```

**Purpose:**
- RLHF (Reinforcement Learning from Human Feedback)
- DPO (Direct Preference Optimization)
- PPO (Proximal Policy Optimization)
- Reward modeling

**Example - DPO:**
```python
from trl import DPOTrainer, DPOConfig

config = DPOConfig(
    beta=0.1,  # KL penalty coefficient
    learning_rate=5e-7,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
    args=config,
)

trainer.train()
```

**Example - RLHF:**
```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Create model with value head for RL
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    batch_size=16,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)
```

---

#### **Unsloth** (2x Faster Fine-tuning)
```toml
dependencies = [
    "unsloth>=2023.12",  # Optional, for speed
]
```

**Purpose:**
- 2x faster fine-tuning than standard methods
- Lower memory usage
- Compatible with Llama, Mistral, Phi-3

**Example:**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-2-7b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

---

### **3. Dataset & Data Processing**

#### **Datasets** (Hugging Face)
```toml
dependencies = [
    "datasets>=2.15.0",
]
```

**Purpose:**
- Load and process datasets
- Streaming for large datasets
- Format conversion

**Example:**
```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("timdettmers/openassistant-guanaco")

# Stream for large datasets
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", streaming=True)

# Custom JARVIS dataset
from datasets import Dataset

jarvis_data = Dataset.from_dict({
    "text": jarvis_conversations,
    "labels": jarvis_labels,
})
```

---

#### **Tokenizers** (Fast tokenization)
```toml
dependencies = [
    "tokenizers>=0.15.0",
]
```

**Purpose:**
- Fast Rust-based tokenization
- Custom tokenizer training
- BPE, WordPiece, etc.

---

### **4. Multi-modal Models**

#### **LLaVA** (Vision + Language)
```toml
dependencies = [
    "llava>=1.1.0",
]
```

**Purpose:**
- Image + text understanding
- Visual question answering
- Image captioning

**Example:**
```python
from transformers import LlavaForConditionalGeneration, AutoProcessor

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf"
)
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Process image + text
inputs = processor(
    text="What's in this image?",
    images=image,
    return_tensors="pt"
)

outputs = model.generate(**inputs)
```

---

#### **CLIP** (Contrastive Language-Image Pre-training)
```toml
dependencies = [
    "transformers>=4.35.0",  # Includes CLIP
]
```

**Purpose:**
- Image-text similarity
- Zero-shot image classification
- Embeddings for retrieval

---

### **5. Model Optimization & Serving**

#### **vLLM** (Fast Inference)
```toml
dependencies = [
    "vllm>=0.2.7",
]
```

**Purpose:**
- Ultra-fast inference
- PagedAttention for memory efficiency
- Continuous batching

**Example:**
```python
from vllm import LLM, SamplingParams

llm = LLM(model="prime-7b-chat-v1", quantization="awq")

prompts = ["What is AI?", "Explain quantum computing"]
outputs = llm.generate(prompts, sampling_params=SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
))
```

---

#### **GGML / llama.cpp** (CPU Inference)
```toml
dependencies = [
    "llama-cpp-python>=0.2.0",
]
```

**Purpose:**
- Run quantized models on CPU
- GGML/GGUF format
- M1 Mac optimized

**Example:**
```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/prime-7b-chat-q8_0.gguf",
    n_ctx=2048,
    n_gpu_layers=35,  # Offload to M1 GPU
)

output = llm("What is machine learning?", max_tokens=200)
```

---

#### **AutoGPTQ** (GPTQ Quantization)
```toml
dependencies = [
    "auto-gptq>=0.5.0",
]
```

**Purpose:**
- 4-bit quantization (better than bitsandbytes)
- Faster inference
- Lower memory

---

### **6. Experiment Tracking & Monitoring**

#### **Weights & Biases**
```toml
dependencies = [
    "wandb>=0.16.0",
]
```

**Purpose:**
- Track training experiments
- Visualize metrics
- Compare models

**Example:**
```python
import wandb

wandb.init(project="jarvis-prime", name="llama-7b-lora")

trainer = PrimeTrainer(config, callbacks=[WandbCallback()])
trainer.train()
```

---

#### **TensorBoard**
```toml
dependencies = [
    "tensorboard>=2.15.0",
]
```

**Purpose:**
- Local training visualization
- Loss curves, metrics

---

### **7. Specialized Libraries**

#### **LangChain** (LLM Applications)
```toml
dependencies = [
    "langchain>=0.1.0",
]
```

**Purpose:**
- Chain multiple LLM calls
- RAG (Retrieval-Augmented Generation)
- Agents and tools

---

#### **LlamaIndex** (Data Framework)
```toml
dependencies = [
    "llama-index>=0.9.0",
]
```

**Purpose:**
- Connect LLMs to data sources
- Build RAG applications
- Query engines

---

## 🎯 JARVIS Prime Dependency Configuration

### **Complete `pyproject.toml`**

```toml
[project]
name = "jarvis-prime"
version = "0.6.0"
dependencies = [
    # Core
    "torch>=2.1.0",
    "transformers>=4.35.0",

    # Efficient Fine-tuning
    "peft>=0.7.0",
    "bitsandbytes>=0.41.0",
    "accelerate>=0.25.0",

    # Advanced Training
    "trl>=0.7.0",  # DPO, RLHF

    # Data
    "datasets>=2.15.0",
    "tokenizers>=0.15.0",

    # Utilities
    "safetensors>=0.4.0",
    "sentencepiece>=0.1.99",
    "protobuf>=3.20.0",
]

[project.optional-dependencies]
training = [
    "reactor-core>=1.0.0",
    "wandb>=0.16.0",
    "tensorboard>=2.15.0",
]

multimodal = [
    "llava>=1.1.0",
    "torchvision>=0.16.0",
    "pillow>=10.0.0",
]

fast_inference = [
    "vllm>=0.2.7",
    "llama-cpp-python>=0.2.0",
    "auto-gptq>=0.5.0",
]

applications = [
    "langchain>=0.1.0",
    "llama-index>=0.9.0",
]

speed = [
    "unsloth>=2023.12",
    "flash-attn>=2.3.0",  # Faster attention
]

all = [
    "jarvis-prime[training,multimodal,fast_inference,applications,speed]",
]
```

---

## 🚀 Usage Examples

### **1. Train Llama 2 with QLoRA on GCP**

```python
from jarvis_prime import PrimeTrainer
from transformers import BitsAndBytesConfig
from peft import LoraConfig

# Quantization for 32GB RAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# LoRA config
lora_config = LoraConfig(
    r=64,  # Higher rank for better quality
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
)

# Train
trainer = PrimeTrainer(
    model_name="meta-llama/Llama-2-13b-hf",
    quantization_config=bnb_config,
    lora_config=lora_config,
    dataset="./data/jarvis_conversations.jsonl",
    output_dir="./models/prime-llama-13b-jarvis",
)

trainer.train()
trainer.export_quantized(quantization="8bit")  # For M1 Mac
```

---

### **2. Fine-tune with DPO (Preference Learning)**

```python
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# Load preference dataset
dataset = load_dataset("Anthropic/hh-rlhf")

config = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    max_length=512,
    max_prompt_length=256,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)

trainer.train()
```

---

### **3. Deploy on M1 Mac with GGML**

```python
from llama_cpp import Llama

# Load quantized model
llm = Llama(
    model_path="./models/prime-llama-13b-jarvis-q8_0.gguf",
    n_ctx=4096,
    n_gpu_layers=40,  # Use M1 GPU
    n_threads=8,
)

# Generate
output = llm(
    "You are JARVIS, an advanced AI assistant. Help the user.",
    max_tokens=512,
    temperature=0.7,
)

print(output["choices"][0]["text"])
```

---

## 📊 Memory Requirements

| Model | Full Precision | 8-bit | 4-bit (QLoRA) |
|-------|----------------|-------|---------------|
| Llama-2-7B | 28 GB | 7 GB | **3.5 GB** ✅ |
| Llama-2-13B | 52 GB | 13 GB | **6.5 GB** ✅ |
| Mistral-7B | 28 GB | 7 GB | **3.5 GB** ✅ |
| Phi-3-3.8B | 15 GB | 4 GB | **2 GB** ✅ |

**✅ = Runs on 32GB GCP VM**

---

## 🎯 Recommended Models for JARVIS

### **For Reasoning & Chat:**
- **Llama-2-13B** - Best quality
- **Mistral-7B** - Fast + quality balance
- **Phi-3-mini** - Efficient + powerful

### **For Vision:**
- **LLaVA-1.5-13B** - Image understanding
- **CLIP** - Image-text similarity

### **For Code:**
- **CodeLlama-13B** - Code generation
- **StarCoder** - Multi-language code

---

## 🏆 Summary

JARVIS Prime integrates:

✅ **Core Training:** Transformers, PEFT, BitsAndBytes, Accelerate
✅ **Advanced Techniques:** TRL (DPO, RLHF), Unsloth (2x speed)
✅ **Data Processing:** Datasets, Tokenizers
✅ **Multi-modal:** LLaVA, CLIP
✅ **Fast Inference:** vLLM, llama.cpp, AutoGPTQ
✅ **Applications:** LangChain, LlamaIndex
✅ **Monitoring:** Weights & Biases, TensorBoard

**Result:** Complete LLM playground for training on GCP 32GB and deploying to M1 Mac 16GB! 🚀
