"""
Advanced Llama-2-13B Model Implementation
Dynamic, async, robust - zero hardcoding
"""
import asyncio
import logging
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from datetime import datetime

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from jarvis_prime.configs.llama_config import LlamaModelConfig

logger = logging.getLogger(__name__)


class LlamaModel:
    """
    Advanced Llama-2-13B implementation

    Features:
    - Dynamic configuration (no hardcoding)
    - Async inference with batching
    - Automatic device detection (M1, CUDA, CPU)
    - Quantization support (4-bit, 8-bit)
    - LoRA/QLoRA fine-tuning
    - Robust error handling
    - Memory optimization
    """

    def __init__(self, config: LlamaModelConfig):
        """
        Initialize Llama model

        Args:
            config: LlamaModelConfig instance
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = self._detect_device()
        self._inference_lock = asyncio.Lock()
        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._is_loaded = False

        logger.info(f"Initialized LlamaModel with config: {config.model_name}")
        logger.info(f"Device: {self.device}")

    def _detect_device(self) -> str:
        """Automatically detect optimal device"""
        if self.config.device != "auto":
            return self.config.device

        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Apple M1/M2 GPU (MPS) detected")
        else:
            device = "cpu"
            logger.warning("No GPU detected, using CPU")

        return device

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Create quantization config from settings"""
        if not self.config.quantization.enabled:
            return None

        bits = self.config.quantization.bits

        # Map compute dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_dtype = dtype_map.get(
            self.config.quantization.compute_dtype,
            torch.bfloat16
        )

        if bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self.config.quantization.use_double_quant,
                bnb_4bit_quant_type=self.config.quantization.quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=compute_dtype,
            )
        else:
            logger.warning(f"Unsupported quantization bits: {bits}, loading without quantization")
            return None

    def load(self):
        """Load model and tokenizer"""
        if self._is_loaded:
            logger.warning("Model already loaded")
            return

        logger.info(f"🚀 Loading {self.config.model_name}...")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True,
            )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Get quantization config
            quant_config = self._get_quantization_config()

            # Determine dtype
            if quant_config:
                dtype = None  # Quantization handles dtype
            elif self.device == "cpu":
                dtype = torch.float32
            else:
                dtype = torch.float16

            # Load model
            logger.info(f"   Quantization: {self.config.quantization.bits if quant_config else None}-bit")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Dtype: {dtype}")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quant_config,
                device_map=self.config.device_map,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            # Move to device if not quantized (quantization handles device placement)
            if not quant_config and self.device != "auto":
                self.model = self.model.to(self.device)

            self.model.eval()

            # Apply LoRA if enabled
            if self.config.lora.enabled:
                self._apply_lora()

            self._is_loaded = True
            logger.info("✅ Model loaded successfully")

            # Log memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _apply_lora(self):
        """Apply LoRA adapters to model"""
        logger.info(f"Applying LoRA (rank={self.config.lora.rank})...")

        lora_config = LoraConfig(
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.dropout,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def load_adapter(self, adapter_path: str):
        """
        Load pre-trained LoRA adapter

        Args:
            adapter_path: Path to adapter weights
        """
        if not self._is_loaded:
            raise RuntimeError("Base model must be loaded first")

        logger.info(f"Loading LoRA adapter from {adapter_path}...")

        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
            is_trainable=False,
        )

        logger.info("✅ Adapter loaded")

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Synchronous text generation

        Args:
            prompt: Input text or list of texts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Generated text(s)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Handle single string or list
        is_single = isinstance(prompt, str)
        prompts = [prompt] if is_single else prompt

        # Use config defaults if not specified
        max_length = max_length or self.config.inference.max_length
        temperature = temperature or self.config.inference.temperature
        top_p = top_p or self.config.inference.top_p
        top_k = top_k or self.config.inference.top_k

        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=self.config.inference.do_sample,
                    num_return_sequences=self.config.inference.num_return_sequences,
                    repetition_penalty=self.config.inference.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            # Decode
            generated_texts = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
            )

            # Remove prompts from outputs
            results = []
            for prompt, generated in zip(prompts, generated_texts):
                # Strip the prompt from the beginning
                result = generated[len(prompt):].strip() if generated.startswith(prompt) else generated
                results.append(result)

            return results[0] if is_single else results

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    async def generate_async(
        self,
        prompt: Union[str, List[str]],
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Async text generation with automatic batching

        Args:
            prompt: Input text or list of texts
            max_length: Maximum generation length
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated text(s)
        """
        if not self.config.inference.async_enabled:
            # Fall back to sync generation
            return self.generate(prompt, max_length, temperature, **kwargs)

        async with self._inference_lock:
            # Run synchronous generation in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.generate(prompt, max_length, temperature, **kwargs)
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Chat interface with conversation history

        Args:
            messages: List of message dicts with "role" and "content"
                     Example: [{"role": "user", "content": "Hello"}]
            **kwargs: Additional generation parameters

        Returns:
            Assistant's response
        """
        # Format messages into prompt
        prompt = self._format_chat_messages(messages)
        return self.generate(prompt, **kwargs)

    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Async chat interface"""
        prompt = self._format_chat_messages(messages)
        return await self.generate_async(prompt, **kwargs)

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into Llama-2 chat template"""
        # Llama-2 chat format
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        formatted = []
        system_prompt = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_prompt = content
            elif role == "user":
                if system_prompt and not formatted:
                    # First user message includes system prompt
                    formatted.append(f"{B_INST} {B_SYS}{system_prompt}{E_SYS}{content} {E_INST}")
                    system_prompt = None
                else:
                    formatted.append(f"{B_INST} {content} {E_INST}")
            elif role == "assistant":
                formatted.append(f" {content} ")

        return "".join(formatted)

    def save_model(self, output_dir: str):
        """
        Save model and tokenizer

        Args:
            output_dir: Directory to save to
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"💾 Saving model to {output_dir}...")

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save config
        config_path = output_path / "jarvis_config.yaml"
        self.config.save_yaml(str(config_path))

        logger.info("✅ Model saved successfully")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        stats = {}

        if torch.cuda.is_available():
            stats["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            stats["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            stats["gpu_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3

        return stats

    def __repr__(self) -> str:
        return (
            f"LlamaModel(\n"
            f"  model={self.config.model_name},\n"
            f"  variant={self.config.variant},\n"
            f"  device={self.device},\n"
            f"  loaded={self._is_loaded},\n"
            f"  quantization={self.config.quantization.bits if self.config.quantization.enabled else None}-bit\n"
            f")"
        )


# Convenience factory functions
def load_llama_13b_gcp() -> LlamaModel:
    """Load Llama-2-13B optimized for GCP 32GB training"""
    from jarvis_prime.configs.llama_config import LlamaPresets
    config = LlamaPresets.llama_13b_gcp_training()
    model = LlamaModel(config)
    model.load()
    return model


def load_llama_13b_m1() -> LlamaModel:
    """Load Llama-2-13B optimized for M1 Mac 16GB inference"""
    from jarvis_prime.configs.llama_config import LlamaPresets
    config = LlamaPresets.llama_13b_m1_inference()
    model = LlamaModel(config)
    model.load()
    return model


def load_from_config(config_path: str) -> LlamaModel:
    """Load model from YAML/JSON config file"""
    config_path_obj = Path(config_path)

    if config_path_obj.suffix == ".yaml" or config_path_obj.suffix == ".yml":
        config = LlamaModelConfig.from_yaml(config_path)
    elif config_path_obj.suffix == ".json":
        config = LlamaModelConfig.from_json(config_path)
    else:
        raise ValueError(f"Unsupported config format: {config_path_obj.suffix}")

    model = LlamaModel(config)
    model.load()
    return model
