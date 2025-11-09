"""
PrimeModel - Unified interface for JARVIS PRIME models
"""
import torch
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PrimeConfig:
    """Configuration for PRIME models"""
    model_name: str
    quantization: Optional[str] = None  # None, "4bit", "8bit"
    device: str = "auto"  # auto, cpu, cuda, mps
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


class PrimeModel:
    """
    Unified interface for JARVIS PRIME models

    Supports:
    - Automatic device detection (M1 MPS, CUDA, CPU)
    - Quantization for memory efficiency
    - Simple from_pretrained() API
    """

    AVAILABLE_MODELS = {
        "prime-7b-chat-v1": {
            "base": "meta-llama/Llama-2-7b-chat-hf",
            "description": "Chat and Q&A optimized for JARVIS",
            "size_gb": 13.5,
            "recommended_ram": 16,
        },
        "prime-7b-vision-v1": {
            "base": "llava-hf/llava-1.5-7b-hf",
            "description": "Multimodal vision + text",
            "size_gb": 14.0,
            "recommended_ram": 16,
        },
        "prime-13b-reasoning-v1": {
            "base": "meta-llama/Llama-2-13b-hf",
            "description": "Advanced reasoning and analysis",
            "size_gb": 26.0,
            "recommended_ram": 32,
        },
    }

    def __init__(self, config: PrimeConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = self._detect_device()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        quantization: Optional[str] = None,
        device: str = "auto",
        **kwargs
    ) -> "PrimeModel":
        """
        Load a pre-trained PRIME model

        Args:
            model_name: Name of the model (e.g., "prime-7b-chat-v1")
            quantization: Quantization mode ("4bit", "8bit", or None)
            device: Device to load model on ("auto", "cpu", "cuda", "mps")
            **kwargs: Additional configuration options

        Returns:
            Loaded PrimeModel instance

        Example:
            >>> model = PrimeModel.from_pretrained("prime-7b-chat-v1", quantization="8bit")
            >>> response = model.generate("What is AI?")
        """
        config = PrimeConfig(
            model_name=model_name,
            quantization=quantization,
            device=device,
            **kwargs
        )

        instance = cls(config)
        instance._load_model()

        return instance

    def _detect_device(self) -> str:
        """Detect optimal device"""
        if self.config.device != "auto":
            return self.config.device

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_model(self):
        """Load model and tokenizer"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.config.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model: {self.config.model_name}. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )

        model_info = self.AVAILABLE_MODELS[self.config.model_name]
        base_model = model_info["base"]

        print(f"🚀 Loading {self.config.model_name}...")
        print(f"   Base: {base_model}")
        print(f"   Device: {self.device}")
        print(f"   Quantization: {self.config.quantization or 'None'}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization if specified
        load_kwargs = {}

        if self.config.quantization == "4bit":
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.quantization == "8bit":
            load_kwargs["load_in_8bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            **load_kwargs
        )

        # Move to device if not quantized (quantization handles device placement)
        if not self.config.quantization:
            self.model = self.model.to(self.device)

        self.model.eval()
        print(f"✅ Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt

        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call from_pretrained() first.")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length or self.config.max_length,
                temperature=temperature or self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
                **kwargs
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from output
        generated_text = generated_text[len(prompt):].strip()

        return generated_text

    def chat(self, messages: list[Dict[str, str]]) -> str:
        """
        Chat interface with conversation history

        Args:
            messages: List of message dicts with "role" and "content"
                      Example: [{"role": "user", "content": "Hello"}]

        Returns:
            Assistant's response
        """
        # Format messages into prompt
        prompt = self._format_chat_messages(messages)

        return self.generate(prompt)

    def _format_chat_messages(self, messages: list[Dict[str, str]]) -> str:
        """Format chat messages into prompt"""
        formatted = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            elif role == "system":
                formatted.append(f"System: {content}")

        formatted.append("Assistant:")
        return "\n".join(formatted)

    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all available PRIME models"""
        return cls.AVAILABLE_MODELS


if __name__ == "__main__":
    # Example usage
    print("Available PRIME models:")
    for name, info in PrimeModel.list_models().items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Size: {info['size_gb']} GB")
        print(f"  Recommended RAM: {info['recommended_ram']} GB")
