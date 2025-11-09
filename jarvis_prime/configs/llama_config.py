"""
Dynamic configuration system for Llama models
Zero hardcoding - all settings loaded from environment/config files
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml
import json


@dataclass
class QuantizationConfig:
    """Quantization settings"""
    enabled: bool = True
    bits: int = 4  # 4, 8, 16
    quant_type: str = "nf4"  # nf4, fp4
    use_double_quant: bool = True
    compute_dtype: str = "bfloat16"  # float16, bfloat16, float32

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        """Load from dictionary"""
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class LoRAConfig:
    """LoRA/QLoRA configuration"""
    enabled: bool = True
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LoRAConfig":
        """Load from dictionary"""
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Training dynamics
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    num_epochs: int = 3
    max_steps: int = -1  # -1 means use num_epochs

    # Optimization
    optimizer: str = "adamw_torch"  # adamw_torch, adamw_8bit, sgd
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"  # linear, cosine, constant

    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # fp16, bf16, no

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3

    # Advanced
    group_by_length: bool = True
    dataloader_num_workers: int = 4
    seed: int = 42

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TrainingConfig":
        """Load from dictionary"""
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class InferenceConfig:
    """Inference settings"""
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.15
    do_sample: bool = True
    num_return_sequences: int = 1

    # Batching
    batch_size: int = 8
    max_batch_size: int = 32

    # Async settings
    async_enabled: bool = True
    max_concurrent_requests: int = 100
    timeout_seconds: float = 30.0

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "InferenceConfig":
        """Load from dictionary"""
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class CheckpointConfig:
    """Checkpointing and recovery (for GCP Spot VMs)"""
    enabled: bool = True
    checkpoint_dir: str = "./checkpoints"
    save_frequency_steps: int = 500
    save_frequency_minutes: int = 30
    max_checkpoints: int = 5

    # GCP Spot VM specific
    preemption_check_interval: int = 60  # seconds
    preemption_signal_file: str = "/tmp/preemption_signal"
    auto_resume: bool = True

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "CheckpointConfig":
        """Load from dictionary"""
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class MonitoringConfig:
    """Monitoring and logging"""
    # Experiment tracking
    wandb_enabled: bool = False
    wandb_project: str = "jarvis-prime"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    tensorboard_enabled: bool = True
    tensorboard_dir: str = "./runs"

    # Metrics
    track_memory: bool = True
    track_gpu_utilization: bool = True
    track_throughput: bool = True

    # Alerts
    alert_on_nan: bool = True
    alert_on_high_loss: bool = True
    high_loss_threshold: float = 10.0

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "MonitoringConfig":
        """Load from dictionary"""
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})


@dataclass
class LlamaModelConfig:
    """
    Complete Llama model configuration
    Zero hardcoding - all values from environment or config files
    """
    # Model identity
    model_name: str = "meta-llama/Llama-2-13b-hf"
    model_type: str = "llama"
    variant: str = "13b"  # 7b, 13b, 70b

    # Device management
    device: str = "auto"  # auto, cuda, mps, cpu
    device_map: str = "auto"  # auto, balanced, sequential

    # Sub-configurations
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Data settings
    max_seq_length: int = 4096
    dataset_path: Optional[str] = None
    dataset_split: str = "train"

    # Output
    output_dir: str = "./outputs"
    model_save_name: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LlamaModelConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        return cls._from_dict(config)

    @classmethod
    def from_json(cls, json_path: str) -> "LlamaModelConfig":
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config = json.load(f)

        return cls._from_dict(config)

    @classmethod
    def from_env(cls) -> "LlamaModelConfig":
        """Load configuration from environment variables"""
        config = {}

        # Model settings
        if model_name := os.getenv("JARVIS_MODEL_NAME"):
            config["model_name"] = model_name
        if variant := os.getenv("JARVIS_MODEL_VARIANT"):
            config["variant"] = variant
        if device := os.getenv("JARVIS_DEVICE"):
            config["device"] = device

        # Quantization
        quant_config = {}
        if os.getenv("JARVIS_QUANT_ENABLED"):
            quant_config["enabled"] = os.getenv("JARVIS_QUANT_ENABLED").lower() == "true"
        if quant_bits := os.getenv("JARVIS_QUANT_BITS"):
            quant_config["bits"] = int(quant_bits)
        if quant_config:
            config["quantization"] = quant_config

        # Training
        train_config = {}
        if batch_size := os.getenv("JARVIS_BATCH_SIZE"):
            train_config["batch_size"] = int(batch_size)
        if lr := os.getenv("JARVIS_LEARNING_RATE"):
            train_config["learning_rate"] = float(lr)
        if train_config:
            config["training"] = train_config

        # Dataset
        if dataset_path := os.getenv("JARVIS_DATASET_PATH"):
            config["dataset_path"] = dataset_path
        if output_dir := os.getenv("JARVIS_OUTPUT_DIR"):
            config["output_dir"] = output_dir

        return cls._from_dict(config)

    @classmethod
    def _from_dict(cls, config: Dict[str, Any]) -> "LlamaModelConfig":
        """Internal method to construct from dictionary"""
        # Handle nested configs
        if "quantization" in config and isinstance(config["quantization"], dict):
            config["quantization"] = QuantizationConfig.from_dict(config["quantization"])
        if "lora" in config and isinstance(config["lora"], dict):
            config["lora"] = LoRAConfig.from_dict(config["lora"])
        if "training" in config and isinstance(config["training"], dict):
            config["training"] = TrainingConfig.from_dict(config["training"])
        if "inference" in config and isinstance(config["inference"], dict):
            config["inference"] = InferenceConfig.from_dict(config["inference"])
        if "checkpoint" in config and isinstance(config["checkpoint"], dict):
            config["checkpoint"] = CheckpointConfig.from_dict(config["checkpoint"])
        if "monitoring" in config and isinstance(config["monitoring"], dict):
            config["monitoring"] = MonitoringConfig.from_dict(config["monitoring"])

        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "variant": self.variant,
            "device": self.device,
            "device_map": self.device_map,
            "max_seq_length": self.max_seq_length,
            "dataset_path": self.dataset_path,
            "dataset_split": self.dataset_split,
            "output_dir": self.output_dir,
            "model_save_name": self.model_save_name,
            "quantization": self.quantization.__dict__,
            "lora": self.lora.__dict__,
            "training": self.training.__dict__,
            "inference": self.inference.__dict__,
            "checkpoint": self.checkpoint.__dict__,
            "monitoring": self.monitoring.__dict__,
        }

    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML"""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def save_json(self, json_path: str):
        """Save configuration to JSON"""
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Preset configurations
class LlamaPresets:
    """Pre-configured Llama model presets"""

    @staticmethod
    def llama_13b_gcp_training() -> LlamaModelConfig:
        """Llama-2-13B optimized for GCP 32GB training"""
        return LlamaModelConfig(
            model_name="meta-llama/Llama-2-13b-hf",
            variant="13b",
            device="auto",
            quantization=QuantizationConfig(bits=4, compute_dtype="bfloat16"),
            lora=LoRAConfig(rank=64, alpha=128),
            training=TrainingConfig(
                batch_size=4,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                gradient_checkpointing=True,
            ),
            checkpoint=CheckpointConfig(enabled=True, save_frequency_minutes=30),
            monitoring=MonitoringConfig(wandb_enabled=True, tensorboard_enabled=True),
        )

    @staticmethod
    def llama_13b_m1_inference() -> LlamaModelConfig:
        """Llama-2-13B optimized for M1 Mac 16GB inference"""
        return LlamaModelConfig(
            model_name="meta-llama/Llama-2-13b-hf",
            variant="13b",
            device="mps",
            quantization=QuantizationConfig(bits=8, compute_dtype="float16"),
            inference=InferenceConfig(
                batch_size=4,
                async_enabled=True,
                max_concurrent_requests=50,
            ),
            monitoring=MonitoringConfig(tensorboard_enabled=False),
        )

    @staticmethod
    def llama_7b_fast_training() -> LlamaModelConfig:
        """Llama-2-7B for fast iteration"""
        return LlamaModelConfig(
            model_name="meta-llama/Llama-2-7b-hf",
            variant="7b",
            quantization=QuantizationConfig(bits=4),
            lora=LoRAConfig(rank=32, alpha=64),
            training=TrainingConfig(
                batch_size=8,
                gradient_accumulation_steps=2,
                num_epochs=1,
            ),
        )
