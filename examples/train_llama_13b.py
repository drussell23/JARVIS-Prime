#!/usr/bin/env python3
"""
Example: Train Llama-2-13B with QLoRA on GCP 32GB Spot VM

This example demonstrates:
- Loading Llama-2-13B with 4-bit quantization
- Applying LoRA adapters
- Training on custom dataset
- Automatic checkpointing for Spot VM recovery
- Monitoring with W&B and TensorBoard
"""
import logging
from pathlib import Path

from jarvis_prime.configs.llama_config import LlamaModelConfig, LlamaPresets
from jarvis_prime.trainer.llama_trainer import LlamaTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train Llama-2-13B with QLoRA"""

    # Option 1: Use preset configuration
    config = LlamaPresets.llama_13b_gcp_training()

    # Option 2: Load from YAML file
    # config = LlamaModelConfig.from_yaml("configs/llama_13b_config.yaml")

    # Option 3: Load from environment variables
    # config = LlamaModelConfig.from_env()

    # Customize configuration
    config.dataset_path = "./data/jarvis_conversations.jsonl"  # Your dataset
    config.output_dir = "./outputs/llama-13b-jarvis"
    config.model_save_name = "llama-13b-jarvis-v1"

    # Training settings
    config.training.num_epochs = 3
    config.training.batch_size = 4
    config.training.gradient_accumulation_steps = 4
    config.training.learning_rate = 2e-4

    # LoRA settings
    config.lora.rank = 64
    config.lora.alpha = 128
    config.lora.dropout = 0.05

    # Monitoring
    config.monitoring.wandb_enabled = True
    config.monitoring.wandb_project = "jarvis-prime"
    config.monitoring.wandb_run_name = "llama-13b-qlora-v1"

    # Checkpointing for Spot VMs
    config.checkpoint.enabled = True
    config.checkpoint.save_frequency_minutes = 30
    config.checkpoint.auto_resume = True

    # Save config for reference
    config.save_yaml("./outputs/llama-13b-jarvis/config.yaml")

    # Create trainer
    logger.info("🚀 Initializing Llama Trainer...")
    trainer = LlamaTrainer(config)

    # Train
    logger.info("🎯 Starting training...")
    trainer.train()

    logger.info("✅ Training complete!")
    logger.info(f"📁 Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
