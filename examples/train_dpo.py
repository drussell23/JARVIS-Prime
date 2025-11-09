#!/usr/bin/env python3
"""
Example: Train Llama-2-13B with DPO (Direct Preference Optimization)

DPO is a simpler alternative to RLHF that directly optimizes for human preferences
without needing a separate reward model.

This example demonstrates:
- Loading preference dataset
- Training with DPO
- Fine-tuning on human feedback
"""
import logging

from jarvis_prime.configs.llama_config import LlamaModelConfig, LlamaPresets
from jarvis_prime.trainer.llama_trainer import LlamaTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train with DPO"""

    # Load config
    config = LlamaPresets.llama_13b_gcp_training()

    # Customize for DPO
    config.output_dir = "./outputs/llama-13b-dpo"
    config.model_save_name = "llama-13b-dpo-v1"

    # DPO works better with lower learning rate
    config.training.learning_rate = 5e-7
    config.training.num_epochs = 1

    # Monitoring
    config.monitoring.wandb_enabled = True
    config.monitoring.wandb_project = "jarvis-prime"
    config.monitoring.wandb_run_name = "llama-13b-dpo-v1"

    # Create trainer
    logger.info("🚀 Initializing DPO Trainer...")
    trainer = LlamaTrainer(config)

    # Train with DPO
    # Dataset should have format:
    # {
    #   "prompt": "...",
    #   "chosen": "preferred response",
    #   "rejected": "less preferred response"
    # }
    logger.info("🎯 Starting DPO training...")
    trainer.train_dpo(
        preference_dataset_path="Anthropic/hh-rlhf",  # Example dataset
        beta=0.1,  # KL penalty coefficient
    )

    logger.info("✅ DPO training complete!")
    logger.info(f"📁 Model saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
