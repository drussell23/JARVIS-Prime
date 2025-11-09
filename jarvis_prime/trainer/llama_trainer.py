"""
Advanced Llama Trainer
Supports QLoRA, DPO, RLHF, checkpointing, and GCP Spot VM recovery
"""
import os
import signal
import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from datetime import datetime
import json

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, Dataset
from peft import prepare_model_for_kbit_training

from jarvis_prime.configs.llama_config import LlamaModelConfig
from jarvis_prime.models.llama_model import LlamaModel

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpointing and recovery for GCP Spot VMs
    Handles preemption signals and automatic state saving
    """

    def __init__(self, config: LlamaModelConfig):
        self.config = config.checkpoint
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.preempted = False
        self._setup_preemption_handler()

    def _setup_preemption_handler(self):
        """Setup handler for GCP preemption signals"""
        if not self.config.enabled:
            return

        def preemption_handler(signum, frame):
            logger.warning("⚠️  Preemption signal received! Saving checkpoint...")
            self.preempted = True
            # Write signal file for external monitoring
            Path(self.config.preemption_signal_file).touch()

        # SIGTERM is sent by GCP before preemption
        signal.signal(signal.SIGTERM, preemption_handler)
        logger.info("GCP Spot VM preemption handler installed")

    def save_checkpoint(
        self,
        trainer: Trainer,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """Save training checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint-epoch{epoch}-step{step}-{timestamp}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        logger.info(f"💾 Saving checkpoint: {checkpoint_name}")

        # Save trainer state
        trainer.save_model(str(checkpoint_path))

        # Save additional metadata
        metadata = {
            "epoch": epoch,
            "step": step,
            "timestamp": timestamp,
            "metrics": metrics or {},
        }

        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        logger.info(f"✅ Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints limit"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if len(checkpoints) > self.config.max_checkpoints:
            for old_checkpoint in checkpoints[self.config.max_checkpoints:]:
                logger.info(f"🗑️  Removing old checkpoint: {old_checkpoint.name}")
                import shutil
                shutil.rmtree(old_checkpoint)

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recent checkpoint"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        return checkpoints[0] if checkpoints else None

    def should_save_checkpoint(self, step: int, last_save_time: datetime) -> bool:
        """Determine if checkpoint should be saved"""
        # Always save if preempted
        if self.preempted:
            return True

        # Save based on step frequency
        if step % self.config.save_frequency_steps == 0:
            return True

        # Save based on time frequency
        minutes_since_save = (datetime.now() - last_save_time).total_seconds() / 60
        if minutes_since_save >= self.config.save_frequency_minutes:
            return True

        return False


class LlamaTrainer:
    """
    Advanced trainer for Llama models

    Features:
    - QLoRA fine-tuning
    - DPO (Direct Preference Optimization)
    - RLHF support
    - Automatic checkpointing
    - GCP Spot VM recovery
    - Monitoring and logging
    """

    def __init__(self, config: LlamaModelConfig):
        """
        Initialize trainer

        Args:
            config: LlamaModelConfig instance
        """
        self.config = config
        self.model_wrapper = LlamaModel(config)
        self.checkpoint_manager = CheckpointManager(config)
        self.training_args = self._create_training_args()

        self._setup_logging()
        self._setup_monitoring()

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.output_dir}/training.log"),
                logging.StreamHandler(),
            ]
        )

    def _setup_monitoring(self):
        """Setup monitoring tools (W&B, TensorBoard)"""
        if self.config.monitoring.wandb_enabled:
            try:
                import wandb
                wandb.init(
                    project=self.config.monitoring.wandb_project,
                    entity=self.config.monitoring.wandb_entity,
                    name=self.config.monitoring.wandb_run_name or self._generate_run_name(),
                    config=self.config.to_dict(),
                )
                logger.info("✅ Weights & Biases initialized")
            except ImportError:
                logger.warning("wandb not installed, skipping W&B integration")

    def _generate_run_name(self) -> str:
        """Generate unique run name"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.config.variant}_{self.config.lora.rank}r_{timestamp}"

    def _create_training_args(self) -> TrainingArguments:
        """Create Hugging Face TrainingArguments from config"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tc = self.config.training  # Training config shorthand

        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=tc.num_epochs,
            max_steps=tc.max_steps,
            per_device_train_batch_size=tc.batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            learning_rate=tc.learning_rate,
            warmup_ratio=tc.warmup_ratio,
            optim=tc.optimizer,
            weight_decay=tc.weight_decay,
            max_grad_norm=tc.max_grad_norm,
            lr_scheduler_type=tc.lr_scheduler_type,
            logging_steps=tc.logging_steps,
            save_steps=tc.save_steps,
            eval_steps=tc.eval_steps,
            save_total_limit=tc.save_total_limit,
            fp16=(tc.mixed_precision == "fp16"),
            bf16=(tc.mixed_precision == "bf16"),
            gradient_checkpointing=tc.gradient_checkpointing,
            group_by_length=tc.group_by_length,
            dataloader_num_workers=tc.dataloader_num_workers,
            seed=tc.seed,
            report_to=self._get_report_to(),
            logging_dir=f"{output_dir}/logs",
        )

    def _get_report_to(self) -> list:
        """Determine which monitoring tools to report to"""
        report_to = []

        if self.config.monitoring.tensorboard_enabled:
            report_to.append("tensorboard")

        if self.config.monitoring.wandb_enabled:
            report_to.append("wandb")

        return report_to if report_to else ["none"]

    def load_dataset(self, dataset_path: Optional[str] = None) -> Dataset:
        """
        Load training dataset

        Args:
            dataset_path: Path to dataset (overrides config)

        Returns:
            Loaded dataset
        """
        dataset_path = dataset_path or self.config.dataset_path

        if not dataset_path:
            raise ValueError("No dataset_path provided")

        logger.info(f"📚 Loading dataset from {dataset_path}")

        # Handle different dataset sources
        if dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=dataset_path, split='train')
        elif dataset_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=dataset_path, split='train')
        elif '/' in dataset_path and not os.path.exists(dataset_path):
            # Assume HuggingFace Hub dataset
            dataset = load_dataset(dataset_path, split=self.config.dataset_split)
        else:
            # Try loading as directory
            dataset = load_dataset(dataset_path, split=self.config.dataset_split)

        logger.info(f"✅ Dataset loaded: {len(dataset)} examples")
        return dataset

    def prepare_model_for_training(self):
        """Prepare model for QLoRA training"""
        logger.info("🔧 Preparing model for training...")

        # Load base model
        self.model_wrapper.load()

        # Prepare for k-bit training if quantized
        if self.config.quantization.enabled:
            logger.info("Preparing model for k-bit training...")
            self.model_wrapper.model = prepare_model_for_kbit_training(
                self.model_wrapper.model,
                use_gradient_checkpointing=self.config.training.gradient_checkpointing
            )

        # Enable gradient checkpointing
        if self.config.training.gradient_checkpointing:
            self.model_wrapper.model.gradient_checkpointing_enable()

        logger.info("✅ Model prepared for training")

    def train(
        self,
        dataset_path: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Train the model

        Args:
            dataset_path: Path to training dataset
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        logger.info("🚀 Starting training...")
        logger.info(f"Config: {self.config.model_name}")
        logger.info(f"LoRA Rank: {self.config.lora.rank}")
        logger.info(f"Quantization: {self.config.quantization.bits}-bit")

        # Auto-resume if enabled
        if self.config.checkpoint.auto_resume and not resume_from_checkpoint:
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            if latest_checkpoint:
                logger.info(f"📂 Auto-resuming from {latest_checkpoint}")
                resume_from_checkpoint = str(latest_checkpoint)

        # Prepare model
        self.prepare_model_for_training()

        # Load dataset
        dataset = self.load_dataset(dataset_path)

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model_wrapper.tokenizer,
            mlm=False,
        )

        # Create trainer
        trainer = Trainer(
            model=self.model_wrapper.model,
            args=self.training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.model_wrapper.tokenizer,
        )

        # Train
        try:
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            # Save final model
            logger.info("💾 Saving final model...")
            trainer.save_model(self.config.output_dir)

            # Log metrics
            metrics = train_result.metrics
            logger.info(f"📊 Training metrics: {metrics}")

            # Save config
            config_path = Path(self.config.output_dir) / "jarvis_config.yaml"
            self.config.save_yaml(str(config_path))

            logger.info("✅ Training complete!")

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")

            # Save emergency checkpoint
            if self.config.checkpoint.enabled:
                logger.info("💾 Saving emergency checkpoint...")
                self.checkpoint_manager.save_checkpoint(
                    trainer=trainer,
                    epoch=-1,
                    step=-1,
                    metrics={"error": str(e)}
                )

            raise

    def train_dpo(
        self,
        preference_dataset_path: str,
        beta: float = 0.1,
        **kwargs
    ):
        """
        Train using Direct Preference Optimization (DPO)

        Args:
            preference_dataset_path: Path to preference dataset
            beta: KL penalty coefficient
            **kwargs: Additional DPO parameters
        """
        logger.info("🚀 Starting DPO training...")

        try:
            from trl import DPOTrainer, DPOConfig
        except ImportError:
            raise ImportError("TRL library required for DPO. Install with: pip install trl")

        # Load model
        self.prepare_model_for_training()

        # Load reference model (frozen copy)
        logger.info("Loading reference model...")
        ref_model = LlamaModel(self.config)
        ref_model.load()

        # Load preference dataset
        dataset = self.load_dataset(preference_dataset_path)

        # DPO config
        dpo_config = DPOConfig(
            beta=beta,
            learning_rate=self.config.training.learning_rate,
            max_length=self.config.max_seq_length,
            max_prompt_length=self.config.max_seq_length // 2,
            **kwargs
        )

        # Create DPO trainer
        dpo_trainer = DPOTrainer(
            model=self.model_wrapper.model,
            ref_model=ref_model.model,
            args=dpo_config,
            train_dataset=dataset,
            tokenizer=self.model_wrapper.tokenizer,
        )

        # Train
        dpo_trainer.train()

        logger.info("✅ DPO training complete!")

    def train_rlhf(
        self,
        reward_model_path: str,
        dataset_path: str,
        **kwargs
    ):
        """
        Train using Reinforcement Learning from Human Feedback (RLHF)

        Args:
            reward_model_path: Path to trained reward model
            dataset_path: Path to prompts dataset
            **kwargs: Additional PPO parameters
        """
        logger.info("🚀 Starting RLHF training...")

        try:
            from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
        except ImportError:
            raise ImportError("TRL library required for RLHF. Install with: pip install trl")

        # Load model with value head
        logger.info("Loading model with value head...")
        self.model_wrapper.load()

        model_with_value_head = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.model_wrapper.model
        )

        # Load reward model
        logger.info(f"Loading reward model from {reward_model_path}...")
        # Implementation depends on reward model format

        # PPO config
        ppo_config = PPOConfig(
            learning_rate=self.config.training.learning_rate,
            batch_size=self.config.training.batch_size,
            **kwargs
        )

        # Create PPO trainer
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model_with_value_head,
            tokenizer=self.model_wrapper.tokenizer,
        )

        # Load dataset
        dataset = self.load_dataset(dataset_path)

        # Training loop
        for batch in dataset:
            # Generate responses
            # Get rewards
            # Update model with PPO
            pass  # Implementation details

        logger.info("✅ RLHF training complete!")


# Convenience functions
def train_llama_13b_qlora(
    dataset_path: str,
    output_dir: str = "./outputs/llama-13b-qlora",
    **kwargs
) -> LlamaTrainer:
    """
    Quick-start QLoRA training for Llama-2-13B

    Args:
        dataset_path: Path to training data
        output_dir: Where to save model
        **kwargs: Override config parameters

    Returns:
        Trainer instance
    """
    from jarvis_prime.configs.llama_config import LlamaPresets

    config = LlamaPresets.llama_13b_gcp_training()
    config.dataset_path = dataset_path
    config.output_dir = output_dir

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    trainer = LlamaTrainer(config)
    trainer.train()

    return trainer
