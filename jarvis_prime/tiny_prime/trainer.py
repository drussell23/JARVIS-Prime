"""Tiny Prime training loop (from scratch) with Spot-resilient checkpointing."""

from __future__ import annotations

import json
import logging
import os
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from jarvis_prime.tiny_prime.config import TinyPrimeConfig

logger = logging.getLogger(__name__)


@dataclass
class _CheckpointState:
    preempted: bool = False


class _PreemptionCallback(TrainerCallback):
    def __init__(self, state: _CheckpointState, signal_file: str):
        self._state = state
        self._signal_file = signal_file

    def on_step_end(self, args, state, control, **kwargs):
        if self._state.preempted:
            # Force a save and stop training as safely as possible.
            control.should_save = True
            control.should_training_stop = True
            try:
                Path(self._signal_file).touch()
            except Exception:
                pass
        return control


class TinyPrimeTrainer:
    def __init__(self, cfg: TinyPrimeConfig):
        self.cfg = cfg
        self._ckpt_state = _CheckpointState()
        self._install_sigterm_handler()

    def _install_sigterm_handler(self) -> None:
        if not bool(self.cfg.get("checkpoint.enabled", True)):
            return

        signal_file = str(self.cfg.get("checkpoint.preemption_signal_file", "/tmp/tiny_prime_preemption_signal"))

        def _handler(signum, frame):
            logger.warning("⚠️  SIGTERM received (Spot preemption likely). Requesting checkpoint...")
            self._ckpt_state.preempted = True
            try:
                Path(signal_file).touch()
            except Exception:
                pass

        signal.signal(signal.SIGTERM, _handler)

    def _training_args(self) -> TrainingArguments:
        # Store incremental checkpoints separately from the final export directory.
        # (Keeps Spot VM recovery robust and easy to manage.)
        ckpt_dir = (
            Path(self.cfg.resolve_path(str(self.cfg.get("checkpoint.checkpoint_dir", "./checkpoints/tiny_prime"))))
            if bool(self.cfg.get("checkpoint.enabled", True))
            else Path(self.cfg.resolve_path_for("training.output_dir"))
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        mp = str(self.cfg.get("training.mixed_precision", "no"))
        fp16 = mp == "fp16"
        bf16 = mp == "bf16"

        return TrainingArguments(
            output_dir=str(ckpt_dir),
            num_train_epochs=float(self.cfg.get("training.num_epochs", 1)),
            max_steps=int(self.cfg.get("training.max_steps", -1)),
            per_device_train_batch_size=int(self.cfg.get("training.batch_size", 32)),
            gradient_accumulation_steps=int(self.cfg.get("training.gradient_accumulation_steps", 1)),
            learning_rate=float(self.cfg.get("training.learning_rate", 3e-4)),
            warmup_ratio=float(self.cfg.get("training.warmup_ratio", 0.03)),
            weight_decay=float(self.cfg.get("training.weight_decay", 0.01)),
            max_grad_norm=float(self.cfg.get("training.max_grad_norm", 1.0)),
            lr_scheduler_type=str(self.cfg.get("training.lr_scheduler_type", "cosine")),
            logging_steps=int(self.cfg.get("training.logging_steps", 25)),
            save_steps=int(self.cfg.get("training.save_steps", 100)),
            eval_steps=int(self.cfg.get("training.eval_steps", 200)),
            save_total_limit=int(self.cfg.get("training.save_total_limit", 3)),
            seed=int(self.cfg.get("training.seed", 42)),
            dataloader_num_workers=int(self.cfg.get("training.dataloader_num_workers", 4)),
            fp16=fp16,
            bf16=bf16,
            gradient_checkpointing=bool(self.cfg.get("training.gradient_checkpointing", False)),
            report_to=(
                ["tensorboard"]
                if bool(self.cfg.get("monitoring.tensorboard_enabled", True))
                else ["none"]
            ),
            logging_dir=str(Path(self.cfg.resolve_path(str(self.cfg.get("monitoring.tensorboard_dir", "./runs"))))),
        )

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        tok_dir = self.cfg.resolve_path_for("tokenizer.output_dir")
        tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)

        # Ensure core tokens exist (robustness across tokenizer setups)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or "</s>"
        if tok.eos_token is None and "</s>" in tok.get_vocab():
            tok.eos_token = "</s>"
        if tok.bos_token is None and "<s>" in tok.get_vocab():
            tok.bos_token = "<s>"

        return tok

    def _build_model(self, tokenizer) -> LlamaForCausalLM:
        model_cfg = LlamaConfig(
            vocab_size=int(self.cfg.get("model.vocab_size", 0) or len(tokenizer)),
            hidden_size=int(self.cfg.get("model.hidden_size", 768)),
            intermediate_size=int(self.cfg.get("model.intermediate_size", 3072)),
            num_hidden_layers=int(self.cfg.get("model.num_hidden_layers", 12)),
            num_attention_heads=int(self.cfg.get("model.num_attention_heads", 12)),
            max_position_embeddings=int(self.cfg.get("model.context_window", 512)),
            rms_norm_eps=float(self.cfg.get("model.rms_norm_eps", 1e-5)),
            rope_theta=float(self.cfg.get("model.rope_theta", 10000)),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        model = LlamaForCausalLM(model_cfg)
        return model

    def _load_dataset(self) -> Dataset:
        ds_path = self.cfg.resolve_path_for("training.dataset_path")
        if ds_path.endswith(".json") or ds_path.endswith(".jsonl"):
            return load_dataset("json", data_files=ds_path, split="train")
        return load_dataset(ds_path, split="train")

    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        logging.basicConfig(level=logging.INFO)

        args = self._training_args()

        tokenizer = self._load_tokenizer()
        model = self._build_model(tokenizer)

        ds = self._load_dataset()

        context_window = int(self.cfg.get("model.context_window", 512))

        def _tok(batch: Dict[str, Any]) -> Dict[str, Any]:
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=context_window,
            )

        # Remove all original columns (including \"text\") to keep the collator tensor-only.
        ds = ds.map(_tok, batched=True, remove_columns=list(ds.column_names))

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        callbacks = []
        if bool(self.cfg.get("checkpoint.enabled", True)):
            callbacks.append(
                _PreemptionCallback(
                    state=self._ckpt_state,
                    signal_file=str(self.cfg.get("checkpoint.preemption_signal_file", "/tmp/tiny_prime_preemption_signal")),
                )
            )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds,
            data_collator=collator,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )

        # Auto-resume
        if resume_from_checkpoint is None and bool(self.cfg.get("checkpoint.auto_resume", True)):
            ckpt_dir = Path(args.output_dir)
            if ckpt_dir.exists():
                candidates = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True)
                if candidates:
                    resume_from_checkpoint = str(candidates[0])

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        out_dir = Path(self.cfg.resolve_path_for("training.output_dir"))
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir / "tokenizer"))

        # Persist effective config
        cfg_path = out_dir / "tiny_prime_config.json"
        cfg_path.write_text(json.dumps(self.cfg.to_dict(), indent=2), encoding="utf-8")

        logger.info(f"✅ Tiny Prime training complete. Saved to {out_dir}")
