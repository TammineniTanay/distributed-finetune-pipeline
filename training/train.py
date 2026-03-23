"""Main training script — QLoRA fine-tuning with DeepSpeed/FSDP.

This is the core of the pipeline. Handles model loading with 4-bit
quantization, LoRA injection, distributed setup, and the actual
training loop via HF SFTTrainer.

Can be launched directly or via the orchestrator.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import load_from_disk, load_dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.config_loader import load_config, get_deepspeed_config, get_fsdp_config
from training.callbacks.custom_callbacks import (
    WandbMetricsCallback,
    GradientMonitorCallback,
    CheckpointCleanupCallback,
    EarlyStoppingWithPatience,
)

log = logging.getLogger(__name__)


# Model Loading

def load_quantized_model(cfg) -> tuple:
    """Load the base model with 4-bit quantization and LoRA adapters.
    
    Returns (model, tokenizer, peft_config).
    """
    # NOTE: trust_remote_code=False is deliberate — dont want arbitrary code execution
    # from random HF repos. If you need a custom model arch, fork it first.
    model_cfg = cfg.training.base_model
    qlora_cfg = cfg.training.qlora
    quant_cfg = qlora_cfg.quantization

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("compute_dtype", "bfloat16")),
    )

    log.info(f"Loading model: {model_cfg['name']} with 4-bit quantization")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision", "main"),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        padding_side="right",
        use_fast=True,
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision", "main"),
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
    )

    # Prepare for k-bit training (freezes base, casts trainable to fp32)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # LoRA configuration
    peft_config = LoraConfig(
        r=qlora_cfg.r,
        lora_alpha=qlora_cfg.lora_alpha,
        lora_dropout=qlora_cfg.lora_dropout,
        target_modules=qlora_cfg.target_modules,
        modules_to_save=qlora_cfg.modules_to_save or None,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # Apply LoRA
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model, tokenizer, peft_config


# Dataset Loading

def load_training_data(cfg, tokenizer) -> tuple:
    """Load and prepare the training and validation datasets."""
    assembly_cfg = cfg.data_curation.get("assembly", {}) if isinstance(cfg.data_curation, dict) else {}
    data_dir = Path(assembly_cfg.get("output_dir", "data/final"))

    # Try loading from disk (Arrow format)
    try:
        train_dataset = load_from_disk(str(data_dir / "train"))
        val_dataset = load_from_disk(str(data_dir / "validation"))
        log.info(f"Loaded datasets: train={len(train_dataset)}, val={len(val_dataset)}")
    except Exception:
        # Fallback to JSONL
        train_file = data_dir / "train.jsonl"
        val_file = data_dir / "validation.jsonl"

        if train_file.exists():
            train_dataset = load_dataset("json", data_files=str(train_file), split="train")
            val_dataset = load_dataset("json", data_files=str(val_file), split="train")
        else:
            raise FileNotFoundError(f"No training data found in {data_dir}")

    return train_dataset, val_dataset


# Training Arguments

def build_training_args(cfg) -> TrainingArguments:
    """Build HuggingFace TrainingArguments from pipeline config."""
    hp = cfg.training.hyperparameters
    dist = cfg.training.distributed
    tracking = cfg.training.tracking

    output_dir = tracking.get("output_dir", "outputs/training")

    # Base args
    args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": hp.get("num_epochs", 3),
        "per_device_train_batch_size": hp.get("per_device_train_batch_size", 4),
        "per_device_eval_batch_size": hp.get("per_device_eval_batch_size", 4),
        "gradient_accumulation_steps": hp.get("gradient_accumulation_steps", 8),
        "learning_rate": hp.get("learning_rate", 2e-4),
        "lr_scheduler_type": hp.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": hp.get("warmup_ratio", 0.03),
        "weight_decay": hp.get("weight_decay", 0.01),
        "max_grad_norm": hp.get("max_grad_norm", 1.0),
        "bf16": hp.get("bf16", True),
        "tf32": hp.get("tf32", True),
        "gradient_checkpointing": hp.get("gradient_checkpointing", True),
        "gradient_checkpointing_kwargs": hp.get("gradient_checkpointing_kwargs", {"use_reentrant": False}),
        "optim": hp.get("optim", "paged_adamw_8bit"),
        "logging_steps": hp.get("logging_steps", 10),
        "save_steps": hp.get("save_steps", 500),
        "eval_steps": hp.get("eval_steps", 500),
        "eval_strategy": "steps",
        "save_total_limit": hp.get("save_total_limit", 3),
        "load_best_model_at_end": hp.get("load_best_model_at_end", True),
        "metric_for_best_model": hp.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": hp.get("greater_is_better", False),
        "report_to": hp.get("report_to", "wandb"),
        "dataloader_num_workers": hp.get("dataloader_num_workers", 4),
        "dataloader_pin_memory": hp.get("dataloader_pin_memory", True),
        "group_by_length": hp.get("group_by_length", True),
        "remove_unused_columns": False,
        "seed": 42,
        "data_seed": 42,
        "ddp_find_unused_parameters": False,
        "run_name": f"qlora-{cfg.project.get('name', 'finetune')}",
    }

    # Distributed strategy
    if dist.strategy == "deepspeed_zero3":
        ds_config = get_deepspeed_config(cfg)
        # Write DeepSpeed config to file
        ds_config_path = Path(output_dir) / "ds_config.json"
        ds_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ds_config_path, "w") as f:
            json.dump(ds_config, f, indent=2)
        args_dict["deepspeed"] = str(ds_config_path)
    elif dist.strategy == "fsdp":
        fsdp_cfg = get_fsdp_config(cfg)
        args_dict.update(fsdp_cfg)

    # W&B configuration
    wandb_cfg = tracking.get("wandb", {})
    if wandb_cfg:
        os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "llm-finetune")
        if wandb_cfg.get("entity"):
            os.environ["WANDB_ENTITY"] = wandb_cfg["entity"]
        os.environ["WANDB_LOG_MODEL"] = str(wandb_cfg.get("log_model", "checkpoint"))

    return TrainingArguments(**args_dict)


# Trainer Setup

def build_trainer(
    model,
    tokenizer,
    peft_config,
    train_dataset,
    val_dataset,
    training_args,
    cfg,
) -> SFTTrainer:
    """Build the SFTTrainer with all callbacks."""
    hp = cfg.training.hyperparameters

    # Custom callbacks
    callbacks = [
        WandbMetricsCallback(),
        GradientMonitorCallback(log_every_n_steps=50),
        CheckpointCleanupCallback(keep_last_n=3),
        EarlyStoppingWithPatience(patience=5, min_delta=0.001),
    ]

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        max_seq_length=hp.get("max_seq_length", 4096),
        dataset_text_field="text",
        neftune_noise_alpha=hp.get("neftune_noise_alpha", 5.0),
        callbacks=callbacks,
        tokenizer=tokenizer,
        packing=False,  # We handle packing in the dataset assembler
    )

    return trainer


# Post-training: merge adapter with base model

def merge_adapter_with_base(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    torch_dtype: str = "bfloat16",
):
    """
    Merge LoRA adapter weights back into the base model.
    Produces a standalone model that doesn't require PEFT at inference.
    """
    log.info(f"Merging adapter {adapter_path} with base model {base_model_name}")

    # Load base model in full precision for merging
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=getattr(torch, torch_dtype),
        device_map="auto",
        trust_remote_code=False,
    )

    # Load and merge adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    # Save merged model
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))

    # Save tokenizer alongside
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(str(output_path))

    log.info(f"Merged model saved to {output_path}")
    return str(output_path)


# Main training entry point

def train(config_path: str = "config/pipeline_config.yaml", overrides: dict = None):
    """
    Main training function. Called by the launcher script.
    
    Usage:
        # Single GPU
        python -m training.train
        
        # Multi-GPU with DeepSpeed
        deepspeed --num_gpus=4 -m training.train
        
        # Multi-GPU with torchrun (FSDP)
        torchrun --nproc_per_node=4 -m training.train
    """
    # Load config
    cfg = load_config(config_path, overrides)

    # Initialize W&B
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        try:
            import wandb
            wandb_cfg = cfg.training.tracking.get("wandb", {})
            wandb.init(
                project=wandb_cfg.get("project", "llm-finetune"),
                tags=wandb_cfg.get("tags", []),
                config=cfg.model_dump(),
            )
        except Exception as e:
            log.warning(f"W&B init failed: {e}")

    # Load model and data
    model, tokenizer, peft_config = load_quantized_model(cfg)
    train_dataset, val_dataset = load_training_data(cfg, tokenizer)

    # Build training args and trainer
    training_args = build_training_args(cfg)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=training_args,
        cfg=cfg,
    )

    # Train!
    log.info("=" * 60)
    log.info("Starting training")
    log.info(f"  Strategy: {cfg.training.distributed.strategy}")
    log.info(f"  GPUs: {cfg.training.distributed.num_gpus}")
    log.info(f"  LoRA rank: {cfg.training.qlora.r}")
    log.info(f"  Batch size (effective): "
                f"{training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * cfg.training.distributed.num_gpus}")
    log.info(f"  Learning rate: {training_args.learning_rate}")
    log.info(f"  Epochs: {training_args.num_train_epochs}")
    log.info("=" * 60)

    train_result = trainer.train()

    # Save final adapter
    final_adapter_dir = Path(training_args.output_dir) / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    # Log final metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Evaluate
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Merge adapter with base model (on rank 0 only)
    if local_rank == 0:
        merged_dir = Path(training_args.output_dir) / "final_merged"
        merge_adapter_with_base(
            base_model_name=cfg.training.base_model["name"],
            adapter_path=str(final_adapter_dir),
            output_path=str(merged_dir),
            torch_dtype=cfg.training.base_model.get("torch_dtype", "bfloat16"),
        )

    log.info("Training complete!")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    train()
