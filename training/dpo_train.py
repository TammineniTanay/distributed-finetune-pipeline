"""DPO (Direct Preference Optimization) trainer.

Runs after SFT to align the model with human preferences without
needing a separate reward model. Uses the TRL DPOTrainer which
implements the DPO loss from Rafailov et al. 2023.

The key insight: DPO reparameterizes the RLHF objective so you
can optimize directly on preference pairs (chosen, rejected)
instead of training a reward model + doing PPO. Way simpler,
surprisingly effective.

Usage:
    python -m training.dpo_train --config config/pipeline_config.yaml
    
    # or via the orchestrator (add dpo stage after sft)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """DPO training configuration."""
    sft_model_path: str          # path to the SFT checkpoint
    dataset_path: str            # JSONL with chosen/rejected pairs
    output_dir: str = "outputs/dpo_model"
    beta: float = 0.1            # KL penalty coefficient — lower = more deviation from ref
    learning_rate: float = 5e-7  # much lower than SFT, we're fine-tuning an already-tuned model
    num_epochs: int = 1          # usually 1-2 epochs is enough
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_length: int = 1024
    max_prompt_length: int = 512
    warmup_ratio: float = 0.1
    lora_r: int = 16             # smaller rank than SFT — we're making small adjustments
    lora_alpha: int = 32
    bf16: bool = True
    loss_type: str = "sigmoid"   # sigmoid (original DPO) or hinge or ipo
    label_smoothing: float = 0.0 # >0 helps with noisy preferences
    # ref model can be None (uses implicit reference from the SFT model)
    ref_model_path: Optional[str] = None

    @classmethod
    def from_config(cls, config: dict) -> "DPOConfig":
        dpo_cfg = config.get("training", {}).get("dpo", {})
        return cls(**{k: v for k, v in dpo_cfg.items() if k in cls.__dataclass_fields__})


def load_preference_dataset(path: str, tokenizer=None):
    """Load preference dataset from JSONL.
    
    Expected format per line:
    {
        "prompt": "What is...",
        "chosen": "Good detailed answer...",
        "rejected": "Bad or wrong answer..."
    }

    Some datasets use "chosen_response"/"rejected_response" — we handle both.
    """
    from datasets import Dataset

    examples = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)

            # normalize field names (different datasets use different conventions)
            prompt = ex.get("prompt", ex.get("question", ex.get("instruction", "")))
            chosen = ex.get("chosen", ex.get("chosen_response", ex.get("preferred", "")))
            rejected = ex.get("rejected", ex.get("rejected_response", ex.get("dispreferred", "")))

            if not (prompt and chosen and rejected):
                continue

            examples.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })

    log.info(f"Loaded {len(examples)} preference pairs from {path}")
    return Dataset.from_list(examples)


def train_dpo(config: dict) -> dict:
    """Run DPO training on top of an SFT model.
    
    This is the main entry point. Loads the SFT model, applies a fresh
    LoRA adapter (smaller rank than SFT since we're making subtle adjustments),
    and trains with DPO loss on preference pairs.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import DPOTrainer, DPOConfig as TRLDPOConfig

    cfg = DPOConfig.from_config(config)

    log.info(f"Loading SFT model from {cfg.sft_model_path}")
    log.info(f"DPO beta={cfg.beta}, lr={cfg.learning_rate}, loss={cfg.loss_type}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # important for DPO batch processing

    # Quantize for memory efficiency (same as SFT)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.sft_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model)

    # Fresh LoRA adapter for DPO (separate from SFT adapter)
    # smaller rank — we're nudging, not overhauling
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Reference model — if not provided, DPOTrainer uses implicit reference
    # (copies of the initial weights). This is the standard approach.
    ref_model = None
    if cfg.ref_model_path:
        log.info(f"Loading explicit reference model from {cfg.ref_model_path}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.ref_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
        )

    # Load preference data
    dataset = load_preference_dataset(cfg.dataset_path, tokenizer)
    log.info(f"Training on {len(dataset)} preference pairs")

    # DPO training args
    training_args = TRLDPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        logging_steps=5,
        save_strategy="epoch",
        remove_unused_columns=False,
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        label_smoothing=cfg.label_smoothing,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        # gradient checkpointing saves memory at cost of ~20% speed
        gradient_checkpointing=True,
        report_to="wandb",
    )

    # Build trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    log.info("Starting DPO training...")
    train_result = trainer.train()

    # Save
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    metrics = train_result.metrics
    log.info(f"DPO training complete. Loss: {metrics.get('train_loss', 'N/A')}")
    log.info(f"Model saved to {cfg.output_dir}")

    # Log reward accuracies — this tells us if the model learned to prefer
    # chosen over rejected. Should be >50%, ideally 70-80%.
    reward_acc = metrics.get("train_rewards/accuracies", None)
    if reward_acc:
        log.info(f"Reward accuracy: {reward_acc:.1%}")

    return {
        "output_dir": cfg.output_dir,
        "metrics": metrics,
        "num_pairs": len(dataset),
    }


# Sample preference data generator (for testing / bootstrapping)
def generate_sample_preferences(
    output_path: str,
    num_pairs: int = 100,
):
    """Generate synthetic preference pairs for testing the DPO pipeline.
    
    In production you'd use human annotations or AI feedback (RLAIF).
    This just creates plausible-looking pairs for pipeline testing.
    """
    import random

    prompts = [
        "Explain how gradient descent works.",
        "What is the difference between L1 and L2 regularization?",
        "How does dropout prevent overfitting?",
        "Explain the attention mechanism in transformers.",
        "What is batch normalization and why is it useful?",
        "Describe the vanishing gradient problem.",
        "What is transfer learning?",
        "Explain the bias-variance tradeoff.",
        "How does a convolutional neural network process images?",
        "What are word embeddings?",
    ]

    pairs = []
    for i in range(num_pairs):
        prompt = random.choice(prompts)
        pairs.append({
            "prompt": prompt,
            "chosen": f"[Detailed, accurate answer to: {prompt}] This involves several key concepts...",
            "rejected": f"[Vague or incorrect answer to: {prompt}] I think it's something about computers.",
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    log.info(f"Generated {num_pairs} sample preference pairs → {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DPO Training")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--generate-sample-data", action="store_true",
                        help="Generate sample preference pairs for testing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.generate_sample_data:
        generate_sample_preferences("data/preferences/sample_pairs.jsonl")
    else:
        # would need to load config here
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        train_dpo(config)
