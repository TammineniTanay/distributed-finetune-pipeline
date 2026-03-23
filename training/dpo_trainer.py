"""DPO (Direct Preference Optimization) trainer.

Runs after SFT to align the model with human preferences. DPO is
simpler than full RLHF because it skips the reward model — it
optimizes the policy directly from preference pairs.

Preference data format (JSONL):
  {"chosen": "good response", "rejected": "bad response", "prompt": "..."}

Paper: https://arxiv.org/abs/2305.18290
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """DPO training configuration."""
    sft_model_path: str
    preference_data_path: str
    output_dir: str = "outputs/dpo_model"
    beta: float = 0.1  # KL penalty coefficient — higher = more conservative
    learning_rate: float = 5e-7  # much lower than SFT, DPO is sensitive
    num_epochs: int = 1  # usually 1-2 epochs is enough
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 1024
    max_prompt_length: int = 512
    warmup_ratio: float = 0.1
    bf16: bool = True
    # loss variants
    loss_type: str = "sigmoid"  # sigmoid (original DPO) | hinge | ipo
    label_smoothing: float = 0.0
    # reference model
    ref_model_path: Optional[str] = None  # None = use SFT model as ref
    # lora — can do DPO on top of existing LoRA adapter
    use_lora: bool = True
    lora_r: int = 16  # smaller than SFT — DPO needs fewer params
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    @classmethod
    def from_config(cls, config: dict) -> "DPOConfig":
        dpo_cfg = config.get("training", {}).get("dpo", {})
        return cls(**{k: v for k, v in dpo_cfg.items() if k in cls.__dataclass_fields__})


def load_preference_data(data_path: str, tokenizer, max_length: int = 1024):
    """Load preference pairs from JSONL.

    Expected format:
        {"prompt": "...", "chosen": "...", "rejected": "..."}

    Returns a HuggingFace Dataset ready for DPOTrainer.
    """
    from datasets import Dataset

    records = []
    with open(data_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            # DPOTrainer expects these exact field names
            records.append({
                "prompt": record["prompt"],
                "chosen": record["chosen"],
                "rejected": record["rejected"],
            })

    log.info(f"Loaded {len(records)} preference pairs from {data_path}")
    return Dataset.from_list(records)


def build_dpo_trainer(cfg: DPOConfig):
    """Set up the DPO trainer with model, ref model, and data.

    Uses TRL's DPOTrainer which handles:
    - implicit reward modeling
    - KL-constrained policy optimization
    - reference model log-prob computation
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from trl import DPOTrainer

    log.info(f"Loading SFT model from {cfg.sft_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the SFT model (this becomes the policy we're optimizing)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.sft_model_path,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=False,
    )

    # Reference model — frozen copy used to compute KL divergence
    # If not specified, DPOTrainer uses the initial policy weights
    ref_model = None
    if cfg.ref_model_path and cfg.ref_model_path != cfg.sft_model_path:
        log.info(f"Loading separate reference model from {cfg.ref_model_path}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.ref_model_path,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
            device_map="auto",
            trust_remote_code=False,
        )

    # LoRA for DPO — typically smaller rank than SFT
    peft_config = None
    if cfg.use_lora:
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

    # Load preference data
    dataset = load_preference_data(cfg.preference_data_path, tokenizer, cfg.max_length)

    training_args = TrainingArguments(
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
        gradient_checkpointing=True,
        report_to="wandb",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=cfg.beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        loss_type=cfg.loss_type,
        label_smoothing=cfg.label_smoothing,
    )

    return trainer


def train_dpo(config: dict) -> dict:
    """Entry point for DPO training stage."""
    cfg = DPOConfig.from_config(config)

    log.info("=" * 50)
    log.info("  DPO Training")
    log.info(f"  beta={cfg.beta}, lr={cfg.learning_rate}, loss={cfg.loss_type}")
    log.info("=" * 50)

    trainer = build_dpo_trainer(cfg)
    result = trainer.train()

    # Save
    trainer.save_model(cfg.output_dir)
    log.info(f"DPO model saved to {cfg.output_dir}")

    metrics = {
        "dpo_train_loss": result.training_loss,
        "dpo_epochs": cfg.num_epochs,
        "dpo_beta": cfg.beta,
        "output_dir": cfg.output_dir,
    }

    # log reward accuracies if available
    try:
        eval_results = trainer.evaluate()
        metrics["dpo_reward_accuracy"] = eval_results.get("eval_rewards/accuracies", None)
        metrics["dpo_reward_margins"] = eval_results.get("eval_rewards/margins", None)
    except Exception:
        pass

    return metrics


def generate_preference_data_from_model(
    model_path: str,
    prompts_path: str,
    output_path: str,
    num_generations: int = 4,
    temperature: float = 0.8,
):
    """Generate preference pairs by sampling multiple responses per prompt
    and ranking them with an LLM judge.

    This is a common workflow: generate N responses, score them, take the
    best as 'chosen' and worst as 'rejected'. Cheaper than human annotation.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Generating preference data from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = []
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["prompt"])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as out:
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            responses = []
            for _ in range(num_generations):
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                text = tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                responses.append(text)

            # Simple heuristic ranking: longer, more detailed = better
            # In production you'd use an LLM judge or reward model here
            scored = sorted(responses, key=lambda r: len(r.split()), reverse=True)

            out.write(json.dumps({
                "prompt": prompt,
                "chosen": scored[0],      # best
                "rejected": scored[-1],    # worst
            }) + "\n")

    log.info(f"Generated {len(prompts)} preference pairs → {output_path}")
