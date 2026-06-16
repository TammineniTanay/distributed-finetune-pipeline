# Keep Colab alive during long training
import IPython
IPython.display.display(IPython.display.Javascript('''
function ClickConnect(){
  console.log("Keeping alive...");
  document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
'''))

!pip install -q transformers datasets accelerate peft trl bitsandbytes
!pip install -q wandb rouge-score matplotlib

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 4-bit NF4 quantization — same config as the full pipeline
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

# LoRA adapter — r=32 (smaller than full pipeline's r=64 to fit T4)
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from datasets import load_dataset

# Using a small instruction dataset for demo
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.shuffle(seed=42).select(range(2000))  # 2K examples for demo

def format_instruction(example):
    if example.get("input", "").strip():
        text = f"""<|user|>\n{example['instruction']}\n{example['input']}</s>\n<|assistant|>\n{example['output']}</s>"""
    else:
        text = f"""<|user|>\n{example['instruction']}</s>\n<|assistant|>\n{example['output']}</s>"""
    return {"text": text}

dataset = dataset.map(format_instruction)
print(f"Training examples: {len(dataset)}")
print(f"\nSample:\n{dataset[0]['text'][:300]}...")

from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./outputs/sft",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    bf16=True,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
    max_grad_norm=1.0,
    optim="paged_adamw_8bit",
    max_length=512,
    dataset_text_field="text",
    neftune_noise_alpha=5.0,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print("Starting QLoRA fine-tuning...")
result = trainer.train()
print(f"\nTraining complete! Final loss: {result.metrics['train_loss']:.4f}")

import matplotlib.pyplot as plt
import numpy as np

# Extract loss from training log
logs = trainer.state.log_history
train_losses = [(l["step"], l["loss"]) for l in logs if "loss" in l]
steps, losses = zip(*train_losses)

# EMA smoothing
def ema(values, alpha=0.1):
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed

smoothed = ema(losses, alpha=0.15)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(steps, losses, alpha=0.3, color="#6366f1", linewidth=1, label="Raw loss")
ax.plot(steps, smoothed, color="#6366f1", linewidth=2.5, label="EMA (α=0.15)")
ax.set_xlabel("Training Steps", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("QLoRA Fine-Tuning Loss — TinyLlama 1.1B", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(alpha=0.2)
ax.set_facecolor("#0a0a0a")
fig.patch.set_facecolor("#0a0a0a")
ax.tick_params(colors="#888")
ax.xaxis.label.set_color("#888")
ax.yaxis.label.set_color("#888")
ax.title.set_color("#ddd")
for spine in ax.spines.values():
    spine.set_color("#333")
ax.legend(facecolor="#111", edgecolor="#333", labelcolor="#aaa")
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nFinal loss: {losses[-1]:.4f}")
print(f"Loss reduction: {losses[0]:.4f} → {losses[-1]:.4f} ({(1-losses[-1]/losses[0])*100:.1f}% decrease)")

# Save LoRA adapter
trainer.save_model("./outputs/sft_adapter")
tokenizer.save_pretrained("./outputs/sft_adapter")
print("LoRA adapter saved")

# Merge adapter with base model (produces standalone model)
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

merged_model = PeftModel.from_pretrained(base_model, "./outputs/sft_adapter")
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained("./outputs/merged_model")
tokenizer.save_pretrained("./outputs/merged_model")
print("Merged model saved (standalone, no PEFT needed at inference)")

# Load the fine-tuned model for inference
ft_model = AutoModelForCausalLM.from_pretrained(
    "./outputs/merged_model",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

test_prompts = [
    "Explain the difference between supervised and unsupervised learning.",
    "Write a Python function to compute the Fibonacci sequence.",
    "What is the attention mechanism in transformers?",
]

print("=" * 70)
print("  INFERENCE RESULTS — Fine-Tuned TinyLlama 1.1B")
print("=" * 70)

for prompt in test_prompts:
    input_text = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(ft_model.device)

    with torch.no_grad():
        outputs = ft_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n📝 Prompt: {prompt}")
    print(f"💬 Response: {response[:500]}")
    print("-" * 70)

import math

# Compute perplexity on a held-out set
eval_dataset = load_dataset("yahma/alpaca-cleaned", split="train")
eval_dataset = eval_dataset.shuffle(seed=99).select(range(200))

total_loss = 0
total_tokens = 0

ft_model.eval()
for ex in eval_dataset:
    text = f"<|user|>\n{ex['instruction']}</s>\n<|assistant|>\n{ex['output']}</s>"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(ft_model.device)

    with torch.no_grad():
        outputs = ft_model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]

avg_loss = total_loss / total_tokens
perplexity = math.exp(avg_loss)

print(f"Evaluation Results (200 held-out examples):")
print(f"  Average loss: {avg_loss:.4f}")
print(f"  Perplexity:   {perplexity:.2f}")

# Final summary
print("=" * 50)
print("  TRAINING SUMMARY")
print("=" * 50)
print(f"  Model:           TinyLlama 1.1B")
print(f"  Method:          QLoRA (NF4, r=32, α=64)")
print(f"  Training data:   2,000 examples")
print(f"  Epochs:          2")
print(f"  Final loss:      {result.metrics['train_loss']:.4f}")
print(f"  Perplexity:      {perplexity:.2f}")
print(f"  GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
print(f"  Training time:   {result.metrics['train_runtime']:.0f}s")
print(f"  Throughput:      {result.metrics['train_samples_per_second']:.1f} samples/s")
print(f"")
print(f"  Full pipeline: github.com/TammineniTanay/distributed-finetune-pipeline")
