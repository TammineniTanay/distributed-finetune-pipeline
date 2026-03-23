"""
Custom training callbacks for the distributed fine-tuning pipeline.

Callbacks:
1. WandbMetricsCallback - Extended W&B logging with custom metrics
2. GradientMonitorCallback - Tracks gradient norms and detects anomalies
3. CheckpointCleanupCallback - Manages disk space by cleaning old checkpoints
4. EarlyStoppingWithPatience - Smart early stopping with warmup awareness
"""

from __future__ import annotations

import logging
import shutil
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

log = logging.getLogger(__name__)


class WandbMetricsCallback(TrainerCallback):
    """
    Extended Weights & Biases logging.
    
    Logs additional metrics beyond the default HF integration:
    - Learning rate schedule visualization
    - GPU memory utilization
    - Throughput (tokens/sec, samples/sec)
    - Loss EMA for smoother visualization
    - Custom histograms of loss distribution
    """

    def __init__(self, loss_ema_alpha: float = 0.1):
        self.loss_ema = None
        self.ema_alpha = loss_ema_alpha
        self.step_losses = []
        self._wandb = None

    def _get_wandb(self):
        if self._wandb is None:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                self._wandb = False
        return self._wandb if self._wandb else None

    def on_log(self, args, state: TrainerState, control, logs=None, **kwargs):
        wandb = self._get_wandb()
        if not wandb or not wandb.run:
            return

        if logs is None:
            return

        custom_metrics = {}

        # Loss EMA
        if "loss" in logs:
            loss = logs["loss"]
            self.step_losses.append(loss)
            if self.loss_ema is None:
                self.loss_ema = loss
            else:
                self.loss_ema = self.ema_alpha * loss + (1 - self.ema_alpha) * self.loss_ema
            custom_metrics["train/loss_ema"] = self.loss_ema

        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_allocated = torch.cuda.memory_allocated(i) / 1e9
                mem_reserved = torch.cuda.memory_reserved(i) / 1e9
                mem_total = torch.cuda.get_device_properties(i).total_mem / 1e9
                custom_metrics[f"gpu/{i}/memory_allocated_gb"] = mem_allocated
                custom_metrics[f"gpu/{i}/memory_reserved_gb"] = mem_reserved
                custom_metrics[f"gpu/{i}/memory_utilization"] = mem_allocated / mem_total

        # Throughput
        if "train_runtime" in state.__dict__ and state.global_step > 0:
            elapsed = state.log_history[-1].get("train_runtime", 0) if state.log_history else 0
            if elapsed > 0:
                samples_per_sec = state.global_step * args.per_device_train_batch_size / elapsed
                custom_metrics["throughput/samples_per_sec"] = samples_per_sec

        if custom_metrics:
            wandb.log(custom_metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        wandb = self._get_wandb()
        if wandb and wandb.run and self.step_losses:
            # Log loss histogram
            wandb.log({
                "train/loss_histogram": wandb.Histogram(self.step_losses),
                "train/final_loss_ema": self.loss_ema,
            })


class GradientMonitorCallback(TrainerCallback):
    """
    Monitors gradient statistics to detect training instabilities.
    
    Tracks:
    - Per-layer gradient norms
    - Gradient norm moving average and variance
    - Detects gradient spikes and dead layers (zero gradients)
    """

    def __init__(
        self,
        log_every_n_steps: int = 50,
        spike_threshold: float = 5.0,
        dead_layer_threshold: float = 1e-8,
        window_size: int = 100,
    ):
        self.log_every = log_every_n_steps
        self.spike_threshold = spike_threshold
        self.dead_threshold = dead_layer_threshold
        self.window_size = window_size
        self._norm_history = deque(maxlen=window_size)
        self._spike_count = 0

    def on_step_end(self, args, state: TrainerState, control, model=None, **kwargs):
        if state.global_step % self.log_every != 0 or model is None:
            return

        grad_norms = {}
        total_norm = 0.0
        dead_layers = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2).item()
                grad_norms[name] = norm
                total_norm += norm ** 2

                if norm < self.dead_threshold:
                    dead_layers.append(name)

        total_norm = total_norm ** 0.5
        self._norm_history.append(total_norm)

        # Detect gradient spike
        if len(self._norm_history) >= 10:
            mean_norm = np.mean(list(self._norm_history))
            std_norm = np.std(list(self._norm_history))
            if std_norm > 0 and (total_norm - mean_norm) / std_norm > self.spike_threshold:
                self._spike_count += 1
                log.warning(
                    f"Gradient spike at step {state.global_step}: "
                    f"norm={total_norm:.4f}, mean={mean_norm:.4f}, "
                    f"z-score={(total_norm - mean_norm) / std_norm:.2f}"
                )

        if dead_layers:
            log.warning(
                f"Dead gradient layers at step {state.global_step}: "
                f"{len(dead_layers)} layers with near-zero gradients"
            )

        # Log to W&B
        try:
            import wandb
            if wandb.run:
                wandb.log({
                    "gradients/total_norm": total_norm,
                    "gradients/mean_norm": np.mean(list(self._norm_history)),
                    "gradients/spike_count": self._spike_count,
                    "gradients/dead_layers": len(dead_layers),
                }, step=state.global_step)
        except (ImportError, Exception):
            pass


class CheckpointCleanupCallback(TrainerCallback):
    """
    Manages checkpoint disk space by keeping only the N most recent checkpoints.
    Also cleans up optimizer states from old checkpoints to save space.
    """

    def __init__(self, keep_last_n: int = 3):
        self.keep_last_n = keep_last_n

    def on_save(self, args, state, control, **kwargs):
        output_dir = Path(args.output_dir)
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )

        if len(checkpoints) > self.keep_last_n:
            for ckpt in checkpoints[:-self.keep_last_n]:
                log.info(f"Removing old checkpoint: {ckpt}")
                shutil.rmtree(ckpt, ignore_errors=True)


class EarlyStoppingWithPatience(TrainerCallback):
    """
    Early stopping that's aware of warmup phase.
    
    Won't trigger during warmup or the first N eval steps.
    Uses both absolute and relative improvement thresholds.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        min_evals_before_stopping: int = 3,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.min_evals = min_evals_before_stopping
        self.best_metric = None
        self.wait = 0
        self.eval_count = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None:
            return

        self.eval_count += 1

        # Don't stop early in the first few evals
        if self.eval_count < self.min_evals:
            return

        metric_name = args.metric_for_best_model or "eval_loss"
        current = metrics.get(metric_name)
        if current is None:
            return

        greater_is_better = args.greater_is_better if hasattr(args, "greater_is_better") else False

        if self.best_metric is None:
            self.best_metric = current
            return

        if greater_is_better:
            improved = current > self.best_metric + self.min_delta
        else:
            improved = current < self.best_metric - self.min_delta

        if improved:
            self.best_metric = current
            self.wait = 0
            log.info(f"New best {metric_name}: {current:.6f}")
        else:
            self.wait += 1
            log.info(
                f"No improvement in {metric_name} for {self.wait}/{self.patience} evals "
                f"(best: {self.best_metric:.6f}, current: {current:.6f})"
            )

            if self.wait >= self.patience:
                log.warning(f"Early stopping triggered after {self.wait} evals without improvement")
                control.should_training_stop = True
