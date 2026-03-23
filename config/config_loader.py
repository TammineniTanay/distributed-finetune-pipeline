"""Config loader — YAML config w/ env var interpolation + pydantic validation.

Honestly this started as a simple yaml.safe_load wrapper and grew into
this monstrosity. Might want to switch to hydra at some point but this
works fine for now.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# matches ${VAR_NAME} or ${VAR_NAME:default_value}
_ENV_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _resolve_env_vars(obj):
    """Recursively resolve ${VAR:default} patterns in config values."""
    if isinstance(obj, str):
        def _replacer(m):
            var_name, default = m.group(1), m.group(2)
            return os.environ.get(var_name, default if default is not None else m.group(0))
        return _ENV_PATTERN.sub(_replacer, obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    return obj


# --- Pydantic models ---
# TODO: these could probably be auto-generated from a schema file
# but manual is fine for now since the config doesn't change often

class QLoRAConfig(BaseModel):
    r: int = Field(ge=1, le=256)
    lora_alpha: int = Field(ge=1)
    lora_dropout: float = Field(ge=0.0, le=1.0)
    target_modules: list[str]
    modules_to_save: list[str] = []
    quantization: dict = {}

    @field_validator("lora_alpha")
    @classmethod
    def alpha_ge_rank(cls, v: int, info) -> int:
        r = info.data.get("r", 1)
        if v < r:
            # not a hard error — some papers use alpha < r but it's usually a mistake
            import warnings
            warnings.warn(f"lora_alpha ({v}) < r ({r}); typically alpha >= r for stable training")
        return v


class DistributedConfig(BaseModel):
    strategy: str = "deepspeed_zero3"
    num_gpus: int = Field(ge=1, le=64)  # 64 should be enough for anyone...
    num_nodes: int = Field(ge=1, le=16)
    deepspeed: dict = {}
    fsdp: dict = {}

    @field_validator("strategy")
    @classmethod
    def valid_strategy(cls, v):
        valid = {"deepspeed_zero3", "deepspeed_zero2", "fsdp", "ddp"}
        if v not in valid:
            raise ValueError(f"strategy must be one of {valid}, got '{v}'")
        return v


class TrainingConfig(BaseModel):
    base_model: dict
    qlora: QLoRAConfig
    hyperparameters: dict
    distributed: DistributedConfig
    axolotl: dict = {}  # only used if running via axolotl instead of our train.py
    tracking: dict = {}


class MergeMethodConfig(BaseModel):
    name: str
    config: dict

    @field_validator("name")
    @classmethod
    def valid_method(cls, v):
        valid = {"ties", "dare_ties", "dare_linear", "slerp", "linear", "task_arithmetic"}
        if v not in valid:
            raise ValueError(f"merge method must be one of {valid}, got '{v}'")
        return v


class ModelMergingConfig(BaseModel):
    methods: list[MergeMethodConfig]
    models_to_merge: list[dict]
    base_model: str
    output_dir: str = "outputs/merged"
    dtype: str = "bfloat16"
    device: str = "cuda"


class EvaluationConfig(BaseModel):
    models_to_evaluate: list[dict]
    benchmarks: list[dict]
    llm_judge: dict = {}
    output_dir: str = "outputs/evaluation"
    report_format: list[str] = ["html", "json"]  # FIXME: html report gen not implemented yet


class InferenceConfig(BaseModel):
    vllm: dict
    load_testing: dict = {}


class PipelineConfig(BaseModel):
    """Root config. Kept intentionally loose for data_curation since
    that schema changes a lot during development."""
    project: dict
    data_curation: dict  # not strictly validated — too many nested optional fields
    training: TrainingConfig
    model_merging: ModelMergingConfig
    evaluation: EvaluationConfig
    inference: InferenceConfig
    aws: dict = {}

    @classmethod
    def from_yaml(cls, config_path, overrides=None):
        """Convenience — load from YAML, resolve env vars, validate."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        raw = _resolve_env_vars(raw)
        if overrides:
            if isinstance(overrides, list):
                # handle ["key=val", ...] format from CLI
                for o in overrides:
                    k, v = o.split("=", 1)
                    keys = k.split(".")
                    d = raw
                    for key in keys[:-1]:
                        d = d.setdefault(key, {})
                    d[keys[-1]] = v
            elif isinstance(overrides, dict):
                for dotted_key, value in overrides.items():
                    keys = dotted_key.split(".")
                    d = raw
                    for k in keys[:-1]:
                        d = d.setdefault(k, {})
                    d[keys[-1]] = value
        return cls(**raw)

    def to_dict(self):
        return self.model_dump()


def load_config(config_path="config/pipeline_config.yaml", overrides=None):
    """Load YAML config, resolve env vars, apply dot-notation overrides."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    raw = _resolve_env_vars(raw)

    # dot-notation overrides like training.qlora.r=128
    if overrides:
        for dotted_key, value in overrides.items():
            keys = dotted_key.split(".")
            d = raw
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    return PipelineConfig(**raw)


def get_deepspeed_config(cfg):
    """Build the ds config dict that gets written to a temp JSON file before training.
    These defaults are tuned for 8xA100 80GB — adjust reduce_bucket_size etc for smaller setups."""
    ds = cfg.training.distributed.deepspeed
    hp = cfg.training.hyperparameters

    micro_bs = hp.get("per_device_train_batch_size", 4)
    grad_accum = hp.get("gradient_accumulation_steps", 1)
    n_gpus = cfg.training.distributed.num_gpus

    return {
        "train_batch_size": micro_bs * grad_accum * n_gpus,
        "train_micro_batch_size_per_gpu": micro_bs,
        "gradient_accumulation_steps": grad_accum,
        "gradient_clipping": hp.get("max_grad_norm", 1.0),
        "zero_optimization": {
            "stage": ds.get("zero_stage", 3),
            "offload_optimizer": {
                "device": "cpu" if ds.get("offload_optimizer") else "none",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu" if ds.get("offload_param") else "none",
                "pin_memory": True,
            },
            "overlap_comm": ds.get("overlap_comm", True),
            "contiguous_gradients": ds.get("contiguous_gradients", True),
            "reduce_bucket_size": int(ds.get("reduce_bucket_size", 5e8)),
            "stage3_prefetch_bucket_size": int(ds.get("stage3_prefetch_bucket_size", 5e8)),
            "stage3_param_persistence_threshold": int(ds.get("stage3_param_persistence_threshold", 1e6)),
            "stage3_max_live_parameters": int(ds.get("stage3_max_live_parameters", 1e9)),
            "stage3_max_reuse_distance": int(ds.get("stage3_max_reuse_distance", 1e9)),
            "gather_16bit_weights_on_model_save": ds.get("gather_16bit_weights_on_model_save", True),
            "round_robin_gradients": ds.get("round_robin_gradients", True),
        },
        "bf16": {"enabled": hp.get("bf16", True)},
        "zero_allow_untested_optimizer": True,  # needed for 8bit adam etc
        "wall_clock_breakdown": False,
        "steps_per_print": hp.get("logging_steps", 10),
    }


def get_fsdp_config(cfg):
    """FSDP kwargs for HF Trainer. Less battle-tested than our DS path tbh."""
    fsdp = cfg.training.distributed.fsdp
    return {
        "fsdp": fsdp.get("sharding_strategy", "FULL_SHARD"),
        "fsdp_config": {
            "fsdp_auto_wrap_policy": fsdp.get("auto_wrap_policy", "transformer_based_wrap"),
            # NOTE: this needs to match the actual layer class name in the model
            "fsdp_transformer_layer_cls_to_wrap": fsdp.get("transformer_layer_cls_to_wrap", "LlamaDecoderLayer"),
            "fsdp_backward_prefetch": fsdp.get("backward_prefetch", "backward_pre"),
            "fsdp_forward_prefetch": fsdp.get("forward_prefetch", True),
            "fsdp_offload_params": fsdp.get("cpu_offload", False),
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_sync_module_states": fsdp.get("sync_module_states", True),
            "fsdp_use_orig_params": fsdp.get("use_orig_params", True),
            "fsdp_activation_checkpointing": fsdp.get("activation_checkpointing", True),
            "fsdp_limit_all_gathers": fsdp.get("limit_all_gathers", True),
        },
    }
