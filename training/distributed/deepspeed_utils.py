"""
DeepSpeed ZeRO-3 configuration and multi-node launch utilities.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def generate_deepspeed_config(
    zero_stage: int = 3,
    train_micro_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    num_gpus: int = 4,
    offload_optimizer: bool = False,
    offload_params: bool = False,
    bf16: bool = True,
) -> dict:
    """Generate a production DeepSpeed config."""
    config = {
        "train_batch_size": train_micro_batch_size * gradient_accumulation_steps * num_gpus,
        "train_micro_batch_size_per_gpu": train_micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "wall_clock_breakdown": False,

        "zero_optimization": {
            "stage": zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 500_000_000,
            "allgather_bucket_size": 500_000_000,
        },

        "bf16": {"enabled": bf16},
        "fp16": {"enabled": not bf16},

        "zero_allow_untested_optimizer": True,

        "comms_logger": {
            "enabled": False,
            "verbose": False,
            "prof_all": False,
            "debug": False,
        },
    }

    # Stage 3 specific settings
    if zero_stage == 3:
        config["zero_optimization"].update({
            "stage3_prefetch_bucket_size": 500_000_000,
            "stage3_param_persistence_threshold": 1_000_000,
            "stage3_max_live_parameters": 1_000_000_000,
            "stage3_max_reuse_distance": 1_000_000_000,
            "gather_16bit_weights_on_model_save": True,
            "round_robin_gradients": True,
        })

    # CPU Offloading
    if offload_optimizer:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
            "buffer_count": 4,
            "fast_init": False,
        }

    if offload_params:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
            "buffer_count": 5,
            "buffer_size": 100_000_000,
            "max_in_cpu": 1_000_000_000,
        }

    # Activation checkpointing
    config["activation_checkpointing"] = {
        "partition_activations": True,
        "cpu_checkpointing": offload_params,
        "contiguous_memory_optimization": True,
        "number_checkpoints": None,
        "synchronize_checkpoint_boundary": False,
        "profile": False,
    }

    return config


def launch_deepspeed_training(
    training_script: str = "training/train.py",
    num_gpus: int = 4,
    num_nodes: int = 1,
    master_addr: str = "localhost",
    master_port: int = 29500,
    hostfile: str | None = None,
    config_path: str = "config/pipeline_config.yaml",
    extra_args: dict | None = None,
) -> subprocess.CompletedProcess:
    """
    Launch distributed training via deepspeed CLI.
    
    Supports both single-node multi-GPU and multi-node setups.
    """
    cmd = ["deepspeed"]

    if num_nodes > 1 and hostfile:
        cmd.extend(["--hostfile", hostfile])
    else:
        cmd.extend(["--num_gpus", str(num_gpus)])

    cmd.extend([
        "--master_addr", master_addr,
        "--master_port", str(master_port),
    ])

    cmd.append(training_script)
    cmd.extend(["--config_path", config_path])

    if extra_args:
        for k, v in extra_args.items():
            cmd.extend([f"--{k}", str(v)])

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NCCL_P2P_DISABLE"] = "0"
    env["NCCL_IB_DISABLE"] = "0"

    print(f"Launching: {' '.join(cmd)}")
    return subprocess.run(cmd, env=env, check=True)


def launch_torchrun_training(
    training_script: str = "training/train.py",
    num_gpus: int = 4,
    num_nodes: int = 1,
    node_rank: int = 0,
    master_addr: str = "localhost",
    master_port: int = 29500,
    config_path: str = "config/pipeline_config.yaml",
) -> subprocess.CompletedProcess:
    """Launch distributed training via torchrun (for FSDP)."""
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        f"--nnodes={num_nodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        training_script,
        "--config_path", config_path,
    ]

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"

    print(f"Launching: {' '.join(cmd)}")
    return subprocess.run(cmd, env=env, check=True)
