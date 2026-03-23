"""Model merging implementations: TIES, DARE, SLERP, linear.

All implemented from scratch (no mergekit dependency) so we can
customize the merge logic and run it programmatically. The mergekit
YAML generator is there as a fallback for CLI usage.

Ref: https://arxiv.org/abs/2306.01708 (TIES)
     https://arxiv.org/abs/2311.03099 (DARE)
"""
from __future__ import annotations

import gc
import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from safetensors.torch import load_file, save_file
from tqdm import tqdm

log = logging.getLogger(__name__)


# Task Vector (delta weights)

class TaskVector:
    """
    Represents the difference between a fine-tuned model and the base model.
    task_vector = finetuned_weights - base_weights
    """

    def __init__(self, base_state: dict[str, torch.Tensor], ft_state: dict[str, torch.Tensor]):
        self.vector: dict[str, torch.Tensor] = {}
        for key in base_state:
            if key in ft_state and base_state[key].shape == ft_state[key].shape:
                self.vector[key] = ft_state[key] - base_state[key]

    def keys(self):
        return self.vector.keys()

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.vector[key]

    def __setitem__(self, key: str, value: torch.Tensor):
        self.vector[key] = value


# Merging Algorithms (from scratch)

# The paper calls this "TrIm, Elect Sign, and merge" which is
# a terrible acronym but the method actually works really well
class TIESMerger:
    """
    TIES-Merging: Trim, Elect Sign, and Merge.
    
    1. Trim: Zero out small-magnitude changes (density-based)
    2. Elect Sign: For each parameter, resolve sign conflicts via majority vote
    3. Merge: Average the consistent-sign parameters
    """

    @staticmethod
    def merge(
        base_state: dict[str, torch.Tensor],
        task_vectors: list[TaskVector],
        weights: list[float],
        density: float = 0.5,
        normalize: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Perform TIES merging."""
        merged = {}
        keys = set()
        for tv in task_vectors:
            keys.update(tv.keys())

        for key in tqdm(keys, desc="TIES merging"):
            if key not in base_state:
                continue

            base_param = base_state[key]
            deltas = []

            for tv, w in zip(task_vectors, weights):
                if key in tv.vector:
                    delta = tv[key].float() * w
                else:
                    delta = torch.zeros_like(base_param, dtype=torch.float32)
                deltas.append(delta)

            # Stack deltas: (num_models, *param_shape)
            stacked = torch.stack(deltas, dim=0)

            # Step 1: TRIM - zero out small-magnitude entries
            if density < 1.0:
                for i in range(len(deltas)):
                    flat = stacked[i].abs().flatten()
                    if flat.numel() == 0:
                        continue
                    k = max(1, int(flat.numel() * (1 - density)))
                    threshold = torch.kthvalue(flat, k).values
                    mask = stacked[i].abs() >= threshold
                    stacked[i] = stacked[i] * mask

            # Step 2: ELECT SIGN - majority vote
            signs = torch.sign(stacked)
            # Sum of signs: positive means more models agree on positive
            sign_sum = signs.sum(dim=0)
            elected_sign = torch.sign(sign_sum)
            # Where sign_sum is 0, keep the sign of the model with largest magnitude
            zero_mask = elected_sign == 0
            if zero_mask.any():
                max_mag_idx = stacked.abs().sum(dim=tuple(range(1, stacked.dim()))).argmax()
                elected_sign[zero_mask] = signs[max_mag_idx][zero_mask]

            # Step 3: MERGE - average only entries matching the elected sign
            aligned = torch.zeros_like(base_param, dtype=torch.float32)
            count = torch.zeros_like(base_param, dtype=torch.float32)

            for i in range(len(deltas)):
                agreement = (torch.sign(stacked[i]) == elected_sign)
                aligned += stacked[i] * agreement
                count += agreement.float()

            count = torch.clamp(count, min=1.0)
            merged_delta = aligned / count

            if normalize:
                # Rescale to match the average norm of input task vectors
                avg_norm = torch.stack([d.norm() for d in deltas]).mean()
                if merged_delta.norm() > 0:
                    merged_delta = merged_delta * (avg_norm / merged_delta.norm())

            merged[key] = (base_param.float() + merged_delta).to(base_param.dtype)

        # Copy any keys not in task vectors
        for key in base_state:
            if key not in merged:
                merged[key] = base_state[key]

        return merged


class DAREMerger:
    """
    DARE (Drop And REscale) merging.
    
    Randomly drops delta parameters with probability (1 - density),
    then rescales remaining parameters to preserve expected value.
    Can be combined with TIES or linear merging.
    """

    @staticmethod
    def apply_dare(
        task_vector: TaskVector,
        density: float = 0.5,
        rescale: bool = True,
        seed: int = 42,
    ) -> TaskVector:
        """Apply DARE to a single task vector."""
        rng = torch.Generator()
        rng.manual_seed(seed)

        dare_tv = TaskVector.__new__(TaskVector)
        dare_tv.vector = {}

        for key, delta in task_vector.vector.items():
            # Generate random mask
            mask = torch.bernoulli(
                torch.full_like(delta, density, dtype=torch.float32),
                generator=rng,
            ).to(delta.dtype)

            dropped = delta * mask

            # Rescale to preserve expected value
            if rescale and density > 0:
                dropped = dropped / density

            dare_tv.vector[key] = dropped

        return dare_tv

    @staticmethod
    def merge_dare_ties(
        base_state: dict[str, torch.Tensor],
        task_vectors: list[TaskVector],
        weights: list[float],
        density: float = 0.5,
        rescale: bool = True,
        normalize: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Apply DARE to each task vector, then merge with TIES."""
        dare_tvs = []
        for i, tv in enumerate(task_vectors):
            dare_tv = DAREMerger.apply_dare(tv, density=density, rescale=rescale, seed=42 + i)
            dare_tvs.append(dare_tv)

        return TIESMerger.merge(
            base_state, dare_tvs, weights,
            density=1.0,  # DARE already handles sparsity
            normalize=normalize,
        )

    @staticmethod
    def merge_dare_linear(
        base_state: dict[str, torch.Tensor],
        task_vectors: list[TaskVector],
        weights: list[float],
        density: float = 0.5,
        rescale: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Apply DARE to each task vector, then linear merge."""
        merged = {}
        keys = set()
        for tv in task_vectors:
            keys.update(tv.keys())

        dare_tvs = []
        for i, tv in enumerate(task_vectors):
            dare_tvs.append(DAREMerger.apply_dare(tv, density=density, rescale=rescale, seed=42 + i))

        for key in tqdm(keys, desc="DARE-Linear merging"):
            if key not in base_state:
                continue

            combined = torch.zeros_like(base_state[key], dtype=torch.float32)
            for tv, w in zip(dare_tvs, weights):
                if key in tv.vector:
                    combined += tv[key].float() * w

            merged[key] = (base_state[key].float() + combined).to(base_state[key].dtype)

        for key in base_state:
            if key not in merged:
                merged[key] = base_state[key]

        return merged


class SLERPMerger:
    """
    Spherical Linear Interpolation (SLERP) for model weights.
    
    Interpolates between two models along a great circle on the
    unit hypersphere in weight space.
    """

    @staticmethod
    def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
        """SLERP between two tensors."""
        v0_flat = v0.flatten().float()
        v1_flat = v1.flatten().float()

        # Normalize
        v0_norm = v0_flat / (v0_flat.norm() + 1e-8)
        v1_norm = v1_flat / (v1_flat.norm() + 1e-8)

        # Compute angle
        dot = torch.clamp(torch.dot(v0_norm, v1_norm), -1.0, 1.0)
        omega = torch.acos(dot)

        if omega.abs() < 1e-6:
            # Vectors are nearly parallel - use linear interpolation
            result = (1 - t) * v0_flat + t * v1_flat
        else:
            sin_omega = torch.sin(omega)
            result = (
                torch.sin((1 - t) * omega) / sin_omega * v0_flat
                + torch.sin(t * omega) / sin_omega * v1_flat
            )

        # Scale to average magnitude
        avg_norm = (1 - t) * v0_flat.norm() + t * v1_flat.norm()
        result = result / (result.norm() + 1e-8) * avg_norm

        return result.reshape(v0.shape).to(v0.dtype)

    @staticmethod
    def merge(
        state_a: dict[str, torch.Tensor],
        state_b: dict[str, torch.Tensor],
        t: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Merge two model state dicts via SLERP."""
        merged = {}
        for key in tqdm(state_a, desc="SLERP merging"):
            if key in state_b and state_a[key].shape == state_b[key].shape:
                if state_a[key].numel() > 1:
                    merged[key] = SLERPMerger.slerp(state_a[key], state_b[key], t)
                else:
                    merged[key] = (1 - t) * state_a[key].float() + t * state_b[key].float()
                    merged[key] = merged[key].to(state_a[key].dtype)
            else:
                merged[key] = state_a[key]
        return merged


class LinearMerger:
    """Simple weighted linear interpolation of model weights."""

    @staticmethod
    def merge(
        base_state: dict[str, torch.Tensor],
        task_vectors: list[TaskVector],
        weights: list[float],
    ) -> dict[str, torch.Tensor]:
        merged = {}
        keys = set()
        for tv in task_vectors:
            keys.update(tv.keys())

        for key in tqdm(keys, desc="Linear merging"):
            if key not in base_state:
                continue

            combined = torch.zeros_like(base_state[key], dtype=torch.float32)
            for tv, w in zip(task_vectors, weights):
                if key in tv.vector:
                    combined += tv[key].float() * w

            merged[key] = (base_state[key].float() + combined).to(base_state[key].dtype)

        for key in base_state:
            if key not in merged:
                merged[key] = base_state[key]

        return merged


# Mergekit YAML Config Generator

class MergekitConfigGenerator:
    """Generate mergekit-compatible YAML configs."""

    @staticmethod
    def generate_ties_config(
        base_model: str,
        models: list[dict],
        density: float = 0.5,
        normalize: bool = True,
        int8_mask: bool = True,
    ) -> dict:
        return {
            "merge_method": "ties",
            "base_model": base_model,
            "parameters": {
                "density": density,
                "normalize": normalize,
                "int8_mask": int8_mask,
            },
            "models": [
                {"model": m["path"], "parameters": {"weight": m.get("weight", 1.0)}}
                for m in models
            ],
            "dtype": "bfloat16",
        }

    @staticmethod
    def generate_dare_config(
        base_model: str,
        models: list[dict],
        density: float = 0.5,
        method: str = "dare_ties",
    ) -> dict:
        return {
            "merge_method": method,
            "base_model": base_model,
            "parameters": {
                "density": density,
                "normalize": True,
            },
            "models": [
                {"model": m["path"], "parameters": {"weight": m.get("weight", 1.0)}}
                for m in models
            ],
            "dtype": "bfloat16",
        }

    @staticmethod
    def generate_slerp_config(
        models: list[dict],
        t: float = 0.5,
    ) -> dict:
        if len(models) != 2:
            raise ValueError("SLERP requires exactly 2 models")
        return {
            "merge_method": "slerp",
            "base_model": models[0]["path"],
            "parameters": {"t": t},
            "models": [{"model": m["path"]} for m in models],
            "dtype": "bfloat16",
        }


# Merging Orchestrator

class ModelMerger:
    """
    Orchestrates model merging using various algorithms.
    
    Supports both native Python merging and mergekit CLI.
    """

    def __init__(self, config: dict):
        self.config = config
        self.base_model = config["base_model"]
        self.models = config["models_to_merge"]
        self.methods = config.get("methods", [])
        self.output_dir = Path(config.get("output_dir", "outputs/merged"))
        self.dtype = config.get("dtype", "bfloat16")
        self.device = config.get("device", "cuda")

    def _load_state_dict(self, model_path: str) -> dict[str, torch.Tensor]:
        """Load model state dict from safetensors or pytorch format."""
        path = Path(model_path)

        # Try safetensors first
        st_files = list(path.glob("*.safetensors"))
        if st_files:
            state = {}
            for f in sorted(st_files):
                state.update(load_file(str(f)))
            return state

        # Try pytorch format
        pt_files = list(path.glob("*.bin"))
        if pt_files:
            state = {}
            for f in sorted(pt_files):
                state.update(torch.load(f, map_location="cpu"))
            return state

        # Try single file
        if (path / "pytorch_model.bin").exists():
            return torch.load(path / "pytorch_model.bin", map_location="cpu")

        raise FileNotFoundError(f"No model files found in {path}")

    def _save_merged_model(self, state_dict: dict, method_name: str, source_model_path: str):
        """Save merged model weights and copy config/tokenizer files."""
        output_path = self.output_dir / method_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Save weights as safetensors
        save_file(state_dict, str(output_path / "model.safetensors"))

        # Copy config and tokenizer from source
        source = Path(source_model_path)
        for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                       "special_tokens_map.json", "generation_config.json"]:
            src_file = source / fname
            if src_file.exists():
                shutil.copy2(src_file, output_path / fname)

        log.info(f"Saved merged model ({method_name}) to {output_path}")
        return str(output_path)

    def merge(self) -> dict[str, str]:
        """
        Execute all configured merging methods.
        Returns dict mapping method_name → output_path.
        """
        results = {}

        # Load base model state dict
        log.info(f"Loading base model: {self.base_model}")
        base_state = self._load_state_dict(self.base_model)

        # Load fine-tuned model state dicts and compute task vectors
        task_vectors = []
        model_states = []
        for model_cfg in self.models:
            log.info(f"Loading model: {model_cfg['name']} from {model_cfg['path']}")
            ft_state = self._load_state_dict(model_cfg["path"])
            task_vectors.append(TaskVector(base_state, ft_state))
            model_states.append(ft_state)

        weights = [m.get("weight", 1.0) for m in self.models]

        for method_cfg in self.methods:
            method_name = method_cfg["name"] if isinstance(method_cfg, dict) else method_cfg.name
            params = method_cfg["config"] if isinstance(method_cfg, dict) else method_cfg.config.copy()

            log.info(f"Merging with method: {method_name}")

            if method_name == "ties":
                merged_state = TIESMerger.merge(
                    base_state, task_vectors, weights,
                    density=params.get("density", 0.5),
                    normalize=params.get("normalize", True),
                )
            elif method_name == "dare_ties":
                merged_state = DAREMerger.merge_dare_ties(
                    base_state, task_vectors, weights,
                    density=params.get("density", 0.5),
                    rescale=params.get("rescale", True),
                    normalize=params.get("normalize", True),
                )
            elif method_name == "dare_linear":
                merged_state = DAREMerger.merge_dare_linear(
                    base_state, task_vectors, weights,
                    density=params.get("density", 0.5),
                    rescale=params.get("rescale", True),
                )
            elif method_name == "slerp":
                if len(model_states) != 2:
                    log.warning("SLERP requires exactly 2 models, skipping")
                    continue
                merged_state = SLERPMerger.merge(
                    model_states[0], model_states[1],
                    t=params.get("t", 0.5),
                )
            elif method_name == "linear":
                merged_state = LinearMerger.merge(base_state, task_vectors, weights)
            else:
                log.warning(f"Unknown merge method: {method_name}, skipping")
                continue

            output_path = self._save_merged_model(
                merged_state, method_name, self.models[0]["path"]
            )
            results[method_name] = output_path

            # Free memory
            del merged_state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Generate mergekit configs
        mergekit_dir = Path(self.config.get("mergekit_config_dir", "model_merging/configs"))
        mergekit_dir.mkdir(parents=True, exist_ok=True)
        self._generate_mergekit_configs(mergekit_dir)

        return results

    def _generate_mergekit_configs(self, output_dir: Path):
        """Generate mergekit YAML configs for each method."""
        gen = MergekitConfigGenerator()
        model_list = [{"path": m["path"], "weight": m.get("weight", 1.0)} for m in self.models]

        for method_cfg in self.methods:
            method_name = method_cfg["name"] if isinstance(method_cfg, dict) else method_cfg.name
            params = method_cfg["config"] if isinstance(method_cfg, dict) else method_cfg.config

            if method_name == "ties":
                cfg = gen.generate_ties_config(self.base_model, model_list, **params)
            elif method_name in ("dare_ties", "dare_linear"):
                cfg = gen.generate_dare_config(self.base_model, model_list, method=method_name, **{k: v for k, v in params.items() if k in ("density",)})
            elif method_name == "slerp":
                cfg = gen.generate_slerp_config(model_list, t=params.get("t", 0.5))
            else:
                continue

            config_file = output_dir / f"{method_name}_config.yaml"
            with open(config_file, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            log.info(f"Generated mergekit config: {config_file}")


def run_merging(config: dict) -> dict:
    """Entry point for the model merging stage."""
    merger = ModelMerger(config["model_merging"])
    return merger.merge()
