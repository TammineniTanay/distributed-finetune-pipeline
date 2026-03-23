"""Hyperparameter sweep — grid and random search over QLoRA training configs.

Sweeps the params that matter most:
  - lora_r: 8, 16, 32, 64, 128
  - learning_rate: 1e-5 to 5e-4 (log scale)
  - warmup_ratio: 0.01 to 0.1
  - neftune_noise_alpha: 0, 5, 10, 15

Usage:
    python -m training.hyperparam_sweep --config config/pipeline_config.yaml --method random --max-trials 10
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class SweepParam:
    name: str           # dot-notation path
    values: list[Any]   # for grid search
    low: float = 0
    high: float = 1
    log_scale: bool = False
    param_type: str = "categorical"  # categorical, continuous, integer


DEFAULT_SPACE = [
    SweepParam("training.qlora.r", [8, 16, 32, 64, 128]),
    SweepParam("training.hyperparameters.learning_rate", [1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
               low=1e-5, high=5e-4, log_scale=True, param_type="continuous"),
    SweepParam("training.hyperparameters.warmup_ratio", [0.01, 0.03, 0.05, 0.1],
               low=0.01, high=0.1, param_type="continuous"),
    SweepParam("training.hyperparameters.neftune_noise_alpha", [0, 5, 10, 15]),
]


class HyperparamSweep:
    def __init__(self, base_config, params=None, method="grid", max_trials=20, seed=42):
        self.base_config = base_config
        self.params = params or DEFAULT_SPACE
        self.method = method
        self.max_trials = max_trials
        self.rng = random.Random(seed)

    def generate_configs(self):
        if self.method == "grid":
            return self._grid()
        return self._random()

    def _grid(self):
        all_combos = list(itertools.product(*[p.values for p in self.params]))
        if len(all_combos) > self.max_trials:
            all_combos = self.rng.sample(all_combos, self.max_trials)

        configs = []
        for combo in all_combos:
            c = copy.deepcopy(self.base_config)
            for param, val in zip(self.params, combo):
                self._set(c, param.name, val)
            configs.append(c)
        return configs

    def _random(self):
        configs = []
        for _ in range(self.max_trials):
            c = copy.deepcopy(self.base_config)
            for p in self.params:
                if p.param_type == "continuous":
                    if p.log_scale:
                        val = math.exp(self.rng.uniform(math.log(p.low), math.log(p.high)))
                    else:
                        val = self.rng.uniform(p.low, p.high)
                elif p.param_type == "integer":
                    val = self.rng.randint(int(p.low), int(p.high))
                else:
                    val = self.rng.choice(p.values)
                self._set(c, p.name, val)
            configs.append(c)
        return configs

    def describe_trial(self, config):
        return {p.name: self._get(config, p.name) for p in self.params}

    def save_plan(self, configs, path="outputs/sweep_plan.json"):
        plan = {
            "method": self.method, "num_trials": len(configs),
            "trials": [{"trial": i, "params": self.describe_trial(c)} for i, c in enumerate(configs)],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(plan, f, indent=2)
        log.info(f"Sweep plan: {len(configs)} trials -> {path}")

    def _set(self, d, key, val):
        keys = key.split(".")
        for k in keys[:-1]: d = d.setdefault(k, {})
        d[keys[-1]] = val

    def _get(self, d, key, default=None):
        for k in key.split("."):
            d = d.get(k, default) if isinstance(d, dict) else default
        return d


def run_sweep(config, method="random", max_trials=10):
    sweep = HyperparamSweep(config, method=method, max_trials=max_trials)
    configs = sweep.generate_configs()
    sweep.save_plan(configs)
    for i, c in enumerate(configs):
        log.info(f"Trial {i}: {sweep.describe_trial(c)}")
    return configs
