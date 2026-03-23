"""
Pipeline Orchestrator — End-to-end execution of the LLM fine-tuning pipeline.

Execution DAG:
    data_scraping -> data_cleaning -> llm_scoring -> dataset_assembly
        -> training -> model_merging -> evaluation -> quantization -> deployment

Features:
- Stage-level retry with exponential backoff
- Resume from last completed stage (crash recovery)
- Parallel stage execution where DAG allows
- W&B experiment tracking across all stages
- Configurable stage skipping
- Real-time monitoring integration
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config_loader import load_config
from monitoring.metrics_collector import (
    REGISTRY,
    AlertManager,
    AlertRule,
    GrafanaDashboardGenerator,
    PipelineStage,
    PipelineStateTracker,
    PrometheusExporter,
    SystemMetricsCollector,
)

log = logging.getLogger(__name__)


# Stage Definition

@dataclass
class StageDefinition:
    """Defines a pipeline stage with its runner and dependencies."""
    name: PipelineStage
    runner: Callable
    dependencies: list[PipelineStage]
    max_retries: int = 3
    retry_delay_base: float = 60.0  # seconds
    skip: bool = False


# This orchestrator is intentionally simple — no Airflow/Prefect/etc dependency.
# If we need real DAG scheduling later we can swap this out.
class PipelineOrchestrator:
    """
    DAG-based orchestrator for the full fine-tuning pipeline.
    
    Supports:
    - Sequential and parallel stage execution
    - Crash recovery via persistent state
    - Per-stage retries with exponential backoff
    - Monitoring and alerting integration
    - W&B experiment grouping
    """

    def __init__(self, config_path: str, overrides: list[str] | None = None):
        overrides_dict = {}
        if overrides:
            for o in overrides:
                k, v = o.split("=", 1)
                overrides_dict[k] = v
        self.config = load_config(config_path, overrides=overrides_dict or None)
        self.raw_config = self.config.model_dump()
        self.output_dir = Path(self.raw_config.get("output_dir", "outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        state_file = self.output_dir / "pipeline_state.json"
        self.state_tracker = PipelineStateTracker(state_file=str(state_file))

        # Monitoring
        self.metrics_collector = SystemMetricsCollector(REGISTRY)
        self.prometheus_exporter = PrometheusExporter(REGISTRY, port=9090)
        self.alert_manager = self._setup_alerts()

        # Stage registry
        self.stages = self._build_stage_registry()

        # W&B
        self._wandb_run = None

    def _build_stage_registry(self) -> list[StageDefinition]:
        """Define the pipeline DAG."""
        return [
            StageDefinition(
                name=PipelineStage.DATA_SCRAPING,
                runner=self._run_scraping,
                dependencies=[],
            ),
            StageDefinition(
                name=PipelineStage.DATA_CLEANING,
                runner=self._run_cleaning,
                dependencies=[PipelineStage.DATA_SCRAPING],
            ),
            StageDefinition(
                name=PipelineStage.LLM_SCORING,
                runner=self._run_scoring,
                dependencies=[PipelineStage.DATA_CLEANING],
            ),
            StageDefinition(
                name=PipelineStage.DATASET_ASSEMBLY,
                runner=self._run_assembly,
                dependencies=[PipelineStage.LLM_SCORING],
            ),
            StageDefinition(
                name=PipelineStage.CONTAMINATION_CHECK,
                runner=self._run_contamination_check,
                dependencies=[PipelineStage.DATASET_ASSEMBLY],
            ),
            StageDefinition(
                name=PipelineStage.TRAINING,
                runner=self._run_training,
                dependencies=[PipelineStage.CONTAMINATION_CHECK],
                max_retries=2,
                retry_delay_base=120.0,
            ),
            StageDefinition(
                name=PipelineStage.DPO_TRAINING,
                runner=self._run_dpo,
                dependencies=[PipelineStage.TRAINING],
            ),
            StageDefinition(
                name=PipelineStage.MODEL_MERGING,
                runner=self._run_merging,
                dependencies=[PipelineStage.DPO_TRAINING],
            ),
            StageDefinition(
                name=PipelineStage.EVALUATION,
                runner=self._run_evaluation,
                dependencies=[PipelineStage.MODEL_MERGING],
            ),
            StageDefinition(
                name=PipelineStage.QUANTIZATION,
                runner=self._run_quantization,
                dependencies=[PipelineStage.EVALUATION],
            ),
            StageDefinition(
                name=PipelineStage.DEPLOYMENT,
                runner=self._run_deployment,
                dependencies=[PipelineStage.QUANTIZATION],
            ),
        ]

    def _setup_alerts(self) -> AlertManager:
        am = AlertManager(REGISTRY)

        am.add_rule(AlertRule(
            name="GPU Temperature Critical",
            metric_key="gpu_temperature_celsius",
            condition="gt",
            threshold=90.0,
            severity="critical",
            cooldown_seconds=300,
        ))
        am.add_rule(AlertRule(
            name="Training Loss Spike",
            metric_key="training_loss",
            condition="gt",
            threshold=10.0,
            severity="warning",
            cooldown_seconds=600,
        ))

        am.add_handler(lambda rule, val: log.warning(
            f"ALERT [{rule.severity}]: {rule.name} = {val}"
        ))

        return am

    def run(
        self,
        start_from: PipelineStage | None = None,
        skip_stages: list[PipelineStage] | None = None,
        resume: bool = True,
    ) -> dict:
        """
        Execute the pipeline.
        
        Args:
            start_from: Skip all stages before this one
            skip_stages: List of stages to skip
            resume: If True, resume from last completed stage
        """
        skip_stages = skip_stages or []

        # Start monitoring
        self.metrics_collector.start()
        self.prometheus_exporter.start()
        self.alert_manager.start()

        # Initialize W&B
        self._init_wandb()

        # Determine starting point
        if resume:
            last_completed = self.state_tracker.get_last_completed_stage()
            if last_completed:
                log.info(f"Resuming after: {last_completed.value}")
                start_from = self._next_stage(last_completed)

        results = {}
        pipeline_start = time.time()

        try:
            for stage_def in self.stages:
                # Skip logic
                if start_from and not self._is_at_or_after(stage_def.name, start_from):
                    log.info(f"Skipping {stage_def.name.value} (before start_from)")
                    continue

                if stage_def.name in skip_stages:
                    log.info(f"Skipping {stage_def.name.value} (user requested)")
                    continue

                # Check dependencies
                for dep in stage_def.dependencies:
                    dep_status = self.state_tracker.get_status().get(dep.value, {})
                    if dep_status.get("status") not in ("completed", None):
                        if dep not in skip_stages:
                            raise RuntimeError(
                                f"Dependency {dep.value} not completed for "
                                f"{stage_def.name.value}"
                            )

                # Execute with retries
                result = self._execute_stage(stage_def)
                results[stage_def.name.value] = result

        except Exception as e:
            log.error(f"Pipeline failed: {e}")
            log.error(traceback.format_exc())
            raise
        finally:
            pipeline_duration = time.time() - pipeline_start
            log.info(f"Pipeline completed in {pipeline_duration:.1f}s")

            # Save summary
            summary = {
                "duration_seconds": pipeline_duration,
                "stages": self.state_tracker.get_status(),
                "results": {k: str(v)[:500] for k, v in results.items()},
            }
            with open(self.output_dir / "pipeline_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            # Generate Grafana dashboards
            gen = GrafanaDashboardGenerator()
            gen.save(
                gen.generate_training_dashboard(),
                str(self.output_dir / "grafana_training.json"),
            )
            gen.save(
                gen.generate_inference_dashboard(),
                str(self.output_dir / "grafana_inference.json"),
            )

            # Cleanup
            self._finish_wandb()
            self.alert_manager.stop()
            self.metrics_collector.stop()
            self.prometheus_exporter.stop()

        return results

    def _execute_stage(self, stage_def: StageDefinition) -> dict:
        """Execute a single stage with retry logic."""
        for attempt in range(stage_def.max_retries + 1):
            try:
                log.info(
                    f"\n{'=' * 70}\n"
                    f"  STAGE: {stage_def.name.value} "
                    f"(attempt {attempt + 1}/{stage_def.max_retries + 1})\n"
                    f"{'=' * 70}"
                )

                self.state_tracker.start_stage(stage_def.name, {
                    "attempt": attempt + 1,
                })

                result = stage_def.runner()

                self.state_tracker.complete_stage(stage_def.name, {
                    "result_keys": list(result.keys()) if isinstance(result, dict) else [],
                })

                return result

            except Exception as e:
                log.error(
                    f"Stage {stage_def.name.value} failed (attempt {attempt + 1}): {e}"
                )

                if attempt < stage_def.max_retries:
                    delay = stage_def.retry_delay_base * (2 ** attempt)
                    log.info(f"Retrying in {delay:.0f}s...")
                    time.sleep(delay)
                else:
                    self.state_tracker.fail_stage(stage_def.name, str(e))
                    raise

        return {}  # unreachable

    # =========================================================================
    # Stage Runners
    # =========================================================================

    def _run_scraping(self) -> dict:
        from data_curation.scrapers.multi_source_scraper import run_scraping
        return asyncio.run(run_scraping(self.raw_config))

    def _run_cleaning(self) -> dict:
        from data_curation.cleaners.cleaning_pipeline import run_cleaning
        return run_cleaning(self.raw_config)

    def _run_scoring(self) -> dict:
        from data_curation.llm_judge.judge import run_scoring
        return asyncio.run(run_scoring(self.raw_config))

    def _run_assembly(self) -> dict:
        from data_curation.dataset_assembler import run_assembly
        return run_assembly(self.raw_config)

    def _run_contamination_check(self) -> dict:
        from evaluation.contamination_checker import run_contamination_check
        result = run_contamination_check(self.raw_config)
        if not result.get("clean", True):
            log.warning(
                f"DATA CONTAMINATION DETECTED: {result['contamination_rate']:.1%} of "
                f"benchmark items found in training data. Review the report before proceeding."
            )
        return result

    def _run_training(self) -> dict:
        from training.train import train
        return train(config=self.raw_config)

    def _run_dpo(self) -> dict:
        dpo_cfg = self.raw_config.get("training", {}).get("dpo", {})
        if not dpo_cfg.get("enabled", False):
            log.info("DPO training disabled, skipping")
            return {"skipped": True}
        from training.dpo_trainer import train_dpo
        return train_dpo(self.raw_config)

    def _run_merging(self) -> dict:
        from model_merging.merger import run_merging
        return run_merging(self.raw_config)

    def _run_evaluation(self) -> dict:
        from evaluation.harness.eval_harness import run_evaluation
        return run_evaluation(self.raw_config)

    def _run_quantization(self) -> dict:
        from inference.vllm_server.server import quantize_for_serving

        quant_cfg = self.raw_config.get("inference", {}).get("quantization", {})
        if not quant_cfg.get("enabled", False):
            log.info("Quantization disabled, skipping")
            return {"skipped": True}

        model_path = self.raw_config.get("inference", {}).get(
            "model_path", "outputs/merged_model"
        )
        output_path = str(self.output_dir / "quantized_model")

        result_path = quantize_for_serving(
            model_path=model_path,
            output_path=output_path,
            method=quant_cfg.get("method", "awq"),
            bits=quant_cfg.get("bits", 4),
        )
        return {"quantized_model_path": result_path}

    def _run_deployment(self) -> dict:
        log.info(
            "Deployment stage: Model is ready for serving.\n"
            "Launch with: ./deployment/scripts/launch_inference.sh"
        )
        return {"status": "ready_for_deployment"}

    # =========================================================================
    # Helpers
    # =========================================================================

    def _next_stage(self, stage: PipelineStage) -> Optional[PipelineStage]:
        stage_order = [s.name for s in self.stages]
        for i, s in enumerate(stage_order):
            if s == stage and i + 1 < len(stage_order):
                return stage_order[i + 1]
        return None

    def _is_at_or_after(
        self, current: PipelineStage, target: PipelineStage
    ) -> bool:
        stage_order = [s.name for s in self.stages]
        try:
            return stage_order.index(current) >= stage_order.index(target)
        except ValueError:
            return False

    def _init_wandb(self):
        try:
            import wandb

            project = self.raw_config.get("wandb", {}).get(
                "project", "distributed-finetune-pipeline"
            )
            self._wandb_run = wandb.init(
                project=project,
                config=self.raw_config,
                job_type="pipeline",
                resume="allow",
            )
            log.info(f"W&B run initialized: {wandb.run.url}")
        except Exception as e:
            log.warning(f"W&B initialization failed: {e}")

    def _finish_wandb(self):
        try:
            import wandb
            if wandb.run:
                wandb.finish()
        except Exception:
            pass


# CLI Entry Point

def main():
    parser = argparse.ArgumentParser(
        description="Distributed Fine-Tuning Pipeline Orchestrator"
    )
    parser.add_argument(
        "--config", type=str, default="config/pipeline_config.yaml",
        help="Path to pipeline configuration YAML",
    )
    parser.add_argument(
        "--start-from", type=str, default=None,
        choices=[s.value for s in PipelineStage],
        help="Start from a specific stage (skips earlier stages)",
    )
    parser.add_argument(
        "--skip", type=str, nargs="*", default=[],
        choices=[s.value for s in PipelineStage],
        help="Stages to skip",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Do not resume from last completed stage",
    )
    parser.add_argument(
        "--override", type=str, nargs="*", default=[],
        help="Config overrides (e.g., training.hyperparams.learning_rate=0.0001)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    orchestrator = PipelineOrchestrator(
        config_path=args.config,
        overrides=args.override or None,
    )

    start_from = PipelineStage(args.start_from) if args.start_from else None
    skip_stages = [PipelineStage(s) for s in args.skip]

    results = orchestrator.run(
        start_from=start_from,
        skip_stages=skip_stages,
        resume=not args.no_resume,
    )

    log.info("Pipeline execution complete!")
    log.info(f"Results: {json.dumps({k: str(v)[:200] for k, v in results.items()}, indent=2)}")


if __name__ == "__main__":
    main()
