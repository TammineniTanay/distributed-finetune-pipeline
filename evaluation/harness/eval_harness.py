"""Evaluation harness — benchmarks merged models across multiple tasks.

Supports MC (log-likelihood), generation (ROUGE/BERTScore), and
wraps lm-evaluation-harness for standard benchmarks. Statistical
significance via paired bootstrap + McNemar.

The LLM judge here is separate from the data curation judge —
this one evaluates model outputs, not training data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

log = logging.getLogger(__name__)


# Evaluation Result Models

@dataclass
class EvalResult:
    """Single model × single benchmark result."""
    model_name: str
    benchmark_name: str
    metrics: dict[str, float]
    num_examples: int
    latency_per_example_ms: float
    raw_predictions: list[dict] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Model A vs B on a specific benchmark+metric."""
    model_a: str
    model_b: str
    benchmark: str
    metric: str
    value_a: float
    value_b: float
    delta: float
    p_value: float
    significant: bool  # p < 0.05


# Model Loader (for evaluation)

class EvalModelLoader:
    """Loads models for evaluation with efficient memory management."""

    @staticmethod
    def load_model(
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        quantization: str | None = None,
    ):
        """Load a model for evaluation."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {
            "torch_dtype": getattr(torch, dtype),
            "device_map": "auto",
            "trust_remote_code": False,
        }

        if quantization == "awq":
            load_kwargs["quantization_config"] = {"bits": 4, "quant_method": "awq"}
        elif quantization == "gptq":
            load_kwargs["quantization_config"] = {"bits": 4, "quant_method": "gptq"}

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        model.eval()

        return model, tokenizer

    @staticmethod
    def unload_model(model):
        """Free GPU memory."""
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Benchmark Runners

class BenchmarkRunner(ABC):
    """Abstract base for benchmark implementations."""

    @abstractmethod
    def run(self, model, tokenizer, config: dict) -> EvalResult:
        ...


class MultipleChoiceBenchmark(BenchmarkRunner):
    """
    Evaluates multiple-choice questions.
    
    Scoring: computes log-likelihood of each choice continuation
    and selects the one with highest probability.
    """

    def run(self, model, tokenizer, config: dict) -> EvalResult:
        data_path = config["data_path"]
        num_few_shot = config.get("num_few_shot", 5)
        model_name = config.get("_model_name", "unknown")
        benchmark_name = config.get("name", "mc_benchmark")

        # Load data
        examples = []
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        correct = 0
        total = 0
        latencies = []
        predictions = []

        for ex in tqdm(examples, desc=f"Evaluating {benchmark_name}"):
            question = ex["question"]
            choices = ex["choices"]       # List of answer strings
            correct_idx = ex["answer"]    # Index of correct answer

            # Build few-shot prefix
            few_shot = ex.get("few_shot_examples", [])[:num_few_shot]
            prefix = ""
            for fs in few_shot:
                prefix += f"Q: {fs['question']}\nA: {fs['choices'][fs['answer']]}\n\n"

            # Score each choice
            choice_scores = []
            start = time.monotonic()

            for choice in choices:
                prompt = f"{prefix}Q: {question}\nA: {choice}"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits

                # Score = sum of log-probs for the answer tokens
                answer_tokens = tokenizer(choice, return_tensors="pt")["input_ids"][0]
                n_answer = len(answer_tokens)

                log_probs = torch.nn.functional.log_softmax(logits[0, -n_answer-1:-1, :], dim=-1)
                score = sum(
                    log_probs[i, answer_tokens[i]].item()
                    for i in range(n_answer)
                )
                choice_scores.append(score)

            elapsed = (time.monotonic() - start) * 1000
            latencies.append(elapsed)

            predicted_idx = int(np.argmax(choice_scores))
            is_correct = predicted_idx == correct_idx
            correct += int(is_correct)
            total += 1

            predictions.append({
                "question": question,
                "predicted": predicted_idx,
                "correct": correct_idx,
                "is_correct": is_correct,
                "scores": choice_scores,
            })

        accuracy = correct / total if total > 0 else 0.0

        return EvalResult(
            model_name=model_name,
            benchmark_name=benchmark_name,
            metrics={"accuracy": accuracy, "correct": correct, "total": total},
            num_examples=total,
            latency_per_example_ms=np.mean(latencies) if latencies else 0,
            raw_predictions=predictions,
        )


# FIXME: generation eval is slow as hell because we cant batch properly
# with variable-length outputs. vLLM batched inference would help here.
class GenerationBenchmark(BenchmarkRunner):
    """
    Evaluates open-ended generation quality.
    
    Metrics: ROUGE-L, BERTScore, LLM-as-judge.
    """

    def run(self, model, tokenizer, config: dict) -> EvalResult:
        data_path = config["data_path"]
        max_new_tokens = config.get("max_new_tokens", 512)
        metrics_list = config.get("metrics", ["rouge_l"])
        model_name = config.get("_model_name", "unknown")
        benchmark_name = config.get("name", "gen_benchmark")

        # Load data
        examples = []
        with open(data_path) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        generations = []
        references = []
        latencies = []

        for ex in tqdm(examples, desc=f"Generating {benchmark_name}"):
            prompt = ex["prompt"]
            reference = ex.get("reference", "")

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            start = time.monotonic()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            elapsed = (time.monotonic() - start) * 1000
            latencies.append(elapsed)

            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            generations.append(generated)
            references.append(reference)

        # Compute metrics
        result_metrics = {}

        if "rouge_l" in metrics_list:
            result_metrics["rouge_l"] = self._compute_rouge(generations, references)

        if "bertscore" in metrics_list:
            result_metrics["bertscore_f1"] = self._compute_bertscore(generations, references)

        predictions = [
            {"prompt": ex["prompt"], "generated": gen, "reference": ref}
            for ex, gen, ref in zip(examples, generations, references)
        ]

        return EvalResult(
            model_name=model_name,
            benchmark_name=benchmark_name,
            metrics=result_metrics,
            num_examples=len(examples),
            latency_per_example_ms=np.mean(latencies) if latencies else 0,
            raw_predictions=predictions,
        )

    def _compute_rouge(self, predictions: list[str], references: list[str]) -> float:
        """Compute ROUGE-L F1 score."""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = [
                scorer.score(ref, pred)["rougeL"].fmeasure
                for pred, ref in zip(predictions, references)
                if ref  # Skip empty references
            ]
            return float(np.mean(scores)) if scores else 0.0
        except ImportError:
            log.warning("rouge_score not installed, skipping ROUGE-L")
            return 0.0

    def _compute_bertscore(self, predictions: list[str], references: list[str]) -> float:
        """Compute BERTScore F1."""
        try:
            from bert_score import score as bert_score
            valid = [(p, r) for p, r in zip(predictions, references) if r]
            if not valid:
                return 0.0
            preds, refs = zip(*valid)
            _, _, f1 = bert_score(list(preds), list(refs), lang="en", verbose=False)
            return float(f1.mean())
        except ImportError:
            log.warning("bert_score not installed, skipping BERTScore")
            return 0.0


class LMEvalHarnessBenchmark(BenchmarkRunner):
    """Run standard benchmarks using the lm-evaluation-harness."""

    def run(self, model, tokenizer, config: dict) -> EvalResult:
        task = config["task"]
        num_few_shot = config.get("num_few_shot", 0)
        model_name = config.get("_model_name", "unknown")

        try:
            import lm_eval
            from lm_eval.models.huggingface import HFLM

            lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")

            results = lm_eval.simple_evaluate(
                model=lm,
                tasks=[task],
                num_fewshot=num_few_shot,
                batch_size="auto",
            )

            task_results = results.get("results", {}).get(task, {})
            metrics = {}
            for k, v in task_results.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)

            return EvalResult(
                model_name=model_name,
                benchmark_name=task,
                metrics=metrics,
                num_examples=task_results.get("num_examples", 0),
                latency_per_example_ms=0,
            )
        except ImportError:
            log.warning("lm-evaluation-harness not installed")
            return EvalResult(
                model_name=model_name,
                benchmark_name=task,
                metrics={"error": -1},
                num_examples=0,
                latency_per_example_ms=0,
            )


# LLM Judge for Generation Evaluation

class EvalLLMJudge:
    """Uses an LLM to judge generation quality with structured rubrics."""

    def __init__(self, config: dict):
        self.model = config.get("model", "gpt-4o")
        self.provider = config.get("provider", "openai")
        self.num_judges = config.get("num_judges", 3)
        self.rubric_path = config.get("rubric_path")
        self.rubric = self._load_rubric()

    def _load_rubric(self) -> dict:
        if self.rubric_path and Path(self.rubric_path).exists():
            with open(self.rubric_path) as f:
                return yaml.safe_load(f)
        return {
            "criteria": [
                {"name": "accuracy", "weight": 0.3, "description": "Factual correctness"},
                {"name": "completeness", "weight": 0.3, "description": "Covers all aspects"},
                {"name": "clarity", "weight": 0.2, "description": "Clear and well-organized"},
                {"name": "helpfulness", "weight": 0.2, "description": "Useful and actionable"},
            ]
        }

    async def judge(self, predictions: list[dict]) -> list[dict]:
        """Judge a list of prediction dicts using the configured LLM."""
        import openai
        client = openai.AsyncOpenAI()

        results = []
        for pred in tqdm(predictions, desc="LLM judging"):
            scores_list = []
            for _ in range(self.num_judges):
                prompt = self._build_judge_prompt(pred)
                try:
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=256,
                        temperature=0.3,
                    )
                    score_data = json.loads(response.choices[0].message.content)
                    scores_list.append(score_data)
                except Exception as e:
                    log.warning(f"Judge error: {e}")

            # Aggregate scores (majority vote / average)
            if scores_list:
                avg_scores = {}
                for criterion in self.rubric["criteria"]:
                    name = criterion["name"]
                    vals = [s.get(name, 0) for s in scores_list if name in s]
                    avg_scores[name] = np.mean(vals) if vals else 0

                composite = sum(
                    avg_scores.get(c["name"], 0) * c["weight"]
                    for c in self.rubric["criteria"]
                )

                pred["judge_scores"] = avg_scores
                pred["judge_composite"] = float(composite)

            results.append(pred)

        return results

    def _build_judge_prompt(self, pred: dict) -> str:
        criteria_desc = "\n".join(
            f"- {c['name']}: {c['description']}"
            for c in self.rubric["criteria"]
        )
        return f"""Evaluate the following AI-generated response on a scale of 1-5 for each criterion.

Criteria:
{criteria_desc}

Prompt: {pred.get('prompt', '')}

Reference (gold answer): {pred.get('reference', 'N/A')}

Model response: {pred.get('generated', '')}

Respond ONLY with JSON: {{"accuracy": N, "completeness": N, "clarity": N, "helpfulness": N}}"""


# Statistical Significance Testing

class SignificanceTester:
    """Significance tests — paired bootstrap and McNemar."""

    @staticmethod
    def paired_bootstrap(
        scores_a: list[float],
        scores_b: list[float],
        n_bootstrap: int = 10000,
        seed: int = 42,
    ) -> float:
        """Paired bootstrap test. Returns p-value."""
        rng = np.random.RandomState(seed)
        n = len(scores_a)
        assert n == len(scores_b), "Score lists must have same length"

        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)
        observed_diff = np.mean(scores_a) - np.mean(scores_b)

        count = 0
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            boot_diff = np.mean(scores_a[idx]) - np.mean(scores_b[idx])
            if boot_diff <= 0:  # Count times difference is in wrong direction
                count += 1

        return count / n_bootstrap

    @staticmethod
    def mcnemar_test(correct_a: list[bool], correct_b: list[bool]) -> float:
        """McNemar's test for paired binary outcomes."""
        a = np.array(correct_a)
        b = np.array(correct_b)

        # Discordant pairs
        b_correct_a_wrong = np.sum(~a & b)
        a_correct_b_wrong = np.sum(a & ~b)

        if b_correct_a_wrong + a_correct_b_wrong == 0:
            return 1.0

        # McNemar's chi-squared
        chi2 = (abs(b_correct_a_wrong - a_correct_b_wrong) - 1) ** 2 / (
            b_correct_a_wrong + a_correct_b_wrong
        )
        return float(1 - stats.chi2.cdf(chi2, df=1))


# Evaluation Orchestrator

class EvaluationPipeline:
    """
    Orchestrates evaluation across all models and benchmarks.
    Generates comparison reports and significance tests.
    """

    RUNNER_MAP = {
        "multiple_choice": MultipleChoiceBenchmark,
        "generation": GenerationBenchmark,
        "lm_eval_harness": LMEvalHarnessBenchmark,
    }

    def __init__(self, config: dict):
        self.config = config
        self.models = config["models_to_evaluate"]
        self.benchmarks = config["benchmarks"]
        self.output_dir = Path(config.get("output_dir", "outputs/evaluation"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, list[EvalResult]] = defaultdict(list)

    def run(self) -> dict:
        """Run all evaluations."""
        all_results = {}

        for model_cfg in self.models:
            model_name = model_cfg["name"]
            model_path = model_cfg["path"]

            log.info(f"{'=' * 60}")
            log.info(f"Evaluating model: {model_name} ({model_path})")
            log.info(f"{'=' * 60}")

            model, tokenizer = EvalModelLoader.load_model(model_path)

            model_results = {}
            for bench_cfg in self.benchmarks:
                bench_name = bench_cfg["name"]
                bench_type = bench_cfg["type"]

                runner_cls = self.RUNNER_MAP.get(bench_type)
                if not runner_cls:
                    log.warning(f"Unknown benchmark type: {bench_type}")
                    continue

                runner = runner_cls()
                bench_cfg_copy = dict(bench_cfg)
                bench_cfg_copy["_model_name"] = model_name

                try:
                    result = runner.run(model, tokenizer, bench_cfg_copy)
                    model_results[bench_name] = result
                    self.results[model_name].append(result)
                    log.info(f"  {bench_name}: {result.metrics}")
                except Exception as e:
                    log.error(f"  {bench_name} failed: {e}")
                    model_results[bench_name] = EvalResult(
                        model_name=model_name,
                        benchmark_name=bench_name,
                        metrics={"error": str(e)},
                        num_examples=0,
                        latency_per_example_ms=0,
                    )

            all_results[model_name] = model_results
            EvalModelLoader.unload_model(model)

        # Generate reports
        self._save_results(all_results)
        comparisons = self._compute_comparisons()
        report = self._generate_report(all_results, comparisons)

        return report

    def _compute_comparisons(self) -> list[ComparisonResult]:
        """Compute pairwise statistical comparisons."""
        comparisons = []
        model_names = list(self.results.keys())
        tester = SignificanceTester()

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name_a, name_b = model_names[i], model_names[j]

                for result_a in self.results[name_a]:
                    # Find matching benchmark in model_b
                    result_b = None
                    for r in self.results[name_b]:
                        if r.benchmark_name == result_a.benchmark_name:
                            result_b = r
                            break

                    if result_b is None:
                        continue

                    for metric in result_a.metrics:
                        if metric in result_b.metrics:
                            val_a = result_a.metrics[metric]
                            val_b = result_b.metrics[metric]

                            # Bootstrap test using raw predictions if available
                            if result_a.raw_predictions and result_b.raw_predictions:
                                scores_a = [p.get("is_correct", 0) for p in result_a.raw_predictions]
                                scores_b = [p.get("is_correct", 0) for p in result_b.raw_predictions]
                                if len(scores_a) == len(scores_b):
                                    p_val = tester.paired_bootstrap(scores_a, scores_b)
                                else:
                                    p_val = 1.0
                            else:
                                p_val = 1.0

                            comparisons.append(ComparisonResult(
                                model_a=name_a,
                                model_b=name_b,
                                benchmark=result_a.benchmark_name,
                                metric=metric,
                                value_a=val_a,
                                value_b=val_b,
                                delta=val_a - val_b,
                                p_value=p_val,
                                significant=p_val < 0.05,
                            ))

        return comparisons

    def _save_results(self, all_results: dict):
        """Save raw results to JSON."""
        serializable = {}
        for model_name, benchmarks in all_results.items():
            serializable[model_name] = {
                bench_name: {
                    "metrics": result.metrics,
                    "num_examples": result.num_examples,
                    "latency_per_example_ms": result.latency_per_example_ms,
                }
                for bench_name, result in benchmarks.items()
            }

        with open(self.output_dir / "eval_results.json", "w") as f:
            json.dump(serializable, f, indent=2)

    def _generate_report(self, all_results: dict, comparisons: list[ComparisonResult]) -> dict:
        """Generate a comprehensive evaluation report."""
        report = {
            "summary": {},
            "per_model": {},
            "comparisons": [],
            "best_model_per_benchmark": {},
        }

        # Per-model summary
        for model_name, benchmarks in all_results.items():
            model_summary = {}
            for bench_name, result in benchmarks.items():
                model_summary[bench_name] = result.metrics
            report["per_model"][model_name] = model_summary

        # Best model per benchmark
        for bench_cfg in self.benchmarks:
            bench_name = bench_cfg["name"]
            metric = bench_cfg.get("metric", "accuracy")
            best_model = None
            best_score = -float("inf")

            for model_name, benchmarks in all_results.items():
                if bench_name in benchmarks:
                    score = benchmarks[bench_name].metrics.get(metric, -float("inf"))
                    if score > best_score:
                        best_score = score
                        best_model = model_name

            report["best_model_per_benchmark"][bench_name] = {
                "model": best_model,
                "score": best_score,
                "metric": metric,
            }

        # Comparisons
        report["comparisons"] = [
            {
                "model_a": c.model_a,
                "model_b": c.model_b,
                "benchmark": c.benchmark,
                "metric": c.metric,
                "value_a": round(c.value_a, 4),
                "value_b": round(c.value_b, 4),
                "delta": round(c.delta, 4),
                "p_value": round(c.p_value, 4),
                "significant": c.significant,
            }
            for c in comparisons
        ]

        # Save report
        with open(self.output_dir / "eval_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Log to W&B
        try:
            import wandb
            if wandb.run:
                wandb.log({"evaluation": report})
        except Exception:
            pass

        log.info(f"Evaluation report saved to {self.output_dir / 'eval_report.json'}")
        return report


def run_evaluation(config: dict) -> dict:
    """Run eval — called by the orchestrator or directly."""
    pipeline = EvaluationPipeline(config["evaluation"])
    return pipeline.run()
