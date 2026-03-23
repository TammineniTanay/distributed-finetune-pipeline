"""LLM-as-judge scoring pipeline.

Uses Claude/GPT-4/local models to score data quality on multiple
dimensions. Caches results in SQLite so you dont pay for the same
doc twice if the pipeline crashes and restarts.

TODO: add support for batch API to cut costs ~50%
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# Data models

@dataclass
class ScoringDimension:
    name: str
    weight: float
    description: str


@dataclass
class JudgeScore:
    doc_id: str
    scores: dict[str, float]          # dimension_name → score
    composite_score: float
    rationale: str
    model_used: str
    latency_ms: float
    token_count: int = 0

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "scores": self.scores,
            "composite_score": round(self.composite_score, 4),
            "rationale": self.rationale,
            "model_used": self.model_used,
            "latency_ms": round(self.latency_ms, 2),
            "token_count": self.token_count,
        }


# Score cache (SQLite-backed)

class ScoreCache:
    """SQLite-backed cache for LLM judge scores to avoid redundant API calls."""

    def __init__(self, cache_dir: str | Path):
        self.db_path = Path(cache_dir) / "judge_cache.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scores (
                    content_hash TEXT PRIMARY KEY,
                    score_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON scores(created_at)")

    def get(self, content_hash: str) -> Optional[JudgeScore]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT score_json FROM scores WHERE content_hash = ?",
                (content_hash,),
            ).fetchone()
            if row:
                data = json.loads(row[0])
                return JudgeScore(**data)
        return None

    def put(self, content_hash: str, score: JudgeScore):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO scores (content_hash, score_json, created_at) VALUES (?, ?, ?)",
                (content_hash, json.dumps(score.to_dict()), time.time()),
            )

    def size(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]


# LLM Provider Abstraction

class LLMProvider(ABC):
    """Abstract base for LLM API providers."""

    @abstractmethod
    async def score(self, prompt: str) -> tuple[str, int]:
        """Send prompt to LLM and return (response_text, token_count)."""
        ...


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        import anthropic
        self.client = anthropic.AsyncAnthropic()
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def score(self, prompt: str) -> tuple[str, int]:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return text, tokens


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, model: str = "gpt-4o"):
        import openai
        self.client = openai.AsyncOpenAI()
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def score(self, prompt: str) -> tuple[str, int]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
        )
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens
        return text, tokens


class LocalProvider(LLMProvider):
    """Local vLLM/TGI server provider."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "local"):
        import openai
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key="dummy")
        self.model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def score(self, prompt: str) -> tuple[str, int]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
        )
        text = response.choices[0].message.content
        tokens = getattr(response.usage, "total_tokens", 0)
        return text, tokens


PROVIDERS = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "local": LocalProvider,
}


# Scoring prompt builder

class ScoringPromptBuilder:
    """Builds structured scoring prompts for the LLM judge."""

    SYSTEM_TEMPLATE = """You are an expert data quality judge for training LLM datasets.
Your task is to evaluate the quality of training examples across multiple dimensions.

Score each dimension from {min_score} to {max_score} (integers only).
Be strict and calibrated — a score of {max_score} should be exceptional.

Scoring dimensions:
{dimensions_desc}

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{
    "scores": {{
        {score_keys}
    }},
    "rationale": "Brief explanation of your scoring (2-3 sentences max)"
}}

Do not include any text outside the JSON object."""

    def __init__(self, dimensions: list[ScoringDimension], score_range: tuple[int, int] = (1, 5)):
        self.dimensions = dimensions
        self.min_score, self.max_score = score_range

    def build_prompt(self, content: str, metadata: dict = None) -> str:
        dimensions_desc = "\n".join(
            f"- {d.name} (weight: {d.weight:.0%}): {d.description}"
            for d in self.dimensions
        )
        score_keys = ",\n        ".join(
            f'"{d.name}": <integer {self.min_score}-{self.max_score}>'
            for d in self.dimensions
        )
        system = self.SYSTEM_TEMPLATE.format(
            min_score=self.min_score,
            max_score=self.max_score,
            dimensions_desc=dimensions_desc,
            score_keys=score_keys,
        )

        user_msg = f"Evaluate this training example:\n\n---\n{content[:4000]}\n---"
        if metadata:
            user_msg += f"\n\nMetadata: {json.dumps(metadata, indent=2)}"

        return f"{system}\n\n{user_msg}"


# LLM Judge

class LLMJudge:
    """
    Multi-dimensional LLM-based quality scorer.
    
    Scores documents asynchronously in batches with caching and calibration.
    """

    def __init__(self, config: dict):
        self.config = config
        self.batch_size = config.get("batch_size", 50)
        self.max_concurrent = config.get("max_concurrent", 10)
        self.min_composite_score = config.get("min_composite_score", 3.5)

        # Dimensions
        self.dimensions = [
            ScoringDimension(**d) for d in config.get("scoring_dimensions", [])
        ]
        self.score_range = tuple(config.get("score_range", [1, 5]))

        # Provider
        provider_name = config.get("provider", "anthropic")
        model = config.get("model", "claude-3-5-sonnet-20241022")
        self.provider = PROVIDERS[provider_name](model=model)

        # Prompt builder
        self.prompt_builder = ScoringPromptBuilder(self.dimensions, self.score_range)

        # Cache
        cache_dir = config.get("cache_dir", "data/judge_cache")
        self.cache = ScoreCache(cache_dir)

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        # Stats
        self.stats = {
            "total_scored": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0.0,
            "score_distribution": {},
            "passed": 0,
            "failed": 0,
            "errors": 0,
        }

    def _parse_response(self, response_text: str) -> dict:
        """Parse the LLM's JSON response, handling potential formatting issues."""
        # Try direct parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        import re
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try extracting JSON object
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse LLM response as JSON: {response_text[:200]}")

    def _compute_composite(self, scores: dict[str, float]) -> float:
        """Compute weighted composite score."""
        total_weight = sum(d.weight for d in self.dimensions)
        weighted_sum = sum(
            scores.get(d.name, 0) * d.weight
            for d in self.dimensions
        )
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    async def _score_single(self, doc_id: str, content: str, metadata: dict = None) -> JudgeScore:
        """Score a single document with caching."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check cache
        cached = self.cache.get(content_hash)
        if cached:
            self.stats["cache_hits"] += 1
            return cached

        # Build prompt and call LLM
        prompt = self.prompt_builder.build_prompt(content, metadata)

        async with self._semaphore:
            start = time.monotonic()
            try:
                response_text, token_count = await self.provider.score(prompt)
                latency_ms = (time.monotonic() - start) * 1000

                parsed = self._parse_response(response_text)
                scores = parsed.get("scores", {})
                rationale = parsed.get("rationale", "")

                # Validate and clamp scores
                for dim in self.dimensions:
                    val = scores.get(dim.name, self.score_range[0])
                    scores[dim.name] = max(self.score_range[0], min(self.score_range[1], int(val)))

                composite = self._compute_composite(scores)

                score = JudgeScore(
                    doc_id=doc_id,
                    scores=scores,
                    composite_score=composite,
                    rationale=rationale,
                    model_used=self.config.get("model", "unknown"),
                    latency_ms=latency_ms,
                    token_count=token_count,
                )

                # Update cache and stats
                self.cache.put(content_hash, score)
                self.stats["api_calls"] += 1
                self.stats["total_tokens"] += token_count

                return score

            except Exception as e:
                logger.error(f"Scoring error for {doc_id}: {e}")
                self.stats["errors"] += 1
                # Return minimum score on error
                scores = {d.name: self.score_range[0] for d in self.dimensions}
                return JudgeScore(
                    doc_id=doc_id,
                    scores=scores,
                    composite_score=float(self.score_range[0]),
                    rationale=f"Error: {str(e)}",
                    model_used=self.config.get("model", "unknown"),
                    latency_ms=0.0,
                )

    async def score_batch(self, documents: list[dict]) -> list[JudgeScore]:
        """Score a batch of documents concurrently."""
        tasks = []
        for doc in documents:
            task = self._score_single(
                doc_id=doc.get("content_hash", doc.get("source_id", "")),
                content=doc.get("content", ""),
                metadata=doc.get("metadata"),
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        scores = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Batch scoring exception: {r}")
                self.stats["errors"] += 1
            else:
                scores.append(r)
                self.stats["total_scored"] += 1
                if r.composite_score >= self.min_composite_score:
                    self.stats["passed"] += 1
                else:
                    self.stats["failed"] += 1

        return scores

    async def run(self, input_path: str | Path, output_dir: str | Path) -> dict:
        """Score all documents and save results."""
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load documents
        documents = []
        for jsonl_file in sorted(input_path.rglob("*.jsonl")):
            with open(jsonl_file) as f:
                for line in f:
                    if line.strip():
                        documents.append(json.loads(line))

        logger.info(f"Scoring {len(documents)} documents with LLM judge")

        # Process in batches
        all_scores = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            batch_scores = await self.score_batch(batch)
            all_scores.extend(batch_scores)

            logger.info(
                f"Batch {i // self.batch_size + 1}: "
                f"scored={len(batch_scores)}, "
                f"passed={sum(1 for s in batch_scores if s.composite_score >= self.min_composite_score)}"
            )

        # Save scored documents (merge scores back into docs)
        scored_output = output_dir / "scored_data.jsonl"
        passed_output = output_dir / "passed_data.jsonl"

        score_map = {s.doc_id: s for s in all_scores}

        with open(scored_output, "w") as f_all, open(passed_output, "w") as f_passed:
            for doc in documents:
                doc_id = doc.get("content_hash", doc.get("source_id", ""))
                score = score_map.get(doc_id)
                if score:
                    doc["judge_scores"] = score.to_dict()
                    doc["composite_score"] = score.composite_score
                    f_all.write(json.dumps(doc) + "\n")

                    if score.composite_score >= self.min_composite_score:
                        f_passed.write(json.dumps(doc) + "\n")

        # Save stats
        if self.stats["api_calls"] > 0:
            self.stats["avg_latency_ms"] = (
                sum(s.latency_ms for s in all_scores) / len(all_scores)
            )

        stats_file = output_dir / "scoring_stats.json"
        with open(stats_file, "w") as f:
            json.dump(self.stats, f, indent=2)

        logger.info(
            f"Scoring complete: {self.stats['total_scored']} scored, "
            f"{self.stats['passed']} passed (>= {self.min_composite_score}), "
            f"{self.stats['failed']} filtered, "
            f"{self.stats['cache_hits']} cache hits"
        )

        return self.stats


# Calibration against human annotations

class JudgeCalibrator:
    """
    Calibrate the LLM judge against human annotations.
    Computes Cohen's kappa for inter-annotator agreement.
    """

    def __init__(self, human_annotations_path: str | Path):
        self.annotations = self._load_annotations(human_annotations_path)

    def _load_annotations(self, path: str | Path) -> dict:
        annotations = {}
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                annotations[data["doc_id"]] = data
        return annotations

    def compute_agreement(self, judge_scores: list[JudgeScore]) -> dict:
        """Compute Cohen's kappa between LLM judge and human annotations."""
        human_labels = []
        judge_labels = []

        for score in judge_scores:
            if score.doc_id in self.annotations:
                human = self.annotations[score.doc_id]
                # Binarize: pass/fail
                human_pass = human.get("composite_score", 0) >= 3.5
                judge_pass = score.composite_score >= 3.5
                human_labels.append(int(human_pass))
                judge_labels.append(int(judge_pass))

        if not human_labels:
            return {"kappa": 0.0, "agreement": 0.0, "n_samples": 0}

        # Cohen's kappa calculation
        n = len(human_labels)
        h = np.array(human_labels)
        j = np.array(judge_labels)

        # Observed agreement
        po = np.mean(h == j)

        # Expected agreement by chance
        p_h1 = np.mean(h)
        p_j1 = np.mean(j)
        pe = p_h1 * p_j1 + (1 - p_h1) * (1 - p_j1)

        kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0

        # Per-dimension agreement
        dimension_agreement = {}
        for dim_name in judge_scores[0].scores.keys() if judge_scores else []:
            dim_human = []
            dim_judge = []
            for score in judge_scores:
                if score.doc_id in self.annotations:
                    human = self.annotations[score.doc_id]
                    h_score = human.get("scores", {}).get(dim_name, 0)
                    j_score = score.scores.get(dim_name, 0)
                    dim_human.append(h_score)
                    dim_judge.append(j_score)

            if dim_human:
                # Pearson correlation for continuous scores
                correlation = np.corrcoef(dim_human, dim_judge)[0, 1]
                mae = np.mean(np.abs(np.array(dim_human) - np.array(dim_judge)))
                dimension_agreement[dim_name] = {
                    "pearson_r": round(float(correlation), 4),
                    "mae": round(float(mae), 4),
                }

        return {
            "cohens_kappa": round(float(kappa), 4),
            "observed_agreement": round(float(po), 4),
            "n_samples": n,
            "dimension_agreement": dimension_agreement,
        }


async def run_scoring(config: dict) -> dict:
    """Entry point for the LLM judge scoring stage."""
    judge = LLMJudge(config["data_curation"]["llm_judge"])
    return await judge.run(
        input_path=config["data_curation"]["cleaning"]["output_dir"],
        output_dir=config["data_curation"]["llm_judge"]["output_dir"],
    )
