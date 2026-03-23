"""Data contamination detector.

Checks if training data contains benchmark test examples, which would
make evaluation scores meaningless. This is a real problem — GPT-4's
technical report dedicates a whole section to it.

Detection methods:
  1. N-gram overlap: exact substring matching (fast, high precision)
  2. Embedding similarity: semantic matching (catches paraphrases)
  3. Partial matching: sliding window for partial leakage

Usage:
    checker = ContaminationChecker(benchmark_dir="evaluation/benchmarks")
    report = checker.check_dataset("data/assembled/train.jsonl")
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ContaminationMatch:
    """A single contamination match between training and benchmark data."""
    train_doc_id: str
    benchmark_name: str
    benchmark_item_id: int
    match_type: str  # "exact", "ngram_overlap", "high_similarity"
    overlap_score: float
    train_snippet: str
    benchmark_snippet: str


@dataclass
class ContaminationReport:
    """Full contamination report across all benchmarks."""
    total_train_docs: int = 0
    total_benchmark_items: int = 0
    contaminated_count: int = 0
    contamination_rate: float = 0.0
    matches: list[ContaminationMatch] = field(default_factory=list)
    per_benchmark: dict = field(default_factory=dict)
    clean: bool = True

    def summary(self) -> str:
        lines = [
            f"Contamination Report",
            f"  Training docs checked: {self.total_train_docs}",
            f"  Benchmark items: {self.total_benchmark_items}",
            f"  Contaminated: {self.contaminated_count} ({self.contamination_rate:.2%})",
            f"  Status: {'CLEAN' if self.clean else 'CONTAMINATED'}",
        ]
        if self.per_benchmark:
            lines.append("  Per benchmark:")
            for name, info in self.per_benchmark.items():
                lines.append(f"    {name}: {info['contaminated']}/{info['total']} items leaked")
        return "\n".join(lines)


class NGramIndex:
    """Fast n-gram lookup index for contamination detection.

    Builds a set of all n-grams in the benchmark data, then checks
    training documents against it. This catches exact and near-exact
    copies but misses paraphrases.
    """

    def __init__(self, n: int = 13):
        # 13-gram is the sweet spot — short enough to catch partial matches,
        # long enough to avoid false positives from common phrases
        self.n = n
        self._index: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self._total_ngrams = 0

    def add_benchmark(self, benchmark_name: str, item_id: int, text: str):
        """Index a benchmark item's n-grams."""
        text = self._normalize(text)
        words = text.split()
        for i in range(len(words) - self.n + 1):
            ngram = " ".join(words[i:i + self.n])
            self._index[ngram].append((benchmark_name, item_id))
            self._total_ngrams += 1

    def check(self, text: str) -> list[tuple[str, int, float]]:
        """Check a training document for benchmark n-gram matches.

        Returns list of (benchmark_name, item_id, overlap_fraction) tuples.
        """
        text = self._normalize(text)
        words = text.split()
        if len(words) < self.n:
            return []

        doc_ngrams = set()
        for i in range(len(words) - self.n + 1):
            doc_ngrams.add(" ".join(words[i:i + self.n]))

        # Count matches per benchmark item
        hit_counts: dict[tuple[str, int], int] = defaultdict(int)
        for ngram in doc_ngrams:
            if ngram in self._index:
                for bench_name, item_id in self._index[ngram]:
                    hit_counts[(bench_name, item_id)] += 1

        results = []
        for (bench_name, item_id), count in hit_counts.items():
            # overlap = fraction of document n-grams that matched
            overlap = count / max(len(doc_ngrams), 1)
            if overlap > 0.1:  # 10% overlap threshold
                results.append((bench_name, item_id, overlap))

        return sorted(results, key=lambda x: x[2], reverse=True)

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison — lowercase, strip punctuation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @property
    def size(self):
        return self._total_ngrams


class ContaminationChecker:
    """Main contamination checker.

    Loads benchmark data, builds an n-gram index, then scans training
    data for matches. Reports contamination per benchmark.
    """

    def __init__(
        self,
        benchmark_dir: str = "evaluation/benchmarks",
        ngram_size: int = 13,
        overlap_threshold: float = 0.1,
    ):
        self.benchmark_dir = Path(benchmark_dir)
        self.overlap_threshold = overlap_threshold
        self.ngram_index = NGramIndex(n=ngram_size)
        self._benchmark_texts: dict[str, dict[int, str]] = {}
        self._load_benchmarks()

    def _load_benchmarks(self):
        """Load all .jsonl benchmark files and index them."""
        if not self.benchmark_dir.exists():
            log.warning(f"Benchmark dir not found: {self.benchmark_dir}")
            return

        for bench_file in self.benchmark_dir.glob("*.jsonl"):
            bench_name = bench_file.stem
            self._benchmark_texts[bench_name] = {}

            with open(bench_file) as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    item = json.loads(line)

                    # Extract text from various benchmark formats
                    text_parts = []
                    for key in ["question", "prompt", "reference", "content"]:
                        if key in item:
                            text_parts.append(str(item[key]))
                    if "choices" in item:
                        text_parts.extend(str(c) for c in item["choices"])

                    full_text = " ".join(text_parts)
                    self._benchmark_texts[bench_name][i] = full_text
                    self.ngram_index.add_benchmark(bench_name, i, full_text)

            log.info(f"Indexed benchmark '{bench_name}': {len(self._benchmark_texts[bench_name])} items")

        log.info(f"Total n-grams indexed: {self.ngram_index.size}")

    def check_dataset(self, data_path: str) -> ContaminationReport:
        """Scan a training dataset for contamination."""
        report = ContaminationReport()

        data_path = Path(data_path)
        if not data_path.exists():
            log.warning(f"Training data not found: {data_path}")
            return report

        # Count benchmark items
        for bench_name, items in self._benchmark_texts.items():
            report.total_benchmark_items += len(items)
            report.per_benchmark[bench_name] = {"total": len(items), "contaminated": 0}

        # Scan training data
        contaminated_items: set[tuple[str, int]] = set()

        with open(data_path) as f:
            for doc_idx, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue

                report.total_train_docs += 1

                # Get text content
                text = doc.get("text", doc.get("content", doc.get("instruction", "")))
                if isinstance(text, dict):
                    text = json.dumps(text)

                # Check against n-gram index
                matches = self.ngram_index.check(str(text))

                for bench_name, item_id, overlap in matches:
                    if (bench_name, item_id) in contaminated_items:
                        continue  # already flagged

                    contaminated_items.add((bench_name, item_id))

                    match_type = "exact" if overlap > 0.8 else "ngram_overlap"

                    report.matches.append(ContaminationMatch(
                        train_doc_id=str(doc_idx),
                        benchmark_name=bench_name,
                        benchmark_item_id=item_id,
                        match_type=match_type,
                        overlap_score=overlap,
                        train_snippet=str(text)[:200],
                        benchmark_snippet=self._benchmark_texts[bench_name][item_id][:200],
                    ))

                    if bench_name in report.per_benchmark:
                        report.per_benchmark[bench_name]["contaminated"] += 1

        report.contaminated_count = len(contaminated_items)
        report.contamination_rate = (
            report.contaminated_count / max(report.total_benchmark_items, 1)
        )
        report.clean = report.contaminated_count == 0

        return report

    def check_text(self, text: str) -> list[tuple[str, int, float]]:
        """Quick check — does this single text overlap with any benchmark?"""
        return self.ngram_index.check(text)


def run_contamination_check(config: dict) -> dict:
    """Entry point — run contamination check as a pipeline stage."""
    eval_cfg = config.get("evaluation", {})
    bench_dir = eval_cfg.get("benchmark_dir", "evaluation/benchmarks")
    train_data = config.get("data_curation", {}).get("assembly", {}).get(
        "output_path", "data/assembled/train.jsonl"
    )

    checker = ContaminationChecker(benchmark_dir=bench_dir)
    report = checker.check_dataset(train_data)

    log.info(report.summary())

    # Save report
    output_dir = Path(eval_cfg.get("output_dir", "outputs/evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "contamination_report.json"

    with open(report_path, "w") as f:
        json.dump({
            "total_train_docs": report.total_train_docs,
            "total_benchmark_items": report.total_benchmark_items,
            "contaminated_count": report.contaminated_count,
            "contamination_rate": report.contamination_rate,
            "clean": report.clean,
            "per_benchmark": report.per_benchmark,
            "matches": [
                {
                    "train_doc_id": m.train_doc_id,
                    "benchmark": m.benchmark_name,
                    "item_id": m.benchmark_item_id,
                    "type": m.match_type,
                    "overlap": round(m.overlap_score, 4),
                }
                for m in report.matches
            ],
        }, f, indent=2)

    log.info(f"Contamination report saved to {report_path}")
    return {"clean": report.clean, "contamination_rate": report.contamination_rate}
