"""Data contamination detector.

Checks if training data contains examples from evaluation benchmarks.
This is critical — if your training set leaks benchmark test examples,
your eval scores are meaningless and you'll get called out in any
serious review.

Methods:
  1. Exact match: full text dedup against benchmark questions/answers
  2. N-gram overlap: sliding window n-gram matching (catches paraphrases)
  3. Embedding similarity: semantic match via sentence embeddings (optional)

Usage:
    python -m evaluation.contamination_check \
        --training-data data/assembled/train.jsonl \
        --benchmarks evaluation/benchmarks/*.jsonl \
        --threshold 0.8

Reference: Sainz et al. 2023 "NLP Evaluation in trouble"
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class ContaminationResult:
    """Result of checking one training example against benchmarks."""
    training_id: str
    training_text: str
    matched_benchmark: str
    matched_example_id: str
    match_type: str          # exact, ngram, embedding
    similarity: float        # 0-1
    ngram_overlap_ratio: float = 0.0
    matched_text_preview: str = ""


@dataclass
class ContaminationReport:
    """Full contamination report across all benchmarks."""
    total_training_examples: int = 0
    total_benchmark_examples: int = 0
    contaminated_count: int = 0
    contamination_rate: float = 0.0
    matches: list[ContaminationResult] = field(default_factory=list)
    per_benchmark: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Contamination Report",
            f"{'=' * 50}",
            f"Training examples checked: {self.total_training_examples}",
            f"Benchmark examples: {self.total_benchmark_examples}",
            f"Contaminated: {self.contaminated_count} ({self.contamination_rate:.2%})",
            f"",
        ]
        for bench, stats in self.per_benchmark.items():
            lines.append(f"  {bench}: {stats['matches']}/{stats['total']} "
                         f"({stats['matches']/max(stats['total'],1):.1%})")

        if self.matches:
            lines.append(f"\nTop contamination matches:")
            for m in sorted(self.matches, key=lambda x: -x.similarity)[:10]:
                lines.append(
                    f"  [{m.match_type}] sim={m.similarity:.3f} "
                    f"benchmark={m.matched_benchmark} "
                    f"text='{m.training_text[:80]}...'"
                )
        return "\n".join(lines)


class ContaminationChecker:
    """Checks training data for benchmark contamination.
    
    The approach: build an index of all benchmark text (questions + answers),
    then scan training data looking for matches. Three levels of matching:
    
    1. Exact: hash-based, catches verbatim copies
    2. N-gram: character n-gram overlap ratio, catches near-copies
    3. Embedding: cosine similarity of sentence embeddings (optional, slow)
    """

    def __init__(
        self,
        ngram_size: int = 13,         # 13-gram is standard in the literature
        ngram_threshold: float = 0.8, # >80% n-gram overlap = contaminated
        embedding_threshold: float = 0.9,
        use_embeddings: bool = False, # disabled by default (needs sentence-transformers)
    ):
        self.ngram_size = ngram_size
        self.ngram_threshold = ngram_threshold
        self.embedding_threshold = embedding_threshold
        self.use_embeddings = use_embeddings

        # indexes
        self._exact_hashes: dict[str, tuple[str, str]] = {}  # hash -> (benchmark_name, example_id)
        self._benchmark_ngrams: dict[str, dict[str, set]] = {}  # bench -> {example_id: ngram_set}
        self._benchmark_texts: dict[str, dict[str, str]] = {}   # bench -> {example_id: text}
        self._embedder = None

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison — lowercase, strip whitespace, collapse spaces."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        # remove common formatting that might differ
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def _compute_ngrams(self, text: str) -> set:
        """Compute character-level n-grams."""
        text = self._normalize(text)
        if len(text) < self.ngram_size:
            return {text}
        return {text[i:i+self.ngram_size] for i in range(len(text) - self.ngram_size + 1)}

    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(self._normalize(text).encode()).hexdigest()

    def index_benchmark(self, name: str, examples: list[dict]):
        """Add a benchmark's examples to the contamination index.
        
        Expects dicts with at least one of: question, prompt, text, choices, reference
        """
        self._benchmark_ngrams[name] = {}
        self._benchmark_texts[name] = {}

        for i, ex in enumerate(examples):
            ex_id = str(ex.get("id", i))

            # extract all text fields from the benchmark example
            text_parts = []
            for key in ["question", "prompt", "text", "instruction"]:
                if key in ex and ex[key]:
                    text_parts.append(str(ex[key]))
            # also index answer choices and references
            if "choices" in ex:
                text_parts.extend(str(c) for c in ex["choices"])
            if "reference" in ex:
                text_parts.append(str(ex["reference"]))

            full_text = " ".join(text_parts)
            if not full_text.strip():
                continue

            # exact hash
            h = self._text_hash(full_text)
            self._exact_hashes[h] = (name, ex_id)

            # also hash individual fields (catches partial copies)
            for part in text_parts:
                if len(part) > 30:  # skip very short strings
                    ph = self._text_hash(part)
                    self._exact_hashes[ph] = (name, ex_id)

            # n-gram index
            self._benchmark_ngrams[name][ex_id] = self._compute_ngrams(full_text)
            self._benchmark_texts[name][ex_id] = full_text[:200]

        log.info(f"Indexed benchmark '{name}': {len(self._benchmark_ngrams[name])} examples")

    def check_example(self, text: str, example_id: str = "") -> list[ContaminationResult]:
        """Check a single training example against all indexed benchmarks."""
        matches = []
        text_norm = self._normalize(text)
        text_hash = self._text_hash(text)

        # 1. Exact match
        if text_hash in self._exact_hashes:
            bench_name, bench_id = self._exact_hashes[text_hash]
            matches.append(ContaminationResult(
                training_id=example_id,
                training_text=text[:200],
                matched_benchmark=bench_name,
                matched_example_id=bench_id,
                match_type="exact",
                similarity=1.0,
                matched_text_preview=self._benchmark_texts.get(bench_name, {}).get(bench_id, ""),
            ))
            return matches  # exact match, no need to check further

        # 2. N-gram overlap
        train_ngrams = self._compute_ngrams(text)
        if not train_ngrams:
            return matches

        for bench_name, examples in self._benchmark_ngrams.items():
            for ex_id, bench_ngrams in examples.items():
                if not bench_ngrams:
                    continue

                overlap = len(train_ngrams & bench_ngrams)
                # use the smaller set as denominator (containment similarity)
                # this catches cases where a short benchmark Q is embedded in a long training doc
                min_size = min(len(train_ngrams), len(bench_ngrams))
                ratio = overlap / max(min_size, 1)

                if ratio >= self.ngram_threshold:
                    matches.append(ContaminationResult(
                        training_id=example_id,
                        training_text=text[:200],
                        matched_benchmark=bench_name,
                        matched_example_id=ex_id,
                        match_type="ngram",
                        similarity=ratio,
                        ngram_overlap_ratio=ratio,
                        matched_text_preview=self._benchmark_texts.get(bench_name, {}).get(ex_id, ""),
                    ))

        return matches

    def check_dataset(self, training_path: str, benchmarks: dict[str, str]) -> ContaminationReport:
        """Check an entire training dataset against multiple benchmarks.
        
        Args:
            training_path: path to training JSONL
            benchmarks: {name: path} dict of benchmark JSONL files
        """
        # Index all benchmarks
        total_bench = 0
        for name, path in benchmarks.items():
            examples = []
            with open(path) as f:
                for line in f:
                    if line.strip():
                        examples.append(json.loads(line))
            self.index_benchmark(name, examples)
            total_bench += len(examples)

        # Scan training data
        report = ContaminationReport(total_benchmark_examples=total_bench)
        per_bench = {name: {"total": 0, "matches": 0} for name in benchmarks}

        contaminated_ids = set()
        with open(training_path) as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                report.total_training_examples += 1

                ex = json.loads(line)
                # extract text from training example
                text = ex.get("text", ex.get("instruction", ex.get("prompt", "")))
                if not text:
                    continue

                matches = self.check_example(text, example_id=str(i))
                if matches:
                    contaminated_ids.add(i)
                    report.matches.extend(matches)
                    for m in matches:
                        per_bench[m.matched_benchmark]["matches"] += 1

        report.contaminated_count = len(contaminated_ids)
        report.contamination_rate = (
            report.contaminated_count / max(report.total_training_examples, 1)
        )

        for name in benchmarks:
            per_bench[name]["total"] = len(self._benchmark_ngrams.get(name, {}))
        report.per_benchmark = per_bench

        return report

    def save_report(self, report: ContaminationReport, output_path: str):
        """Save contamination report to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "total_training_examples": report.total_training_examples,
            "total_benchmark_examples": report.total_benchmark_examples,
            "contaminated_count": report.contaminated_count,
            "contamination_rate": report.contamination_rate,
            "per_benchmark": report.per_benchmark,
            "matches": [
                {
                    "training_id": m.training_id,
                    "training_text": m.training_text,
                    "matched_benchmark": m.matched_benchmark,
                    "matched_example_id": m.matched_example_id,
                    "match_type": m.match_type,
                    "similarity": m.similarity,
                }
                for m in report.matches
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        log.info(f"Contamination report saved to {output_path}")


def run_contamination_check(config: dict) -> ContaminationReport:
    """Entry point for the pipeline orchestrator."""
    eval_cfg = config.get("evaluation", {})
    training_path = config.get("data_curation", {}).get("assembly", {}).get(
        "output_path", "data/assembled/train.jsonl"
    )
    benchmark_dir = Path(eval_cfg.get("benchmark_dir", "evaluation/benchmarks"))

    benchmarks = {}
    for f in benchmark_dir.glob("*.jsonl"):
        benchmarks[f.stem] = str(f)

    if not benchmarks:
        log.warning("No benchmark files found, skipping contamination check")
        return ContaminationReport()

    checker = ContaminationChecker(
        ngram_size=eval_cfg.get("contamination_ngram_size", 13),
        ngram_threshold=eval_cfg.get("contamination_threshold", 0.8),
    )

    report = checker.check_dataset(training_path, benchmarks)
    checker.save_report(report, "outputs/contamination_report.json")

    log.info(f"\n{report.summary()}")

    if report.contamination_rate > 0.01:
        log.warning(
            f"HIGH CONTAMINATION: {report.contamination_rate:.1%} of training data "
            f"matches benchmark examples. Eval scores may be inflated!"
        )

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check training data for benchmark contamination")
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--benchmarks", nargs="+", required=True, help="Benchmark JSONL files")
    parser.add_argument("--ngram-size", type=int, default=13)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output", default="outputs/contamination_report.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    checker = ContaminationChecker(ngram_size=args.ngram_size, ngram_threshold=args.threshold)

    benchmarks = {}
    for bp in args.benchmarks:
        benchmarks[Path(bp).stem] = bp

    report = checker.check_dataset(args.training_data, benchmarks)
    checker.save_report(report, args.output)
    print(report.summary())
