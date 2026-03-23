"""Data cleaning: dedup + PII scrubbing + quality filters.

This is the most compute-heavy part of data curation — the MinHash
LSH dedup alone can take hours on multi-million doc corpora. Consider
running with --workers flag for multiprocessing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

log = logging.getLogger(__name__)


# MinHash LSH Deduplication

# TODO(tanay): benchmark this against datasketch library — ours might be slower
# for very large corpora (>10M docs) but avoids the dep
class MinHashLSH:
    """
    MinHash Locality-Sensitive Hashing for near-duplicate detection.
    
    Uses a banded LSH scheme: documents that share a hash in ANY band
    are candidate duplicates, then verified with full Jaccard similarity.
    """

    def __init__(
        self,
        num_perm: int = 128,
        bands: int = 32,
        rows: int = 4,
        similarity_threshold: float = 0.85,
        ngram_size: int = 5,
    ):
        assert bands * rows == num_perm, f"bands * rows must equal num_perm: {bands}*{rows} != {num_perm}"
        self.num_perm = num_perm
        self.bands = bands
        self.rows = rows
        self.threshold = similarity_threshold
        self.ngram_size = ngram_size

        # Generate hash coefficients: h(x) = (a*x + b) % p
        self._max_hash = (1 << 32) - 1
        self._mersenne_prime = (1 << 61) - 1
        rng = np.random.RandomState(42)
        self._a = rng.randint(1, self._mersenne_prime, size=num_perm, dtype=np.int64)
        self._b = rng.randint(0, self._mersenne_prime, size=num_perm, dtype=np.int64)

        # LSH index: band_id -> {band_hash: set of doc_ids}
        self._buckets: list[dict[int, set]] = [dict() for _ in range(bands)]
        self._signatures: dict[str, np.ndarray] = {}

    def _shingle(self, text: str) -> set[int]:
        """Convert text to a set of character n-gram hashes."""
        text = text.lower().strip()
        shingles = set()
        for i in range(len(text) - self.ngram_size + 1):
            shingle = text[i : i + self.ngram_size]
            h = struct.unpack("<I", hashlib.md5(shingle.encode()).digest()[:4])[0]
            shingles.add(h)
        return shingles

    def _compute_signature(self, shingles: set[int]) -> np.ndarray:
        """Compute the MinHash signature for a set of shingles."""
        sig = np.full(self.num_perm, self._mersenne_prime, dtype=np.int64)
        for shingle in shingles:
            hashes = (self._a * shingle + self._b) % self._mersenne_prime
            sig = np.minimum(sig, hashes)
        return sig

    def _jaccard_from_signatures(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        return float(np.mean(sig1 == sig2))

    def insert(self, doc_id: str, text: str) -> bool:
        """
        Insert a document. Returns True if the document is unique
        (not a near-duplicate of any existing document).
        """
        shingles = self._shingle(text)
        if not shingles:
            return False

        sig = self._compute_signature(shingles)

        # Check for duplicates in LSH buckets
        candidates = set()
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = start + self.rows
            band_hash = hash(sig[start:end].tobytes())

            bucket = self._buckets[band_idx]
            if band_hash in bucket:
                candidates.update(bucket[band_hash])

        # Verify candidates with full signature comparison
        for candidate_id in candidates:
            if candidate_id in self._signatures:
                sim = self._jaccard_from_signatures(sig, self._signatures[candidate_id])
                if sim >= self.threshold:
                    log.debug(f"Duplicate detected: {doc_id} ≈ {candidate_id} (sim={sim:.3f})")
                    return False  # Duplicate

        # Not a duplicate - insert into index
        self._signatures[doc_id] = sig
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = start + self.rows
            band_hash = hash(sig[start:end].tobytes())
            self._buckets[band_idx].setdefault(band_hash, set()).add(doc_id)

        return True

    @property
    def size(self) -> int:
        return len(self._signatures)


# PII Detection and Removal

class PIIRemover:
    """Regex-based PII detection and redaction."""

    PATTERNS = {
        "email": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        ),
        "phone": re.compile(
            r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}"
        ),
        "ssn": re.compile(
            r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
        ),
        "ip_address": re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        ),
        "credit_card": re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
        ),
    }

    def __init__(self, patterns: list[str], replacement: str = "[REDACTED]"):
        self.active_patterns = {
            name: pattern
            for name, pattern in self.PATTERNS.items()
            if name in patterns
        }
        self.replacement = replacement

    def remove(self, text: str) -> tuple[str, dict[str, int]]:
        """Remove PII from text. Returns cleaned text and count of removals."""
        counts = {}
        for name, pattern in self.active_patterns.items():
            matches = pattern.findall(text)
            if matches:
                counts[name] = len(matches)
                text = pattern.sub(f"{self.replacement}_{name.upper()}", text)
        return text, counts


# Text Quality Filters

@dataclass
class QualityMetrics:
    """Quality metrics computed for a single document."""
    char_count: int = 0
    word_count: int = 0
    line_count: int = 0
    avg_word_length: float = 0.0
    repetition_ratio: float = 0.0
    special_char_ratio: float = 0.0
    uppercase_ratio: float = 0.0
    url_density: float = 0.0
    has_code: bool = False
    language: str = "unknown"
    language_confidence: float = 0.0
    passed: bool = True
    rejection_reasons: list[str] = field(default_factory=list)


class TextQualityFilter:
    """
    Multi-criteria text quality filter.
    
    Checks: length, word count, repetition, language, boilerplate, etc.
    """

    URL_PATTERN = re.compile(r"https?://\S+")
    CODE_PATTERN = re.compile(r"(?:```[\s\S]*?```|def\s+\w+|class\s+\w+|import\s+\w+|function\s+\w+)")
    BOILERPLATE_PATTERNS = [
        re.compile(r"cookie\s*(?:policy|consent|notice)", re.I),
        re.compile(r"privacy\s*policy", re.I),
        re.compile(r"terms\s*(?:of|and)\s*(?:service|use|conditions)", re.I),
        re.compile(r"subscribe\s*(?:to|for)\s*(?:our|the)\s*newsletter", re.I),
        re.compile(r"all\s*rights\s*reserved", re.I),
        re.compile(r"©\s*\d{4}", re.I),
    ]

    def __init__(self, config: dict):
        self.min_chars = config.get("min_length_chars", 100)
        self.max_chars = config.get("max_length_chars", 32768)
        self.min_words = config.get("min_word_count", 20)
        self.max_repetition = config.get("max_repetition_ratio", 0.3)
        self.remove_boilerplate = config.get("remove_boilerplate", True)
        self.lang_filter = config.get("language_filter", "en")
        self.lang_threshold = config.get("language_confidence_threshold", 0.8)

    def _compute_repetition_ratio(self, text: str, n: int = 10) -> float:
        """Compute the fraction of text covered by repeated n-grams."""
        words = text.lower().split()
        if len(words) < n:
            return 0.0

        ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        return repeated / max(len(ngrams), 1)

    def _detect_language(self, text: str) -> tuple[str, float]:
        """Detect language using a simple heuristic (or lingua/langdetect)."""
        try:
            from langdetect import detect_langs
            results = detect_langs(text[:2000])  # Use first 2000 chars
            if results:
                return results[0].lang, results[0].prob
        except Exception:
            pass
        return "unknown", 0.0

    def _remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate text patterns."""
        lines = text.split("\n")
        filtered = []
        for line in lines:
            if any(p.search(line) for p in self.BOILERPLATE_PATTERNS):
                continue
            filtered.append(line)
        return "\n".join(filtered)

    def analyze(self, text: str) -> QualityMetrics:
        """Compute quality metrics and determine if document passes filters."""
        metrics = QualityMetrics()

        # Pre-process
        if self.remove_boilerplate:
            text = self._remove_boilerplate(text)

        metrics.char_count = len(text)
        words = text.split()
        metrics.word_count = len(words)
        metrics.line_count = text.count("\n") + 1

        if words:
            metrics.avg_word_length = sum(len(w) for w in words) / len(words)

        # Check length
        if metrics.char_count < self.min_chars:
            metrics.passed = False
            metrics.rejection_reasons.append(f"too_short ({metrics.char_count} < {self.min_chars})")

        if metrics.char_count > self.max_chars:
            metrics.passed = False
            metrics.rejection_reasons.append(f"too_long ({metrics.char_count} > {self.max_chars})")

        if metrics.word_count < self.min_words:
            metrics.passed = False
            metrics.rejection_reasons.append(f"too_few_words ({metrics.word_count} < {self.min_words})")

        # Repetition
        metrics.repetition_ratio = self._compute_repetition_ratio(text)
        if metrics.repetition_ratio > self.max_repetition:
            metrics.passed = False
            metrics.rejection_reasons.append(
                f"too_repetitive ({metrics.repetition_ratio:.2f} > {self.max_repetition})"
            )

        # Special characters
        if text:
            alpha_count = sum(c.isalpha() for c in text)
            metrics.special_char_ratio = 1.0 - (alpha_count / len(text))
            if metrics.special_char_ratio > 0.5:
                metrics.passed = False
                metrics.rejection_reasons.append(f"too_many_special_chars ({metrics.special_char_ratio:.2f})")

            metrics.uppercase_ratio = sum(c.isupper() for c in text) / max(alpha_count, 1)
            if metrics.uppercase_ratio > 0.8:
                metrics.passed = False
                metrics.rejection_reasons.append(f"too_much_uppercase ({metrics.uppercase_ratio:.2f})")

        # URLs
        urls = self.URL_PATTERN.findall(text)
        metrics.url_density = len(urls) / max(metrics.word_count, 1)
        if metrics.url_density > 0.3:
            metrics.passed = False
            metrics.rejection_reasons.append(f"too_many_urls ({metrics.url_density:.2f})")

        # Code detection
        metrics.has_code = bool(self.CODE_PATTERN.search(text))

        # Language
        metrics.language, metrics.language_confidence = self._detect_language(text)
        if self.lang_filter and metrics.language != self.lang_filter:
            if metrics.language_confidence >= self.lang_threshold:
                metrics.passed = False
                metrics.rejection_reasons.append(
                    f"wrong_language ({metrics.language}, confidence={metrics.language_confidence:.2f})"
                )

        return metrics


# Cleaning Pipeline

class CleaningPipeline:
    """
    End-to-end cleaning pipeline that processes raw scraped documents.
    
    Pipeline stages:
    1. Deduplication (MinHash LSH)
    2. Text quality filtering
    3. PII removal
    4. Final formatting
    """

    def __init__(self, config: dict):
        self.config = config
        dedup_cfg = config.get("deduplication", {})
        self.dedup = MinHashLSH(
            num_perm=dedup_cfg.get("num_perm", 128),
            bands=dedup_cfg.get("bands", 32),
            rows=dedup_cfg.get("rows", 4),
            similarity_threshold=dedup_cfg.get("similarity_threshold", 0.85),
        )
        self.quality_filter = TextQualityFilter(config.get("text_processing", {}))

        pii_cfg = config.get("pii_removal", {})
        self.pii_remover = PIIRemover(
            patterns=pii_cfg.get("patterns", []),
            replacement=pii_cfg.get("replacement", "[REDACTED]"),
        ) if pii_cfg.get("enabled", True) else None

        self.output_dir = Path(config.get("output_dir", "data/cleaned"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Stats
        self.stats = {
            "total_input": 0,
            "duplicates_removed": 0,
            "quality_filtered": 0,
            "pii_redactions": {},
            "total_output": 0,
            "rejection_reasons": Counter(),
        }

    def _load_documents(self, input_dir: str | Path) -> Generator[dict, None, None]:
        """Stream documents from JSONL files in the input directory."""
        input_path = Path(input_dir)
        for jsonl_file in sorted(input_path.rglob("*.jsonl")):
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    def process_document(self, doc: dict) -> Optional[dict]:
        """Process a single document through the cleaning pipeline."""
        self.stats["total_input"] += 1
        content = doc.get("content", "")
        doc_id = doc.get("content_hash") or doc.get("source_id", "")

        # Stage 1: Deduplication
        is_unique = self.dedup.insert(doc_id, content)
        if not is_unique:
            self.stats["duplicates_removed"] += 1
            return None

        # Stage 2: Quality filtering
        metrics = self.quality_filter.analyze(content)
        if not metrics.passed:
            self.stats["quality_filtered"] += 1
            for reason in metrics.rejection_reasons:
                self.stats["rejection_reasons"][reason.split("(")[0].strip()] += 1
            return None

        # Stage 3: PII removal
        if self.pii_remover:
            content, pii_counts = self.pii_remover.remove(content)
            for pii_type, count in pii_counts.items():
                self.stats["pii_redactions"][pii_type] = (
                    self.stats["pii_redactions"].get(pii_type, 0) + count
                )

        # Stage 4: Update document
        doc["content"] = content
        doc["quality_metrics"] = {
            "char_count": metrics.char_count,
            "word_count": metrics.word_count,
            "repetition_ratio": round(metrics.repetition_ratio, 4),
            "language": metrics.language,
            "language_confidence": round(metrics.language_confidence, 4),
            "has_code": metrics.has_code,
        }

        self.stats["total_output"] += 1
        return doc

    def run(self, input_dir: str | Path) -> dict:
        """Run the full cleaning pipeline."""
        log.info(f"Starting cleaning pipeline, input: {input_dir}")

        output_file = self.output_dir / "cleaned_data.jsonl"
        batch = []
        batch_size = 10000

        with open(output_file, "w") as f:
            for doc in self._load_documents(input_dir):
                result = self.process_document(doc)
                if result:
                    f.write(json.dumps(result) + "\n")

                if self.stats["total_input"] % 10000 == 0:
                    log.info(
                        f"Processed {self.stats['total_input']} docs | "
                        f"Kept: {self.stats['total_output']} | "
                        f"Dupes: {self.stats['duplicates_removed']} | "
                        f"Filtered: {self.stats['quality_filtered']}"
                    )

        # Save stats
        stats_file = self.output_dir / "cleaning_stats.json"
        self.stats["rejection_reasons"] = dict(self.stats["rejection_reasons"])
        with open(stats_file, "w") as f:
            json.dump(self.stats, f, indent=2)

        log.info(
            f"Cleaning complete: {self.stats['total_input']} → {self.stats['total_output']} documents "
            f"({self.stats['duplicates_removed']} dupes, {self.stats['quality_filtered']} filtered)"
        )

        return self.stats


def run_cleaning(config: dict) -> dict:
    """Entry point for the cleaning stage."""
    pipeline = CleaningPipeline(config["data_curation"]["cleaning"])
    return pipeline.run(config["data_curation"]["scraping"]["output_dir"])
