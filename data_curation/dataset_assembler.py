"""Dataset assembly — format conversion, chat templates, sequence packing.

Takes scored/filtered JSONL and produces Arrow datasets ready for
HF Trainer. The packing logic uses first-fit decreasing which wastes
~5-8% of tokens but is way simpler than optimal bin packing.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Chat Templates

CHAT_TEMPLATES = {
    "chatml": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
        "bos": "",
        "eos": "<|im_end|>",
    },
    "llama": {
        "system": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
        "bos": "<|begin_of_text|>",
        "eos": "<|eot_id|>",
    },
    "alpaca": {
        "system": "",
        "user": "### Instruction:\n{content}\n\n### Response:\n",
        "assistant": "{content}\n\n",
        "bos": "",
        "eos": "</s>",
    },
    "vicuna": {
        "system": "{content}\n\n",
        "user": "USER: {content}\n",
        "assistant": "ASSISTANT: {content}</s>\n",
        "bos": "",
        "eos": "</s>",
    },
}


class ChatTemplateFormatter:
    """Formats conversations using the specified chat template."""

    def __init__(self, template_name: str = "chatml"):
        if template_name not in CHAT_TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}. Options: {list(CHAT_TEMPLATES)}")
        self.template = CHAT_TEMPLATES[template_name]
        self.name = template_name

    def format_conversation(self, messages: list[dict]) -> str:
        """
        Format a list of messages into a single string.
        
        Each message should have 'role' (system/user/assistant) and 'content'.
        """
        parts = [self.template["bos"]] if self.template["bos"] else []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in self.template:
                parts.append(self.template[role].format(content=content))

        return "".join(parts)

    def format_instruction_pair(
        self,
        instruction: str,
        response: str,
        system_prompt: str = "",
    ) -> str:
        """Format a simple instruction-response pair."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": instruction})
        messages.append({"role": "assistant", "content": response})
        return self.format_conversation(messages)


# Sequence Packing

class SequencePacker:
    """
    Pack multiple short sequences into a single sequence up to max_length.
    
    Uses first-fit decreasing bin packing for optimal utilization.
    Separates packed sequences with an EOS token to prevent attention leakage.
    """

    def __init__(self, max_length: int, separator: str = "\n<|endoftext|>\n"):
        self.max_length = max_length
        self.separator = separator
        self.sep_len = len(separator.split())  # Approximate token count

    def pack(self, sequences: list[str]) -> list[str]:
        """Pack sequences into bins. Returns list of packed sequences."""
        # Estimate lengths (rough word-count approximation)
        items = [(seq, len(seq.split())) for seq in sequences]
        # Sort by length descending (first-fit decreasing)
        items.sort(key=lambda x: x[1], reverse=True)

        bins: list[tuple[list[str], int]] = []  # (sequences, current_length)

        for seq, length in items:
            if length > self.max_length:
                # Truncate to fit
                words = seq.split()[:self.max_length]
                bins.append(([" ".join(words)], self.max_length))
                continue

            placed = False
            for i, (bin_seqs, bin_len) in enumerate(bins):
                if bin_len + self.sep_len + length <= self.max_length:
                    bin_seqs.append(seq)
                    bins[i] = (bin_seqs, bin_len + self.sep_len + length)
                    placed = True
                    break

            if not placed:
                bins.append(([seq], length))

        packed = [self.separator.join(seqs) for seqs, _ in bins]

        utilization = sum(len(p.split()) for p in packed) / (len(packed) * self.max_length) if packed else 0
        logger.info(
            f"Packed {len(sequences)} sequences into {len(bins)} bins "
            f"(utilization: {utilization:.1%})"
        )

        return packed


# Document to conversation converter

class DocumentConverter:
    """Convert raw documents into instruction-response conversation format."""

    CONVERSION_TEMPLATES = [
        {
            "type": "qa",
            "system": "You are a knowledgeable AI assistant. Answer questions accurately and thoroughly.",
            "instruction_prefix": "Based on the following context, answer the question.\n\nContext: {context}\n\nQuestion: ",
            "questions": [
                "What are the key points discussed in this content?",
                "Explain the main concepts covered here.",
                "Summarize the most important information from this text.",
                "What practical implications does this content have?",
            ],
        },
        {
            "type": "explanation",
            "system": "You are an expert educator. Explain concepts clearly and in depth.",
            "instruction_prefix": "Explain the following topic in detail:\n\n",
        },
        {
            "type": "analysis",
            "system": "You are an analytical AI. Provide thorough analysis of the given content.",
            "instruction_prefix": "Analyze the following and provide key insights:\n\n",
        },
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def convert(self, doc: dict) -> list[dict]:
        """Convert a document into one or more conversation examples."""
        content = doc.get("content", "")
        source = doc.get("source", "")
        conversations = []

        # Source-specific conversion
        if source in ("stackoverflow",):
            # Already has Q&A structure
            parts = content.split("=" * 40)
            if len(parts) >= 2:
                question = parts[0].strip()
                answer = parts[1].replace("Answers:", "").strip()
                conversations.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful programming assistant."},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ]
                })
        elif source in ("huggingface",):
            # May already be in conversation format
            conversations.append({
                "messages": [
                    {"role": "user", "content": content.split("\n\n")[0] if "\n\n" in content else content[:500]},
                    {"role": "assistant", "content": content},
                ]
            })
        else:
            # General content → generate QA pairs
            template = self.rng.choice(self.CONVERSION_TEMPLATES)
            if template["type"] == "qa":
                question = self.rng.choice(template["questions"])
                conversations.append({
                    "messages": [
                        {"role": "system", "content": template["system"]},
                        {"role": "user", "content": f"{template['instruction_prefix']}{content[:2000]}\n\n{question}"},
                        {"role": "assistant", "content": content},
                    ]
                })
            else:
                conversations.append({
                    "messages": [
                        {"role": "system", "content": template["system"]},
                        {"role": "user", "content": f"{template['instruction_prefix']}{content[:500]}"},
                        {"role": "assistant", "content": content},
                    ]
                })

        return conversations


# Dataset Assembler

class DatasetAssembler:
    """
    Assembles the final training dataset from scored documents.
    
    Pipeline:
    1. Load scored/filtered documents
    2. Convert to conversation format
    3. Apply chat template
    4. Optionally pack sequences
    5. Stratified split into train/val/test
    6. Save in requested format
    """

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        np.random.seed(seed)

        assembly = config.get("assembly", {})
        self.train_ratio = assembly.get("train_split", 0.90)
        self.val_ratio = assembly.get("val_split", 0.05)
        self.test_ratio = assembly.get("test_split", 0.05)
        self.max_seq_length = assembly.get("max_sequence_length", 4096)
        self.do_packing = assembly.get("packing", True)
        self.output_format = assembly.get("output_format", "arrow")
        self.output_dir = Path(assembly.get("output_dir", "data/final"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.formatter = ChatTemplateFormatter(assembly.get("chat_template", "chatml"))
        self.converter = DocumentConverter(seed=seed)
        self.packer = SequencePacker(max_length=self.max_seq_length) if self.do_packing else None

    def _load_scored_docs(self, scored_dir: str | Path) -> list[dict]:
        """Load passed documents from the scoring stage."""
        docs = []
        passed_file = Path(scored_dir) / "passed_data.jsonl"
        if passed_file.exists():
            with open(passed_file) as f:
                for line in f:
                    if line.strip():
                        docs.append(json.loads(line))
        logger.info(f"Loaded {len(docs)} scored documents")
        return docs

    def _stratified_split(self, examples: list[dict]) -> tuple[list, list, list]:
        """Split examples with stratification by score bins."""
        # Bin by composite score
        for ex in examples:
            score = ex.get("composite_score", 3.5)
            ex["_score_bin"] = min(int(score), 5)

        # Group by bin
        bins = {}
        for ex in examples:
            b = ex["_score_bin"]
            bins.setdefault(b, []).append(ex)

        train, val, test = [], [], []
        for bin_examples in bins.values():
            self.rng.shuffle(bin_examples)
            n = len(bin_examples)
            n_val = max(1, int(n * self.val_ratio))
            n_test = max(1, int(n * self.test_ratio))
            n_train = n - n_val - n_test

            train.extend(bin_examples[:n_train])
            val.extend(bin_examples[n_train : n_train + n_val])
            test.extend(bin_examples[n_train + n_val :])

        # Clean up temp field
        for ex in train + val + test:
            ex.pop("_score_bin", None)

        return train, val, test

    def _save_split(self, examples: list[dict], split_name: str):
        """Save a split in the requested format."""
        output_path = self.output_dir / split_name

        if self.output_format == "jsonl":
            outfile = output_path.with_suffix(".jsonl")
            with open(outfile, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

        elif self.output_format in ("arrow", "parquet"):
            try:
                from datasets import Dataset
                ds = Dataset.from_list(examples)
                if self.output_format == "arrow":
                    ds.save_to_disk(str(output_path))
                else:
                    ds.to_parquet(str(output_path.with_suffix(".parquet")))
            except ImportError:
                # Fallback to JSONL
                outfile = output_path.with_suffix(".jsonl")
                with open(outfile, "w") as f:
                    for ex in examples:
                        f.write(json.dumps(ex) + "\n")

        logger.info(f"Saved {split_name}: {len(examples)} examples")

    def assemble(self) -> dict:
        """Run the full assembly pipeline."""
        scored_dir = self.config.get("llm_judge", {}).get("output_dir", "data/scored")
        docs = self._load_scored_docs(scored_dir)

        # Convert to conversations
        all_examples = []
        for doc in docs:
            conversations = self.converter.convert(doc)
            for conv in conversations:
                text = self.formatter.format_conversation(conv["messages"])
                example = {
                    "text": text,
                    "messages": conv["messages"],
                    "composite_score": doc.get("composite_score", 0),
                    "source": doc.get("source", "unknown"),
                }
                all_examples.append(example)

        logger.info(f"Generated {len(all_examples)} training examples from {len(docs)} documents")

        # Pack sequences if enabled
        if self.packer:
            texts = [ex["text"] for ex in all_examples]
            packed_texts = self.packer.pack(texts)
            all_examples = [{"text": t} for t in packed_texts]
            logger.info(f"After packing: {len(all_examples)} sequences")

        # Stratified split
        train, val, test = self._stratified_split(all_examples)

        # Save
        self._save_split(train, "train")
        self._save_split(val, "validation")
        self._save_split(test, "test")

        stats = {
            "total_examples": len(all_examples),
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "packing_enabled": self.do_packing,
            "chat_template": self.formatter.name,
            "output_format": self.output_format,
        }

        stats_file = self.output_dir / "assembly_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        return stats


def run_assembly(config: dict) -> dict:
    """Entry point for dataset assembly."""
    assembler = DatasetAssembler(
        config=config["data_curation"],
        seed=config.get("project", {}).get("seed", 42),
    )
    return assembler.assemble()
