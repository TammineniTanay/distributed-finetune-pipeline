"""Data versioning — tracks which dataset version produced which model.

Every time the data pipeline runs, it generates a version manifest
with hashes of the input data, cleaning params, and assembly config.
This gets stored alongside the model checkpoint so you can always
trace back from a model to the exact data it was trained on.

No DVC dependency — just SHA256 hashes + JSON manifests.

Usage:
    # generate manifest for current dataset
    python -m data_curation.versioning snapshot data/assembled/

    # compare two versions
    python -m data_curation.versioning diff v1_manifest.json v2_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class DataVersion:
    version_id: str
    created_at: str
    num_files: int = 0
    total_bytes: int = 0
    total_examples: int = 0
    file_hashes: dict[str, str] = field(default_factory=dict)  # filename -> sha256
    config_hash: str = ""          # hash of the cleaning/assembly config used
    cleaning_config: dict = field(default_factory=dict)
    assembly_config: dict = field(default_factory=dict)
    parent_version: str = ""       # previous version id (for diffing)
    notes: str = ""


def hash_file(path: str, chunk_size: int = 8192) -> str:
    """SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_dict(d: dict) -> str:
    """Deterministic hash of a dict."""
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def create_snapshot(
    data_dir: str,
    config: dict = None,
    notes: str = "",
    output_path: str = None,
) -> DataVersion:
    """Create a version snapshot of a dataset directory.
    
    Hashes every file, counts examples (for JSONL), records the
    config that produced this data.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # generate version id from timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_hash = hash_dict(config) if config else "no_config"
    version_id = f"v_{ts}_{config_hash}"

    file_hashes = {}
    total_bytes = 0
    total_examples = 0

    for f in sorted(data_path.rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(data_path))
            file_hashes[rel] = hash_file(str(f))
            total_bytes += f.stat().st_size

            # count JSONL lines
            if f.suffix == ".jsonl":
                with open(f) as fh:
                    total_examples += sum(1 for line in fh if line.strip())

    version = DataVersion(
        version_id=version_id,
        created_at=datetime.now().isoformat(),
        num_files=len(file_hashes),
        total_bytes=total_bytes,
        total_examples=total_examples,
        file_hashes=file_hashes,
        config_hash=config_hash,
        cleaning_config=config.get("data_curation", {}).get("cleaning", {}) if config else {},
        assembly_config=config.get("data_curation", {}).get("assembly", {}) if config else {},
        notes=notes,
    )

    # save manifest
    if output_path is None:
        output_path = str(data_path / f"VERSION_{version_id}.json")

    with open(output_path, "w") as f:
        json.dump(asdict(version), f, indent=2)

    log.info(f"Data snapshot: {version_id} ({version.num_files} files, "
             f"{total_bytes/1e6:.1f}MB, {total_examples} examples)")
    return version


def diff_versions(v1_path: str, v2_path: str) -> dict:
    """Compare two data version manifests."""
    with open(v1_path) as f:
        v1 = json.load(f)
    with open(v2_path) as f:
        v2 = json.load(f)

    h1, h2 = v1["file_hashes"], v2["file_hashes"]
    all_files = set(list(h1.keys()) + list(h2.keys()))

    added = [f for f in all_files if f in h2 and f not in h1]
    removed = [f for f in all_files if f in h1 and f not in h2]
    modified = [f for f in all_files if f in h1 and f in h2 and h1[f] != h2[f]]
    unchanged = [f for f in all_files if f in h1 and f in h2 and h1[f] == h2[f]]

    return {
        "v1": v1["version_id"],
        "v2": v2["version_id"],
        "added": added,
        "removed": removed,
        "modified": modified,
        "unchanged_count": len(unchanged),
        "examples_delta": v2.get("total_examples", 0) - v1.get("total_examples", 0),
        "bytes_delta": v2.get("total_bytes", 0) - v1.get("total_bytes", 0),
        "config_changed": v1.get("config_hash") != v2.get("config_hash"),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    snap = sub.add_parser("snapshot")
    snap.add_argument("data_dir")
    snap.add_argument("--output", default=None)

    d = sub.add_parser("diff")
    d.add_argument("v1")
    d.add_argument("v2")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.cmd == "snapshot":
        v = create_snapshot(args.data_dir, output_path=args.output)
        print(f"Version: {v.version_id}")
        print(f"Files: {v.num_files}, Examples: {v.total_examples}, Size: {v.total_bytes/1e6:.1f}MB")
    elif args.cmd == "diff":
        result = diff_versions(args.v1, args.v2)
        print(f"Diff: {result['v1']} -> {result['v2']}")
        print(f"  Added: {len(result['added'])}, Removed: {len(result['removed'])}, Modified: {len(result['modified'])}")
        print(f"  Examples delta: {result['examples_delta']:+d}")
        print(f"  Config changed: {result['config_changed']}")
