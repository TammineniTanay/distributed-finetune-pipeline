"""Experiment tracker — local experiment management and comparison.

Tracks every pipeline run with its config, metrics, and artifacts.
Compare runs side-by-side to see what changed and what improved.
No server needed — just JSON files on disk.

Usage:
    python -m experiments.tracker list
    python -m experiments.tracker compare run_001 run_002
    python -m experiments.tracker show run_001
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments/runs")


@dataclass
class RunMetadata:
    run_id: str
    created_at: str
    status: str = "running"
    duration_seconds: float = 0.0
    config_hash: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None


class ExperimentTracker:
    def __init__(self, base_dir=EXPERIMENTS_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run(self, config: dict, tags=None, notes="") -> str:
        existing = sorted(self.base_dir.iterdir()) if self.base_dir.exists() else []
        run_num = len(existing) + 1
        run_id = f"run_{run_num:03d}"
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        meta = RunMetadata(run_id=run_id, created_at=datetime.now().isoformat(),
                           config_hash=config_hash, tags=tags or [], notes=notes)

        self._save(run_dir / "metadata.json", asdict(meta))
        self._save(run_dir / "config.json", config)
        self._save(run_dir / "metrics.json", {})
        log.info(f"Created run: {run_id} (hash: {config_hash})")
        return run_id

    def log_metrics(self, run_id: str, stage: str, metrics: dict):
        run_dir = self.base_dir / run_id
        all_m = self._load(run_dir / "metrics.json")
        all_m[stage] = metrics
        self._save(run_dir / "metrics.json", all_m)

        meta = self._load(run_dir / "metadata.json")
        if "train_loss" in metrics: meta["train_loss"] = metrics["train_loss"]
        if "eval_loss" in metrics: meta["eval_loss"] = metrics["eval_loss"]
        self._save(run_dir / "metadata.json", meta)

    def complete_run(self, run_id: str, status="completed"):
        run_dir = self.base_dir / run_id
        meta = self._load(run_dir / "metadata.json")
        created = datetime.fromisoformat(meta["created_at"])
        meta["status"] = status
        meta["duration_seconds"] = (datetime.now() - created).total_seconds()
        self._save(run_dir / "metadata.json", meta)

    def list_runs(self):
        runs = []
        for d in sorted(self.base_dir.iterdir()):
            mp = d / "metadata.json"
            if mp.exists(): runs.append(self._load(mp))
        return runs

    def compare_runs(self, id_a: str, id_b: str) -> dict:
        a = self._load_run(id_a)
        b = self._load_run(id_b)
        return {
            "run_a": id_a, "run_b": id_b,
            "config_changes": self._diff(a["config"], b["config"]),
            "metric_deltas": self._compare_metrics(a["metrics"], b["metrics"]),
        }

    def _load_run(self, run_id):
        d = self.base_dir / run_id
        return {"config": self._load(d / "config.json"), "metrics": self._load(d / "metrics.json")}

    def _diff(self, a, b, prefix=""):
        diffs = {}
        for k in set(list(a.keys()) + list(b.keys())):
            fk = f"{prefix}.{k}" if prefix else k
            va, vb = a.get(k), b.get(k)
            if va == vb: continue
            elif isinstance(va, dict) and isinstance(vb, dict):
                diffs.update(self._diff(va, vb, fk))
            else:
                diffs[fk] = {"old": va, "new": vb}
        return diffs

    def _compare_metrics(self, ma, mb):
        deltas = {}
        for stage in set(list(ma.keys()) + list(mb.keys())):
            sa, sb = ma.get(stage, {}), mb.get(stage, {})
            for m in set(list(sa.keys()) + list(sb.keys())):
                va, vb = sa.get(m), sb.get(m)
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    d = vb - va
                    deltas[f"{stage}/{m}"] = {
                        "run_a": va, "run_b": vb, "delta": round(d, 6),
                        "pct": round(d / abs(va) * 100, 2) if va else 0,
                    }
        return deltas

    def format_comparison(self, comp):
        lines = [f"Compare: {comp['run_a']} vs {comp['run_b']}", "=" * 50]
        if comp["config_changes"]:
            lines.append("Config changes:")
            for k, v in comp["config_changes"].items():
                lines.append(f"  {k}: {v['old']} -> {v['new']}")
        else:
            lines.append("Config: IDENTICAL")
        lines.append("\nMetric deltas:")
        for m, d in comp["metric_deltas"].items():
            sign = "+" if d["delta"] > 0 else ""
            lines.append(f"  {m}: {d['run_a']:.4f} -> {d['run_b']:.4f} ({sign}{d['delta']:.4f})")
        return "\n".join(lines)

    def _save(self, p, d):
        with open(p, "w") as f: json.dump(d, f, indent=2, default=str)
    def _load(self, p):
        return json.load(open(p)) if p.exists() else {}


if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("list")
    s = sub.add_parser("show"); s.add_argument("run_id")
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    args = parser.parse_args()
    t = ExperimentTracker()
    if args.cmd == "list":
        for r in t.list_runs():
            loss = r.get("train_loss", "N/A")
            print(f"  {r['run_id']}  {r['status']:<10}  loss={loss}  hash={r.get('config_hash','')}")
    elif args.cmd == "compare":
        print(t.format_comparison(t.compare_runs(args.a, args.b)))
    elif args.cmd == "show":
        d = t.base_dir / args.run_id
        print(json.dumps(t._load(d / "metrics.json"), indent=2))
