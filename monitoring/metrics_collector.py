"""Monitoring: Prometheus metrics, GPU telemetry, pipeline state tracking.

Homegrown monitoring because prometheus_client was overkill for what
we need and added unnecessary deps. The MetricRegistry produces
Prometheus-compatible text format that any scraper can consume.

TODO: might want to just use prometheus_client eventually, this was
a fun exercise but maintenance burden is real
"""

from __future__ import annotations

import json
import logging
import os
import platform
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)


# Metric Types

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    name: str
    type: MetricType
    help: str
    labels: dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    _buckets: list[float] = field(default_factory=list)
    _observations: list[float] = field(default_factory=list)
    _sum: float = 0.0
    _count: int = 0


class MetricRegistry:
    """Thread-safe metric registry with Prometheus-compatible output."""

    def __init__(self):
        self._metrics: dict[str, Metric] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, help: str, labels: dict | None = None) -> str:
        key = self._key(name, labels)
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = Metric(
                    name=name, type=MetricType.COUNTER, help=help,
                    labels=labels or {},
                )
        return key

    def gauge(self, name: str, help: str, labels: dict | None = None) -> str:
        key = self._key(name, labels)
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = Metric(
                    name=name, type=MetricType.GAUGE, help=help,
                    labels=labels or {},
                )
        return key

    def histogram(
        self, name: str, help: str,
        buckets: list[float] | None = None,
        labels: dict | None = None,
    ) -> str:
        key = self._key(name, labels)
        if buckets is None:
            buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = Metric(
                    name=name, type=MetricType.HISTOGRAM, help=help,
                    labels=labels or {}, _buckets=sorted(buckets),
                )
        return key

    def inc(self, key: str, value: float = 1.0) -> None:
        with self._lock:
            if key in self._metrics:
                self._metrics[key].value += value

    def set(self, key: str, value: float) -> None:
        with self._lock:
            if key in self._metrics:
                self._metrics[key].value = value

    def observe(self, key: str, value: float) -> None:
        with self._lock:
            if key in self._metrics:
                m = self._metrics[key]
                m._observations.append(value)
                m._sum += value
                m._count += 1
                if len(m._observations) > 100_000:
                    m._observations = m._observations[-50_000:]

    def format_prometheus(self):
        lines = []
        seen_help = set()
        with self._lock:
            for key, m in sorted(self._metrics.items()):
                if m.name not in seen_help:
                    lines.append(f"# HELP {m.name} {m.help}")
                    lines.append(f"# TYPE {m.name} {m.type.value}")
                    seen_help.add(m.name)

                label_str = self._format_labels(m.labels)

                if m.type == MetricType.HISTOGRAM:
                    bucket_counts = [0] * len(m._buckets)
                    for obs in m._observations:
                        for i, b in enumerate(m._buckets):
                            if obs <= b:
                                bucket_counts[i] += 1

                    cumulative = 0
                    for i, b in enumerate(m._buckets):
                        cumulative += bucket_counts[i]
                        le_labels = {**m.labels, "le": str(b)}
                        lines.append(
                            f'{m.name}_bucket{self._format_labels(le_labels)} {cumulative}'
                        )
                    inf_labels = {**m.labels, "le": "+Inf"}
                    lines.append(
                        f'{m.name}_bucket{self._format_labels(inf_labels)} {m._count}'
                    )
                    lines.append(f"{m.name}_sum{label_str} {m._sum}")
                    lines.append(f"{m.name}_count{label_str} {m._count}")
                else:
                    lines.append(f"{m.name}{label_str} {m.value}")

        return "\n".join(lines) + "\n"

    def snapshot(self):
        with self._lock:
            return {k: m.value for k, m in self._metrics.items()}

    def _key(self, name: str, labels: dict | None) -> str:
        if not labels:
            return name
        label_parts = sorted(f"{k}={v}" for k, v in labels.items())
        return f"{name}{{{','.join(label_parts)}}}"

    def _format_labels(self, labels: dict) -> str:
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(parts) + "}"


# Global registry
REGISTRY = MetricRegistry()


# System Metrics Collector (GPU / CPU / Memory / Disk)

class SystemMetricsCollector:
    """
    Collects system telemetry at configurable intervals.
    
    - GPU: utilization, memory used/total, temperature, power (via pynvml)
    - CPU: utilization per core, load average
    - Memory: RSS, virtual, swap
    - Disk: read/write bytes, IOPS
    - Network: bytes sent/received
    """

    def __init__(self, registry: MetricRegistry, interval: float = 5.0):
        self.registry = registry
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._gpu_available = False
        self._setup_metrics()

    def _setup_metrics(self):
        self._gpu_util = self.registry.gauge(
            "gpu_utilization_percent", "GPU compute utilization"
        )
        self._gpu_mem_used = self.registry.gauge(
            "gpu_memory_used_bytes", "GPU memory used"
        )
        self._gpu_mem_total = self.registry.gauge(
            "gpu_memory_total_bytes", "GPU memory total"
        )
        self._gpu_temp = self.registry.gauge(
            "gpu_temperature_celsius", "GPU temperature"
        )
        self._gpu_power = self.registry.gauge(
            "gpu_power_watts", "GPU power draw"
        )
        self._cpu_util = self.registry.gauge(
            "cpu_utilization_percent", "CPU utilization"
        )
        self._mem_used = self.registry.gauge(
            "memory_used_bytes", "System memory used"
        )
        self._mem_total = self.registry.gauge(
            "memory_total_bytes", "System memory total"
        )
        self._disk_read = self.registry.counter(
            "disk_read_bytes_total", "Total disk bytes read"
        )
        self._disk_write = self.registry.counter(
            "disk_write_bytes_total", "Total disk bytes written"
        )

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        log.info(f"System metrics collector started (interval={self.interval}s)")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _collect_loop(self):
        self._init_gpu()
        while not self._stop_event.is_set():
            try:
                self._collect_gpu()
                self._collect_cpu_memory()
                self._collect_disk()
            except Exception as e:
                log.warning(f"Metrics collection error: {e}")
            self._stop_event.wait(self.interval)

    def _init_gpu(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._gpu_available = True
            self._gpu_count = pynvml.nvmlDeviceGetCount()
            self._gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(self._gpu_count)
            ]
            log.info(f"GPU monitoring initialized: {self._gpu_count} device(s)")
        except Exception:
            self._gpu_available = False

    def _collect_gpu(self):
        # print(f"GPU {i}: {util.gpu}% util, {mem.used/1e9:.1f}GB")  # noisy
        if not self._gpu_available:
            return
        try:
            import pynvml
            for i, handle in enumerate(self._gpu_handles):
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    power = 0.0

                labels = {"gpu": str(i)}
                self.registry.set(
                    self.registry.gauge(
                        "gpu_utilization_percent",
                        "GPU compute utilization", labels
                    ),
                    util.gpu,
                )
                self.registry.set(
                    self.registry.gauge(
                        "gpu_memory_used_bytes",
                        "GPU memory used", labels
                    ),
                    mem.used,
                )
                self.registry.set(
                    self.registry.gauge(
                        "gpu_temperature_celsius",
                        "GPU temperature", labels
                    ),
                    temp,
                )
                self.registry.set(
                    self.registry.gauge(
                        "gpu_power_watts",
                        "GPU power draw", labels
                    ),
                    power,
                )
        except Exception as e:
            log.debug(f"GPU metrics error: {e}")

    def _collect_cpu_memory(self):
        try:
            import psutil
            self.registry.set(self._cpu_util, psutil.cpu_percent())
            mem = psutil.virtual_memory()
            self.registry.set(self._mem_used, mem.used)
            self.registry.set(self._mem_total, mem.total)
        except ImportError:
            # Fallback: read /proc on Linux
            if platform.system() == "Linux":
                try:
                    with open("/proc/meminfo") as f:
                        info = {}
                        for line in f:
                            parts = line.split()
                            if len(parts) >= 2:
                                info[parts[0].rstrip(":")] = int(parts[1]) * 1024
                    total = info.get("MemTotal", 0)
                    available = info.get("MemAvailable", 0)
                    self.registry.set(self._mem_used, total - available)
                    self.registry.set(self._mem_total, total)
                except Exception:
                    pass

    def _collect_disk(self):
        try:
            import psutil
            disk = psutil.disk_io_counters()
            if disk:
                self.registry.set(self._disk_read, disk.read_bytes)
                self.registry.set(self._disk_write, disk.write_bytes)
        except ImportError:
            pass


# Pipeline State Tracker

class PipelineStage(Enum):
    DATA_SCRAPING = "data_scraping"
    DATA_CLEANING = "data_cleaning"
    LLM_SCORING = "llm_scoring"
    DATASET_ASSEMBLY = "dataset_assembly"
    CONTAMINATION_CHECK = "contamination_check"
    TRAINING = "training"
    DPO_TRAINING = "dpo_training"
    HYPERPARAM_SWEEP = "hyperparam_sweep"
    MODEL_MERGING = "model_merging"
    EVALUATION = "evaluation"
    QUANTIZATION = "quantization"
    DEPLOYMENT = "deployment"


@dataclass
class StageRecord:
    stage: PipelineStage
    status: str  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class PipelineStateTracker:
    """
    Tracks pipeline execution state across stages.
    Persists state to disk for crash recovery.
    """

    def __init__(self, state_file: str = "pipeline_state.json"):
        self.state_file = Path(state_file)
        self.stages: dict[str, StageRecord] = {}
        self._lock = threading.Lock()
        self._callbacks: list[Callable] = []
        self._load_state()

    def register_callback(self, fn: Callable[[StageRecord], None]):
        self._callbacks.append(fn)

    def start_stage(self, stage: PipelineStage, metadata: dict | None = None):
        with self._lock:
            record = StageRecord(
                stage=stage,
                status="running",
                start_time=time.time(),
                metadata=metadata or {},
            )
            self.stages[stage.value] = record
            self._persist()
            log.info(f"Pipeline stage started: {stage.value}")
            for cb in self._callbacks:
                try:
                    cb(record)
                except Exception:
                    pass

    def complete_stage(self, stage: PipelineStage, metadata: dict | None = None):
        with self._lock:
            if stage.value in self.stages:
                record = self.stages[stage.value]
                record.status = "completed"
                record.end_time = time.time()
                if metadata:
                    record.metadata.update(metadata)
                self._persist()
                log.info(
                    f"Pipeline stage completed: {stage.value} "
                    f"({record.duration_seconds:.1f}s)"
                )
                for cb in self._callbacks:
                    try:
                        cb(record)
                    except Exception:
                        pass

    def fail_stage(self, stage: PipelineStage, error: str):
        with self._lock:
            if stage.value in self.stages:
                record = self.stages[stage.value]
                record.status = "failed"
                record.end_time = time.time()
                record.error = error
                self._persist()
                log.error(f"Pipeline stage failed: {stage.value} — {error}")
                for cb in self._callbacks:
                    try:
                        cb(record)
                    except Exception:
                        pass

    def get_status(self) -> dict:
        with self._lock:
            return {
                name: {
                    "status": r.status,
                    "duration_seconds": r.duration_seconds,
                    "error": r.error,
                    "metadata": r.metadata,
                }
                for name, r in self.stages.items()
            }

    def get_last_completed_stage(self) -> Optional[PipelineStage]:
        completed = [
            r for r in self.stages.values()
            if r.status == "completed" and r.end_time
        ]
        if not completed:
            return None
        latest = max(completed, key=lambda r: r.end_time)
        return latest.stage

    def _persist(self):
        try:
            data = {}
            for name, r in self.stages.items():
                data[name] = {
                    "status": r.status,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "error": r.error,
                    "metadata": r.metadata,
                }
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to persist pipeline state: {e}")

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                for name, info in data.items():
                    try:
                        stage = PipelineStage(name)
                    except ValueError:
                        continue
                    self.stages[name] = StageRecord(
                        stage=stage,
                        status=info.get("status", "unknown"),
                        start_time=info.get("start_time"),
                        end_time=info.get("end_time"),
                        error=info.get("error"),
                        metadata=info.get("metadata", {}),
                    )
                log.info(f"Loaded pipeline state from {self.state_file}")
            except Exception as e:
                log.warning(f"Failed to load pipeline state: {e}")


# Alert Manager

@dataclass
class AlertRule:
    name: str
    metric_key: str
    condition: str  # "gt", "lt", "eq"
    threshold: float
    severity: str = "warning"  # warning, critical
    cooldown_seconds: float = 300.0
    message_template: str = ""
    _last_fired: float = 0.0


class AlertManager:
    """
    Evaluates alert rules against the metric registry and dispatches
    notifications via pluggable backends (log, webhook, Slack, PagerDuty).
    """

    def __init__(self, registry: MetricRegistry):
        self.registry = registry
        self.rules: list[AlertRule] = []
        self._handlers: list[Callable[[AlertRule, float], None]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)

    def add_handler(self, handler: Callable[[AlertRule, float], None]):
        self._handlers.append(handler)

    def start(self, check_interval: float = 30.0):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._eval_loop, args=(check_interval,), daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _eval_loop(self, interval: float):
        while not self._stop_event.is_set():
            snapshot = self.registry.snapshot()
            now = time.time()

            for rule in self.rules:
                value = snapshot.get(rule.metric_key)
                if value is None:
                    continue

                triggered = False
                if rule.condition == "gt" and value > rule.threshold:
                    triggered = True
                elif rule.condition == "lt" and value < rule.threshold:
                    triggered = True
                elif rule.condition == "eq" and value == rule.threshold:
                    triggered = True

                if triggered and (now - rule._last_fired) > rule.cooldown_seconds:
                    rule._last_fired = now
                    msg = rule.message_template.format(
                        name=rule.name, value=value, threshold=rule.threshold
                    ) if rule.message_template else (
                        f"[{rule.severity.upper()}] {rule.name}: "
                        f"{rule.metric_key}={value} {rule.condition} {rule.threshold}"
                    )
                    log.warning(f"ALERT: {msg}")
                    for handler in self._handlers:
                        try:
                            handler(rule, value)
                        except Exception as e:
                            log.error(f"Alert handler error: {e}")

            self._stop_event.wait(interval)


# Grafana Dashboard Generator

class GrafanaDashboardGenerator:
    """Generates Grafana JSON dashboard models for the pipeline."""

    def __init__(self, datasource: str = "Prometheus"):
        self.datasource = datasource

    def generate_training_dashboard(self) -> dict:
        return {
            "dashboard": {
                "title": "LLM Fine-Tuning Pipeline",
                "tags": ["ml", "training", "llm"],
                "timezone": "browser",
                "refresh": "10s",
                "panels": [
                    self._panel(
                        "Training Loss", 0, 0, 12, 8,
                        [
                            'training_loss',
                            'training_loss_ema',
                        ],
                    ),
                    self._panel(
                        "Learning Rate", 12, 0, 12, 8,
                        ['training_learning_rate'],
                    ),
                    self._panel(
                        "GPU Utilization (%)", 0, 8, 8, 8,
                        ['gpu_utilization_percent'],
                    ),
                    self._panel(
                        "GPU Memory (GB)", 8, 8, 8, 8,
                        ['gpu_memory_used_bytes / 1073741824'],
                    ),
                    self._panel(
                        "GPU Temperature (°C)", 16, 8, 8, 8,
                        ['gpu_temperature_celsius'],
                    ),
                    self._panel(
                        "Throughput (tokens/s)", 0, 16, 12, 8,
                        ['training_throughput_tokens_per_sec'],
                    ),
                    self._panel(
                        "Gradient Norms", 12, 16, 12, 8,
                        ['gradient_norm_total'],
                    ),
                    self._stat_panel(
                        "Current Epoch", 0, 24, 4, 4,
                        'training_epoch',
                    ),
                    self._stat_panel(
                        "Global Step", 4, 24, 4, 4,
                        'training_global_step',
                    ),
                    self._stat_panel(
                        "Total Requests (Inference)", 8, 24, 4, 4,
                        'inference_total_requests',
                    ),
                ],
            },
        }

    def generate_inference_dashboard(self) -> dict:
        return {
            "dashboard": {
                "title": "LLM Inference Monitoring",
                "tags": ["ml", "inference", "vllm"],
                "timezone": "browser",
                "refresh": "5s",
                "panels": [
                    self._panel(
                        "Request Latency (ms)", 0, 0, 12, 8,
                        ['inference_latency_ms'],
                    ),
                    self._panel(
                        "Requests/sec", 12, 0, 12, 8,
                        ['rate(inference_total_requests[1m])'],
                    ),
                    self._panel(
                        "Token Throughput", 0, 8, 12, 8,
                        [
                            'rate(inference_tokens_in_total[1m])',
                            'rate(inference_tokens_out_total[1m])',
                        ],
                    ),
                    self._panel(
                        "Error Rate", 12, 8, 12, 8,
                        ['rate(inference_errors_total[1m])'],
                    ),
                    self._panel(
                        "GPU Utilization", 0, 16, 12, 8,
                        ['gpu_utilization_percent'],
                    ),
                    self._panel(
                        "GPU Memory", 12, 16, 12, 8,
                        ['gpu_memory_used_bytes'],
                    ),
                ],
            },
        }

    def save(self, dashboard: dict, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dashboard, f, indent=2)
        log.info(f"Grafana dashboard saved to {output_path}")

    def _panel(
        self, title: str, x: int, y: int, w: int, h: int,
        queries: list[str],
    ) -> dict:
        return {
            "type": "timeseries",
            "title": title,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [
                {
                    "expr": q,
                    "datasource": self.datasource,
                    "legendFormat": q.split("{")[0] if "{" in q else q,
                }
                for q in queries
            ],
        }

    def _stat_panel(
        self, title: str, x: int, y: int, w: int, h: int,
        query: str,
    ) -> dict:
        return {
            "type": "stat",
            "title": title,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [
                {"expr": query, "datasource": self.datasource}
            ],
        }


# Prometheus HTTP Exporter

class PrometheusExporter:
    """
    Lightweight HTTP server that exposes /metrics for Prometheus scraping.
    Runs in a background thread.
    """

    def __init__(self, registry: MetricRegistry, port: int = 9090):
        self.registry = registry
        self.port = port
        self._server = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        from http.server import HTTPServer, BaseHTTPRequestHandler

        registry = self.registry

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/metrics":
                    body = registry.format_prometheus().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/health":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress access logs

        self._server = HTTPServer(("0.0.0.0", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        log.info(f"Prometheus exporter started on port {self.port}")

    def stop(self):
        if self._server:
            self._server.shutdown()
