# Distributed LLM Fine-Tuning Pipeline

End-to-end production pipeline for fine-tuning, merging, evaluating, and deploying large language models. Built for Llama 3 8B but adaptable to any HuggingFace-compatible model.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Pipeline Orchestrator                          │
│  (DAG execution · crash recovery · retry logic · W&B tracking)     │
├─────────┬──────────┬───────────┬──────────┬──────────┬────────────┤
│  Data   │  Data    │   LLM     │ Dataset  │Training  │  Model     │
│Scraping │Cleaning  │  Judge    │ Assembly │(QLoRA)   │ Merging    │
│         │          │           │          │          │            │
│ arXiv   │ MinHash  │ Multi-dim │ ChatML   │DeepSpeed │ TIES       │
│ GitHub  │  LSH     │  scoring  │ Llama    │ ZeRO-3   │ DARE       │
│ SO      │ PII      │ Calibrate │ Alpaca   │ FSDP     │ SLERP      │
│ HF      │ Quality  │  vs human │ Packing  │ NEFTune  │ Linear     │
│ Custom  │ LangDet  │           │ Splits   │Flash Attn│            │
├─────────┴──────────┴───────────┴──────────┼──────────┼────────────┤
│              Evaluation Harness            │Inference │ Deployment │
│                                            │          │            │
│  MC benchmarks · Generation · LM-eval      │  vLLM    │ Docker     │
│  LLM-as-judge · Statistical significance   │  AWQ/GPTQ│ AWS Spot   │
│  MMLU · HellaSwag · TruthfulQA · GSM8K     │  FastAPI │ Terraform  │
├────────────────────────────────────────────┴──────────┴────────────┤
│                     Monitoring Stack                               │
│  Prometheus · Grafana · GPU/CPU/Memory · Alerts · Pipeline State   │
└────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU(s) with CUDA 12.4+
- Docker & Docker Compose (for containerized deployment)

### Installation

```bash
git clone https://github.com/TammineniTanay/distributed-finetune-pipeline.git
cd distributed-finetune-pipeline

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention (requires CUDA toolkit)
pip install flash-attn --no-build-isolation
```

### Configuration

Edit `config/pipeline_config.yaml` to customize:
- Data sources and quality thresholds
- Base model and QLoRA hyperparameters
- Distributed training strategy (DeepSpeed / FSDP)
- Model merging methods and weights
- Evaluation benchmarks
- Inference serving configuration

### Run the Full Pipeline

```bash
# Full pipeline (all stages)
python pipeline.py --config config/pipeline_config.yaml

# Resume from last completed stage
python pipeline.py --config config/pipeline_config.yaml

# Start from a specific stage
python pipeline.py --start-from training

# Skip specific stages
python pipeline.py --skip data_scraping data_cleaning

# Override config values
python pipeline.py --override training.hyperparams.learning_rate=0.0001
```

### Run Individual Stages

```bash
# Training only (distributed)
./deployment/scripts/launch_training.sh config/pipeline_config.yaml 8 1

# Inference server
./deployment/scripts/launch_inference.sh outputs/merged_model 2 8000
```

## Pipeline Stages

### 1. Data Curation

**Scraping** (`data_curation/scrapers/multi_source_scraper.py`)
- Async multi-source scraping: arXiv, GitHub, StackOverflow, HuggingFace datasets, custom URLs
- Adaptive rate limiting with token bucket + 429 backoff
- SQLite-backed resumability for crash recovery
- JSONL batch output

**Cleaning** (`data_curation/cleaners/cleaning_pipeline.py`)
- MinHash LSH deduplication (128 permutations, 32 bands, configurable Jaccard threshold)
- PII removal: email, phone, SSN, IP, credit card → placeholder tokens
- Quality filtering: length, word count, repetition ratio, special characters, URL density
- Language detection

**LLM Judge** (`data_curation/llm_judge/judge.py`)
- Multi-dimensional scoring: relevance, quality, complexity, diversity, instruction-following
- Provider-agnostic: Anthropic, OpenAI, or local models
- SQLite score cache to avoid redundant API calls
- Calibration against human annotations (Cohen's kappa, Pearson correlation)

**Assembly** (`data_curation/dataset_assembler.py`)
- Chat template formatting: ChatML, Llama, Alpaca, Vicuna
- First-fit decreasing sequence packing for GPU utilization
- Stratified train/val/test splits
- Arrow, Parquet, or JSONL output

### 2. Training

**QLoRA Fine-Tuning** (`training/train.py`)
- 4-bit NormalFloat quantization with double quantization
- LoRA: r=64, alpha=128, targeting q/k/v/o/gate/up/down projections
- DeepSpeed ZeRO-3 or FSDP distributed strategies
- NEFTune noisy embeddings for regularization
- Flash Attention 2
- Cosine LR schedule with warmup

**Custom Callbacks** (`training/callbacks/custom_callbacks.py`)
- W&B metrics: loss EMA, GPU utilization, throughput, loss histogram
- Gradient monitoring: per-layer norms, spike detection (z-score), dead layer detection
- Checkpoint cleanup: keeps last N checkpoints
- Early stopping with warmup awareness

**Distributed Utilities** (`training/distributed/deepspeed_utils.py`)
- ZeRO-3 config generation with optional CPU/NVMe offload
- DeepSpeed and torchrun launchers with multi-node hostfile support

### 3. Model Merging

**From-scratch implementations** (`model_merging/merger.py`)
- **TIES**: Trim (density sparsification) → Elect Sign (majority vote) → Merge (aligned average)
- **DARE**: Random drop + rescale, combined with TIES or linear
- **SLERP**: Spherical linear interpolation between weight spaces
- **Linear**: Weighted task vector combination
- Mergekit YAML config generation for CLI workflows

### 4. Evaluation

**Multi-benchmark harness** (`evaluation/harness/eval_harness.py`)
- Multiple choice: log-likelihood scoring across answer choices
- Generation: ROUGE-L, BERTScore, LLM-as-judge
- Standard benchmarks via lm-evaluation-harness (MMLU, HellaSwag, TruthfulQA, GSM8K, HumanEval)
- Multi-judge ensemble with structured rubrics and majority vote aggregation
- Statistical significance: paired bootstrap, McNemar's test
- Side-by-side comparison reports across all models

### 5. Inference & Deployment

**vLLM Serving** (`inference/vllm_server/server.py`)
- Tensor parallel serving (TP=2 default)
- AWQ/GPTQ quantization for memory-efficient inference
- Prefix caching and chunked prefill
- FastAPI wrapper with health checks, Prometheus metrics, request logging

**Load Testing** (`inference/load_testing/locustfile.py`)
- Locust-based load tests with short/medium/long prompts
- Streaming and burst traffic patterns
- Time-to-first-token (TTFT) tracking

**Docker** (`deployment/docker/`)
- Multi-stage builds: CUDA devel for training, CUDA runtime for inference
- Docker Compose with GPU reservations, shared volumes, monitoring stack

**AWS Infrastructure** (`deployment/aws/main.tf`)
- Terraform IaC: VPC, S3, spot fleets (p4d.24xlarge), autoscaling (g5.2xlarge)
- Spot interruption handler with emergency checkpoint saves
- Periodic S3 checkpoint sync

### 6. Monitoring

**Metrics** (`monitoring/metrics_collector.py`)
- Thread-safe metric registry (counter, gauge, histogram)
- Prometheus-compatible `/metrics` endpoint
- GPU telemetry via pynvml: utilization, memory, temperature, power
- CPU, memory, disk I/O collection
- Pipeline state tracking with JSON persistence
- Alert rules with cooldown and pluggable handlers

**Dashboards**
- Auto-generated Grafana JSON dashboards for training and inference
- Prometheus scrape configs and alert rules included

## Project Structure

```
distributed-finetune-pipeline/
├── config/
│   ├── pipeline_config.yaml      # Master configuration
│   ├── config_loader.py          # Pydantic-validated loader
│   └── custom_urls.txt           # Custom scraping URLs
├── data_curation/
│   ├── scrapers/multi_source_scraper.py
│   ├── cleaners/cleaning_pipeline.py
│   ├── llm_judge/judge.py
│   ├── dataset_assembler.py
│   └── versioning.py             # Dataset version tracking
├── training/
│   ├── train.py                  # SFT training (QLoRA)
│   ├── dpo_train.py              # DPO alignment training
│   ├── hyperparam_sweep.py       # Grid/random search
│   ├── callbacks/custom_callbacks.py
│   ├── distributed/deepspeed_utils.py
│   └── configs/axolotl_config.yaml
├── model_merging/
│   └── merger.py                 # TIES, DARE, SLERP, Linear
├── evaluation/
│   ├── harness/eval_harness.py
│   ├── contamination_check.py    # Benchmark leakage detection
│   └── benchmarks/               # Sample data + rubrics
├── experiments/
│   └── tracker.py                # Run tracking + comparison
├── inference/
│   ├── vllm_server/server.py
│   └── load_testing/locustfile.py
├── monitoring/
│   ├── metrics_collector.py
│   ├── prometheus.yml
│   └── alert_rules.yml
├── deployment/
│   ├── docker/                   # Dockerfiles + Compose
│   ├── aws/                      # Terraform IaC
│   └── scripts/                  # Launch scripts
├── docs/
│   └── MODEL_CARD.md             # Model card template
├── tests/
│   └── test_pipeline.py
├── .github/workflows/ci.yml      # GitHub Actions CI
├── pipeline.py                   # Main orchestrator
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Beyond SFT: DPO Alignment

The pipeline includes a DPO (Direct Preference Optimization) stage that runs after SFT to align the model with human preferences — without needing a separate reward model.

```bash
# Generate sample preference pairs (for testing)
python -m training.dpo_train --generate-sample-data

# Run DPO training
python -m training.dpo_train --config config/pipeline_config.yaml
```

DPO uses a much lower learning rate (5e-7) and smaller LoRA rank (16) than SFT since it's making subtle adjustments to an already-capable model.

## Data Contamination Detection

Before trusting eval results, check if training data leaked benchmark examples:

```bash
python -m evaluation.contamination_check \
    --training-data data/assembled/train.jsonl \
    --benchmarks evaluation/benchmarks/*.jsonl \
    --threshold 0.8
```

Uses 13-gram character overlap (standard in the literature) to catch both exact copies and paraphrased benchmark questions.

## Experiment Tracking

Every pipeline run gets tracked with full config snapshots and metrics:

```bash
# List all runs
python -m experiments.tracker list

# Compare two runs (config diff + metric deltas)
python -m experiments.tracker compare run_001 run_002

# Show run details
python -m experiments.tracker show run_001
```

## Hyperparameter Sweep

Systematic search over LoRA rank, learning rate, warmup, and NEFTune noise:

```bash
# Random search (10 trials)
python -m training.hyperparam_sweep --config config/pipeline_config.yaml --method random --max-trials 10

# Grid search
python -m training.hyperparam_sweep --method grid --max-trials 20 --dry-run
```

## Data Versioning

Track which dataset version produced which model:

```bash
# Snapshot current dataset
python -m data_curation.versioning snapshot data/assembled/

# Compare versions
python -m data_curation.versioning diff v1_manifest.json v2_manifest.json
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test classes
pytest tests/test_pipeline.py::TestModelMerging -v
pytest tests/test_pipeline.py::TestMonitoring -v
```

## CI/CD

GitHub Actions runs on every push to `main`:
- Ruff linting
- Python syntax validation (all files)
- YAML validation
- Unit tests (CPU-only components)

## Docker Quick Start

```bash
cd deployment/docker

# Training (GPU required)
docker compose --profile training up --build

# Inference
docker compose --profile inference up --build

# Full stack with monitoring
docker compose --profile inference up --build

# Load testing
docker compose --profile loadtest up
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `WANDB_API_KEY` | Weights & Biases API key | For tracking |
| `HF_TOKEN` | HuggingFace access token | For gated models |
| `OPENAI_API_KEY` | OpenAI API key | For LLM judge |
| `ANTHROPIC_API_KEY` | Anthropic API key | For LLM judge |
| `AWS_ACCESS_KEY_ID` | AWS credentials | For deployment |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials | For deployment |
| `GRAFANA_PASSWORD` | Grafana admin password | For monitoring |

## License

MIT
