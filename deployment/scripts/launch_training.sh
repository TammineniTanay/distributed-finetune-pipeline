#!/bin/bash
set -euo pipefail

# ==============================================================================
# launch_training.sh — Launch distributed training on multi-GPU
#
# Usage:
#   ./launch_training.sh [CONFIG_PATH] [NUM_GPUS] [NUM_NODES]
#
# Examples:
#   ./launch_training.sh                                    # defaults
#   ./launch_training.sh config/pipeline_config.yaml 8 1    # 8 GPUs, 1 node
#   ./launch_training.sh config/pipeline_config.yaml 8 2    # 8 GPUs, 2 nodes
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${1:-config/pipeline_config.yaml}"
NUM_GPUS="${2:-$(nvidia-smi --list-gpus | wc -l)}"
NUM_NODES="${3:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"

echo "=============================================="
echo "  Distributed Fine-Tuning Pipeline Launcher"
echo "=============================================="
echo "Config:      $CONFIG"
echo "GPUs:        $NUM_GPUS"
echo "Nodes:       $NUM_NODES"
echo "Master:      $MASTER_ADDR:$MASTER_PORT"
echo "Node Rank:   $NODE_RANK"
echo "=============================================="

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"

# Set distributed environment
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# Determine launcher: DeepSpeed or torchrun
USE_DEEPSPEED="${USE_DEEPSPEED:-1}"

if [ "$USE_DEEPSPEED" = "1" ]; then
    echo "Launching with DeepSpeed..."

    HOSTFILE=""
    if [ "$NUM_NODES" -gt 1 ] && [ -f "hostfile" ]; then
        HOSTFILE="--hostfile hostfile"
    fi

    deepspeed \
        --num_gpus "$NUM_GPUS" \
        --num_nodes "$NUM_NODES" \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        $HOSTFILE \
        training/train.py \
        --config "$CONFIG"
else
    echo "Launching with torchrun..."

    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --nnodes="$NUM_NODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        training/train.py \
        --config "$CONFIG"
fi

echo "Training complete!"
