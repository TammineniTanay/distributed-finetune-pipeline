#!/bin/bash
set -euo pipefail

# ==============================================================================
# launch_inference.sh — Launch vLLM inference server
#
# Usage:
#   ./launch_inference.sh [MODEL_PATH] [NUM_GPUS] [PORT]
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL_PATH="${1:-outputs/merged_model}"
NUM_GPUS="${2:-2}"
PORT="${3:-8000}"
WRAPPER_PORT="${4:-8080}"
QUANTIZATION="${QUANTIZATION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

echo "=============================================="
echo "  vLLM Inference Server Launcher"
echo "=============================================="
echo "Model:       $MODEL_PATH"
echo "GPUs:        $NUM_GPUS"
echo "Port:        $PORT"
echo "Wrapper:     $WRAPPER_PORT"
echo "Quantization: ${QUANTIZATION:-none}"
echo "=============================================="

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))

# Build vLLM command
VLLM_ARGS=(
    --model "$MODEL_PATH"
    --tensor-parallel-size "$NUM_GPUS"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization 0.90
    --max-num-batched-tokens 8192
    --max-num-seqs 256
    --enable-prefix-caching
    --enable-chunked-prefill
    --host 0.0.0.0
    --port "$PORT"
)

if [ -n "$QUANTIZATION" ]; then
    VLLM_ARGS+=(--quantization "$QUANTIZATION")
fi

echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server "${VLLM_ARGS[@]}" &
VLLM_PID=$!

# Wait for health
echo "Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    sleep 2
done

# Start API wrapper
echo "Starting API wrapper on port $WRAPPER_PORT..."
python -c "
from inference.vllm_server.server import create_api_app, VLLMServerConfig
import uvicorn

config = VLLMServerConfig(
    model_path='$MODEL_PATH',
    tensor_parallel_size=$NUM_GPUS,
    port=$PORT,
)
app = create_api_app(config)
uvicorn.run(app, host='0.0.0.0', port=$WRAPPER_PORT)
" &

# Trap for clean shutdown
trap "kill $VLLM_PID 2>/dev/null; wait" EXIT SIGTERM SIGINT

wait $VLLM_PID
