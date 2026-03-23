#!/bin/bash
set -euo pipefail

# ==============================================================================
# EC2 Training Instance Bootstrap Script
# Configures the Deep Learning AMI for distributed fine-tuning
# ==============================================================================

exec > >(tee /var/log/user-data.log) 2>&1
echo "=== Bootstrap started at $(date) ==="

# Environment variables
export AWS_DEFAULT_REGION="${region}"
export S3_BUCKET="${s3_bucket}"
export WANDB_API_KEY="${wandb_key}"
export HF_TOKEN="${hf_token}"
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export OMP_NUM_THREADS=8

# Mount NVMe SSD (p4d instances have local NVMe)
NVME_DEVICE="/dev/nvme1n1"
MOUNT_POINT="/data"
if [ -b "$NVME_DEVICE" ]; then
    echo "Formatting and mounting NVMe SSD..."
    mkfs.xfs -f "$NVME_DEVICE"
    mkdir -p "$MOUNT_POINT"
    mount "$NVME_DEVICE" "$MOUNT_POINT"
    chmod 777 "$MOUNT_POINT"
    echo "$NVME_DEVICE $MOUNT_POINT xfs defaults,noatime 0 0" >> /etc/fstab
fi

# Create workspace
WORKSPACE="$MOUNT_POINT/pipeline"
mkdir -p "$WORKSPACE"/{checkpoints,data,outputs,logs}
cd "$WORKSPACE"

# Pull pipeline code from S3
aws s3 sync "s3://$S3_BUCKET/code/" "$WORKSPACE/" --exclude "*.pyc"

# Pull cached datasets if available
aws s3 sync "s3://$S3_BUCKET/data/" "$WORKSPACE/data/" || true

# Install Python dependencies (DL AMI has conda)
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

pip install --no-cache-dir \
    transformers>=4.44.0 \
    datasets>=2.21.0 \
    accelerate>=0.33.0 \
    peft>=0.12.0 \
    trl>=0.10.0 \
    bitsandbytes>=0.43.0 \
    deepspeed>=0.14.0 \
    wandb \
    pynvml \
    psutil \
    aiohttp

# Install Flash Attention 2 (if not pre-installed)
pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn already installed or build failed"

# Login to Hugging Face
python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# Spot interruption handler: saves checkpoint to S3 on termination
cat > /usr/local/bin/spot-handler.sh << 'HANDLER_EOF'
#!/bin/bash
# Polls EC2 metadata for spot interruption notice
while true; do
    STATUS=$(curl -sf -o /dev/null -w "%{http_code}" \
        http://169.254.169.254/latest/meta-data/spot/instance-action)
    if [ "$STATUS" = "200" ]; then
        echo "$(date) — Spot interruption detected! Saving checkpoint..."
        # Send SIGUSR1 to training process to trigger emergency checkpoint
        pkill -USR1 -f "training.train" || true
        sleep 30
        # Sync checkpoints to S3
        aws s3 sync /data/pipeline/checkpoints/ \
            "s3://${S3_BUCKET}/checkpoints/emergency-$(date +%s)/" \
            --only-show-errors
        echo "Emergency checkpoint saved to S3"
        break
    fi
    sleep 5
done
HANDLER_EOF
chmod +x /usr/local/bin/spot-handler.sh
nohup /usr/local/bin/spot-handler.sh &

# Periodic checkpoint sync to S3 (every 30 min)
cat > /usr/local/bin/checkpoint-sync.sh << 'SYNC_EOF'
#!/bin/bash
while true; do
    sleep 1800
    aws s3 sync /data/pipeline/checkpoints/ \
        "s3://${S3_BUCKET}/checkpoints/" \
        --only-show-errors 2>/dev/null || true
done
SYNC_EOF
chmod +x /usr/local/bin/checkpoint-sync.sh
nohup /usr/local/bin/checkpoint-sync.sh &

echo "=== Bootstrap complete at $(date) ==="
echo "Ready for distributed training launch."
