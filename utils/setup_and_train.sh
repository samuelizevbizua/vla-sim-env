#!/bin/bash
# setup_and_train.sh - Full setup + optimised training on RunPod RTX 4090
# Direct download to local /tmp for maximum I/O performance.

set -e  # exit on error, but conditional checks prevent unnecessary exits

echo "🚀 Starting full setup and training (idempotent, optimised)..."

# ------------------------------
# 1. Environment variables
# ------------------------------
export CUDA_VISIBLE_DEVICES=0
export HF_TOKEN="your_huggingface_token_here"  # Set your Hugging Face token here

# ------------------------------
# 2. Install Miniforge if missing
# ------------------------------
if [ ! -d "/root/miniforge3" ]; then
    echo "📦 Installing Miniforge..."
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b -p /root/miniforge3
    rm Miniforge3-Linux-x86_64.sh
    source /root/miniforge3/etc/profile.d/conda.sh
    conda init bash
else
    source /root/miniforge3/etc/profile.d/conda.sh
fi

# ------------------------------
# 3. Create conda environment if missing
# ------------------------------
if ! conda env list | grep -q "lerobot_env"; then
    echo "🐍 Creating conda environment 'lerobot_env' with Python 3.12..."
    conda create -n lerobot_env python=3.12 -y
fi
conda activate lerobot_env

# ------------------------------
# 4. Install LeRobot if not already installed
# ------------------------------
if ! python -c "import lerobot" 2>/dev/null; then
    echo "📚 Installing LeRobot..."
    pip install --upgrade pip
    pip install 'lerobot[dataset,smolvla] @ git+https://github.com/huggingface/lerobot.git'
    pip install datasets av
else
    echo "✅ LeRobot already installed."
fi

# ------------------------------
# 5. Verify GPU
# ------------------------------
echo "🔍 Verifying GPU..."
python -c "import torch; assert torch.cuda.is_available(); print('✅ GPU ready: {}'.format(torch.cuda.get_device_name(0)))"

# ------------------------------
# 6. Download dataset directly to fast local storage (/tmp)
# ------------------------------
DATASET_FAST="/tmp/dataset"
if [ ! -f "$DATASET_FAST/meta/info.json" ]; then
    echo "📥 Downloading dataset directly to fast local NVMe ($DATASET_FAST)..."
    mkdir -p "$DATASET_FAST"
    hf download leumals/ur5e_webots_pickplace --repo-type dataset --local-dir "$DATASET_FAST"
else
    echo "✅ Dataset already exists at $DATASET_FAST"
fi

# ------------------------------
# 7. Kill any previous training process
# ------------------------------
pkill -f lerobot_train 2>/dev/null || true
sleep 2

# ------------------------------
# 8. Launch training with optimised data loader
# ------------------------------
cd /workspace
mkdir -p lerobot
cd lerobot

echo "🏋️ Starting training with optimised data loader (num_workers=12, prefetch=8)..."
nohup python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=local/ur5e_pickplace \
    --dataset.root="$DATASET_FAST" \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base \
    --output_dir=./outputs/smolvla_finetuned \
    --job_name=ur5e_pickplace_training \
    --batch_size=16 \
    --steps=15000 \
    --policy.device=cuda \
    --policy.use_amp=true \
    --policy.optimizer_lr=1e-5 \
    --policy.optimizer_weight_decay=0.01 \
    --policy.scheduler_warmup_steps=500 \
    --save_freq=1000 \
    --log_freq=50 \
    --wandb.enable=false \
    --save_checkpoint=true \
    --policy.push_to_hub=true \
    --policy.repo_id=leumals/smolvla_ur5e_finetuned \
    --policy.private=false \
    --num_workers=12 \
    --prefetch_factor=8 \
    > /workspace/training.log 2>&1 &

PID=$!
echo "✅ Training launched with PID $PID"
echo "📊 Monitor logs: tail -f /workspace/training.log"
echo "🖥️ Monitor GPU: watch -n 1 nvidia-smi"