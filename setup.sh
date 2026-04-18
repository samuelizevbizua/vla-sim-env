#!/usr/bin/env bash
set -e

# Copy .env from template if needed
if [ -f ".env.template" ] && [ ! -f ".env" ]; then
  cp .env.template .env
fi

# Remove existing virtual environment for a clean setup
if [ -d ".venv" ]; then
  rm -rf .venv
fi

# Remove existing pycache directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Create a fresh virtual environment with Python 3.12
python3.12 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# ===== Install LeRobot in the same environment =====
if [ -d "lerobot" ]; then
  rm -rf lerobot
fi

# Clone the LeRobot 
sitory
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[smolvla,dataset,viz]"
cd ..

echo "🎉 Setup complete."