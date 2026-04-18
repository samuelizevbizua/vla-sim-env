#!/usr/bin/env bash
set -e

# Remove existing virtual environment for a clean setup
if [ -d ".lerobot_venv" ]; then
  rm -rf .lerobot_venv
fi

# Create virtual environment with Python 3.12
python3.12 -m venv .lerobot_venv

source .lerobot_venv/bin/activate

# Clone LeRobot repository
if [ ! -d "lerobot/.git" ]; then
  git clone https://github.com/huggingface/lerobot.git
fi

# Move into the folder
cd lerobot

# Install LeRobot with SmolVLA support (run from repo root)
pip install -e ".[smolvla]"
