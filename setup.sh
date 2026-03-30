#!/usr/bin/env bash
set -e

# Remove existing virtual environment for a clean setup
if [ -d ".venv" ]; then
  rm -rf .venv
fi

# Remove existing pycache directories
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Create a fresh virtual environment with Python 3.11
python3.11 -m venv .venv

# Activate the virtual environment
# shellcheck source=/dev/null
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
