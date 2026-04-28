#!/bin/bash

# ==============================================================================
# LeRobot Video Conversion & Hub Uploader
# ==============================================================================

# 1. Export your Hugging Face Token (Requires 'Write' permissions)
export HF_TOKEN="your huggingface_token_here"

echo "🎥 Starting Video Conversion and Upload Pipeline..."
echo "Please wait while LeRobot compresses the images into libx264 MP4s..."

# 2. Convert and Push in one clean command
lerobot-edit-dataset \
    --repo_id local/ur5e_pickplace_v3 \
    --root /Users/samuelize/Desktop/msc-proj/recorded_dataset_v3 \
    --new_repo_id leumals/ur5e_pickplace_v3 \
    --operation.type convert_image_to_video \
    --push_to_hub true

echo "✅ All done! Your highly-optimized video dataset is now live on Hugging Face."