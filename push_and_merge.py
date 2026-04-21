#!/usr/bin/env python3
"""
Push local dataset to Hub, convert images to videos, then merge with Berkeley UR5 dataset.
"""

import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ========== CONFIGURATION ==========
HF_USERNAME = "leumals"  # change to your Hugging Face username
LOCAL_REPO_ID = "local/ur5e_pickplace"
LOCAL_ROOT = "./recorded_dataset"
HUB_DATASET_ID = f"{HF_USERNAME}/ur5e_webots_pickplace"
MERGED_DATASET_ID = f"{HF_USERNAME}/ur5e_merged"

# ========== 1. Push local dataset to Hub ==========
print("Step 1: Pushing local dataset to Hugging Face Hub...")
ds = LeRobotDataset(repo_id=LOCAL_REPO_ID, root=LOCAL_ROOT)
ds.repo_id = HUB_DATASET_ID
ds.push_to_hub(private=False)
print(f"Dataset pushed to: https://huggingface.co/datasets/{HUB_DATASET_ID}")

# ========== 2. Convert images to videos (to match Berkeley's video dtype) ==========
print("\nStep 2: Converting images to videos...")
# We need to use the CLI tool because the Python API for conversion is not exposed.
# Run the conversion command (this will create a new video-based dataset on the Hub).
convert_cmd = (
    f"lerobot-edit-dataset "
    f"--repo_id {HUB_DATASET_ID} "
    f"--operation.type convert_image_to_video "
    f"--operation.vcodec libsvtav1 "
    f"--new_repo_id {HUB_DATASET_ID}_video "
    f"--push_to_hub True"
)
print(f"Running: {convert_cmd}")
os.system(convert_cmd)

# Wait a moment for the conversion to complete (optional, but good)
import time
time.sleep(5)

# ========== 3. Merge the converted dataset with Berkeley UR5 ==========
print("\nStep 3: Merging with Berkeley UR5 dataset...")
merge_cmd = (
    f"lerobot-edit-dataset "
    f"--operation.type merge "
    f"--operation.repo_ids \"['lerobot/berkeley_autolab_ur5', '{HUB_DATASET_ID}_video']\" "
    f"--new_repo_id {MERGED_DATASET_ID} "
    f"--push_to_hub True"
)
print(f"Running: {merge_cmd}")
os.system(merge_cmd)

print(f"\n✅ Done! Merged dataset available at: https://huggingface.co/datasets/{MERGED_DATASET_ID}")