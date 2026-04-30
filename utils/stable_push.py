import os
from huggingface_hub import HfApi
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# 1. Force the ultra-fast Rust engine natively!
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# 2. Insert your newly generated write token here!
HF_TOKEN = os.getenv("HF_TOKEN")

print("🚀 Initializing Stable Push with Rust Engine...")
api = HfApi(token=HF_TOKEN)

print("📦 Uploading completed local dataset...")
api.upload_large_folder(
    folder_path="/Users/samuelize/Desktop/msc-proj/recorded_dataset_v4",
    repo_id="leumals/ur5e_pickplace_v2", #ill name it v2 online since theres already a v1
    repo_type="dataset"
)

print("✅ Upload completely finished!")