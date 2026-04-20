import os
from dotenv import load_dotenv

# Load .env file (if you have HF_TOKEN stored there)
load_dotenv()

# from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ds = LeRobotDataset(repo_id="local/ur5e_pickplace", root="/Users/samuelize/Desktop/msc-proj/recorded_dataset")

# print(f"Total episodes: {ds.meta.total_episodes}")
# print(f"Total frames: {ds.num_frames}")
# print(f"Features: {list(ds.features.keys())}")

# # Load the first frame
# frame = ds[0]
# print("Keys in first frame:", frame.keys())

# # Check if images are present
# if "observation.images.top" in frame:
#     print("Top image shape:", frame["observation.images.top"].shape)
# else:
#     print("No top image in frame!")

# if "observation.images.wrist" in frame:
#     print("Wrist image shape:", frame["observation.images.wrist"].shape)
# else:
#     print("No wrist image in frame!")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load your local dataset
ds = LeRobotDataset(
    repo_id="local/ur5e_pickplace",
    root="/Users/samuelize/Desktop/msc-proj/recorded_dataset"
)

# Override the repo_id to your desired Hub ID
ds.repo_id = "leumals/ur5e_webots_pickplace"
ds.push_to_hub(private=False)