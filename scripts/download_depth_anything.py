#!/usr/bin/env python3
"""Download Depth Anything V2 Small model weights from HuggingFace."""

from huggingface_hub import hf_hub_download
import os

files = [
    (
        "models/depth/depth_anything_v2_vits.safetensors",
        "Kijai/DepthAnythingV2-safetensors",
        "depth_anything_v2_vits_fp32.safetensors",
    ),
]

for local_path, repo_id, filename in files:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        print(f"  Already exists: {local_path}")
        continue

    print(f"  Downloading {filename} from {repo_id}...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=os.path.dirname(local_path),
    )
    # Rename if the downloaded filename differs from our target
    downloaded = os.path.join(os.path.dirname(local_path), filename)
    if downloaded != local_path and os.path.exists(downloaded):
        os.rename(downloaded, local_path)
        print(f"  Renamed to: {local_path}")

    print(f"  Saved: {local_path}")

print("Done!")
