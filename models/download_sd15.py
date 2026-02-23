#!/usr/bin/env python3
"""Download required model files for Stable Diffusion 1.5 inference.

Requires: pip install huggingface-hub

Files downloaded:
  models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors           (~2.1 GB)
  models/clip/vocab.json                                             (~940 KB)
  models/clip/merges.txt                                             (~512 KB)
  models/controlnet/control_v11f1p_sd15_depth_fp16.safetensors      (~723 MB)
"""

import hashlib
import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    print("Error: huggingface-hub is required. Install with:")
    print("  pip install huggingface-hub")
    sys.exit(1)

MODELS_DIR = Path(__file__).resolve().parent

# (local_relative_path, repo_id, filename_in_repo)
FILES = [
    (
        "checkpoints/v1-5-pruned-emaonly-fp16.safetensors",
        "Comfy-Org/stable-diffusion-v1-5-archive",
        "v1-5-pruned-emaonly-fp16.safetensors",
    ),
    (
        "clip/vocab.json",
        "openai/clip-vit-large-patch14",
        "vocab.json",
    ),
    (
        "clip/merges.txt",
        "openai/clip-vit-large-patch14",
        "merges.txt",
    ),
    (
        "controlnet/control_v11f1p_sd15_depth_fp16.safetensors",
        "comfyanonymous/ControlNet-v1-1_fp16_safetensors",
        "control_v11f1p_sd15_depth_fp16.safetensors",
    ),
]


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_expected_sha256(repo_id: str, filename: str) -> str | None:
    """Fetch the expected SHA-256 from HuggingFace LFS metadata.

    Returns None for non-LFS files (small files stored as git blobs).
    """
    api = HfApi()
    info = api.model_info(repo_id, files_metadata=True)
    for sibling in info.siblings:
        if sibling.rfilename == filename and sibling.lfs:
            return sibling.lfs.sha256
    return None


def download_one(local_rel: str, repo_id: str, repo_filename: str) -> None:
    """Download a single file if missing or hash-mismatched."""
    local_path = MODELS_DIR / local_rel

    # For LFS files, verify SHA-256 against HuggingFace metadata.
    expected_sha = get_expected_sha256(repo_id, repo_filename)

    if local_path.exists() and expected_sha:
        actual = sha256_file(local_path)
        if actual == expected_sha:
            print(f"  {local_rel} — up to date (sha256 verified)")
            return
        print(f"  {local_rel} — hash mismatch, re-downloading...")
    elif local_path.exists():
        # Non-LFS file (small); download to temp and compare content.
        cached = hf_hub_download(repo_id=repo_id, filename=repo_filename)
        if open(cached, "rb").read() == open(local_path, "rb").read():
            print(f"  {local_rel} — up to date")
            return
        print(f"  {local_rel} — content changed, updating...")
    else:
        print(f"  {local_rel} — downloading from {repo_id}...")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo_id, filename=repo_filename)
    shutil.copy2(cached, local_path)
    print(f"  {local_rel} — done")


def main() -> None:
    print("Downloading SD 1.5 models...\n")
    for local_rel, repo_id, repo_filename in FILES:
        download_one(local_rel, repo_id, repo_filename)
    print("\nAll SD 1.5 models ready!")


if __name__ == "__main__":
    main()
