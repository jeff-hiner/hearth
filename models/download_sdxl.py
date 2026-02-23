#!/usr/bin/env python3
"""Download required model files for Stable Diffusion XL inference.

Requires: pip install huggingface-hub safetensors torch

Files downloaded:
  models/checkpoints/sd_xl_base_1.0.safetensors                (~6.9 GB)
  models/vae/sdxl-vae-fp16-fix.safetensors                      (~335 MB)
  models/clip/vocab.json                                         (~940 KB)
  models/clip/merges.txt                                         (~512 KB)
  models/controlnet/controlnet-depth-sdxl-1.0-fp16.safetensors  (~2.5 GB, converted)
"""

import hashlib
import shutil
import sys
from pathlib import Path

# Add scripts/ to path for converter import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    print("Error: huggingface-hub is required. Install with:")
    print("  pip install huggingface-hub")
    sys.exit(1)

MODELS_DIR = Path(__file__).resolve().parent

# (local_relative_path, repo_id, filename_in_repo)
# Note: the VAE repo filename differs from the local name.
FILES = [
    (
        "checkpoints/sd_xl_base_1.0.safetensors",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "sd_xl_base_1.0.safetensors",
    ),
    (
        "vae/sdxl-vae-fp16-fix.safetensors",
        "madebyollin/sdxl-vae-fp16-fix",
        "sdxl_vae.safetensors",
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
        with open(cached, "rb") as a, open(local_path, "rb") as b:
            if a.read() == b.read():
                print(f"  {local_rel} — up to date")
                return
        print(f"  {local_rel} — content changed, updating...")
    else:
        print(f"  {local_rel} — downloading from {repo_id}...")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo_id, filename=repo_filename)
    shutil.copy2(cached, local_path)
    print(f"  {local_rel} — done")


# Files that need diffusers->A1111 conversion after download.
# (local_relative_path, repo_id, filename_in_repo)
CONVERT_FILES = [
    (
        "controlnet/controlnet-depth-sdxl-1.0-fp16.safetensors",
        "diffusers/controlnet-depth-sdxl-1.0",
        "diffusion_pytorch_model.fp16.safetensors",
    ),
]


def download_and_convert(local_rel: str, repo_id: str, repo_filename: str) -> None:
    """Download a diffusers ControlNet and convert to A1111 format."""
    from convert_controlnet_diffusers import convert

    local_path = MODELS_DIR / local_rel
    if local_path.exists():
        print(f"  {local_rel} — already exists (delete to re-convert)")
        return

    print(f"  {local_rel} — downloading from {repo_id}...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo_id, filename=repo_filename)

    print(f"  {local_rel} — converting diffusers -> A1111 format...")
    convert(cached, str(local_path))
    print(f"  {local_rel} — done")


def main() -> None:
    print("Downloading SDXL models...\n")
    for local_rel, repo_id, repo_filename in FILES:
        download_one(local_rel, repo_id, repo_filename)
    for local_rel, repo_id, repo_filename in CONVERT_FILES:
        download_and_convert(local_rel, repo_id, repo_filename)
    print("\nAll SDXL models ready!")


if __name__ == "__main__":
    main()
