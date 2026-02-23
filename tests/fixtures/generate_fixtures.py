#!/usr/bin/env python3
"""
Generate VAE decoder test fixtures using PyTorch/diffusers.

This script creates reference data for testing the Rust VAE decoder implementation.
It generates random latent tensors, decodes them with the diffusers VAE, and saves
both the inputs (.npy) and expected outputs (.png).

Requirements:
    pip install torch diffusers numpy pillow safetensors

Usage:
    python generate_fixtures.py --model-path /path/to/sd-v1-5.safetensors

The model can be any SD 1.5 checkpoint in safetensors format (e.g., from civitai
or huggingface). Only the VAE weights are used.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def convert_sd_vae_key_to_diffusers(key: str) -> str:
    """Convert a Stable Diffusion VAE key to diffusers format.

    SD checkpoints use a different naming convention than diffusers:
    - SD: decoder.mid.block_1.norm1.weight
    - diffusers: decoder.mid_block.resnets.0.norm1.weight

    Also, SD up blocks are in reverse order:
    - SD up.0 = final block (128 ch) -> diffusers up_blocks.3
    - SD up.3 = first block (512 ch) -> diffusers up_blocks.0
    """
    import re

    # Mid block resnets: mid.block_1 -> mid_block.resnets.0
    # Note: SD uses 1-indexed, diffusers uses 0-indexed
    key = re.sub(r"\.mid\.block_(\d+)\.", lambda m: f".mid_block.resnets.{int(m.group(1)) - 1}.", key)

    # Mid block attention: mid.attn_1 -> mid_block.attentions.0
    key = re.sub(r"\.mid\.attn_(\d+)\.", lambda m: f".mid_block.attentions.{int(m.group(1)) - 1}.", key)

    # Up blocks: up.X -> up_blocks.(3-X) (reversed order)
    # SD has 4 up blocks numbered 0-3, with 0 being the final (smallest) block
    key = re.sub(r"\.up\.(\d+)\.block\.(\d+)\.", lambda m: f".up_blocks.{3 - int(m.group(1))}.resnets.{m.group(2)}.", key)
    key = re.sub(r"\.up\.(\d+)\.upsample\.", lambda m: f".up_blocks.{3 - int(m.group(1))}.upsamplers.0.", key)

    # Down blocks: down.X.block.Y -> down_blocks.X.resnets.Y (same order)
    key = re.sub(r"\.down\.(\d+)\.block\.(\d+)\.", r".down_blocks.\1.resnets.\2.", key)
    key = re.sub(r"\.down\.(\d+)\.downsample\.", r".down_blocks.\1.downsamplers.0.", key)

    # Attention projections: q/k/v/proj_out -> to_q/to_k/to_v/to_out.0
    key = re.sub(r"\.q\.(weight|bias)$", r".to_q.\1", key)
    key = re.sub(r"\.k\.(weight|bias)$", r".to_k.\1", key)
    key = re.sub(r"\.v\.(weight|bias)$", r".to_v.\1", key)
    key = re.sub(r"\.proj_out\.(weight|bias)$", r".to_out.0.\1", key)

    # Attention norm: norm -> group_norm
    key = re.sub(r"(\.attentions\.\d+)\.norm\.", r"\1.group_norm.", key)

    # Skip connection: nin_shortcut -> conv_shortcut
    key = key.replace(".nin_shortcut.", ".conv_shortcut.")

    # Output norm: norm_out -> conv_norm_out
    key = key.replace(".norm_out.", ".conv_norm_out.")

    return key


def convert_sd_vae_state_dict(state_dict: dict) -> dict:
    """Convert an SD VAE state dict to diffusers format.

    This handles both key renaming and tensor reshaping (attention weights
    are 4D Conv2d in SD but 2D Linear in diffusers).
    """
    converted = {}
    for key, value in state_dict.items():
        new_key = convert_sd_vae_key_to_diffusers(key)

        # Attention weights need to be reshaped from [out, in, 1, 1] to [out, in]
        if ".to_q.weight" in new_key or ".to_k.weight" in new_key or ".to_v.weight" in new_key or ".to_out.0.weight" in new_key:
            if value.dim() == 4 and value.shape[2] == 1 and value.shape[3] == 1:
                value = value.squeeze(-1).squeeze(-1)

        converted[new_key] = value
    return converted


def load_vae_from_checkpoint(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load VAE decoder from a full SD checkpoint."""
    from diffusers import AutoencoderKL
    from safetensors.torch import load_file

    # Load the full checkpoint
    state_dict = load_file(checkpoint_path)

    # Extract VAE weights (they have "first_stage_model." prefix in full checkpoints)
    vae_state_dict = {}
    prefix = "first_stage_model."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            vae_state_dict[new_key] = value

    # If no first_stage_model prefix found, try vae prefix or assume it's a VAE-only checkpoint
    if not vae_state_dict:
        prefix = "vae."
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                vae_state_dict[new_key] = value

    if not vae_state_dict:
        # Assume it's already a VAE-only checkpoint
        vae_state_dict = state_dict

    # Convert SD state dict to diffusers format (key names and tensor shapes)
    converted_state_dict = convert_sd_vae_state_dict(vae_state_dict)

    # Create VAE with SD 1.5 config
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        act_fn="silu",
        latent_channels=4,
        norm_num_groups=32,
        sample_size=512,
        scaling_factor=0.18215,
    )

    # Load weights - use strict=True now that we have proper key mapping
    missing, unexpected = vae.load_state_dict(converted_state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys in VAE: {missing[:5]}..." if len(missing) > 5 else f"Warning: Missing keys: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys in VAE: {unexpected[:5]}..." if len(unexpected) > 5 else f"Warning: Unexpected keys: {unexpected}")

    vae = vae.to(device)
    vae.eval()

    # Patch diffusers ResNet blocks to match ComfyUI/original SD behavior (no sqrt(2) scaling)
    # Diffusers adds output_scale_factor=sqrt(2) which ComfyUI doesn't have
    patch_resnet_scale_factor(vae)

    return vae


def patch_resnet_scale_factor(vae: torch.nn.Module) -> None:
    """Patch all ResNet blocks to use output_scale_factor=1.0.

    Diffusers uses output_scale_factor=sqrt(2) for up_blocks, but ComfyUI and the
    original LDM/SD code don't have this scaling. We patch it out to match ComfyUI.
    """
    for module in vae.modules():
        if hasattr(module, "output_scale_factor"):
            module.output_scale_factor = 1.0


def generate_deterministic_latent(height: int, width: int, seed: int = 42) -> np.ndarray:
    """Generate a deterministic random latent tensor."""
    rng = np.random.default_rng(seed)
    # Latent shape: [1, 4, height, width] (batch=1, channels=4)
    latent = rng.standard_normal((1, 4, height, width)).astype(np.float32)
    return latent


def decode_latent(vae: torch.nn.Module, latent: np.ndarray, device: torch.device) -> np.ndarray:
    """Decode a latent tensor to RGB image using the VAE."""
    with torch.no_grad():
        latent_tensor = torch.from_numpy(latent).to(device)

        # Scale latents before decoding (diffusers vae.decode does NOT scale internally)
        # This matches what our Rust VaeDecoder.forward() does
        latent_tensor = latent_tensor / 0.18215

        decoded = vae.decode(latent_tensor).sample

        # Convert to numpy, clamp to [0, 1], then to uint8
        decoded = decoded.cpu().numpy()
        decoded = np.clip(decoded, -1.0, 1.0)
        # VAE output is in [-1, 1], convert to [0, 1]
        decoded = (decoded + 1.0) / 2.0

        return decoded


def save_as_png(tensor: np.ndarray, path: Path) -> None:
    """Save a [1, 3, H, W] float tensor as PNG."""
    # Remove batch dimension: [3, H, W]
    img = tensor[0]
    # Transpose to [H, W, 3]
    img = np.transpose(img, (1, 2, 0))
    # Convert to uint8
    img = (img * 255.0).astype(np.uint8)
    # Save
    Image.fromarray(img).save(path)


def main():
    parser = argparse.ArgumentParser(description="Generate VAE decoder test fixtures")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to SD 1.5 checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory for fixtures (default: same as script)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for latent generation (default: 42)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading VAE from {args.model_path}...")
    vae = load_vae_from_checkpoint(args.model_path, device)

    # Generate fixtures at two sizes:
    # 1. 8x8 latent -> 64x64 image (small, fast to test)
    # 2. 64x64 latent -> 512x512 image (full size)
    sizes = [
        (8, 8, "8x8"),
        (64, 64, "64x64"),
    ]

    for height, width, name in sizes:
        print(f"\nGenerating {name} fixtures...")

        # Generate deterministic latent
        latent = generate_deterministic_latent(height, width, seed=args.seed)

        # Save latent as .npy
        latent_path = output_dir / f"latent_{name}.npy"
        np.save(latent_path, latent)
        print(f"  Saved latent to {latent_path} (shape: {latent.shape})")

        # Decode and save as PNG
        decoded = decode_latent(vae, latent, device)
        png_path = output_dir / f"expected_{name}.png"
        save_as_png(decoded, png_path)
        output_height, output_width = decoded.shape[2], decoded.shape[3]
        print(f"  Saved expected output to {png_path} (size: {output_width}x{output_height})")

    print("\nDone! Fixtures generated successfully.")
    print("\nTo run the Rust tests against these fixtures:")
    print("  cargo test vae_decoder")


if __name__ == "__main__":
    main()
