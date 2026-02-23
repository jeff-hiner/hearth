#!/usr/bin/env python3
"""
Generate SDXL pipeline test fixtures using PyTorch/diffusers.

Creates reference intermediate tensors at each pipeline stage for comparison
against the Rust implementation. All operations run on CPU in float32.

Requirements:
    pip install torch diffusers transformers safetensors numpy

Usage:
    python generate_sdxl_fixtures.py --model-path /path/to/sd_xl_base_1.0.safetensors

Outputs (in same directory as script):
    sdxl_tokens.npy          - [77] int32 tokenized "a cat"
    sdxl_clip_l_hidden.npy   - [1, 77, 768] float32 CLIP-L penultimate layer
    sdxl_clip_g_hidden.npy   - [1, 77, 1280] float32 OpenCLIP-G penultimate layer
    sdxl_clip_g_pooled.npy   - [1, 1280] float32 OpenCLIP-G pooled output
    sdxl_context.npy         - [1, 77, 2048] float32 concatenated context
    sdxl_y.npy               - [1, 2816] float32 vector conditioning
    sdxl_unet_input.npy      - [1, 4, 16, 16] float32 fixed UNet input
    sdxl_unet_output.npy     - [1, 4, 16, 16] float32 UNet output
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch


def timestep_embedding(timesteps, dim, max_period=10000):
    """Fourier embedding matching sgm's timestep_embedding (cos-first ordering).

    Args:
        timesteps: 1D tensor of scalar values
        dim: embedding dimension (must be even)
        max_period: controls frequency range

    Returns:
        [len(timesteps), dim] tensor with [cos, sin] ordering
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def main():
    parser = argparse.ArgumentParser(description="Generate SDXL pipeline test fixtures")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to sd_xl_base_1.0.safetensors",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Output directory for fixtures (default: same as script)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    dtype = torch.float32

    # -------------------------------------------------------------------------
    # Load pipeline from single-file checkpoint
    # -------------------------------------------------------------------------
    print(f"Loading SDXL pipeline from {args.model_path}...")
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_single_file(
        str(args.model_path),
        torch_dtype=dtype,
    ).to(device)

    # Extract components
    tokenizer_l = pipe.tokenizer        # CLIPTokenizer for CLIP-L
    tokenizer_g = pipe.tokenizer_2      # CLIPTokenizer for OpenCLIP-G
    text_encoder_l = pipe.text_encoder   # CLIPTextModel (CLIP-L)
    text_encoder_g = pipe.text_encoder_2 # CLIPTextModelWithProjection (OpenCLIP-G)
    unet = pipe.unet

    text_encoder_l.eval()
    text_encoder_g.eval()
    unet.eval()

    prompt = "a cat"

    # -------------------------------------------------------------------------
    # Stage 1: Tokenize
    # -------------------------------------------------------------------------
    print("\n--- Stage 1: Tokenization ---")

    # CLIP-L tokenizer (same vocab as OpenCLIP-G for SDXL)
    tokens_l = tokenizer_l(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    token_ids = tokens_l.input_ids[0]  # [77]

    print(f"  Tokens (first 10): {token_ids[:10].tolist()}")
    print(f"  Token shape: {token_ids.shape}")

    # Save as int32 for easy loading in Rust
    np.save(output_dir / "sdxl_tokens.npy", token_ids.numpy().astype(np.int32))
    print(f"  Saved sdxl_tokens.npy")

    # Also tokenize with G tokenizer to check if they differ
    tokens_g = tokenizer_g(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    token_ids_g = tokens_g.input_ids[0]
    if not torch.equal(token_ids, token_ids_g):
        print(f"  WARNING: CLIP-L and OpenCLIP-G tokens differ!")
        print(f"    CLIP-L: {token_ids[:10].tolist()}")
        print(f"    CLIP-G: {token_ids_g[:10].tolist()}")
        np.save(output_dir / "sdxl_tokens_g.npy", token_ids_g.numpy().astype(np.int32))
    else:
        print(f"  CLIP-L and OpenCLIP-G tokens match")

    # -------------------------------------------------------------------------
    # Stage 2: CLIP-L hidden states (penultimate layer, no final_layer_norm)
    # -------------------------------------------------------------------------
    print("\n--- Stage 2: CLIP-L hidden states ---")

    with torch.no_grad():
        clip_l_out = text_encoder_l(
            tokens_l.input_ids.to(device),
            output_hidden_states=True,
        )
        # Penultimate layer = hidden_states[-2] (layer 10 of 12)
        clip_l_hidden = clip_l_out.hidden_states[-2]  # [1, 77, 768]

    print(f"  Shape: {clip_l_hidden.shape}")
    print(f"  [0,0,:5]: {clip_l_hidden[0, 0, :5].tolist()}")
    print(f"  Mean: {clip_l_hidden.mean().item():.6f}")
    print(f"  Abs max: {clip_l_hidden.abs().max().item():.6f}")

    np.save(output_dir / "sdxl_clip_l_hidden.npy", clip_l_hidden.numpy())
    print(f"  Saved sdxl_clip_l_hidden.npy")

    # -------------------------------------------------------------------------
    # Stage 3: OpenCLIP-G hidden states + pooled output
    # -------------------------------------------------------------------------
    print("\n--- Stage 3: OpenCLIP-G hidden states + pooled ---")

    with torch.no_grad():
        # Use G tokenizer's tokens
        clip_g_out = text_encoder_g(
            tokens_g.input_ids.to(device),
            output_hidden_states=True,
        )
        # Penultimate layer = hidden_states[-2] (layer 30 of 32)
        clip_g_hidden = clip_g_out.hidden_states[-2]  # [1, 77, 1280]
        # Pooled output (already projected through text_projection by diffusers)
        clip_g_pooled = clip_g_out.text_embeds  # [1, 1280]

    print(f"  Hidden shape: {clip_g_hidden.shape}")
    print(f"  Hidden [0,0,:5]: {clip_g_hidden[0, 0, :5].tolist()}")
    print(f"  Hidden mean: {clip_g_hidden.mean().item():.6f}")
    print(f"  Pooled shape: {clip_g_pooled.shape}")
    print(f"  Pooled [:5]: {clip_g_pooled[0, :5].tolist()}")

    np.save(output_dir / "sdxl_clip_g_hidden.npy", clip_g_hidden.numpy())
    np.save(output_dir / "sdxl_clip_g_pooled.npy", clip_g_pooled.numpy())
    print(f"  Saved sdxl_clip_g_hidden.npy, sdxl_clip_g_pooled.npy")

    # -------------------------------------------------------------------------
    # Stage 4: Build conditioning vectors
    # -------------------------------------------------------------------------
    print("\n--- Stage 4: Conditioning vectors ---")

    # Context = cat(clip_l_hidden, clip_g_hidden) along feature dim
    context = torch.cat([clip_l_hidden, clip_g_hidden], dim=-1)  # [1, 77, 2048]
    print(f"  Context shape: {context.shape}")
    print(f"  Context [0,0,:5]: {context[0, 0, :5].tolist()}")

    # y = cat(pooled, fourier_embeds)
    # Fourier embeddings for: original_size=(1024,1024), crop=(0,0), target=(1024,1024)
    scalars = torch.tensor([1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0])
    fourier_parts = []
    for s in scalars:
        emb = timestep_embedding(s.unsqueeze(0), 256)  # [1, 256]
        fourier_parts.append(emb)

    fourier_cat = torch.cat(fourier_parts, dim=-1)  # [1, 1536]
    y = torch.cat([clip_g_pooled, fourier_cat], dim=-1)  # [1, 2816]

    print(f"  y shape: {y.shape}")
    print(f"  y (pooled part) [:5]: {y[0, :5].tolist()}")
    print(f"  y (fourier part) [1280:1285]: {y[0, 1280:1285].tolist()}")

    np.save(output_dir / "sdxl_context.npy", context.numpy())
    np.save(output_dir / "sdxl_y.npy", y.numpy())
    print(f"  Saved sdxl_context.npy, sdxl_y.npy")

    # -------------------------------------------------------------------------
    # Stage 5: Single UNet forward pass with fixed inputs
    # -------------------------------------------------------------------------
    print("\n--- Stage 5: UNet forward pass ---")

    # Fixed deterministic input
    unet_input = torch.ones(1, 4, 16, 16) * 0.5
    timestep_val = torch.tensor([500.0])

    # SDXL UNet expects:
    # - sample: [B, 4, H, W]
    # - timestep: scalar or [B]
    # - encoder_hidden_states: [B, 77, 2048]  (context)
    # - added_cond_kwargs: {"text_embeds": [B, 1280], "time_ids": [B, 6]}
    #
    # time_ids encodes: [original_h, original_w, crop_top, crop_left, target_h, target_w]
    time_ids = torch.tensor([[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]])

    print(f"  Input shape: {unet_input.shape}")
    print(f"  Timestep: {timestep_val.item()}")

    with torch.no_grad():
        unet_output = unet(
            unet_input.to(device),
            timestep_val.to(device),
            encoder_hidden_states=context.to(device),
            added_cond_kwargs={
                "text_embeds": clip_g_pooled.to(device),
                "time_ids": time_ids.to(device),
            },
        ).sample

    print(f"  Output shape: {unet_output.shape}")
    print(f"  Output [0,0,0,:5]: {unet_output[0, 0, 0, :5].tolist()}")
    print(f"  Output mean: {unet_output.mean().item():.6f}")
    print(f"  Output abs max: {unet_output.abs().max().item():.6f}")
    has_nan = torch.isnan(unet_output).any().item()
    has_inf = torch.isinf(unet_output).any().item()
    print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")

    np.save(output_dir / "sdxl_unet_input.npy", unet_input.numpy())
    np.save(output_dir / "sdxl_unet_output.npy", unet_output.numpy())
    print(f"  Saved sdxl_unet_input.npy, sdxl_unet_output.npy")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n=== Summary ===")
    files = [
        "sdxl_tokens.npy",
        "sdxl_clip_l_hidden.npy",
        "sdxl_clip_g_hidden.npy",
        "sdxl_clip_g_pooled.npy",
        "sdxl_context.npy",
        "sdxl_y.npy",
        "sdxl_unet_input.npy",
        "sdxl_unet_output.npy",
    ]
    for f in files:
        p = output_dir / f
        if p.exists():
            arr = np.load(p)
            print(f"  {f}: shape={arr.shape}, dtype={arr.dtype}")

    print("\nDone! Run comparison with:")
    print("  cargo run --release --example compare_sdxl")


if __name__ == "__main__":
    main()
