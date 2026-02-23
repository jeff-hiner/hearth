#!/usr/bin/env python3
"""Convert a diffusers-format ControlNet safetensors to A1111/ComfyUI format.

Diffusers ControlNet checkpoints use a different key naming convention than the
A1111/ldm format that hearth expects (control_model.* prefix). This script
performs a mechanical key rename + optional weight reshape.

Supports both SD 1.5 and SDXL ControlNets. Auto-detects the variant from the
checkpoint contents.

Usage:
    python scripts/convert_controlnet_diffusers.py input.safetensors output.safetensors

Requires: pip install safetensors torch
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

try:
    import safetensors.torch as st
    import torch
except ImportError:
    print("Error: safetensors and torch are required. Install with:")
    print("  pip install safetensors torch")
    sys.exit(1)

# Architecture configs: (channel_mult, layers_per_block, transformer_depth)
SD15_CONFIG = ([1, 2, 4, 4], 2, [1, 1, 1, 0])
SDXL_CONFIG = ([1, 2, 4], 2, [0, 2, 10])


def detect_variant(state_dict: dict) -> str:
    """Detect SD 1.5 vs SDXL from the checkpoint keys."""
    if "add_embedding.linear_1.weight" in state_dict:
        return "sdxl"
    return "sd15"


def build_input_block_map(channel_mult, layers_per_block, transformer_depth):
    """Build the flat input_blocks index mapping.

    Returns dict: (level, kind, layer_in_level) -> flat_index
    where kind is 'resnet', 'attn', or 'downsample'.
    """
    mapping = {}
    flat_idx = 1  # 0 is conv_in

    for level, mult in enumerate(channel_mult):
        has_attn = transformer_depth[level] > 0
        is_last = level == len(channel_mult) - 1

        for layer in range(layers_per_block):
            mapping[("resnet", level, layer)] = flat_idx
            if has_attn:
                mapping[("attn", level, layer)] = flat_idx
            flat_idx += 1

        if not is_last:
            mapping[("downsample", level, 0)] = flat_idx
            flat_idx += 1

    return mapping


# ResNet sub-key mapping: diffusers -> A1111
RESNET_KEY_MAP = {
    "norm1": "in_layers.0",
    "conv1": "in_layers.2",
    "norm2": "out_layers.0",
    "conv2": "out_layers.3",
    "time_emb_proj": "emb_layers.1",
    "conv_shortcut": "skip_connection",
}

# Mid block sub-key mapping for resnets
MID_RESNET_MAP = {
    "mid_block.resnets.0": "control_model.middle_block.0",
    "mid_block.resnets.1": "control_model.middle_block.2",
}

# Mid block attention
MID_ATTN_MAP = {
    "mid_block.attentions.0": "control_model.middle_block.1",
}


def convert_resnet_key(suffix: str) -> str:
    """Convert a resnet sub-key from diffusers to A1111 format."""
    for diffusers_sub, a1111_sub in RESNET_KEY_MAP.items():
        if suffix.startswith(diffusers_sub):
            rest = suffix[len(diffusers_sub):]
            return a1111_sub + rest
    return suffix  # passthrough (shouldn't happen)


def convert_key(key: str, block_map: dict, is_sd15: bool) -> str:
    """Convert a single diffusers key to A1111 format."""
    prefix = "control_model"

    # conv_in
    if key.startswith("conv_in."):
        return f"{prefix}.input_blocks.0.0.{key[len('conv_in.'):]}"

    # Time embedding
    if key.startswith("time_embedding.linear_1."):
        return f"{prefix}.time_embed.0.{key[len('time_embedding.linear_1.'):]}"
    if key.startswith("time_embedding.linear_2."):
        return f"{prefix}.time_embed.2.{key[len('time_embedding.linear_2.'):]}"

    # SDXL add_embedding (label_emb)
    if key.startswith("add_embedding.linear_1."):
        return f"{prefix}.label_emb.0.0.{key[len('add_embedding.linear_1.'):]}"
    if key.startswith("add_embedding.linear_2."):
        return f"{prefix}.label_emb.0.2.{key[len('add_embedding.linear_2.'):]}"

    # Hint encoder
    if key.startswith("controlnet_cond_embedding.conv_in."):
        return f"{prefix}.input_hint_block.0.{key[len('controlnet_cond_embedding.conv_in.'):]}"
    if key.startswith("controlnet_cond_embedding.blocks."):
        rest = key[len("controlnet_cond_embedding.blocks."):]
        block_idx = int(rest.split(".")[0])
        sub = rest[len(str(block_idx)) + 1:]
        # blocks.0 -> hint_block.2, blocks.1 -> 4, etc.
        a1111_idx = (block_idx + 1) * 2
        return f"{prefix}.input_hint_block.{a1111_idx}.{sub}"
    if key.startswith("controlnet_cond_embedding.conv_out."):
        return f"{prefix}.input_hint_block.14.{key[len('controlnet_cond_embedding.conv_out.'):]}"

    # Zero convolutions
    if key.startswith("controlnet_down_blocks."):
        rest = key[len("controlnet_down_blocks."):]
        idx = int(rest.split(".")[0])
        sub = rest[len(str(idx)) + 1:]
        return f"{prefix}.zero_convs.{idx}.0.{sub}"
    if key.startswith("controlnet_mid_block."):
        return f"{prefix}.middle_block_out.0.{key[len('controlnet_mid_block.'):]}"

    # Down blocks -> input_blocks
    if key.startswith("down_blocks."):
        rest = key[len("down_blocks."):]
        parts = rest.split(".", 3)  # level.kind.layer_idx.sub_key
        level = int(parts[0])
        kind = parts[1]

        if kind == "resnets":
            layer = int(parts[2])
            sub = parts[3]
            flat_idx = block_map[("resnet", level, layer)]
            converted_sub = convert_resnet_key(sub)
            return f"{prefix}.input_blocks.{flat_idx}.0.{converted_sub}"
        elif kind == "attentions":
            layer = int(parts[2])
            sub = parts[3]
            flat_idx = block_map[("attn", level, layer)]
            return f"{prefix}.input_blocks.{flat_idx}.1.{sub}"
        elif kind == "downsamplers":
            # downsamplers.0.conv.{w,b}
            sub = parts[3]  # "conv.weight" or "conv.bias"
            assert sub.startswith("conv."), f"unexpected downsample key: {key}"
            wb = sub[len("conv."):]
            flat_idx = block_map[("downsample", level, 0)]
            return f"{prefix}.input_blocks.{flat_idx}.0.op.{wb}"

    # Mid block resnets
    for diff_prefix, a1111_prefix in MID_RESNET_MAP.items():
        if key.startswith(diff_prefix + "."):
            rest = key[len(diff_prefix) + 1:]
            converted_sub = convert_resnet_key(rest)
            return f"{a1111_prefix}.{converted_sub}"

    # Mid block attention
    for diff_prefix, a1111_prefix in MID_ATTN_MAP.items():
        if key.startswith(diff_prefix + "."):
            rest = key[len(diff_prefix) + 1:]
            return f"{a1111_prefix}.{rest}"

    raise ValueError(f"Unknown key: {key}")


def maybe_reshape_proj(key: str, tensor: torch.Tensor, is_sd15: bool) -> torch.Tensor:
    """Reshape proj_in/proj_out from Linear [out, in] to Conv1x1 [out, in, 1, 1].

    Only needed for SD 1.5 where the A1111 format uses Conv1x1.
    SDXL uses Linear in both formats.
    """
    if not is_sd15:
        return tensor
    if ("proj_in.weight" in key or "proj_out.weight" in key) and tensor.dim() == 2:
        return tensor.unsqueeze(-1).unsqueeze(-1)
    return tensor


def convert(input_path: str, output_path: str) -> None:
    """Convert a diffusers ControlNet to A1111 format."""
    print(f"Loading {input_path}...")
    state_dict = st.load_file(input_path)

    variant = detect_variant(state_dict)
    is_sd15 = variant == "sd15"
    config = SD15_CONFIG if is_sd15 else SDXL_CONFIG
    channel_mult, layers_per_block, transformer_depth = config
    print(f"Detected variant: {variant.upper()}")
    print(f"  channel_mult={channel_mult}, layers={layers_per_block}, "
          f"transformer_depth={transformer_depth}")

    block_map = build_input_block_map(channel_mult, layers_per_block, transformer_depth)

    converted = OrderedDict()
    for key in sorted(state_dict.keys()):
        new_key = convert_key(key, block_map, is_sd15)
        tensor = maybe_reshape_proj(new_key, state_dict[key], is_sd15)
        converted[new_key] = tensor

    print(f"Converted {len(converted)} keys")
    print(f"Saving to {output_path}...")
    st.save_file(converted, output_path)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert diffusers ControlNet safetensors to A1111/ComfyUI format"
    )
    parser.add_argument("input", help="Input diffusers safetensors file")
    parser.add_argument("output", help="Output A1111-format safetensors file")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: {args.input} not found")
        sys.exit(1)

    convert(args.input, args.output)


if __name__ == "__main__":
    main()
