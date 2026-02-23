#!/usr/bin/env python3
"""Reference Depth Anything V2 inference for comparison with Rust implementation."""

import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
from PIL import Image

MODEL_PATH = "models/depth/depth_anything_v2_vits_fp16.safetensors"
IMAGE_PATH = "output/astronaut_f16.png"
RESOLUTION = 518

# Load image and preprocess (must match Rust implementation exactly)
img = Image.open(IMAGE_PATH).convert("RGB")
img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
pixels = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 518, 518]

# ImageNet normalization
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
tensor = (tensor - mean) / std

print(f"Input tensor: {tensor.shape}, dtype={tensor.dtype}")
print(f"Input range: [{tensor.min():.4f}, {tensor.max():.4f}]")

# Load weights
f = safe_open(MODEL_PATH, framework="pt", device="cpu")

def get(name):
    return f.get_tensor(name).float()

# === ENCODER ===
patch_embed_w = get("pretrained.patch_embed.proj.weight")  # [384, 3, 14, 14]
patch_embed_b = get("pretrained.patch_embed.proj.bias")    # [384]
cls_token = get("pretrained.cls_token")  # [1, 1, 384]
pos_embed = get("pretrained.pos_embed")  # [1, 1370, 384]

# Patch embedding
x = F.conv2d(tensor, patch_embed_w, patch_embed_b, stride=14)  # [1, 384, 37, 37]
print(f"After patch_embed: {x.shape}")
x = x.flatten(2).transpose(1, 2)  # [1, 1369, 384]

# Prepend CLS
x = torch.cat([cls_token.expand(1, -1, -1), x], dim=1)  # [1, 1370, 384]
x = x + pos_embed

# Transformer blocks
intermediate_layers = [2, 5, 8, 11]
features = []
for i in range(12):
    prefix = f"pretrained.blocks.{i}"
    # LayerNorm + Attention
    norm1_w = get(f"{prefix}.norm1.weight")
    norm1_b = get(f"{prefix}.norm1.bias")
    ls1 = get(f"{prefix}.ls1.gamma")

    x_norm = F.layer_norm(x, [384], norm1_w, norm1_b, eps=1e-6)

    # Self-attention
    qkv_w = get(f"{prefix}.attn.qkv.weight")
    qkv_b = get(f"{prefix}.attn.qkv.bias")
    proj_w = get(f"{prefix}.attn.proj.weight")
    proj_b = get(f"{prefix}.attn.proj.bias")

    qkv = F.linear(x_norm, qkv_w, qkv_b)
    B, N, _ = qkv.shape
    qkv = qkv.reshape(B, N, 3, 6, 64).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = (q @ k.transpose(-2, -1)) * (64 ** -0.5)
    attn = attn.softmax(dim=-1)
    out = (attn @ v).transpose(1, 2).reshape(B, N, 384)
    out = F.linear(out, proj_w, proj_b)

    x = x + out * ls1

    # LayerNorm + MLP
    norm2_w = get(f"{prefix}.norm2.weight")
    norm2_b = get(f"{prefix}.norm2.bias")
    ls2 = get(f"{prefix}.ls2.gamma")

    x_norm = F.layer_norm(x, [384], norm2_w, norm2_b, eps=1e-6)

    fc1_w = get(f"{prefix}.mlp.fc1.weight")
    fc1_b = get(f"{prefix}.mlp.fc1.bias")
    fc2_w = get(f"{prefix}.mlp.fc2.weight")
    fc2_b = get(f"{prefix}.mlp.fc2.bias")

    mlp_out = F.linear(x_norm, fc1_w, fc1_b)
    mlp_out = F.gelu(mlp_out)
    mlp_out = F.linear(mlp_out, fc2_w, fc2_b)

    x = x + mlp_out * ls2

    if i in intermediate_layers:
        features.append(x.clone())

# Apply final norm to last feature
norm_w = get("pretrained.norm.weight")
norm_b = get("pretrained.norm.bias")
features[-1] = F.layer_norm(features[-1], [384], norm_w, norm_b, eps=1e-6)

for i, feat in enumerate(features):
    print(f"Feature {i}: {feat.shape}, range=[{feat.min():.4f}, {feat.max():.4f}]")

# === DECODER ===
patch_h, patch_w = 37, 37

# Reassemble
layers = []
out_channels = [48, 96, 192, 384]
for i in range(4):
    feat = features[i]
    # Strip CLS
    feat = feat[:, 1:, :]  # [1, 1369, 384]
    # Reshape to spatial
    feat = feat.transpose(1, 2).reshape(1, 384, patch_h, patch_w)
    # Project
    proj_w = get(f"depth_head.projects.{i}.weight")
    proj_b = get(f"depth_head.projects.{i}.bias")
    feat = F.conv2d(feat, proj_w, proj_b)
    # Resize
    if i == 0:
        w = get("depth_head.resize_layers.0.weight")
        b = get("depth_head.resize_layers.0.bias")
        feat = F.conv_transpose2d(feat, w, b, stride=4, padding=0)
    elif i == 1:
        w = get("depth_head.resize_layers.1.weight")
        b = get("depth_head.resize_layers.1.bias")
        feat = F.conv_transpose2d(feat, w, b, stride=2, padding=0)
    elif i == 2:
        pass  # identity
    elif i == 3:
        w = get("depth_head.resize_layers.3.weight")
        b = get("depth_head.resize_layers.3.bias")
        feat = F.conv2d(feat, w, b, stride=2, padding=1)
    layers.append(feat)
    print(f"Reassembled layer {i}: {feat.shape}, range=[{feat.min():.4f}, {feat.max():.4f}]")

# layer_rn
for i in range(4):
    w = get(f"depth_head.scratch.layer{i+1}_rn.weight")
    layers[i] = F.conv2d(layers[i], w, padding=1)
    print(f"layer_{i+1}_rn: {layers[i].shape}, range=[{layers[i].min():.4f}, {layers[i].max():.4f}]")

# RefineNet fusion
def rcu(x, prefix):
    w1 = get(f"{prefix}.conv1.weight")
    b1 = get(f"{prefix}.conv1.bias")
    w2 = get(f"{prefix}.conv2.weight")
    b2 = get(f"{prefix}.conv2.bias")
    out = F.relu(x)
    out = F.conv2d(out, w1, b1, padding=1)
    out = F.relu(out)
    out = F.conv2d(out, w2, b2, padding=1)
    return out + x

def refinenet(path_in, layer_rn, target_size, idx):
    prefix = f"depth_head.scratch.refinenet{idx}"
    if layer_rn is not None:
        res = rcu(layer_rn, f"{prefix}.resConfUnit1")
        output = path_in + res
    else:
        output = path_in
    output = rcu(output, f"{prefix}.resConfUnit2")
    if target_size is not None:
        output = F.interpolate(output, size=target_size, mode="bilinear", align_corners=True)
    else:
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
    # out_conv
    w = get(f"{prefix}.out_conv.weight")
    b = get(f"{prefix}.out_conv.bias")
    output = F.conv2d(output, w, b)
    return output

size_3 = layers[2].shape[2:]
size_2 = layers[1].shape[2:]
size_1 = layers[0].shape[2:]

path_4 = refinenet(layers[3], None, size_3, 4)
print(f"path_4: {path_4.shape}, range=[{path_4.min():.4f}, {path_4.max():.4f}]")
path_3 = refinenet(path_4, layers[2], size_2, 3)
print(f"path_3: {path_3.shape}, range=[{path_3.min():.4f}, {path_3.max():.4f}]")
path_2 = refinenet(path_3, layers[1], size_1, 2)
print(f"path_2: {path_2.shape}, range=[{path_2.min():.4f}, {path_2.max():.4f}]")
path_1 = refinenet(path_2, layers[0], None, 1)
print(f"path_1: {path_1.shape}, range=[{path_1.min():.4f}, {path_1.max():.4f}]")

# Check path_1 ch0 at same positions as Rust
ch0 = path_1[0, 0].detach().numpy()
print(f"path_1 ch0: (0,0)={ch0[0,0]:.4f} (74,74)={ch0[74,74]:.4f} (148,148)={ch0[148,148]:.4f} (220,220)={ch0[220,220]:.4f} (295,295)={ch0[295,295]:.4f}")

# Output head
oc1_w = get("depth_head.scratch.output_conv1.weight")
oc1_b = get("depth_head.scratch.output_conv1.bias")
out = F.conv2d(path_1, oc1_w, oc1_b, padding=1)
print(f"after output_conv1: {out.shape}, range=[{out.min():.4f}, {out.max():.4f}]")

# Detailed output_conv1 diagnostics
oc1_data = out[0, 0].detach().numpy()
print("output_conv1 ch0 diagonal:")
for i in range(0, 296, 37):
    print(f"  ({i},{i}): {oc1_data[i,i]:.4f}")
print(f"output_conv1 ch0 row 148: ", end="")
for c in range(0, 296, 37):
    print(f"({c})={oc1_data[148,c]:.4f} ", end="")
print()

# Upsample to input resolution (patch_h * 14, patch_w * 14) = (518, 518)
out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=True)
print(f"after upsample: {out.shape}")
up_data = out[0, 0].detach().numpy()
print(f"after bilinear ch0 row 100: ", end="")
for c in range(0, 518, 74):
    print(f"({c})={up_data[100,c]:.4f} ", end="")
print()
print(f"after bilinear ch0 row 400: ", end="")
for c in range(0, 518, 74):
    print(f"({c})={up_data[400,c]:.4f} ", end="")
print()

oc2_0_w = get("depth_head.scratch.output_conv2.0.weight")
oc2_0_b = get("depth_head.scratch.output_conv2.0.bias")
out = F.conv2d(out, oc2_0_w, oc2_0_b, padding=1)
out = F.relu(out)

# Detailed after conv2_0+relu diagnostics
c20_data = out[0, 0].detach().numpy()
print(f"after conv2_0+relu ch0 row 100: ", end="")
for c in range(0, 518, 74):
    print(f"({c})={c20_data[100,c]:.4f} ", end="")
print()
print(f"after conv2_0+relu ch0 row 400: ", end="")
for c in range(0, 518, 74):
    print(f"({c})={c20_data[400,c]:.4f} ", end="")
print()

oc2_2_w = get("depth_head.scratch.output_conv2.2.weight")
oc2_2_b = get("depth_head.scratch.output_conv2.2.bias")
print(f"output_conv2.2 bias: {oc2_2_b.item():.6f}")
out = F.conv2d(out, oc2_2_w, oc2_2_b)
out = F.relu(out)

print(f"Final output: {out.shape}, range=[{out.min():.4f}, {out.max():.4f}]")

# Diagonal scan
depth = out[0, 0].detach().numpy()
print("Diagonal scan:")
for i in range(0, 518, 37):
    print(f"  ({i}, {i}): {depth[i, i]:.4f}")

# Save reference output
depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
depth_u16 = (depth_norm * 65535).astype(np.uint16)
Image.fromarray(depth_u16).save("output/astronaut_depth_ref.png")
print("Saved reference depth to output/astronaut_depth_ref.png")
