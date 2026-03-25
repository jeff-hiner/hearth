# Hearth

A portable Rust inference engine for Stable Diffusion, built on Vulkan.

Hearth lets you run diffusion models on any Vulkan-capable GPU, via a single binary.
No Python, no screwing around with venv or torch versioning, and no CUDA!

By default, the models and flows work in f16 space. This allows the
workflows to work correctly and run at a reasonable speed on most hardware.

The ComfyUI API is *early development.* Many workflows won't work yet, because
the nodes aren't written. Contributions and bug reports are welcome.

## Features

- Runs on **any Vulkan-capable GPU** (NVIDIA, AMD, Intel). Want to run Stable Diffusion
  on your laptop's integrated GPU? You probably can! (Given enough time.)
- **Single binary** means no Python runtime, no dependency management hell
- Supports **both SD 1.5 and SDXL** txt2img with LoRA and ControlNet support
- Runs with the most common samplers (Euler, DPM++ SDE, etc) and schedulers (Normal, Karras)
- Preliminary **img2img support** (still needs polish)
- **ComfyUI and A1111/Forge compatible APIs**

## Requirements

- Rust (latest stable, edition 2024)
- Vulkan-capable GPU with up-to-date drivers

## Model Setup

Models live in `models/` relative to the working directory. I've provided some
helper scripts to download everything from HuggingFace (requires `pip install huggingface-hub`),
or you can just download the models yourself and place them accordingly.

```bash
python models/download_sd15.py   # SD 1.5 (~2.8 GB)
python models/download_sdxl.py   # SDXL   (~9.7 GB)
```

### Required Files

**CLIP tokenizer** (shared by SD 1.5 and SDXL):

| File | Source |
|------|--------|
| `models/clip/vocab.json` | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) |
| `models/clip/merges.txt` | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) |

**SD 1.5**:

| File | Size | Source |
|------|------|--------|
| `models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors` | 2.1 GB | [Comfy-Org/stable-diffusion-v1-5-archive](https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive) |

**SDXL**:

| File | Size | Source |
|------|------|--------|
| `models/checkpoints/sd_xl_base_1.0.safetensors` | 6.9 GB | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| `models/vae/sdxl-vae-fp16-fix.safetensors` | 335 MB | [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) |

**ControlNet** (optional):

| File | Size | Source |
|------|------|--------|
| `models/controlnet/control_v11f1p_sd15_depth_fp16.safetensors` | 723 MB | [comfyanonymous/ControlNet-v1-1_fp16_safetensors](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors) |
| `models/controlnet/controlnet-depth-sdxl-1.0-fp16.safetensors` | 2.5 GB | [diffusers/controlnet-depth-sdxl-1.0](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0) (converted by download script) |

**Depth Anything V2** (optional, for depth estimation):

| File | Source |
|------|--------|
| `models/depth/depth_anything_v2_vits_fp16.safetensors` | [depth-anything/Depth-Anything-V2-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small) |

**LoRA** (optional): place `.safetensors` LoRA files in `models/loras/`. Both
diffusers and ldm key formats are auto-detected.

## Quick Start: Generate an Image

The fastest way to verify your setup is with the generation examples. These
run end-to-end (load models, encode prompt, sample, decode, save PNG) with no
server involved.

```bash
# SD 1.5 requires checkpoint + CLIP tokenizer files (~2.8 GB total)
cargo run --release --example generate -- "a cat sitting on a couch"

# SDXL requires checkpoint + VAE + CLIP tokenizer files (~9.7 GB total)
cargo run --release --example generate_xl -- "a mountain landscape at sunset"
```

The first time you run, the binary has to build Vulkan shaders. Be patient.

Both examples accept options for tuning the output. Run with `--help` for options.

Example with LoRA and ControlNet:

```bash
cargo run --release --example generate -- "studio ghibli style, a forest" \
  --lora models/loras/ghibli.safetensors --lora-strength 0.8 \
  --cn-model models/controlnet/control_v11f1p_sd15_depth_fp16.safetensors \
  --cn-image depth_map.png --cn-weight 0.6
```

## Running the Server

Once generation works, you can run the full inference server:

```bash
cargo run --release
```

This starts two API endpoints:

- ComfyUI API at `127.0.0.1:8188`
- A1111 API at `127.0.0.1:7860`

Point a ComfyUI frontend at `127.0.0.1:8188` or an A1111-compatible client
(e.g. StableProjectorz) at `127.0.0.1:7860`.

## Other Examples

```bash
# Check GPU detection and capabilities
cargo run --example gpu_info

# Estimate depth from an image (requires depth model)
cargo run --release --example estimate_depth -- input.png -o depth.png
```

## Development

```bash
cargo clippy --tests     # Lint (must pass before commits)
cargo test               # Run tests
cargo +nightly fmt       # Format
```

### Shader Caching

Hearth uses two layers of GPU shader caching. When modifying kernel source, **both**
must be cleared or stale shaders will be silently reused:

1. `cargo clean` regenerates compiled cubecl IR
2. Deleting the cubecl pipeline cache forces Vulkan shader recompilation
   - Linux: `~/.local/share/cubecl/pipeline_cache`
   - macOS: `~/Library/Application Support/cubecl/pipeline_cache`
   - Windows: `%LOCALAPPDATA%\cubecl\pipeline_cache`
