# Test Fixtures

Reference data for validating the Rust VAE decoder against PyTorch/diffusers output.

## Files

- `generate_fixtures.py` - Python script to generate fixtures
- `latent_8x8.npy` - Small test input (shape: [1, 4, 8, 8])
- `expected_8x8.png` - PyTorch reference output (64x64 RGB)
- `latent_64x64.npy` - Full-size test input (shape: [1, 4, 64, 64])
- `expected_64x64.png` - PyTorch reference output (512x512 RGB)

## Generating Fixtures

Requires Python with PyTorch and diffusers:

```bash
pip install torch diffusers numpy pillow safetensors
```

Run with any SD 1.5 checkpoint:

```bash
python generate_fixtures.py --model-path /path/to/sd-v1-5.safetensors
```

The script uses a fixed random seed (42) for reproducibility.

## Testing

The Rust tests load `latent_*.npy`, decode with our VAE implementation, convert to
uint8 PNG format, and compare against `expected_*.png` with ±1 tolerance per channel.

This validates the same f32→u8 conversion path used by the SaveImage node.
