//! LoRA (Low-Rank Adaptation) loading and weight merging.
//!
//! Supports Kohya/A1111-format LoRA safetensors files with static weight
//! merging for both UNet and CLIP text encoders (SD 1.5 + SDXL).
//!
//! # Algorithm
//!
//! For each target weight in the model:
//! 1. Compute the LoRA key by stripping the model prefix and replacing `.` with `_`
//! 2. Load `{key}.lora_down.weight` and `{key}.lora_up.weight`
//! 3. Compute `delta = strength * (alpha / rank) * (up @ down)`
//! 4. Add `delta` to the existing weight in-place

use crate::{
    model_loader::{LoadError, SafeTensorsFile, load_tensor_2d, load_tensor_4d},
    types::Backend,
};
use burn::{module::Param, nn::Linear, prelude::*};
use safetensors::SafeTensors;

/// LoRA key naming format.
///
/// Different training tools produce different key naming schemes for the same layers.
/// The model-level `apply_lora` detects this and uses appropriate prefixes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoraFormat {
    /// CompVis/ldm format: `input_blocks.1.1`, `middle_block`, `output_blocks.0.0`
    Ldm,
    /// HuggingFace diffusers format: `down_blocks.0.attentions.0`, `mid_block`, `up_blocks.0`
    Diffusers,
}

/// Detect the LoRA key naming format by inspecting UNet keys.
///
/// Returns `None` if the LoRA contains no UNet keys matching either known format.
pub(crate) fn detect_unet_lora_format(
    lora: &SafeTensors<'_>,
    lora_prefix: &str,
) -> Option<LoraFormat> {
    let diffusers_probe = format!("{lora_prefix}_down_blocks_");
    let ldm_probe = format!("{lora_prefix}_input_blocks_");
    for name in lora.names() {
        if name.starts_with(&diffusers_probe) {
            return Some(LoraFormat::Diffusers);
        }
        if name.starts_with(&ldm_probe) {
            return Some(LoraFormat::Ldm);
        }
    }
    None
}

/// A loaded LoRA safetensors file with memory-mapped access.
pub struct LoraFile {
    /// The underlying safetensors file.
    file: SafeTensorsFile,
}

impl LoraFile {
    /// Open a LoRA safetensors file.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, LoadError> {
        let file = SafeTensorsFile::open(path)?;
        Ok(Self { file })
    }

    /// Parse and return a SafeTensors view.
    pub fn tensors(&self) -> Result<SafeTensors<'_>, LoadError> {
        self.file.tensors()
    }
}

/// Compute the LoRA key base from a LoRA prefix and model-relative path.
///
/// Replaces `.` with `_` in the model path and prepends the LoRA prefix.
///
/// # Example
/// ```ignore
/// lora_key_base("lora_unet", "input_blocks.1.1.transformer_blocks.0.attn1.to_q")
/// // -> "lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q"
/// ```
pub(crate) fn lora_key_base(lora_prefix: &str, model_path: &str) -> String {
    let sanitized = model_path.replace('.', "_");
    format!("{lora_prefix}_{sanitized}")
}

/// Load the LoRA scale factor (alpha / rank) for a given key base.
///
/// If an `.alpha` tensor exists, computes `alpha / rank`. Otherwise returns 1.0
/// (alpha defaults to rank when absent).
fn load_lora_scale(lora: &SafeTensors<'_>, key_base: &str, rank: usize) -> f32 {
    let alpha_key = format!("{key_base}.alpha");
    match lora.tensor(&alpha_key) {
        Ok(view) => {
            // Alpha is stored as a scalar tensor (f32 or f16)
            let data = view.data();
            let alpha = match view.dtype() {
                safetensors::Dtype::F32 => {
                    let bytes: [u8; 4] = data[..4].try_into().expect("alpha too short");
                    f32::from_le_bytes(bytes)
                }
                safetensors::Dtype::F16 => {
                    let bytes: [u8; 2] = data[..2].try_into().expect("alpha too short");
                    half::f16::from_le_bytes(bytes).to_f32()
                }
                safetensors::Dtype::BF16 => {
                    let bytes: [u8; 2] = data[..2].try_into().expect("alpha too short");
                    half::bf16::from_le_bytes(bytes).to_f32()
                }
                _ => rank as f32,
            };
            alpha / rank as f32
        }
        Err(_) => 1.0,
    }
}

/// Apply a LoRA delta to a [`Linear`] layer.
///
/// Loads `{key_base}.lora_down.weight` `[rank, in]` and
/// `{key_base}.lora_up.weight` `[out, rank]`, computes the delta
/// `strength * scale * (up @ down)`, and adds it to the linear weight.
///
/// Burn stores linear weights as `[in, out]` (transposed from PyTorch),
/// so the delta is transposed before addition.
///
/// Returns `true` if LoRA tensors were found and applied.
pub(crate) fn apply_lora_linear(
    linear: &mut Linear<Backend>,
    key_base: &str,
    lora: &SafeTensors<'_>,
    strength: f32,
    device: &Device<Backend>,
) -> Result<bool, LoadError> {
    let down_key = format!("{key_base}.lora_down.weight");
    let up_key = format!("{key_base}.lora_up.weight");

    // Check if this LoRA targets this layer
    if lora.tensor(&down_key).is_err() {
        return Ok(false);
    }

    let down = load_tensor_2d(lora, &down_key, device)?; // [rank, in]
    let up = load_tensor_2d(lora, &up_key, device)?; // [out, rank]

    let rank = down.shape().dims[0];
    let scale = load_lora_scale(lora, key_base, rank);

    // delta = up @ down -> [out, in]
    let delta = up.matmul(down);

    // Burn stores weights as [in, out], so transpose the delta
    let delta = delta.transpose() * (strength * scale);

    // Add delta to existing weight
    let weight = linear.weight.val() + delta;
    linear.weight = Param::from_tensor(weight);

    Ok(true)
}

/// Apply a LoRA delta to a [`Conv2d`](burn::nn::conv::Conv2d) layer.
///
/// For conv LoRA:
/// - `down` is `[rank, in_ch, kH, kW]`
/// - `up` is `[out_ch, rank, 1, 1]`
///
/// The delta is computed as:
/// `up.flatten(rank, out) @ down.flatten(rank, in*kH*kW)` -> `[out, in*kH*kW]`
/// then reshaped to `[out, in, kH, kW]`.
///
/// Conv2d weights in both PyTorch and Burn use `[out, in, kH, kW]` layout,
/// so no transposition is needed.
///
/// Returns `true` if LoRA tensors were found and applied.
pub(crate) fn apply_lora_conv2d(
    conv: &mut burn::nn::conv::Conv2d<Backend>,
    key_base: &str,
    lora: &SafeTensors<'_>,
    strength: f32,
    device: &Device<Backend>,
) -> Result<bool, LoadError> {
    let down_key = format!("{key_base}.lora_down.weight");
    let up_key = format!("{key_base}.lora_up.weight");

    // Check if this LoRA targets this layer
    if lora.tensor(&down_key).is_err() {
        return Ok(false);
    }

    let down = load_tensor_4d(lora, &down_key, device)?; // [rank, in, kH, kW]
    let up = load_tensor_4d(lora, &up_key, device)?; // [out, rank, 1, 1]

    let [rank, in_ch, kh, kw] = down.shape().dims();
    let [out_ch, _, _, _] = up.shape().dims();

    let scale = load_lora_scale(lora, key_base, rank);

    // Flatten for matmul
    let up_2d = up.reshape([out_ch, rank]); // [out, rank]
    let down_2d = down.reshape([rank, in_ch * kh * kw]); // [rank, in*kH*kW]

    // delta = up @ down -> [out, in*kH*kW]
    let delta = up_2d.matmul(down_2d);
    let delta = delta.reshape([out_ch, in_ch, kh, kw]) * (strength * scale);

    // Conv weights are [out, in, kH, kW] in both PyTorch and Burn — no transpose
    let weight = conv.weight.val() + delta;
    conv.weight = Param::from_tensor(weight);

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_base_construction() {
        let key = lora_key_base(
            "lora_unet",
            "input_blocks.1.1.transformer_blocks.0.attn1.to_q",
        );
        assert_eq!(
            key,
            "lora_unet_input_blocks_1_1_transformer_blocks_0_attn1_to_q"
        );
    }

    #[test]
    fn key_base_clip() {
        let key = lora_key_base("lora_te", "text_model.encoder.layers.0.self_attn.q_proj");
        assert_eq!(key, "lora_te_text_model_encoder_layers_0_self_attn_q_proj");
    }
}
