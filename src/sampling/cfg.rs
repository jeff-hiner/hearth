//! Shared classifier-free guidance helper.

use crate::{
    controlnet::{ControlNetOutput, ControlNetUnit},
    types::Backend,
    unet::{DenoisingUnet, UnetConditioning, UnetConfig},
};
use burn::prelude::*;

/// Predict noise with classifier-free guidance using batched inference.
///
/// Combines positive and negative conditioning into a single batched UNet call
/// to maximize GPU utilization.
///
/// # Arguments
/// * `unet` - The UNet denoiser model
/// * `scheduler_scale` - Scaling factor from `NoiseSchedule::scale_model_input`
/// * `latent` - Current noisy latent `[B, 4, H, W]` (unscaled)
/// * `timestep` - Current UNet timestep
/// * `cond_batched` - Pre-batched positive+negative conditioning
/// * `cfg_scale` - Classifier-free guidance scale
pub(super) fn predict_noise_cfg_batched<U: DenoisingUnet>(
    unet: &U,
    scheduler_scale: f32,
    latent: &Tensor<Backend, 4>,
    timestep: &Tensor<Backend, 1>,
    cond_batched: &U::Cond,
    cfg_scale: f32,
) -> Tensor<Backend, 4> {
    let latent_scaled = latent.clone() * scheduler_scale;

    // Batch latent: [B, 4, H, W] -> [2*B, 4, H, W]
    // Zero-copy when B=1: just sets stride[0]=0 and doubles shape[0]
    let latent_batched = latent_scaled.repeat_dim(0, 2);

    // Single batched UNet forward pass
    let noise_batched = unet.forward(latent_batched, timestep.clone(), cond_batched);

    // Split result: [2*B, 4, H, W] -> ([B, 4, H, W], [B, 4, H, W])
    let [noise_pos, noise_neg]: [Tensor<Backend, 4>; 2] = noise_batched
        .chunk(2, 0)
        .try_into()
        .expect("chunk returned wrong count");

    // CFG: noise = noise_neg + cfg_scale * (noise_pos - noise_neg)
    //            = noise_neg * (1 - cfg_scale) + noise_pos * cfg_scale
    noise_neg * (1.0 - cfg_scale) + noise_pos * cfg_scale
}

/// Predict noise with CFG and ControlNet residuals.
///
/// Runs active ControlNets with positive conditioning, sums their weighted
/// residuals, duplicates for CFG batch, then runs the UNet with residuals.
///
/// # Arguments
/// * `unet` - The UNet model (not trait — needs `compute_emb` + `forward_with_emb`)
/// * `controlnets` - ControlNet units with hint images and settings
/// * `scheduler_scale` - Scaling factor from `NoiseSchedule::scale_model_input`
/// * `latent` - Current noisy latent `[B, 4, H, W]` (unscaled)
/// * `timestep` - Current UNet timestep
/// * `positive` - Positive conditioning
/// * `negative` - Negative conditioning
/// * `cfg_scale` - Classifier-free guidance scale
/// * `step` - Current sampling step index
/// * `num_steps` - Total number of sampling steps
#[expect(
    clippy::too_many_arguments,
    reason = "controlled CFG needs all parameters"
)]
pub(super) fn predict_noise_cfg_controlled<C: UnetConfig, Cond: UnetConditioning>(
    unet: &crate::unet::Unet<C>,
    controlnets: &[ControlNetUnit<'_, C>],
    scheduler_scale: f32,
    latent: &Tensor<Backend, 4>,
    timestep: f32,
    positive: &Cond,
    negative: &Cond,
    cfg_scale: f32,
    step: usize,
    num_steps: usize,
) -> Tensor<Backend, 4> {
    let latent_scaled = latent.clone() * scheduler_scale;
    let device = latent_scaled.device();
    let [batch, _, _, _] = latent_scaled.shape().dims();

    let context_pos = positive.context();
    let context_neg = negative.context();
    let y_pos = positive.y();
    let y_neg = negative.y();

    // Compute time embedding once (shared between ControlNets and UNet)
    let timestep_1: Tensor<Backend, 1> = Tensor::full([batch], timestep, &device);
    let emb = unet.compute_emb(timestep_1, y_pos);

    // Run active ControlNets with positive conditioning
    let mut cn_outputs = Vec::new();
    for cn in controlnets {
        if !cn.is_active(step, num_steps) {
            continue;
        }

        let cn_timestep: Tensor<Backend, 1> = Tensor::full([batch], timestep, &device);
        let output = cn.model.forward(
            latent_scaled.clone(),
            &cn.hint,
            cn_timestep,
            context_pos,
            y_pos,
        );
        cn_outputs.push(output.scale(cn.weight));
    }

    // Sum all ControlNet residuals
    let residuals = ControlNetOutput::sum(cn_outputs);

    // Duplicate residuals for CFG batch (positive + negative both get residuals)
    // Zero-copy when B=1: repeat_dim just sets stride[0]=0
    let batched_residuals = residuals.map(|r| {
        let down_residuals = r
            .down_residuals
            .into_iter()
            .map(|t| t.repeat_dim(0, 2))
            .collect();
        let mid_residual = r.mid_residual.repeat_dim(0, 2);
        ControlNetOutput {
            down_residuals,
            mid_residual,
        }
    });

    // Batch latent and embeddings for CFG
    let latent_batched = latent_scaled.repeat_dim(0, 2);
    let emb_neg = unet.compute_emb(Tensor::full([batch], timestep, &device), y_neg);
    let emb_batched = Tensor::cat(vec![emb, emb_neg], 0);
    let context_batched = Tensor::cat(vec![context_pos.clone(), context_neg.clone()], 0);

    // UNet forward with residuals
    let noise_batched = unet.forward_with_emb(
        latent_batched,
        emb_batched,
        &context_batched,
        batched_residuals.as_ref(),
    );

    // Split and CFG combine
    let [noise_pos, noise_neg]: [Tensor<Backend, 4>; 2] = noise_batched
        .chunk(2, 0)
        .try_into()
        .expect("chunk returned wrong count");

    noise_neg * (1.0 - cfg_scale) + noise_pos * cfg_scale
}
