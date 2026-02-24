//! `KSamplerAdvanced` node — runs partial diffusion sampling with explicit step control.

use super::ksampler::{
    build_sd15_units, build_sdxl_units, extract_sd15_conditioning, extract_sdxl_conditioning,
    sample_sd15, sample_sd15_controlled, sample_sdxl, sample_sdxl_controlled,
};
use crate::{
    node::{
        Node, ResolvedInput, SlotDef, ValueType, context::ExecutionContext, error::NodeError,
        value::NodeValue, variant::UnetVariant,
    },
    sampling::{DiffusionSchedule, NoiseSchedule, SamplerKind, SchedulerKind},
    types::{Backend, ConditioningValue, ControlNetRef, Latent},
};
use burn::tensor::{Distribution, Tensor};
use std::sync::atomic::Ordering;

/// Runs denoising diffusion sampling with explicit step range control.
///
/// ComfyUI equivalent: `KSamplerAdvanced`
///
/// Unlike [`KSampler`](super::KSampler), this node replaces the `denoise` parameter
/// with `add_noise`, `start_at_step`, and `end_at_step`, allowing partial denoising
/// ranges for multi-pass workflows (e.g. SDXL base+refiner, img2img).
#[derive(Debug)]
pub(crate) struct KSamplerAdvanced {
    /// Whether to inject initial noise before sampling.
    add_noise: bool,
    /// First step to execute (0-indexed).
    start_at_step: usize,
    /// Step to stop at (exclusive; 0 = run all steps).
    end_at_step: usize,
    /// Random seed.
    seed: u64,
    /// Number of sampling steps.
    steps: usize,
    /// Classifier-free guidance scale.
    cfg: f32,
    /// Sampling algorithm.
    sampler_name: SamplerKind,
    /// Sigma schedule.
    scheduler: SchedulerKind,
}

impl KSamplerAdvanced {
    /// Create a new `KSamplerAdvanced` node.
    #[expect(clippy::too_many_arguments, reason = "mirrors ComfyUI node inputs")]
    pub(crate) fn new(
        add_noise: bool,
        start_at_step: usize,
        end_at_step: usize,
        seed: u64,
        steps: usize,
        cfg: f32,
        sampler_name: SamplerKind,
        scheduler: SchedulerKind,
    ) -> Self {
        Self {
            add_noise,
            start_at_step,
            end_at_step,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
        }
    }
}

impl Node for KSamplerAdvanced {
    fn type_name(&self) -> &'static str {
        "KSamplerAdvanced"
    }

    fn inputs(&self) -> &'static [SlotDef] {
        static INPUTS: [SlotDef; 4] = [
            SlotDef::required("model", ValueType::Model),
            SlotDef::required("positive", ValueType::Conditioning),
            SlotDef::required("negative", ValueType::Conditioning),
            SlotDef::required("latent_image", ValueType::Latent),
        ];
        &INPUTS
    }

    fn outputs(&self) -> &'static [SlotDef] {
        static OUTPUTS: [SlotDef; 1] = [SlotDef::required("LATENT", ValueType::Latent)];
        &OUTPUTS
    }

    fn execute(
        &self,
        inputs: &[ResolvedInput],
        ctx: &mut ExecutionContext,
    ) -> Result<Vec<NodeValue>, NodeError> {
        let model_handle = match inputs[0].require("KSamplerAdvanced")? {
            NodeValue::Model(h) => *h,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "model",
                    expected: "MODEL",
                    got: other.type_name(),
                });
            }
        };

        let positive = match inputs[1].require("KSamplerAdvanced")? {
            NodeValue::Conditioning(c) => c,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "positive",
                    expected: "CONDITIONING",
                    got: other.type_name(),
                });
            }
        };

        let negative = match inputs[2].require("KSamplerAdvanced")? {
            NodeValue::Conditioning(c) => c,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "negative",
                    expected: "CONDITIONING",
                    got: other.type_name(),
                });
            }
        };

        let latent = match inputs[3].require("KSamplerAdvanced")? {
            NodeValue::Latent(l) => l,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "latent_image",
                    expected: "LATENT",
                    got: other.type_name(),
                });
            }
        };

        // Set seed
        let device = ctx.device().clone();
        <Backend as burn::tensor::backend::Backend>::seed(&device, self.seed);

        // Build full noise schedule
        let noise_schedule = NoiseSchedule::linear(1000, 0.00085, 0.012);
        let full_schedule = self.scheduler.schedule(&noise_schedule, self.steps);

        // Resolve end_at_step: 0 means run all steps
        let end = if self.end_at_step == 0 {
            self.steps
        } else {
            self.end_at_step.min(self.steps)
        };
        let start = self.start_at_step.min(end);

        // Slice the schedule to [start..end]
        // sigmas has length steps+1, timesteps has length steps
        let sliced_sigmas = full_schedule.sigmas[start..=end].to_vec();
        let sliced_timesteps = full_schedule.timesteps[start..end].to_vec();
        let schedule = DiffusionSchedule {
            sigmas: sliced_sigmas,
            timesteps: sliced_timesteps,
        };

        // Optionally add noise
        let [batch, ch, h, w] = latent.shape();
        let noisy_latent = if self.add_noise {
            // Scale by sigma at start_at_step
            let sigma = schedule.sigmas[0];
            let noise: Tensor<Backend, 4> =
                Tensor::random([batch, ch, h, w], Distribution::Normal(0.0, 1.0), &device);
            latent.samples.clone() + noise * sigma
        } else {
            latent.samples.clone()
        };

        tracing::info!(
            steps = self.steps,
            cfg = self.cfg,
            sampler = %self.sampler_name,
            scheduler = %self.scheduler,
            start_at_step = start,
            end_at_step = end,
            add_noise = self.add_noise,
            "sampling (advanced)"
        );

        // Extract ControlNet refs from positive conditioning metadata
        let cn_refs: Vec<ControlNetRef> = positive
            .entries
            .first()
            .and_then(|e| e.meta.get("control_net"))
            .and_then(|v| match v {
                ConditioningValue::ControlNetStack(stack) => Some(stack.clone()),
                _ => None,
            })
            .unwrap_or_default();

        // Build progress callback from the context's progress function.
        // Extract reference before borrowing ctx.models so the borrow checker
        // can see these are disjoint field borrows.
        let cancel = ctx.cancel.clone();
        let progress_fn = ctx.progress.as_deref();
        let progress_cb = |current: usize, total: usize| -> bool {
            if let Some(f) = progress_fn {
                f(current, total);
            }
            !cancel.load(Ordering::Relaxed)
        };

        let unet = ctx
            .models
            .borrow_unet(model_handle)
            .map_err(|e| NodeError::Execution {
                message: format!("failed to borrow UNet: {e}"),
            })?;

        let denoised = if cn_refs.is_empty() {
            match unet {
                UnetVariant::Sd15(unet) => {
                    let pos = extract_sd15_conditioning(positive)?;
                    let neg = extract_sd15_conditioning(negative)?;
                    sample_sd15(
                        unet,
                        noisy_latent,
                        &pos,
                        &neg,
                        &noise_schedule,
                        &schedule,
                        self.cfg,
                        self.sampler_name,
                        Some(&progress_cb),
                    )
                }
                UnetVariant::Sdxl(unet) => {
                    let pos = extract_sdxl_conditioning(positive)?;
                    let neg = extract_sdxl_conditioning(negative)?;
                    sample_sdxl(
                        unet,
                        noisy_latent,
                        &pos,
                        &neg,
                        &noise_schedule,
                        &schedule,
                        self.cfg,
                        self.sampler_name,
                        Some(&progress_cb),
                    )
                }
            }
        } else {
            match unet {
                UnetVariant::Sd15(unet) => {
                    let pos = extract_sd15_conditioning(positive)?;
                    let neg = extract_sd15_conditioning(negative)?;
                    let units = build_sd15_units(&cn_refs, &ctx.models)?;
                    sample_sd15_controlled(
                        unet,
                        &units,
                        noisy_latent,
                        &pos,
                        &neg,
                        &noise_schedule,
                        &schedule,
                        self.cfg,
                        self.sampler_name,
                        Some(&progress_cb),
                    )
                }
                UnetVariant::Sdxl(unet) => {
                    let pos = extract_sdxl_conditioning(positive)?;
                    let neg = extract_sdxl_conditioning(negative)?;
                    let units = build_sdxl_units(&cn_refs, &ctx.models)?;
                    sample_sdxl_controlled(
                        unet,
                        &units,
                        noisy_latent,
                        &pos,
                        &neg,
                        &noise_schedule,
                        &schedule,
                        self.cfg,
                        self.sampler_name,
                        Some(&progress_cb),
                    )
                }
            }
        };

        Ok(vec![NodeValue::Latent(Latent::new(denoised))])
    }
}
