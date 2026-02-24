//! `KSampler` node — runs the diffusion sampling loop.

use crate::{
    clip::{Sd15Conditioning, SdxlConditioning},
    controlnet::ControlNetUnit,
    node::{
        Node, ResolvedInput, SlotDef, ValueType,
        context::ExecutionContext,
        error::NodeError,
        value::NodeValue,
        variant::{ControlNetVariant, UnetVariant},
    },
    sampling::{
        DiffusionSchedule, DpmPp2mSampler, DpmPpSdeSampler, EulerASampler, EulerSampler,
        NoiseSchedule, ProgressCallback, SamplerKind, SchedulerKind,
    },
    types::{Backend, ConditioningValue, ControlNetRef, Latent},
};
use burn::{
    prelude::Backend as _,
    tensor::{Distribution, Tensor},
};
use std::sync::atomic::Ordering;

/// Runs denoising diffusion sampling on a latent tensor.
///
/// ComfyUI equivalent: `KSampler`
///
/// Takes a UNet model, positive/negative conditioning, a latent, and sampling
/// parameters. Returns the denoised latent.
#[derive(Debug)]
pub struct KSampler {
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
    /// Denoise strength (1.0 = full denoise from noise).
    denoise: f32,
}

impl KSampler {
    /// Create a new KSampler node.
    pub fn new(
        seed: u64,
        steps: usize,
        cfg: f32,
        sampler_name: SamplerKind,
        scheduler: SchedulerKind,
        denoise: f32,
    ) -> Self {
        Self {
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            denoise,
        }
    }
}

impl Node for KSampler {
    fn type_name(&self) -> &'static str {
        "KSampler"
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
        let model_handle = match inputs[0].require("KSampler")? {
            NodeValue::Model(h) => *h,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "model",
                    expected: "MODEL",
                    got: other.type_name(),
                });
            }
        };

        let positive = match inputs[1].require("KSampler")? {
            NodeValue::Conditioning(c) => c,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "positive",
                    expected: "CONDITIONING",
                    got: other.type_name(),
                });
            }
        };

        let negative = match inputs[2].require("KSampler")? {
            NodeValue::Conditioning(c) => c,
            other => {
                return Err(NodeError::TypeMismatch {
                    slot: "negative",
                    expected: "CONDITIONING",
                    got: other.type_name(),
                });
            }
        };

        let latent = match inputs[3].require("KSampler")? {
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
        Backend::seed(&device, self.seed);

        // Build noise schedule
        // SDXL and SD 1.5 both use these DDPM parameters
        let noise_schedule = NoiseSchedule::linear(1000, 0.00085, 0.012);
        let schedule = self.scheduler.schedule(&noise_schedule, self.steps);

        // Scale latent by sigma_max for txt2img (denoise=1.0)
        let sigma_max = schedule.sigmas[0];
        let [batch, ch, h, w] = latent.shape();
        let noisy_latent = if (self.denoise - 1.0).abs() < 1e-6 {
            // Full denoise: generate fresh noise scaled by sigma_max
            Tensor::random(
                [batch, ch, h, w],
                Distribution::Normal(0.0, sigma_max as f64),
                &device,
            )
        } else {
            // Partial denoise: add noise to existing latent
            let noise: Tensor<Backend, 4> =
                Tensor::random([batch, ch, h, w], Distribution::Normal(0.0, 1.0), &device);
            latent.samples.clone() + noise * sigma_max * self.denoise
        };

        tracing::info!(
            steps = self.steps,
            cfg = self.cfg,
            sampler = %self.sampler_name,
            scheduler = %self.scheduler,
            "sampling"
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

        // Borrow UNet and dispatch by variant.
        // borrow_* takes &self (LRU uses Cell), so we can hold multiple borrows
        // simultaneously — needed when ControlNets are active.
        let unet = ctx
            .models
            .borrow_unet(model_handle)
            .map_err(|e| NodeError::Execution {
                message: format!("failed to borrow UNet: {e}"),
            })?;

        let denoised = if cn_refs.is_empty() {
            // Standard path: no ControlNets
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
            // ControlNet path
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

/// Extract SD 1.5 conditioning from the generic `Conditioning` type.
pub(super) fn extract_sd15_conditioning(
    cond: &crate::types::Conditioning,
) -> Result<Sd15Conditioning, NodeError> {
    let entry = cond.entries.first().ok_or(NodeError::Execution {
        message: "empty conditioning".to_string(),
    })?;
    Ok(Sd15Conditioning::new(entry.embedding.clone()))
}

/// Extract SDXL conditioning from the generic `Conditioning` type.
///
/// The `y` vector is stored in metadata by `CLIPTextEncode`.
pub(super) fn extract_sdxl_conditioning(
    cond: &crate::types::Conditioning,
) -> Result<SdxlConditioning, NodeError> {
    let entry = cond.entries.first().ok_or(NodeError::Execution {
        message: "empty conditioning".to_string(),
    })?;
    let y = match entry.meta.get("y") {
        Some(ConditioningValue::Tensor(t)) => t.clone(),
        _ => {
            return Err(NodeError::Execution {
                message: "SDXL conditioning missing 'y' (pooled) tensor in metadata".to_string(),
            });
        }
    };
    Ok(SdxlConditioning {
        context: entry.embedding.clone(),
        y,
    })
}

/// Run sampling with an SD 1.5 UNet.
#[expect(clippy::too_many_arguments, reason = "mirrors sampler API")]
pub(super) fn sample_sd15(
    unet: &crate::unet::Sd15Unet2D,
    latent: Tensor<Backend, 4>,
    positive: &Sd15Conditioning,
    negative: &Sd15Conditioning,
    noise_schedule: &NoiseSchedule,
    schedule: &DiffusionSchedule,
    cfg_scale: f32,
    sampler_kind: SamplerKind,
    progress: ProgressCallback<'_>,
) -> Tensor<Backend, 4> {
    match sampler_kind {
        SamplerKind::Euler => {
            let sampler = EulerSampler::new(noise_schedule.clone());
            sampler.sample_with_schedule(
                unet, latent, positive, negative, schedule, cfg_scale, progress,
            )
        }
        SamplerKind::EulerA => {
            let sampler = EulerASampler::new(noise_schedule.clone());
            sampler.sample_with_schedule(
                unet, latent, positive, negative, schedule, cfg_scale, progress,
            )
        }
        SamplerKind::DpmPp2m => {
            let sampler = DpmPp2mSampler::new(noise_schedule.clone());
            sampler.sample_with_schedule(
                unet, latent, positive, negative, schedule, cfg_scale, progress,
            )
        }
        SamplerKind::DpmPpSde => {
            let sampler = DpmPpSdeSampler::new(noise_schedule.clone());
            sampler.sample_with_schedule(
                unet, latent, positive, negative, schedule, cfg_scale, progress,
            )
        }
    }
}

/// Run sampling with an SDXL UNet.
#[expect(clippy::too_many_arguments, reason = "mirrors sampler API")]
pub(super) fn sample_sdxl(
    unet: &crate::unet::SdxlUnet2D,
    latent: Tensor<Backend, 4>,
    positive: &SdxlConditioning,
    negative: &SdxlConditioning,
    noise_schedule: &NoiseSchedule,
    schedule: &DiffusionSchedule,
    cfg_scale: f32,
    sampler_kind: SamplerKind,
    progress: ProgressCallback<'_>,
) -> Tensor<Backend, 4> {
    match sampler_kind {
        SamplerKind::Euler => {
            let sampler = EulerSampler::new(noise_schedule.clone());
            sampler.sample_with_schedule(
                unet, latent, positive, negative, schedule, cfg_scale, progress,
            )
        }
        SamplerKind::EulerA => {
            let sampler = EulerASampler::new(noise_schedule.clone());
            sampler.sample_with_schedule(
                unet, latent, positive, negative, schedule, cfg_scale, progress,
            )
        }
        SamplerKind::DpmPp2m => {
            let sampler = DpmPp2mSampler::new(noise_schedule.clone());
            sampler.sample_with_schedule(
                unet, latent, positive, negative, schedule, cfg_scale, progress,
            )
        }
        SamplerKind::DpmPpSde => {
            let sampler = DpmPpSdeSampler::new(noise_schedule.clone());
            sampler.sample_with_schedule(
                unet, latent, positive, negative, schedule, cfg_scale, progress,
            )
        }
    }
}

/// Build [`ControlNetUnit`]s for SD 1.5 from conditioning metadata refs.
pub(super) fn build_sd15_units<'a>(
    refs: &[ControlNetRef],
    models: &'a crate::model_manager::ModelManager,
) -> Result<Vec<ControlNetUnit<'a, crate::unet::Sd15Unet>>, NodeError> {
    let mut units = Vec::with_capacity(refs.len());
    for cn_ref in refs {
        let variant =
            models
                .borrow_controlnet(cn_ref.handle)
                .map_err(|e| NodeError::Execution {
                    message: format!("failed to borrow ControlNet: {e}"),
                })?;
        match variant {
            ControlNetVariant::Sd15(model) => {
                units.push(ControlNetUnit {
                    model,
                    hint: cn_ref.hint.clone(),
                    weight: cn_ref.strength,
                    start: cn_ref.start_percent,
                    end: cn_ref.end_percent,
                });
            }
            ControlNetVariant::Sdxl(_) => {
                return Err(NodeError::Execution {
                    message: "SDXL ControlNet used with SD 1.5 UNet".to_string(),
                });
            }
        }
    }
    Ok(units)
}

/// Build [`ControlNetUnit`]s for SDXL from conditioning metadata refs.
pub(super) fn build_sdxl_units<'a>(
    refs: &[ControlNetRef],
    models: &'a crate::model_manager::ModelManager,
) -> Result<Vec<ControlNetUnit<'a, crate::unet::SdxlUnet>>, NodeError> {
    let mut units = Vec::with_capacity(refs.len());
    for cn_ref in refs {
        let variant =
            models
                .borrow_controlnet(cn_ref.handle)
                .map_err(|e| NodeError::Execution {
                    message: format!("failed to borrow ControlNet: {e}"),
                })?;
        match variant {
            ControlNetVariant::Sdxl(model) => {
                units.push(ControlNetUnit {
                    model,
                    hint: cn_ref.hint.clone(),
                    weight: cn_ref.strength,
                    start: cn_ref.start_percent,
                    end: cn_ref.end_percent,
                });
            }
            ControlNetVariant::Sd15(_) => {
                return Err(NodeError::Execution {
                    message: "SD 1.5 ControlNet used with SDXL UNet".to_string(),
                });
            }
        }
    }
    Ok(units)
}

/// Run controlled sampling with an SD 1.5 UNet.
#[expect(clippy::too_many_arguments, reason = "mirrors sampler API")]
pub(super) fn sample_sd15_controlled(
    unet: &crate::unet::Sd15Unet2D,
    controlnets: &[ControlNetUnit<'_, crate::unet::Sd15Unet>],
    latent: Tensor<Backend, 4>,
    positive: &Sd15Conditioning,
    negative: &Sd15Conditioning,
    noise_schedule: &NoiseSchedule,
    schedule: &DiffusionSchedule,
    cfg_scale: f32,
    sampler_kind: SamplerKind,
    progress: ProgressCallback<'_>,
) -> Tensor<Backend, 4> {
    match sampler_kind {
        SamplerKind::Euler => {
            let sampler = EulerSampler::new(noise_schedule.clone());
            sampler.sample_with_schedule_controlled(
                unet,
                controlnets,
                latent,
                positive,
                negative,
                schedule,
                cfg_scale,
                progress,
            )
        }
        SamplerKind::EulerA => {
            let sampler = EulerASampler::new(noise_schedule.clone());
            sampler.sample_with_schedule_controlled(
                unet,
                controlnets,
                latent,
                positive,
                negative,
                schedule,
                cfg_scale,
                progress,
            )
        }
        SamplerKind::DpmPp2m => {
            let sampler = DpmPp2mSampler::new(noise_schedule.clone());
            sampler.sample_with_schedule_controlled(
                unet,
                controlnets,
                latent,
                positive,
                negative,
                schedule,
                cfg_scale,
                progress,
            )
        }
        SamplerKind::DpmPpSde => {
            let sampler = DpmPpSdeSampler::new(noise_schedule.clone());
            sampler.sample_with_schedule_controlled(
                unet,
                controlnets,
                latent,
                positive,
                negative,
                schedule,
                cfg_scale,
                progress,
            )
        }
    }
}

/// Run controlled sampling with an SDXL UNet.
#[expect(clippy::too_many_arguments, reason = "mirrors sampler API")]
pub(super) fn sample_sdxl_controlled(
    unet: &crate::unet::SdxlUnet2D,
    controlnets: &[ControlNetUnit<'_, crate::unet::SdxlUnet>],
    latent: Tensor<Backend, 4>,
    positive: &SdxlConditioning,
    negative: &SdxlConditioning,
    noise_schedule: &NoiseSchedule,
    schedule: &DiffusionSchedule,
    cfg_scale: f32,
    sampler_kind: SamplerKind,
    progress: ProgressCallback<'_>,
) -> Tensor<Backend, 4> {
    match sampler_kind {
        SamplerKind::Euler => {
            let sampler = EulerSampler::new(noise_schedule.clone());
            sampler.sample_with_schedule_controlled(
                unet,
                controlnets,
                latent,
                positive,
                negative,
                schedule,
                cfg_scale,
                progress,
            )
        }
        SamplerKind::EulerA => {
            let sampler = EulerASampler::new(noise_schedule.clone());
            sampler.sample_with_schedule_controlled(
                unet,
                controlnets,
                latent,
                positive,
                negative,
                schedule,
                cfg_scale,
                progress,
            )
        }
        SamplerKind::DpmPp2m => {
            let sampler = DpmPp2mSampler::new(noise_schedule.clone());
            sampler.sample_with_schedule_controlled(
                unet,
                controlnets,
                latent,
                positive,
                negative,
                schedule,
                cfg_scale,
                progress,
            )
        }
        SamplerKind::DpmPpSde => {
            let sampler = DpmPpSdeSampler::new(noise_schedule.clone());
            sampler.sample_with_schedule_controlled(
                unet,
                controlnets,
                latent,
                positive,
                negative,
                schedule,
                cfg_scale,
                progress,
            )
        }
    }
}
