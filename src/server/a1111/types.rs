//! A1111 API request/response types.

use serde::{Deserialize, Serialize};
use serde_with::{DisplayFromStr, PickFirst, TryFromInto, serde_as};
use std::collections::HashMap;
use strum::{Display, EnumString, FromRepr};

/// Resize mode for ControlNet units and img2img requests.
///
/// Clients may send this as either a `usize` (`0`, `1`, `2`) or the A1111
/// human-readable string (`"Just Resize"`, `"Crop and Resize"`, …).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Display, EnumString, FromRepr)]
#[repr(usize)]
pub enum ResizeMode {
    #[default]
    #[strum(serialize = "Just Resize")]
    JustResize = 0,
    #[strum(serialize = "Crop and Resize")]
    CropAndResize = 1,
    #[strum(serialize = "Resize and Fill")]
    ResizeAndFill = 2,
}

impl TryFrom<usize> for ResizeMode {
    type Error = &'static str;
    fn try_from(v: usize) -> Result<Self, Self::Error> {
        Self::from_repr(v).ok_or("unknown resize_mode variant")
    }
}

impl From<ResizeMode> for usize {
    fn from(m: ResizeMode) -> usize {
        m as usize
    }
}

/// Control mode for ControlNet units.
///
/// Clients may send this as either a `usize` or the A1111 human-readable string.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Display, EnumString, FromRepr)]
#[repr(usize)]
pub enum ControlMode {
    #[default]
    #[strum(serialize = "Balanced")]
    Balanced = 0,
    #[strum(serialize = "My prompt is more important")]
    PromptPriority = 1,
    #[strum(serialize = "ControlNet is more important")]
    ControlNetPriority = 2,
}

impl TryFrom<usize> for ControlMode {
    type Error = &'static str;
    fn try_from(v: usize) -> Result<Self, Self::Error> {
        Self::from_repr(v).ok_or("unknown control_mode variant")
    }
}

impl From<ControlMode> for usize {
    fn from(m: ControlMode) -> usize {
        m as usize
    }
}

/// Request body for `POST /sdapi/v1/txt2img`.
#[derive(Debug, Serialize, Deserialize)]
pub struct Txt2ImgRequest {
    /// Positive prompt.
    #[serde(default)]
    pub prompt: String,
    /// Negative prompt.
    #[serde(default)]
    pub negative_prompt: String,
    /// Image width in pixels.
    #[serde(default = "default_512")]
    pub width: u32,
    /// Image height in pixels.
    #[serde(default = "default_512")]
    pub height: u32,
    /// Number of sampling steps.
    #[serde(default = "default_20")]
    pub steps: u32,
    /// Classifier-free guidance scale.
    #[serde(default = "default_cfg")]
    pub cfg_scale: f32,
    /// Sampling algorithm name.
    #[serde(default = "default_sampler")]
    pub sampler_name: String,
    /// Sigma schedule name.
    #[serde(default = "default_scheduler")]
    pub scheduler: String,
    /// Random seed (-1 for random).
    #[serde(default = "default_seed")]
    pub seed: i64,
    /// Denoise strength (1.0 = full denoise for txt2img).
    #[serde(default = "default_denoise")]
    pub denoising_strength: f32,
    /// Batch size (number of images to generate).
    #[serde(default = "default_1")]
    pub batch_size: u32,
    /// Number of iterations (each generates `batch_size` images).
    #[serde(default = "default_1")]
    pub n_iter: u32,
    /// Tiling mode (not yet supported).
    #[serde(default)]
    pub tiling: bool,
    /// Override checkpoint filename (optional).
    #[serde(default)]
    pub override_settings: Option<OverrideSettings>,
    /// Extension scripts (ControlNet, etc.).
    #[serde(default)]
    pub alwayson_scripts: Option<AlwaysOnScripts>,
}

/// Optional settings overrides in txt2img request.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct OverrideSettings {
    /// Checkpoint to load.
    #[serde(default)]
    pub sd_model_checkpoint: Option<String>,
}

/// Generation info echoed back in `Txt2ImgResponse`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Txt2ImgInfo {
    /// Positive prompt used.
    pub prompt: String,
    /// Negative prompt used.
    pub negative_prompt: String,
    /// Random seed used.
    pub seed: u64,
    /// Number of sampling steps.
    pub steps: u32,
    /// Classifier-free guidance scale.
    pub cfg_scale: f32,
    /// Sampler algorithm name.
    pub sampler_name: String,
    /// Schedule name.
    pub scheduler: String,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Checkpoint filename.
    pub sd_model_checkpoint: String,
}

/// Response body for `POST /sdapi/v1/txt2img`.
#[derive(Debug, Serialize, Deserialize)]
pub struct Txt2ImgResponse {
    /// Base64-encoded PNG images.
    pub images: Vec<String>,
    /// Request parameters (echoed back).
    pub parameters: Txt2ImgInfo,
    /// Info string (JSON-encoded generation info).
    pub info: String,
}

/// Entry in `GET /sdapi/v1/sd-models` response.
#[derive(Debug, Serialize, Deserialize)]
pub struct SdModel {
    /// Display title (format: "model_name [hash]").
    pub title: String,
    /// Model name (filename stem).
    pub model_name: String,
    /// Filename (relative to checkpoints dir).
    pub filename: String,
    /// Short hash (first 10 chars of sha256).
    pub hash: String,
    /// Full SHA-256 hex digest.
    pub sha256: String,
    /// Companion .yaml config path (not used by Hearth).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<String>,
}

/// Entry in `GET /sdapi/v1/samplers` response.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct SamplerItem {
    pub(super) name: String,
    pub(super) aliases: Vec<String>,
    pub(super) options: SamplerOptions,
}

/// Options for a sampler entry.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(super) struct SamplerOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) scheduler: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) second_order: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) brownian_noise: Option<bool>,
}

/// Entry in `GET /sdapi/v1/schedulers` response.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct SchedulerItem {
    pub(super) name: String,
    pub(super) label: String,
    pub(super) aliases: Vec<String>,
}

/// Response for `GET /sdapi/v1/progress` (A1111 nested format).
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct A1111ProgressResponse {
    progress: f32,
    eta_relative: f32,
    state: A1111ProgressState,
    current_image: Option<String>,
    textinfo: Option<String>,
}

impl A1111ProgressResponse {
    /// Convert internal [`ProgressInfo`](crate::server::ProgressInfo) to A1111 format.
    pub(super) fn from_progress(info: &crate::server::ProgressInfo) -> Self {
        Self {
            progress: info.progress,
            eta_relative: info.eta_relative,
            state: A1111ProgressState {
                skipped: false,
                interrupted: false,
                job: if info.active {
                    "txt2img".to_string()
                } else {
                    String::new()
                },
                job_count: if info.active { 1 } else { 0 },
                job_timestamp: String::new(),
                job_no: 0,
                sampling_step: info.current_step,
                sampling_steps: info.total_steps,
            },
            current_image: None,
            textinfo: None,
        }
    }
}

/// Nested state in A1111 progress response.
#[derive(Debug, Serialize, Deserialize)]
struct A1111ProgressState {
    skipped: bool,
    interrupted: bool,
    job: String,
    job_count: u32,
    job_timestamp: String,
    job_no: u32,
    sampling_step: usize,
    sampling_steps: usize,
}

/// Request body for `POST /sdapi/v1/options` (partial update).
///
/// All fields are optional — only `Some(...)` fields are applied.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(super) struct SetOptionsRequest {
    #[serde(default)]
    pub(super) sd_model_checkpoint: Option<String>,
    #[serde(default)]
    pub(super) sd_vae: Option<String>,
}

/// Wrapper for A1111's `alwayson_scripts` request field.
///
/// Extensions like ControlNet send their parameters nested under named keys.
/// Unknown extension keys are captured in `unknown` and logged as warnings
/// so we notice when clients send data we silently drop.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct AlwaysOnScripts {
    /// ControlNet extension arguments.
    ///
    /// The A1111 ControlNet extension registers with `title() = "ControlNet"`,
    /// so clients (including StableProjectorz) send the key as `"ControlNet"`.
    /// The alias ensures we accept both casings.
    #[serde(default, alias = "ControlNet")]
    pub controlnet: Option<ControlNetArgs>,

    /// Any extension keys we don't explicitly handle.
    ///
    /// Captured so we can warn about them rather than silently dropping.
    #[serde(flatten)]
    pub unknown: HashMap<String, serde_json::Value>,
}

/// ControlNet extension arguments within `alwayson_scripts`.
#[derive(Debug, Serialize, Deserialize)]
pub struct ControlNetArgs {
    /// One entry per ControlNet unit.
    #[serde(default)]
    pub args: Vec<ControlNetUnit>,
}

/// A single ControlNet unit in an A1111 request.
#[serde_as]
#[derive(Debug, Serialize, Deserialize)]
pub struct ControlNetUnit {
    /// Whether this unit is active.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Preprocessor module name ("none" for pre-processed images).
    #[serde(default = "default_module_none")]
    pub module: String,
    /// ControlNet model filename.
    #[serde(default)]
    pub model: String,
    /// Effect strength (0.0–1.0).
    #[serde(default = "default_denoise")]
    pub weight: f32,
    /// Base64-encoded control image.
    #[serde(default)]
    pub image: Option<String>,
    /// Resize mode — accepts `0`/`1`/`2` or `"Just Resize"`/`"Crop and Resize"`/`"Resize and Fill"`.
    #[serde_as(as = "PickFirst<(TryFromInto<usize>, DisplayFromStr)>")]
    #[serde(default)]
    pub resize_mode: ResizeMode,
    /// Step fraction at which ControlNet activates (0.0–1.0).
    #[serde(default)]
    pub guidance_start: f32,
    /// Step fraction at which ControlNet deactivates (0.0–1.0).
    #[serde(default = "default_denoise")]
    pub guidance_end: f32,
    /// Control mode — accepts `0`/`1`/`2` or `"Balanced"`/`"My prompt is more important"`/`"ControlNet is more important"`.
    #[serde_as(as = "PickFirst<(TryFromInto<usize>, DisplayFromStr)>")]
    #[serde(default)]
    pub control_mode: ControlMode,
    /// Whether to auto-detect preprocessor resolution.
    #[serde(default)]
    pub pixel_perfect: bool,
}

/// Request body for `POST /sdapi/v1/img2img`.
#[serde_as]
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Img2ImgRequest {
    /// Base64-encoded init images.
    pub(super) init_images: Vec<String>,
    /// Base64-encoded mask image (white = inpaint region).
    #[serde(default)]
    pub(super) mask: Option<String>,
    /// Positive prompt.
    #[serde(default)]
    pub(super) prompt: String,
    /// Negative prompt.
    #[serde(default)]
    pub(super) negative_prompt: String,
    /// Image width in pixels.
    #[serde(default = "default_512")]
    pub(super) width: u32,
    /// Image height in pixels.
    #[serde(default = "default_512")]
    pub(super) height: u32,
    /// Number of sampling steps.
    #[serde(default = "default_20")]
    pub(super) steps: u32,
    /// Classifier-free guidance scale.
    #[serde(default = "default_cfg")]
    pub(super) cfg_scale: f32,
    /// Sampling algorithm name.
    #[serde(default = "default_sampler")]
    pub(super) sampler_name: String,
    /// Sigma schedule name.
    #[serde(default = "default_scheduler")]
    pub(super) scheduler: String,
    /// Random seed (-1 for random).
    #[serde(default = "default_seed")]
    pub(super) seed: i64,
    /// Denoise strength (0.0 = no change, 1.0 = full denoise).
    #[serde(default = "default_img2img_denoise")]
    pub(super) denoising_strength: f32,
    /// Batch size.
    #[serde(default = "default_1")]
    pub(super) batch_size: u32,
    /// Number of iterations (batches).
    #[serde(default = "default_1")]
    pub(super) n_iter: u32,
    /// Resize mode for init image — accepts integer or string form.
    #[serde_as(as = "PickFirst<(TryFromInto<usize>, DisplayFromStr)>")]
    #[serde(default)]
    pub(super) resize_mode: ResizeMode,
    /// How to fill the masked area before denoising (0=fill, 1=original, 2=latent noise, 3=latent nothing).
    #[serde(default)]
    pub(super) inpainting_fill: u32,
    /// Whether to inpaint only the masked region at full resolution.
    #[serde(default)]
    pub(super) inpaint_full_res: u32,
    /// Padding (pixels) around the masked region when using inpaint_full_res.
    #[serde(default = "default_32")]
    pub(super) inpaint_full_res_padding: u32,
    /// Whether to invert the mask (0=no, 1=yes).
    #[serde(default)]
    pub(super) inpainting_mask_invert: u32,
    /// Gaussian blur radius for the mask.
    #[serde(default = "default_4")]
    pub(super) mask_blur: u32,
    /// Whether to include init images in the response.
    #[serde(default)]
    pub(super) include_init_images: bool,
    /// Tiling mode.
    #[serde(default)]
    pub(super) tiling: bool,
    /// Override checkpoint filename (optional).
    #[serde(default)]
    pub(super) override_settings: Option<OverrideSettings>,
    /// Extension scripts (ControlNet, etc.).
    #[serde(default)]
    pub(super) alwayson_scripts: Option<AlwaysOnScripts>,
}

const fn default_512() -> u32 {
    512
}
const fn default_20() -> u32 {
    20
}
const fn default_cfg() -> f32 {
    7.0
}
fn default_sampler() -> String {
    "euler".to_string()
}
fn default_scheduler() -> String {
    "normal".to_string()
}
const fn default_seed() -> i64 {
    -1
}
const fn default_denoise() -> f32 {
    1.0
}
const fn default_img2img_denoise() -> f32 {
    0.75
}
const fn default_1() -> u32 {
    1
}
const fn default_true() -> bool {
    true
}
fn default_module_none() -> String {
    "none".to_string()
}
const fn default_32() -> u32 {
    32
}
const fn default_4() -> u32 {
    4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn txt2img_request_deserialize() {
        let json = serde_json::json!({
            "prompt": "a cat",
            "negative_prompt": "blurry",
            "width": 768,
            "height": 768,
            "steps": 10,
            "cfg_scale": 5.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "seed": 42,
            "denoising_strength": 0.8,
            "batch_size": 2,
            "override_settings": {
                "sd_model_checkpoint": "model.safetensors"
            }
        });

        let req: Txt2ImgRequest = serde_json::from_value(json).expect("deserialize");
        assert_eq!(req.prompt, "a cat");
        assert_eq!(req.negative_prompt, "blurry");
        assert_eq!(req.width, 768);
        assert_eq!(req.height, 768);
        assert_eq!(req.steps, 10);
        assert_eq!(req.cfg_scale, 5.0);
        assert_eq!(req.seed, 42);
        assert_eq!(req.batch_size, 2);
        assert_eq!(
            req.override_settings
                .as_ref()
                .unwrap()
                .sd_model_checkpoint
                .as_deref(),
            Some("model.safetensors")
        );
    }

    #[test]
    fn txt2img_request_defaults() {
        let json = serde_json::json!({});
        let req: Txt2ImgRequest = serde_json::from_value(json).expect("deserialize");
        assert_eq!(req.width, 512);
        assert_eq!(req.height, 512);
        assert_eq!(req.steps, 20);
        assert_eq!(req.cfg_scale, 7.0);
        assert_eq!(req.seed, -1);
        assert_eq!(req.denoising_strength, 1.0);
        assert_eq!(req.batch_size, 1);
    }

    #[test]
    fn txt2img_info_round_trip() {
        let info = Txt2ImgInfo {
            prompt: "hello".to_string(),
            negative_prompt: "bad".to_string(),
            seed: 123,
            steps: 20,
            cfg_scale: 7.0,
            sampler_name: "euler".to_string(),
            scheduler: "normal".to_string(),
            width: 512,
            height: 512,
            sd_model_checkpoint: "model.safetensors".to_string(),
        };
        let json = serde_json::to_string(&info).expect("serialize");
        let back: Txt2ImgInfo = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(info, back);
    }

    #[test]
    fn txt2img_response_round_trip() {
        let info = Txt2ImgInfo {
            prompt: "test".to_string(),
            negative_prompt: String::new(),
            seed: 1,
            steps: 10,
            cfg_scale: 7.0,
            sampler_name: "euler".to_string(),
            scheduler: "normal".to_string(),
            width: 512,
            height: 512,
            sd_model_checkpoint: "ckpt.safetensors".to_string(),
        };
        let resp = Txt2ImgResponse {
            images: vec!["abc123".to_string()],
            parameters: info,
            info: "{}".to_string(),
        };
        let json = serde_json::to_string(&resp).expect("serialize");
        let back: Txt2ImgResponse = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.images, resp.images);
        assert_eq!(back.parameters, resp.parameters);
        assert_eq!(back.info, resp.info);
    }

    #[test]
    fn sd_model_round_trip() {
        let model = SdModel {
            title: "my_model [abcdef1234]".to_string(),
            model_name: "my_model".to_string(),
            filename: "my_model.safetensors".to_string(),
            hash: "abcdef1234".to_string(),
            sha256: "abcdef1234567890".to_string(),
            config: None,
        };
        let json = serde_json::to_string(&model).expect("serialize");
        let back: SdModel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.title, model.title);
        assert_eq!(back.model_name, model.model_name);
        assert_eq!(back.filename, model.filename);
        assert_eq!(back.hash, model.hash);
        assert_eq!(back.sha256, model.sha256);
        assert_eq!(back.config, None);
    }

    #[test]
    fn sd_model_config_omitted_when_none() {
        let model = SdModel {
            title: "test".to_string(),
            model_name: "test".to_string(),
            filename: "test.safetensors".to_string(),
            hash: "abc".to_string(),
            sha256: "abcdef".to_string(),
            config: None,
        };
        let json = serde_json::to_string(&model).expect("serialize");
        assert!(!json.contains("config"));
    }

    #[test]
    fn sampler_item_round_trip() {
        let item = SamplerItem {
            name: "Euler".to_string(),
            aliases: vec!["euler".to_string()],
            options: SamplerOptions {
                second_order: Some(false),
                ..Default::default()
            },
        };
        let json = serde_json::to_string(&item).expect("serialize");
        let back: SamplerItem = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.name, "Euler");
        assert_eq!(back.aliases, vec!["euler"]);
        // scheduler was None → should not appear in JSON
        assert!(!json.contains("scheduler"));
    }

    #[test]
    fn scheduler_item_round_trip() {
        let item = SchedulerItem {
            name: "karras".to_string(),
            label: "Karras".to_string(),
            aliases: vec![],
        };
        let json = serde_json::to_string(&item).expect("serialize");
        let back: SchedulerItem = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.name, "karras");
        assert_eq!(back.label, "Karras");
    }

    #[test]
    fn a1111_progress_response_round_trip() {
        let resp = A1111ProgressResponse::from_progress(&crate::server::ProgressInfo {
            progress: 0.5,
            eta_relative: 2.0,
            current_step: 10,
            total_steps: 20,
            active: true,
        });
        let json = serde_json::to_string(&resp).expect("serialize");
        let back: A1111ProgressResponse = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.progress, 0.5);
        assert_eq!(back.state.sampling_step, 10);
    }

    #[test]
    fn set_options_partial_deserialize() {
        let json = serde_json::json!({ "sd_model_checkpoint": "model.safetensors" });
        let req: SetOptionsRequest = serde_json::from_value(json).expect("deserialize");
        assert_eq!(
            req.sd_model_checkpoint.as_deref(),
            Some("model.safetensors")
        );
        assert_eq!(req.sd_vae, None);
    }

    #[test]
    fn img2img_request_defaults() {
        let json = serde_json::json!({ "init_images": ["abc123"] });
        let req: Img2ImgRequest = serde_json::from_value(json).expect("deserialize");
        assert_eq!(req.init_images, vec!["abc123"]);
        assert_eq!(req.mask, None);
        assert_eq!(req.width, 512);
        assert_eq!(req.height, 512);
        assert_eq!(req.steps, 20);
        assert_eq!(req.cfg_scale, 7.0);
        assert_eq!(req.denoising_strength, 0.75);
        assert_eq!(req.seed, -1);
        assert_eq!(req.batch_size, 1);
        assert_eq!(req.n_iter, 1);
        assert_eq!(req.resize_mode, ResizeMode::JustResize);
        assert_eq!(req.inpainting_fill, 0);
        assert_eq!(req.inpaint_full_res, 0);
        assert_eq!(req.inpaint_full_res_padding, 32);
        assert_eq!(req.inpainting_mask_invert, 0);
        assert_eq!(req.mask_blur, 4);
        assert!(!req.include_init_images);
        assert!(!req.tiling);
    }

    #[test]
    fn img2img_request_with_mask() {
        let json = serde_json::json!({
            "init_images": ["img_b64"],
            "mask": "mask_b64",
            "denoising_strength": 0.5,
            "inpainting_fill": 1,
            "inpaint_full_res": 1,
            "inpainting_mask_invert": 1,
            "mask_blur": 8,
        });
        let req: Img2ImgRequest = serde_json::from_value(json).expect("deserialize");
        assert_eq!(req.mask.as_deref(), Some("mask_b64"));
        assert_eq!(req.denoising_strength, 0.5);
        assert_eq!(req.inpainting_fill, 1);
        assert_eq!(req.inpaint_full_res, 1);
        assert_eq!(req.inpainting_mask_invert, 1);
        assert_eq!(req.mask_blur, 8);
    }

    #[test]
    fn img2img_request_ignores_unknown_fields() {
        // StableProjectorz may send fields we don't model; serde should not reject them
        let json = serde_json::json!({
            "init_images": ["x"],
            "some_future_field": true,
            "alwayson_scripts": {}
        });
        let req: Img2ImgRequest =
            serde_json::from_value(json).expect("should tolerate unknown fields");
        assert_eq!(req.init_images, vec!["x"]);
    }

    #[test]
    fn controlnet_unit_defaults() {
        let json = serde_json::json!({});
        let unit: ControlNetUnit = serde_json::from_value(json).expect("deserialize");
        assert!(unit.enabled);
        assert_eq!(unit.module, "none");
        assert_eq!(unit.model, "");
        assert_eq!(unit.weight, 1.0);
        assert_eq!(unit.image, None);
        assert_eq!(unit.resize_mode, ResizeMode::JustResize);
        assert_eq!(unit.guidance_start, 0.0);
        assert_eq!(unit.guidance_end, 1.0);
        assert_eq!(unit.control_mode, ControlMode::Balanced);
        assert!(!unit.pixel_perfect);
    }

    #[test]
    fn controlnet_unit_round_trip() {
        let unit = ControlNetUnit {
            enabled: true,
            module: "none".to_string(),
            model: "control_v11p_sd15_canny.safetensors".to_string(),
            weight: 0.8,
            image: Some("base64data".to_string()),
            resize_mode: ResizeMode::CropAndResize,
            guidance_start: 0.1,
            guidance_end: 0.9,
            control_mode: ControlMode::ControlNetPriority,
            pixel_perfect: true,
        };
        let json = serde_json::to_string(&unit).expect("serialize");
        let back: ControlNetUnit = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.model, "control_v11p_sd15_canny.safetensors");
        assert_eq!(back.weight, 0.8);
        assert_eq!(back.guidance_start, 0.1);
        assert_eq!(back.guidance_end, 0.9);
        assert!(back.pixel_perfect);
    }

    #[test]
    fn alwayson_scripts_with_controlnet() {
        let json = serde_json::json!({
            "controlnet": {
                "args": [
                    {
                        "enabled": true,
                        "module": "none",
                        "model": "control_v11p_sd15_canny.safetensors",
                        "weight": 0.9,
                        "image": "b64img",
                        "guidance_start": 0.0,
                        "guidance_end": 1.0
                    },
                    {
                        "enabled": false,
                        "model": "control_v11f1p_sd15_depth.safetensors"
                    }
                ]
            }
        });
        let scripts: AlwaysOnScripts = serde_json::from_value(json).expect("deserialize");
        let cn = scripts.controlnet.expect("controlnet present");
        assert_eq!(cn.args.len(), 2);
        assert!(cn.args[0].enabled);
        assert_eq!(cn.args[0].weight, 0.9);
        assert!(!cn.args[1].enabled);
    }

    #[test]
    fn alwayson_scripts_empty() {
        let json = serde_json::json!({});
        let scripts: AlwaysOnScripts = serde_json::from_value(json).expect("deserialize");
        assert!(scripts.controlnet.is_none());
    }

    #[test]
    fn alwayson_scripts_capital_controlnet() {
        // A1111's ControlNet extension registers with title() = "ControlNet",
        // so SPZ and other clients send the key with that exact casing.
        let json = serde_json::json!({
            "ControlNet": {
                "args": [{
                    "enabled": true,
                    "module": "none",
                    "model": "control_v11f1p_sd15_depth.safetensors",
                    "weight": 1.0,
                    "image": "b64img"
                }]
            }
        });
        let scripts: AlwaysOnScripts = serde_json::from_value(json).expect("deserialize");
        let cn = scripts
            .controlnet
            .expect("ControlNet key should be accepted");
        assert_eq!(cn.args.len(), 1);
        assert_eq!(cn.args[0].model, "control_v11f1p_sd15_depth.safetensors");
        assert!(scripts.unknown.is_empty());
    }

    #[test]
    fn alwayson_scripts_unknown_extensions_captured() {
        let json = serde_json::json!({
            "controlnet": { "args": [] },
            "SomeOtherExtension": { "args": [1, 2, 3] },
            "ADetailer": { "args": [] }
        });
        let scripts: AlwaysOnScripts = serde_json::from_value(json).expect("deserialize");
        assert!(scripts.controlnet.is_some());
        assert_eq!(scripts.unknown.len(), 2);
        assert!(scripts.unknown.contains_key("SomeOtherExtension"));
        assert!(scripts.unknown.contains_key("ADetailer"));
    }

    #[test]
    fn txt2img_request_with_controlnet() {
        let json = serde_json::json!({
            "prompt": "a cat",
            "alwayson_scripts": {
                "controlnet": {
                    "args": [{
                        "model": "control_v11p_sd15_canny.safetensors",
                        "image": "b64img"
                    }]
                }
            }
        });
        let req: Txt2ImgRequest = serde_json::from_value(json).expect("deserialize");
        let scripts = req.alwayson_scripts.expect("alwayson_scripts present");
        let cn = scripts.controlnet.expect("controlnet present");
        assert_eq!(cn.args.len(), 1);
        assert_eq!(cn.args[0].model, "control_v11p_sd15_canny.safetensors");
    }

    #[test]
    fn controlnet_unit_string_resize_and_control_mode() {
        // StableProjectorz sends these as human-readable strings
        let json = serde_json::json!({
            "resize_mode": "Crop and Resize",
            "control_mode": "ControlNet is more important",
        });
        let unit: ControlNetUnit = serde_json::from_value(json).expect("deserialize");
        assert_eq!(unit.resize_mode, ResizeMode::CropAndResize);
        assert_eq!(unit.control_mode, ControlMode::ControlNetPriority);
    }

    #[test]
    fn controlnet_unit_integer_resize_and_control_mode() {
        let json = serde_json::json!({
            "resize_mode": 2,
            "control_mode": 1,
        });
        let unit: ControlNetUnit = serde_json::from_value(json).expect("deserialize");
        assert_eq!(unit.resize_mode, ResizeMode::ResizeAndFill);
        assert_eq!(unit.control_mode, ControlMode::PromptPriority);
    }
}
