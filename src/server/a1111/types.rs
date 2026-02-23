//! A1111 API request/response types.

use serde::{Deserialize, Serialize};

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
    /// Override checkpoint filename (optional).
    #[serde(default)]
    pub override_settings: Option<OverrideSettings>,
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
    /// Display title.
    pub title: String,
    /// Model name (filename stem).
    pub model_name: String,
    /// Filename (relative to checkpoints dir).
    pub filename: String,
}

fn default_512() -> u32 {
    512
}
fn default_20() -> u32 {
    20
}
fn default_cfg() -> f32 {
    7.0
}
fn default_sampler() -> String {
    "euler".to_string()
}
fn default_scheduler() -> String {
    "normal".to_string()
}
fn default_seed() -> i64 {
    -1
}
fn default_denoise() -> f32 {
    1.0
}
fn default_1() -> u32 {
    1
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
            title: "My Model".to_string(),
            model_name: "my_model".to_string(),
            filename: "my_model.safetensors".to_string(),
        };
        let json = serde_json::to_string(&model).expect("serialize");
        let back: SdModel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.title, model.title);
        assert_eq!(back.model_name, model.model_name);
        assert_eq!(back.filename, model.filename);
    }
}
