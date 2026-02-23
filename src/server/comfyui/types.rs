//! ComfyUI protocol request/response types.

use crate::sampling::{SamplerKind, SchedulerKind};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::PathBuf};

/// Request body for `POST /prompt`.
#[derive(Debug, Deserialize)]
pub(super) struct PromptRequest {
    /// The workflow graph — a map from node ID strings to node definitions.
    pub prompt: HashMap<String, ComfyNode>,
    /// Optional client ID for WebSocket correlation.
    #[expect(dead_code, reason = "parsed for future WebSocket correlation")]
    #[serde(default)]
    pub client_id: Option<String>,
}

/// A connection to another node's output slot.
///
/// Wire format: `["node_id", output_slot]` (a 2-element JSON array).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(super) struct Link(pub String, pub u64);

/// A node input that is either a literal value or a link to another node's output.
///
/// Link variant is tried first so `["1", 0]` is not mistaken for a literal array.
/// Not yet used — exists for future "convert widget to input" support.
#[expect(
    dead_code,
    reason = "will be used when widget-to-input conversion is added"
)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub(super) enum Input<T> {
    /// A link to another node's output.
    Link(Link),
    /// A literal value.
    Value(T),
}

/// A single node in a ComfyUI workflow, adjacently tagged by `class_type`.
#[derive(Debug, Deserialize)]
#[serde(tag = "class_type", content = "inputs")]
pub(super) enum ComfyNode {
    /// Load a checkpoint model from disk.
    CheckpointLoaderSimple(CheckpointLoaderInputs),
    /// Encode text with a CLIP model.
    #[serde(rename = "CLIPTextEncode")]
    ClipTextEncode(ClipTextEncodeInputs),
    /// Create an empty latent image (noise).
    EmptyLatentImage(EmptyLatentInputs),
    /// Run the KSampler denoising loop.
    KSampler(KSamplerInputs),
    /// Run the KSamplerAdvanced denoising loop with explicit step control.
    KSamplerAdvanced(KSamplerAdvancedInputs),
    /// Load and apply a LoRA to model and CLIP.
    LoraLoader(LoraLoaderInputs),
    /// Load a ControlNet model from disk.
    ControlNetLoader(ControlNetLoaderInputs),
    /// Apply a ControlNet to conditioning.
    ControlNetApply(ControlNetApplyInputs),
    /// Decode latents to pixel space with a VAE.
    VAEDecode(VaeDecodeInputs),
    /// Encode pixel images to latent space with a VAE.
    VAEEncode(VaeEncodeInputs),
    /// Save output images to disk.
    SaveImage(SaveImageInputs),
}

impl ComfyNode {
    /// Return all link fields with their input slot indices.
    ///
    /// Slot indices match the order of `inputs()` on the corresponding [`Node`] impl.
    pub(super) fn links(&self) -> Vec<(usize, &Link)> {
        match self {
            Self::CheckpointLoaderSimple(_) => vec![],
            Self::ClipTextEncode(i) => vec![(0, &i.clip)],
            Self::EmptyLatentImage(_) => vec![],
            Self::KSampler(i) => vec![
                (0, &i.model),
                (1, &i.positive),
                (2, &i.negative),
                (3, &i.latent_image),
            ],
            Self::KSamplerAdvanced(i) => vec![
                (0, &i.model),
                (1, &i.positive),
                (2, &i.negative),
                (3, &i.latent_image),
            ],
            Self::LoraLoader(i) => vec![(0, &i.model), (1, &i.clip)],
            Self::ControlNetLoader(_) => vec![],
            Self::ControlNetApply(i) => {
                vec![(0, &i.conditioning), (1, &i.control_net), (2, &i.image)]
            }
            Self::VAEDecode(i) => vec![(0, &i.samples), (1, &i.vae)],
            Self::VAEEncode(i) => vec![(0, &i.pixels), (1, &i.vae)],
            Self::SaveImage(i) => vec![(0, &i.images)],
        }
    }
}

/// Inputs for [`CheckpointLoaderSimple`](crate::node::CheckpointLoaderSimple).
#[derive(Debug, Deserialize)]
pub(super) struct CheckpointLoaderInputs {
    /// Checkpoint filename relative to the models directory.
    pub ckpt_name: PathBuf,
}

/// Inputs for [`ClipTextEncode`](crate::node::ClipTextEncode).
#[derive(Debug, Deserialize)]
pub(super) struct ClipTextEncodeInputs {
    /// CLIP model (slot 0).
    pub clip: Link,
    /// Text prompt to encode.
    pub text: String,
}

/// Inputs for [`EmptyLatentImage`](crate::node::EmptyLatentImage).
#[derive(Debug, Deserialize)]
pub(super) struct EmptyLatentInputs {
    /// Image width in pixels.
    pub width: u64,
    /// Image height in pixels.
    pub height: u64,
    /// Batch size.
    #[serde(default = "default_1")]
    pub batch_size: u64,
}

/// Inputs for [`KSampler`](crate::node::KSampler).
#[derive(Debug, Deserialize)]
pub(super) struct KSamplerInputs {
    /// Diffusion model (slot 0).
    pub model: Link,
    /// Positive conditioning (slot 1).
    pub positive: Link,
    /// Negative conditioning (slot 2).
    pub negative: Link,
    /// Latent image to denoise (slot 3).
    pub latent_image: Link,
    /// Random seed.
    #[serde(default)]
    pub seed: u64,
    /// Number of sampling steps.
    #[serde(default = "default_20")]
    pub steps: u64,
    /// Classifier-free guidance scale.
    #[serde(default = "default_7")]
    pub cfg: f64,
    /// Sampling algorithm.
    #[serde(default = "default_euler")]
    pub sampler_name: SamplerKind,
    /// Sigma schedule.
    #[serde(default = "default_normal")]
    pub scheduler: SchedulerKind,
    /// Denoising strength.
    #[serde(default = "default_1f")]
    pub denoise: f64,
}

/// Inputs for [`KSamplerAdvanced`](crate::node::KSamplerAdvanced).
#[derive(Debug, Deserialize)]
pub(super) struct KSamplerAdvancedInputs {
    /// Diffusion model (slot 0).
    pub model: Link,
    /// Positive conditioning (slot 1).
    pub positive: Link,
    /// Negative conditioning (slot 2).
    pub negative: Link,
    /// Latent image to denoise (slot 3).
    pub latent_image: Link,
    /// Whether to inject initial noise (`"enable"` or `"disable"`).
    #[serde(default = "default_enable")]
    pub add_noise: String,
    /// Random seed.
    #[serde(default)]
    pub seed: u64,
    /// Number of sampling steps.
    #[serde(default = "default_20")]
    pub steps: u64,
    /// Classifier-free guidance scale.
    #[serde(default = "default_7")]
    pub cfg: f64,
    /// Sampling algorithm.
    #[serde(default = "default_euler")]
    pub sampler_name: SamplerKind,
    /// Sigma schedule.
    #[serde(default = "default_normal")]
    pub scheduler: SchedulerKind,
    /// First step to execute (0-indexed).
    #[serde(default)]
    pub start_at_step: u64,
    /// Step to stop at (exclusive; 0 = run all steps).
    #[serde(default)]
    pub end_at_step: u64,
}

/// Inputs for [`LoraLoader`](crate::node::LoraLoader).
#[derive(Debug, Deserialize)]
pub(super) struct LoraLoaderInputs {
    /// Model (slot 0). ComfyUI uses uppercase `MODEL` in the wire format.
    #[serde(rename = "MODEL")]
    pub model: Link,
    /// CLIP (slot 1). ComfyUI uses uppercase `CLIP` in the wire format.
    #[serde(rename = "CLIP")]
    pub clip: Link,
    /// LoRA filename relative to the models directory.
    pub lora_name: PathBuf,
    /// LoRA strength applied to the UNet model.
    #[serde(default = "default_1f")]
    pub strength_model: f64,
    /// LoRA strength applied to the CLIP encoder.
    #[serde(default = "default_1f")]
    pub strength_clip: f64,
}

/// Inputs for [`ControlNetLoader`](crate::node::nodes::controlnet_loader::ControlNetLoader).
#[derive(Debug, Deserialize)]
pub(super) struct ControlNetLoaderInputs {
    /// ControlNet filename relative to the models directory.
    pub control_net_name: PathBuf,
}

/// Inputs for [`ControlNetApply`](crate::node::nodes::controlnet_apply::ControlNetApply).
#[derive(Debug, Deserialize)]
pub(super) struct ControlNetApplyInputs {
    /// Conditioning to apply ControlNet to (slot 0).
    pub conditioning: Link,
    /// ControlNet model (slot 1).
    pub control_net: Link,
    /// Control image (slot 2).
    pub image: Link,
    /// ControlNet application strength.
    #[serde(default = "default_1f")]
    pub strength: f64,
}

/// Inputs for [`VaeDecode`](crate::node::VaeDecode).
#[derive(Debug, Deserialize)]
pub(super) struct VaeDecodeInputs {
    /// Latent samples (slot 0).
    pub samples: Link,
    /// VAE model (slot 1).
    pub vae: Link,
}

/// Inputs for [`VaeEncode`](crate::node::VaeEncode).
#[derive(Debug, Deserialize)]
pub(super) struct VaeEncodeInputs {
    /// Pixel images (slot 0).
    pub pixels: Link,
    /// VAE model (slot 1).
    pub vae: Link,
}

/// Inputs for [`SaveImage`](crate::node::SaveImage).
#[derive(Debug, Deserialize)]
pub(super) struct SaveImageInputs {
    /// Decoded images (slot 0).
    pub images: Link,
    /// Output filename prefix.
    #[serde(default = "default_comfyui")]
    pub filename_prefix: String,
}

fn default_1() -> u64 {
    1
}
fn default_20() -> u64 {
    20
}
fn default_7() -> f64 {
    7.0
}
fn default_1f() -> f64 {
    1.0
}
fn default_euler() -> SamplerKind {
    SamplerKind::Euler
}
fn default_normal() -> SchedulerKind {
    SchedulerKind::Normal
}
fn default_enable() -> String {
    "enable".to_string()
}
fn default_comfyui() -> String {
    "ComfyUI".to_string()
}

/// Response body for `POST /prompt`.
#[derive(Debug, Serialize)]
pub(super) struct PromptResponse {
    /// Assigned prompt ID.
    pub prompt_id: String,
}

/// Queue status response.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct QueueStatus {
    /// Number of queued prompts.
    pub queue_pending: Vec<serde_json::Value>,
    /// Currently running prompts.
    pub queue_running: Vec<serde_json::Value>,
}

/// Generic WebSocket message envelope.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct WsMessage<T> {
    /// Message type (e.g. `"status"`, `"progress"`).
    #[serde(rename = "type")]
    pub msg_type: String,
    /// Message payload.
    pub data: T,
}

/// WebSocket status message payload.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct WsStatusData {
    /// Execution status.
    pub status: WsExecInfo,
}

/// Execution info within a status message.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct WsExecInfo {
    /// Queue remaining info.
    pub exec_info: WsQueueRemaining,
}

/// Queue remaining count.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct WsQueueRemaining {
    /// Number of items remaining in queue.
    pub queue_remaining: u32,
}

/// WebSocket progress message payload.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct WsProgressData {
    /// Current step.
    pub value: usize,
    /// Total steps.
    pub max: usize,
}

/// A single entry in `GET /history/{prompt_id}` response.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct HistoryEntry {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ws_status_message_round_trip() {
        let msg = WsMessage {
            msg_type: "status".to_string(),
            data: WsStatusData {
                status: WsExecInfo {
                    exec_info: WsQueueRemaining { queue_remaining: 0 },
                },
            },
        };
        let json = serde_json::to_string(&msg).expect("serialize");
        // Verify "type" key is used in JSON (not "msg_type")
        let raw: serde_json::Value = serde_json::from_str(&json).expect("parse");
        assert!(raw.get("type").is_some(), "expected 'type' key in JSON");
        assert!(raw.get("msg_type").is_none(), "should not have 'msg_type'");

        let back: WsMessage<WsStatusData> = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.msg_type, "status");
        assert_eq!(back.data.status.exec_info.queue_remaining, 0);
    }

    #[test]
    fn ws_progress_message_round_trip() {
        let msg = WsMessage {
            msg_type: "progress".to_string(),
            data: WsProgressData { value: 5, max: 20 },
        };
        let json = serde_json::to_string(&msg).expect("serialize");
        let back: WsMessage<WsProgressData> = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.msg_type, "progress");
        assert_eq!(back.data.value, 5);
        assert_eq!(back.data.max, 20);
    }

    #[test]
    fn prompt_request_deserialize() {
        let json = serde_json::json!({
            "prompt": {
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {
                        "ckpt_name": "model.safetensors"
                    }
                },
                "2": {
                    "class_type": "KSampler",
                    "inputs": {
                        "model": ["1", 0],
                        "positive": ["3", 0],
                        "negative": ["4", 0],
                        "latent_image": ["5", 0],
                        "seed": 42
                    }
                }
            },
            "client_id": "abc-123"
        });
        let req: PromptRequest = serde_json::from_value(json).expect("deserialize");
        assert_eq!(req.prompt.len(), 2);
        assert!(
            matches!(req.prompt["1"], ComfyNode::CheckpointLoaderSimple(_)),
            "expected CheckpointLoaderSimple"
        );
        assert!(
            matches!(req.prompt["2"], ComfyNode::KSampler(_)),
            "expected KSampler"
        );
    }

    #[test]
    fn queue_status_round_trip() {
        let qs = QueueStatus {
            queue_pending: vec![],
            queue_running: vec![],
        };
        let json = serde_json::to_string(&qs).expect("serialize");
        let back: QueueStatus = serde_json::from_str(&json).expect("deserialize");
        assert!(back.queue_pending.is_empty());
        assert!(back.queue_running.is_empty());
    }

    #[test]
    fn fixture_txt2img_basic() {
        let json = include_str!("../../../tests/fixtures/comfyui/input/txt2img_basic.json");
        let req: PromptRequest = serde_json::from_str(json).expect("deserialize txt2img_basic");
        assert_eq!(req.prompt.len(), 7);

        // Verify KSampler (node "5")
        let ComfyNode::KSampler(ref ks) = req.prompt["5"] else {
            panic!("node 5 should be KSampler");
        };
        assert_eq!(ks.seed, 42);
        assert_eq!(ks.steps, 4);
        assert_eq!(ks.model.0, "1");
        assert_eq!(ks.model.1, 0);
        assert_eq!(ks.positive.0, "2");
        assert_eq!(ks.negative.0, "3");
        assert_eq!(ks.latent_image.0, "4");

        // Verify CheckpointLoaderSimple (node "1")
        let ComfyNode::CheckpointLoaderSimple(ref ckpt) = req.prompt["1"] else {
            panic!("node 1 should be CheckpointLoaderSimple");
        };
        assert_eq!(ckpt.ckpt_name, PathBuf::from("sd_xl_base_1.0.safetensors"));

        // Verify SaveImage (node "7")
        let ComfyNode::SaveImage(ref save) = req.prompt["7"] else {
            panic!("node 7 should be SaveImage");
        };
        assert_eq!(save.images.0, "6");
        assert_eq!(save.filename_prefix, "comfyui_test");
    }

    #[test]
    fn fixture_txt2img_with_lora() {
        let json = include_str!("../../../tests/fixtures/comfyui/input/txt2img_with_lora.json");
        let req: PromptRequest = serde_json::from_str(json).expect("deserialize txt2img_with_lora");
        assert_eq!(req.prompt.len(), 8);

        // Verify LoraLoader (node "2")
        let ComfyNode::LoraLoader(ref lora) = req.prompt["2"] else {
            panic!("node 2 should be LoraLoader");
        };
        assert_eq!(lora.lora_name, PathBuf::from("style_ghibli.safetensors"));
        assert!((lora.strength_model - 0.8).abs() < f64::EPSILON);
        assert!((lora.strength_clip - 0.8).abs() < f64::EPSILON);
        assert_eq!(lora.model.0, "1");
        assert_eq!(lora.model.1, 0);
        assert_eq!(lora.clip.0, "1");
        assert_eq!(lora.clip.1, 1);
    }

    #[test]
    fn fixture_txt2img_controlnet() {
        let json = include_str!("../../../tests/fixtures/comfyui/input/txt2img_controlnet.json");
        let req: PromptRequest =
            serde_json::from_str(json).expect("deserialize txt2img_controlnet");

        // Verify ControlNetApply (node "5")
        let ComfyNode::ControlNetApply(ref cna) = req.prompt["5"] else {
            panic!("node 5 should be ControlNetApply");
        };
        assert_eq!(cna.conditioning.0, "3");
        assert_eq!(cna.control_net.0, "2");
        assert_eq!(cna.image.0, "10");
        assert!((cna.strength - 0.9).abs() < f64::EPSILON);

        // Verify ControlNetLoader (node "2")
        let ComfyNode::ControlNetLoader(ref cnl) = req.prompt["2"] else {
            panic!("node 2 should be ControlNetLoader");
        };
        assert_eq!(
            cnl.control_net_name,
            PathBuf::from("control_v11p_sd15_canny.safetensors")
        );
    }

    #[test]
    fn ksampler_advanced_deserialize() {
        let json = serde_json::json!({
            "prompt": {
                "1": {
                    "class_type": "KSamplerAdvanced",
                    "inputs": {
                        "model": ["10", 0],
                        "positive": ["11", 0],
                        "negative": ["12", 0],
                        "latent_image": ["13", 0],
                        "add_noise": "enable",
                        "seed": 123,
                        "steps": 25,
                        "cfg": 8.0,
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "start_at_step": 0,
                        "end_at_step": 20
                    }
                }
            }
        });
        let req: PromptRequest = serde_json::from_value(json).expect("deserialize");
        let ComfyNode::KSamplerAdvanced(ref ksa) = req.prompt["1"] else {
            panic!("node 1 should be KSamplerAdvanced");
        };
        assert_eq!(ksa.model.0, "10");
        assert_eq!(ksa.positive.0, "11");
        assert_eq!(ksa.negative.0, "12");
        assert_eq!(ksa.latent_image.0, "13");
        assert_eq!(ksa.add_noise, "enable");
        assert_eq!(ksa.seed, 123);
        assert_eq!(ksa.steps, 25);
        assert!((ksa.cfg - 8.0).abs() < f64::EPSILON);
        assert_eq!(ksa.start_at_step, 0);
        assert_eq!(ksa.end_at_step, 20);
    }

    #[test]
    fn ksampler_advanced_deserialize_defaults() {
        let json = serde_json::json!({
            "prompt": {
                "1": {
                    "class_type": "KSamplerAdvanced",
                    "inputs": {
                        "model": ["10", 0],
                        "positive": ["11", 0],
                        "negative": ["12", 0],
                        "latent_image": ["13", 0]
                    }
                }
            }
        });
        let req: PromptRequest = serde_json::from_value(json).expect("deserialize");
        let ComfyNode::KSamplerAdvanced(ref ksa) = req.prompt["1"] else {
            panic!("node 1 should be KSamplerAdvanced");
        };
        assert_eq!(ksa.add_noise, "enable");
        assert_eq!(ksa.steps, 20);
        assert!((ksa.cfg - 7.0).abs() < f64::EPSILON);
        assert_eq!(ksa.start_at_step, 0);
        assert_eq!(ksa.end_at_step, 0);
    }

    #[test]
    fn fixture_ws_status_output() {
        let expected = include_str!("../../../tests/fixtures/comfyui/output/ws_status.json");
        let msg = WsMessage {
            msg_type: "status".to_string(),
            data: WsStatusData {
                status: WsExecInfo {
                    exec_info: WsQueueRemaining { queue_remaining: 0 },
                },
            },
        };
        let actual: serde_json::Value = serde_json::to_value(&msg).expect("serialize");
        let expected: serde_json::Value =
            serde_json::from_str(expected).expect("parse ws_status fixture");
        assert_eq!(actual, expected);
    }

    #[test]
    fn fixture_ws_progress_output() {
        let expected = include_str!("../../../tests/fixtures/comfyui/output/ws_progress.json");
        let msg = WsMessage {
            msg_type: "progress".to_string(),
            data: WsProgressData { value: 2, max: 4 },
        };
        let actual: serde_json::Value = serde_json::to_value(&msg).expect("serialize");
        let expected: serde_json::Value =
            serde_json::from_str(expected).expect("parse ws_progress fixture");
        assert_eq!(actual, expected);
    }
}
