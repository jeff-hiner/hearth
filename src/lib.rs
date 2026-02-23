//! Hearth - A portable Rust backend for diffusion model inference
//!
//! This crate provides ComfyUI-compatible types and inference capabilities
//! using the Burn ML framework with wgpu backend for portable GPU compute.

#![warn(unreachable_pub)]
#![recursion_limit = "256"]

pub mod clip;
pub mod controlnet;
pub mod depth;
pub(crate) mod layers;
pub mod lora;
pub mod model_loader;
pub(crate) mod model_manager;
pub mod node;
pub mod sampling;
pub mod server;
pub mod startup;
pub mod types;
pub mod unet;
pub mod vae;
