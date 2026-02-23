//! Model loading utilities for safetensors checkpoints
//!
//! Handles loading SD 1.5/SDXL model weights from safetensors files.

use crate::types::Backend;
use burn::tensor::{DType, Device, Tensor, TensorData};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::{collections::HashMap, fs::File, path::Path};

/// Error type for model loading operations.
#[derive(Debug)]
pub enum LoadError {
    /// An I/O error occurred.
    Io(std::io::Error),
    /// Failed to parse safetensors format.
    SafeTensors(safetensors::SafeTensorError),
    /// Tensor has an unsupported data type.
    UnsupportedDtype(Dtype),
    /// Requested tensor was not found in the file.
    TensorNotFound(String),
    /// Tensor shape did not match expectations.
    ShapeMismatch {
        /// The expected shape.
        expected: Vec<usize>,
        /// The actual shape found.
        got: Vec<usize>,
    },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "IO error: {e}"),
            LoadError::SafeTensors(e) => write!(f, "SafeTensors error: {e}"),
            LoadError::UnsupportedDtype(dt) => write!(f, "Unsupported dtype: {dt:?}"),
            LoadError::TensorNotFound(name) => write!(f, "Tensor not found: {name}"),
            LoadError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {expected:?}, got {got:?}")
            }
        }
    }
}

impl std::error::Error for LoadError {}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

impl From<safetensors::SafeTensorError> for LoadError {
    fn from(e: safetensors::SafeTensorError) -> Self {
        LoadError::SafeTensors(e)
    }
}

/// Component prefixes in SD checkpoints.
pub mod prefix {
    /// UNet diffusion model prefix.
    pub const UNET: &str = "model.diffusion_model.";
    /// VAE (first stage model) prefix.
    pub const VAE: &str = "first_stage_model.";
    /// CLIP text encoder prefix (SD 1.5).
    pub const CLIP: &str = "cond_stage_model.";

    /// SDXL CLIP-L text encoder prefix (within merged checkpoint).
    pub const SDXL_CLIP_L: &str = "conditioner.embedders.0.transformer.text_model";
    /// SDXL OpenCLIP-G text encoder prefix (within merged checkpoint).
    pub const SDXL_CLIP_G: &str = "conditioner.embedders.1.model";
}

/// A loaded safetensors file with memory-mapped access.
pub struct SafeTensorsFile {
    _file: File,
    mmap: Mmap,
}

impl SafeTensorsFile {
    /// Open a safetensors file with memory mapping.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, LoadError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file) }?;
        Ok(Self { _file: file, mmap })
    }

    /// Parse and return a SafeTensors view.
    pub fn tensors(&self) -> Result<SafeTensors<'_>, LoadError> {
        Ok(SafeTensors::deserialize(&self.mmap)?)
    }

    /// Get the file size in bytes.
    pub fn size(&self) -> usize {
        self.mmap.len()
    }
}

/// Information about model components found in a checkpoint.
#[derive(Debug, Default)]
pub struct CheckpointInfo {
    /// Number of UNet tensors found.
    pub unet_tensors: usize,
    /// Number of VAE tensors found.
    pub vae_tensors: usize,
    /// Number of CLIP tensors found.
    pub clip_tensors: usize,
    /// Number of other tensors found.
    pub other_tensors: usize,
    /// Total file size in bytes.
    pub total_size: usize,
}

impl CheckpointInfo {
    /// Analyze a checkpoint file and return component information.
    pub fn from_file(file: &SafeTensorsFile) -> Result<Self, LoadError> {
        let tensors = file.tensors()?;
        let mut info = Self {
            total_size: file.size(),
            ..Default::default()
        };

        for (name, _) in tensors.tensors() {
            if name.starts_with(prefix::UNET) {
                info.unet_tensors += 1;
            } else if name.starts_with(prefix::VAE) {
                info.vae_tensors += 1;
            } else if name.starts_with(prefix::CLIP) {
                info.clip_tensors += 1;
            } else {
                info.other_tensors += 1;
            }
        }

        Ok(info)
    }

    /// Returns true if the checkpoint contains UNet weights.
    pub fn has_unet(&self) -> bool {
        self.unet_tensors > 0
    }

    /// Returns true if the checkpoint contains VAE weights.
    pub fn has_vae(&self) -> bool {
        self.vae_tensors > 0
    }

    /// Returns true if the checkpoint contains CLIP weights.
    pub fn has_clip(&self) -> bool {
        self.clip_tensors > 0
    }
}

/// Build a [`TensorData`] from raw safetensors bytes.
///
/// Constructs a `TensorData` with the checkpoint's native dtype, then converts
/// to the backend's float type. For an f16 backend loading an f16 checkpoint
/// this is a no-op; for mixed cases the conversion happens once at load time
/// rather than repeatedly during inference.
fn tensor_data_from_raw(
    data: &[u8],
    shape: Vec<usize>,
    dtype: Dtype,
) -> Result<TensorData, LoadError> {
    let burn_dtype = match dtype {
        Dtype::F16 => DType::F16,
        Dtype::BF16 => DType::BF16,
        Dtype::F32 => DType::F32,
        other => return Err(LoadError::UnsupportedDtype(other)),
    };
    let td = TensorData::from_bytes_vec(data.to_vec(), shape, burn_dtype);
    Ok(td.convert::<<Backend as burn::tensor::backend::Backend>::FloatElem>())
}

/// Load a tensor from a safetensors file, flattening to 1D.
pub fn load_tensor_1d(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &Device<Backend>,
) -> Result<Tensor<Backend, 1>, LoadError> {
    let view = tensors
        .tensor(name)
        .map_err(|_| LoadError::TensorNotFound(name.to_string()))?;

    let total_elements: usize = view.shape().iter().product();
    let td = tensor_data_from_raw(view.data(), vec![total_elements], view.dtype())?;
    Ok(Tensor::from_data(td, device))
}

/// Load a 2D tensor (e.g., weight matrices).
pub fn load_tensor_2d(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &Device<Backend>,
) -> Result<Tensor<Backend, 2>, LoadError> {
    let view = tensors
        .tensor(name)
        .map_err(|_| LoadError::TensorNotFound(name.to_string()))?;

    let shape = view.shape();
    if shape.len() != 2 {
        return Err(LoadError::ShapeMismatch {
            expected: vec![0, 0], // 2D
            got: shape.to_vec(),
        });
    }

    let td = tensor_data_from_raw(view.data(), shape.to_vec(), view.dtype())?;
    Ok(Tensor::from_data(td, device))
}

/// Load a 4D tensor (e.g., conv weights).
pub fn load_tensor_4d(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &Device<Backend>,
) -> Result<Tensor<Backend, 4>, LoadError> {
    let view = tensors
        .tensor(name)
        .map_err(|_| LoadError::TensorNotFound(name.to_string()))?;

    let shape = view.shape();
    if shape.len() != 4 {
        return Err(LoadError::ShapeMismatch {
            expected: vec![0, 0, 0, 0], // 4D
            got: shape.to_vec(),
        });
    }

    let td = tensor_data_from_raw(view.data(), shape.to_vec(), view.dtype())?;
    Ok(Tensor::from_data(td, device))
}

/// Get all tensor names matching a given prefix.
pub fn tensor_names_with_prefix(tensors: &SafeTensors<'_>, prefix: &str) -> Vec<String> {
    tensors
        .tensors()
        .into_iter()
        .filter_map(|(name, _)| {
            if name.starts_with(prefix) {
                Some(name)
            } else {
                None
            }
        })
        .collect()
}

/// Group tensor names by their layer structure.
pub fn group_by_layer(names: &[String], prefix: &str) -> HashMap<String, Vec<String>> {
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();

    for name in names {
        let short = name.strip_prefix(prefix).unwrap_or(name);
        // Extract layer identifier (e.g., "input_blocks.0" or "output_blocks.5")
        let layer = short.split('.').take(2).collect::<Vec<_>>().join(".");

        groups.entry(layer).or_default().push(name.clone());
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_PATH: &str = "models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors";

    #[test]
    fn open_checkpoint() {
        let path = Path::new(MODEL_PATH);
        if !path.exists() {
            println!("Model not found, skipping");
            return;
        }

        let file = SafeTensorsFile::open(path).expect("Failed to open file");
        println!("File size: {} bytes", file.size());

        let info = CheckpointInfo::from_file(&file).expect("Failed to analyze");
        println!("UNet tensors: {}", info.unet_tensors);
        println!("VAE tensors: {}", info.vae_tensors);
        println!("CLIP tensors: {}", info.clip_tensors);

        assert!(info.has_unet());
        assert!(info.has_vae());
        assert!(info.has_clip());
    }

    #[test]
    fn load_small_tensor() {
        let path = Path::new(MODEL_PATH);
        if !path.exists() {
            println!("Model not found, skipping");
            return;
        }

        let file = SafeTensorsFile::open(path).expect("Failed to open file");
        let tensors = file.tensors().expect("Failed to parse");
        let device = Default::default();

        // Load a small tensor (time embedding bias)
        let tensor = load_tensor_1d(&tensors, "model.diffusion_model.time_embed.0.bias", &device)
            .expect("Failed to load tensor");

        let shape = tensor.shape();
        println!("Loaded tensor shape: {:?}", shape.dims);
        assert_eq!(shape.dims, [1280]); // SD 1.5 time embed dim
    }

    #[test]
    fn load_weight_matrix() {
        let path = Path::new(MODEL_PATH);
        if !path.exists() {
            println!("Model not found, skipping");
            return;
        }

        let file = SafeTensorsFile::open(path).expect("Failed to open file");
        let tensors = file.tensors().expect("Failed to parse");
        let device = Default::default();

        // Load a 2D weight matrix
        let tensor = load_tensor_2d(
            &tensors,
            "model.diffusion_model.time_embed.0.weight",
            &device,
        )
        .expect("Failed to load tensor");

        let shape = tensor.shape();
        println!("Loaded weight shape: {:?}", shape.dims);
        assert_eq!(shape.dims, [1280, 320]); // time_embed linear layer
    }

    #[test]
    fn load_conv_weight() {
        let path = Path::new(MODEL_PATH);
        if !path.exists() {
            println!("Model not found, skipping");
            return;
        }

        let file = SafeTensorsFile::open(path).expect("Failed to open file");
        let tensors = file.tensors().expect("Failed to parse");
        let device = Default::default();

        // Load a 4D conv weight
        let tensor = load_tensor_4d(
            &tensors,
            "model.diffusion_model.input_blocks.0.0.weight",
            &device,
        )
        .expect("Failed to load tensor");

        let shape = tensor.shape();
        println!("Loaded conv weight shape: {:?}", shape.dims);
        assert_eq!(shape.dims, [320, 4, 3, 3]); // First conv: 4 in channels, 320 out
    }
}
