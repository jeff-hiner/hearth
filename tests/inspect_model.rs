//! Inspect SD 1.5 model structure
//!
//! Run with: cargo test --test inspect_model -- --nocapture

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::{collections::HashMap, fs::File, path::Path};

const MODEL_PATH: &str = "models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors";

/// Information about a tensor in the model
#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    shape: Vec<usize>,
    dtype: Dtype,
    size_bytes: usize,
}

impl TensorInfo {
    fn from_view(name: &str, view: &TensorView<'_>) -> Self {
        Self {
            name: name.to_string(),
            shape: view.shape().to_vec(),
            dtype: view.dtype(),
            size_bytes: view.data().len(),
        }
    }
}

fn format_bytes(bytes: usize) -> String {
    const GB: usize = 1024 * 1024 * 1024;
    const MB: usize = 1024 * 1024;
    const KB: usize = 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Determine which component a tensor belongs to based on its name
fn classify_tensor(name: &str) -> &'static str {
    if name.starts_with("model.diffusion_model.") {
        "unet"
    } else if name.starts_with("first_stage_model.") {
        "vae"
    } else if name.starts_with("cond_stage_model.") {
        "clip"
    } else if name.starts_with("model.") {
        "other_model"
    } else {
        "other"
    }
}

#[test]
fn inspect_sd15_structure() {
    let path = Path::new(MODEL_PATH);
    if !path.exists() {
        println!("Model not found at {MODEL_PATH}");
        println!("Download from: https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive");
        return;
    }

    // Memory-map the file for efficient access
    let file = File::open(path).expect("Failed to open model file");
    let mmap = unsafe { Mmap::map(&file) }.expect("Failed to mmap file");

    println!("File size: {}", format_bytes(mmap.len()));

    // Parse safetensors
    let tensors = SafeTensors::deserialize(&mmap).expect("Failed to parse safetensors");

    // Group tensors by component
    let mut components: HashMap<&str, Vec<TensorInfo>> = HashMap::new();

    for (name, view) in tensors.tensors() {
        let component = classify_tensor(&name);
        let info = TensorInfo::from_view(&name, &view);
        components.entry(component).or_default().push(info);
    }

    println!("\n=== SD 1.5 Checkpoint Structure ===\n");

    for (component, tensors) in components.iter() {
        let total_size: usize = tensors.iter().map(|t| t.size_bytes).sum();
        println!(
            "{}: {} tensors, {}",
            component,
            tensors.len(),
            format_bytes(total_size)
        );
    }

    println!("\n=== Sample Tensors ===\n");

    // Show a few example tensors from each component
    for (component, tensors) in components.iter() {
        println!("--- {} ---", component);
        for info in tensors.iter().take(5) {
            println!("  {}: {:?} {:?}", info.name, info.shape, info.dtype);
        }
        if tensors.len() > 5 {
            println!("  ... and {} more", tensors.len() - 5);
        }
        println!();
    }
}

#[test]
fn list_unet_structure() {
    let path = Path::new(MODEL_PATH);
    if !path.exists() {
        println!("Model not found, skipping");
        return;
    }

    let file = File::open(path).expect("Failed to open model file");
    let mmap = unsafe { Mmap::map(&file) }.expect("Failed to mmap file");
    let tensors = SafeTensors::deserialize(&mmap).expect("Failed to parse safetensors");

    println!("\n=== UNet Layer Structure ===\n");

    // Group by layer type
    let mut input_blocks: Vec<String> = Vec::new();
    let mut output_blocks: Vec<String> = Vec::new();
    let mut middle_blocks: Vec<String> = Vec::new();
    let mut other: Vec<String> = Vec::new();

    const PREFIX: &str = "model.diffusion_model.";

    for (name, _) in tensors.tensors() {
        if !name.starts_with(PREFIX) {
            continue;
        }

        let short_name = name[PREFIX.len()..].to_string();

        if short_name.starts_with("input_blocks") {
            input_blocks.push(short_name);
        } else if short_name.starts_with("output_blocks") {
            output_blocks.push(short_name);
        } else if short_name.starts_with("middle_block") {
            middle_blocks.push(short_name);
        } else {
            other.push(short_name);
        }
    }

    println!("Input blocks: {} tensors", input_blocks.len());
    println!("Middle block: {} tensors", middle_blocks.len());
    println!("Output blocks: {} tensors", output_blocks.len());
    println!("Other (time_embed, out, etc.): {} tensors", other.len());

    println!("\n--- Other UNet tensors ---");
    for name in other.iter().take(20) {
        println!("  {}", name);
    }
}

#[test]
fn check_tensor_dtypes() {
    let path = Path::new(MODEL_PATH);
    if !path.exists() {
        println!("Model not found, skipping");
        return;
    }

    let file = File::open(path).expect("Failed to open model file");
    let mmap = unsafe { Mmap::map(&file) }.expect("Failed to mmap file");
    let tensors = SafeTensors::deserialize(&mmap).expect("Failed to parse safetensors");

    // Dtype doesn't implement Hash, so count manually
    let mut f16_count = 0usize;
    let mut f32_count = 0usize;
    let mut other_count = 0usize;

    for (_, tensor) in tensors.tensors() {
        match tensor.dtype() {
            Dtype::F16 => f16_count += 1,
            Dtype::F32 => f32_count += 1,
            _ => other_count += 1,
        }
    }

    println!("\n=== Tensor Data Types ===\n");
    println!("F16: {} tensors", f16_count);
    println!("F32: {} tensors", f32_count);
    if other_count > 0 {
        println!("Other: {} tensors", other_count);
    }
}
