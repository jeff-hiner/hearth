//! Direct test for BlackboxAccelerated attention kernel
//!
//! This bypasses burn's fusion layer to test cubek directly.
//!
//! Run with: cargo test --test blackbox_attention_test --features gpu-vulkan-f16 -- --nocapture

#![cfg(feature = "gpu-vulkan-f16")]

use cubecl::{CubeElement, Runtime, prelude::TensorHandleRef};
use cubek::attention::{
    definition::{AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions},
    launch::{BlueprintStrategy, Strategy, launch_ref},
};

/// List all supported CMMA configurations on this device
#[test]
fn test_list_cmma_configs() {
    let device = cubecl_wgpu::WgpuDevice::DefaultDevice;
    let client = cubecl_wgpu::WgpuRuntime::client(&device);

    let props = client.properties();

    println!("\n=== Supported CMMA configurations ===");
    for config in &props.features.cmma {
        println!("  {:?}", config);
    }
    println!("Total: {} configurations", props.features.cmma.len());

    println!("\n=== Supported MMA (manual) configurations ===");
    for config in &props.features.mma {
        println!("  {:?}", config);
    }
    println!("Total: {} configurations", props.features.mma.len());

    println!("\n=== Hardware properties ===");
    println!("  num_tensor_cores: {:?}", props.hardware.num_tensor_cores);
    println!(
        "  min_tensor_cores_dim: {:?}",
        props.hardware.min_tensor_cores_dim
    );
    println!("  plane_size_min: {}", props.hardware.plane_size_min);
    println!("  plane_size_max: {}", props.hardware.plane_size_max);
}

/// Test BlackboxAccelerated attention kernel directly
#[test]
fn test_blackbox_accelerated_direct() {
    // Get the Vulkan/WGPU runtime client directly
    let device = cubecl_wgpu::WgpuDevice::DefaultDevice;
    let client = cubecl_wgpu::WgpuRuntime::client(&device);

    // Match the failing test dimensions: batch=1, heads=1, seq=16, head_dim=512
    let batch = 1usize;
    let heads = 1usize;
    let seq_q = 16usize;
    let seq_kv = 16usize;
    let head_dim = 512usize;

    let total_q_elems = batch * heads * seq_q * head_dim;
    let total_kv_elems = batch * heads * seq_kv * head_dim;
    let total_out_elems = batch * heads * seq_q * head_dim;

    println!("\nBlackboxAccelerated Direct Test");
    println!("Shape: [{batch}, {heads}, {seq_q}, {head_dim}]");

    // Create input data (f16)
    let q_data: Vec<half::f16> = (0..total_q_elems)
        .map(|i| half::f16::from_f32((i as f32 % 100.0) * 0.01 - 0.5))
        .collect();
    let k_data: Vec<half::f16> = (0..total_kv_elems)
        .map(|i| half::f16::from_f32((i as f32 % 100.0) * 0.01 - 0.5))
        .collect();
    let v_data: Vec<half::f16> = (0..total_kv_elems)
        .map(|i| half::f16::from_f32((i as f32 % 100.0) * 0.01))
        .collect();
    let out_data: Vec<half::f16> = vec![half::f16::ZERO; total_out_elems];

    // Create GPU buffers using create_from_slice with byte slices
    let q_handle = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&q_data));
    let k_handle = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&k_data));
    let v_handle = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&v_data));
    let out_handle = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&out_data));

    // Create tensor references
    let shape_q = vec![batch, heads, seq_q, head_dim];
    let strides_q = vec![heads * seq_q * head_dim, seq_q * head_dim, head_dim, 1];
    let shape_kv = vec![batch, heads, seq_kv, head_dim];
    let strides_kv = vec![heads * seq_kv * head_dim, seq_kv * head_dim, head_dim, 1];
    let shape_out = vec![batch, heads, seq_q, head_dim];
    let strides_out = vec![heads * seq_q * head_dim, seq_q * head_dim, head_dim, 1];

    // Line size for f16 is 2 bytes
    let line_size_f16 = 4usize; // 4 elements per line for f16

    let q_ref =
        unsafe { TensorHandleRef::from_raw_parts(&q_handle, &strides_q, &shape_q, line_size_f16) };
    let k_ref = unsafe {
        TensorHandleRef::from_raw_parts(&k_handle, &strides_kv, &shape_kv, line_size_f16)
    };
    let v_ref = unsafe {
        TensorHandleRef::from_raw_parts(&v_handle, &strides_kv, &shape_kv, line_size_f16)
    };
    let out_ref = unsafe {
        TensorHandleRef::from_raw_parts(&out_handle, &strides_out, &shape_out, line_size_f16)
    };

    // Set up attention types (all f16)
    let attention_types = AttentionGlobalTypes {
        query: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        key: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        value: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        mask: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U8)),
        out: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
    };

    let attention_options = AttentionOptions {
        causal: false,
        accumulator_precision: AccumulatorPrecision::Strict(cubecl::ir::StorageType::Scalar(
            cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
        )),
    };

    println!("Launching BlackboxAccelerated kernel...");

    // Launch the kernel directly
    let result = launch_ref::<cubecl_wgpu::WgpuRuntime>(
        Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(())),
        &client,
        &q_ref,
        &k_ref,
        &v_ref,
        &None, // No mask
        &out_ref,
        &attention_types,
        attention_options,
        None, // No padding, use head_dim for softmax scale
    );

    match result {
        Ok(()) => {
            println!("Kernel launched successfully!");

            // Read back results
            let output_bytes = client.read_one(out_handle);
            let output_f16: &[half::f16] = bytemuck::cast_slice(&output_bytes);

            println!(
                "Output (first 8 values): {:?}",
                &output_f16[..8.min(output_f16.len())]
                    .iter()
                    .map(|x| x.to_f32())
                    .collect::<Vec<_>>()
            );

            // Verify output is not all zeros
            let sum: f32 = output_f16.iter().map(|x| x.to_f32().abs()).sum();
            println!("Output sum: {sum}");
            assert!(sum > 0.001, "Expected non-zero output, got sum={sum}");

            println!("BlackboxAccelerated direct test PASSED!");
        }
        Err(e) => {
            println!("Kernel setup error: {e:?}");
            panic!("BlackboxAccelerated kernel failed to launch: {e:?}");
        }
    }
}

/// Test BlackboxAccelerated with f32 to see if issue is f16-specific
#[test]
#[ignore = "f32 CMMA is not supported on most GPUs"]
fn test_blackbox_accelerated_f32() {
    let device = cubecl_wgpu::WgpuDevice::DefaultDevice;
    let client = cubecl_wgpu::WgpuRuntime::client(&device);

    // Try smaller dimensions
    let batch = 1usize;
    let heads = 1usize;
    let seq_q = 4usize;
    let seq_kv = 4usize;
    let head_dim = 64usize;

    let total_q_elems = batch * heads * seq_q * head_dim;
    let total_kv_elems = batch * heads * seq_kv * head_dim;
    let total_out_elems = batch * heads * seq_q * head_dim;

    println!("\nBlackboxAccelerated F32 Direct Test");
    println!("Shape: [{batch}, {heads}, {seq_q}, {head_dim}]");

    // Create input data (f32)
    let q_data: Vec<f32> = (0..total_q_elems)
        .map(|i| (i as f32 % 100.0) * 0.01 - 0.5)
        .collect();
    let k_data: Vec<f32> = (0..total_kv_elems)
        .map(|i| (i as f32 % 100.0) * 0.01 - 0.5)
        .collect();
    let v_data: Vec<f32> = (0..total_kv_elems)
        .map(|i| (i as f32 % 100.0) * 0.01)
        .collect();
    let out_data: Vec<f32> = vec![0.0; total_out_elems];

    // Create GPU buffers
    let q_handle = client.create_from_slice(f32::as_bytes(&q_data));
    let k_handle = client.create_from_slice(f32::as_bytes(&k_data));
    let v_handle = client.create_from_slice(f32::as_bytes(&v_data));
    let out_handle = client.create_from_slice(f32::as_bytes(&out_data));

    // Create tensor references
    let shape_q = vec![batch, heads, seq_q, head_dim];
    let strides_q = vec![heads * seq_q * head_dim, seq_q * head_dim, head_dim, 1];
    let shape_kv = vec![batch, heads, seq_kv, head_dim];
    let strides_kv = vec![heads * seq_kv * head_dim, seq_kv * head_dim, head_dim, 1];
    let shape_out = vec![batch, heads, seq_q, head_dim];
    let strides_out = vec![heads * seq_q * head_dim, seq_q * head_dim, head_dim, 1];

    let line_size = 4usize;

    let q_ref =
        unsafe { TensorHandleRef::from_raw_parts(&q_handle, &strides_q, &shape_q, line_size) };
    let k_ref =
        unsafe { TensorHandleRef::from_raw_parts(&k_handle, &strides_kv, &shape_kv, line_size) };
    let v_ref =
        unsafe { TensorHandleRef::from_raw_parts(&v_handle, &strides_kv, &shape_kv, line_size) };
    let out_ref = unsafe {
        TensorHandleRef::from_raw_parts(&out_handle, &strides_out, &shape_out, line_size)
    };

    // Set up attention types (all f32)
    let attention_types = AttentionGlobalTypes {
        query: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F32,
        )),
        key: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F32,
        )),
        value: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F32,
        )),
        mask: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U8)),
        out: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F32,
        )),
    };

    let attention_options = AttentionOptions {
        causal: false,
        accumulator_precision: AccumulatorPrecision::Strict(cubecl::ir::StorageType::Scalar(
            cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
        )),
    };

    println!("Launching BlackboxAccelerated kernel (f32)...");

    let result = launch_ref::<cubecl_wgpu::WgpuRuntime>(
        Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(())),
        &client,
        &q_ref,
        &k_ref,
        &v_ref,
        &None,
        &out_ref,
        &attention_types,
        attention_options,
        None, // No padding, use head_dim for softmax scale
    );

    match result {
        Ok(()) => {
            println!("Kernel launched successfully!");

            let output_bytes = client.read_one(out_handle);
            let output_f32: &[f32] = bytemuck::cast_slice(&output_bytes);

            println!(
                "Output (first 8 values): {:?}",
                &output_f32[..8.min(output_f32.len())]
            );

            let sum: f32 = output_f32.iter().map(|x| x.abs()).sum();
            println!("Output sum: {sum}");
            assert!(sum > 0.001, "Expected non-zero output, got sum={sum}");

            println!("BlackboxAccelerated F32 direct test PASSED!");
        }
        Err(e) => {
            println!("Kernel setup error: {e:?}");
            panic!("BlackboxAccelerated kernel failed to launch: {e:?}");
        }
    }
}

/// Test BlackboxAccelerated with head_dim=64 and original_head_dim=40 (simulating padding)
/// This tests whether the softmax scale is computed correctly when using padded dimensions.
#[test]
fn test_blackbox_accelerated_with_original_head_dim() {
    let device = cubecl_wgpu::WgpuDevice::DefaultDevice;
    let client = cubecl_wgpu::WgpuRuntime::client(&device);

    // Simulate SD1.5: original head_dim=40, padded to 64
    let batch = 1usize;
    let heads = 1usize;
    let seq_q = 16usize;
    let seq_kv = 16usize;
    let padded_head_dim = 64usize; // Padded dimension (divisible by 16)
    let original_head_dim = 40usize; // Original dimension for softmax scale

    println!("\nBlackboxAccelerated with original_head_dim Test");
    println!(
        "Shape: [{batch}, {heads}, {seq_q}, {padded_head_dim}] (original_head_dim={original_head_dim})"
    );

    let total_elems = batch * heads * seq_q * padded_head_dim;
    let total_kv_elems = batch * heads * seq_kv * padded_head_dim;
    let total_out_elems = batch * heads * seq_q * padded_head_dim;

    // Create padded input data (f16) - zeros in padded region
    let mut q_data: Vec<half::f16> = vec![half::f16::ZERO; total_elems];
    let mut k_data: Vec<half::f16> = vec![half::f16::ZERO; total_kv_elems];
    let mut v_data: Vec<half::f16> = vec![half::f16::ZERO; total_kv_elems];

    // Fill only the original (non-padded) region
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_q {
                for d in 0..original_head_dim {
                    let idx = b * heads * seq_q * padded_head_dim
                        + h * seq_q * padded_head_dim
                        + s * padded_head_dim
                        + d;
                    let i = s * original_head_dim + d;
                    q_data[idx] = half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.1);
                }
            }
            for s in 0..seq_kv {
                for d in 0..original_head_dim {
                    let idx = b * heads * seq_kv * padded_head_dim
                        + h * seq_kv * padded_head_dim
                        + s * padded_head_dim
                        + d;
                    let i = s * original_head_dim + d;
                    k_data[idx] = half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.1);
                    v_data[idx] = half::f16::from_f32((i % 11) as f32 * 0.05);
                }
            }
        }
    }

    let out_data: Vec<half::f16> = vec![half::f16::ZERO; total_out_elems];

    let q_handle = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&q_data));
    let k_handle = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&k_data));
    let v_handle = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&v_data));
    let out_handle = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&out_data));

    let shape = vec![batch, heads, seq_q, padded_head_dim];
    let strides = vec![
        heads * seq_q * padded_head_dim,
        seq_q * padded_head_dim,
        padded_head_dim,
        1,
    ];
    let shape_kv = vec![batch, heads, seq_kv, padded_head_dim];
    let strides_kv = vec![
        heads * seq_kv * padded_head_dim,
        seq_kv * padded_head_dim,
        padded_head_dim,
        1,
    ];

    let line_size = 4usize;

    let q_ref = unsafe { TensorHandleRef::from_raw_parts(&q_handle, &strides, &shape, line_size) };
    let k_ref =
        unsafe { TensorHandleRef::from_raw_parts(&k_handle, &strides_kv, &shape_kv, line_size) };
    let v_ref =
        unsafe { TensorHandleRef::from_raw_parts(&v_handle, &strides_kv, &shape_kv, line_size) };
    let out_ref =
        unsafe { TensorHandleRef::from_raw_parts(&out_handle, &strides, &shape, line_size) };

    let attention_types = AttentionGlobalTypes {
        query: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        key: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        value: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        mask: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U8)),
        out: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
    };

    let attention_options = AttentionOptions {
        causal: false,
        accumulator_precision: AccumulatorPrecision::Strict(cubecl::ir::StorageType::Scalar(
            cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
        )),
    };

    println!("Launching with original_head_dim={original_head_dim}...");

    let result = launch_ref::<cubecl_wgpu::WgpuRuntime>(
        Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(())),
        &client,
        &q_ref,
        &k_ref,
        &v_ref,
        &None,
        &out_ref,
        &attention_types,
        attention_options,
        Some(original_head_dim), // Pass original head_dim for correct softmax scale
    );

    match result {
        Ok(()) => {
            println!("Kernel launched successfully!");

            let output_bytes = client.read_one(out_handle);
            let output_f16: &[half::f16] = bytemuck::cast_slice(&output_bytes);

            // Check first row (first seq position, first original_head_dim values)
            let first_8: Vec<f32> = output_f16[..8].iter().map(|x| x.to_f32()).collect();
            println!("Output (first 8 values): {:?}", first_8);

            // Check the padded region is still zero (should be, since V was zero there)
            let padded_start = original_head_dim;
            let padded_vals: Vec<f32> = output_f16[padded_start..padded_head_dim]
                .iter()
                .map(|x| x.to_f32())
                .collect();
            println!("Padded region (should be ~0): {:?}", padded_vals);

            // Check for NaN/Inf
            let has_nan = output_f16.iter().any(|x| x.to_f32().is_nan());
            let has_inf = output_f16.iter().any(|x| x.to_f32().is_infinite());
            assert!(!has_nan, "Output contains NaN");
            assert!(!has_inf, "Output contains Inf");

            let sum: f32 = output_f16.iter().map(|x| x.to_f32().abs()).sum();
            println!("Output sum: {sum}");
            assert!(sum > 0.001, "Expected non-zero output");

            println!("BlackboxAccelerated with original_head_dim test PASSED!");
        }
        Err(e) => {
            panic!("Kernel failed: {e:?}");
        }
    }
}

/// Test with multiple heads and larger sequence to match SD1.5 more closely
#[test]
fn test_blackbox_accelerated_multi_head() {
    let device = cubecl_wgpu::WgpuDevice::DefaultDevice;
    let client = cubecl_wgpu::WgpuRuntime::client(&device);

    // Closer to SD1.5: 8 heads, seq=256, head_dim=64 (padded from 40)
    let batch = 1usize;
    let heads = 8usize;
    let seq_q = 256usize;
    let seq_kv = 256usize;
    let padded_head_dim = 64usize;
    let original_head_dim = 40usize;

    println!("\nBlackboxAccelerated Multi-Head Test");
    println!(
        "Shape: [{batch}, {heads}, {seq_q}, {padded_head_dim}] (original_head_dim={original_head_dim})"
    );

    let total_elems = batch * heads * seq_q * padded_head_dim;
    let total_kv_elems = batch * heads * seq_kv * padded_head_dim;

    // Create padded input data
    let mut q_data: Vec<half::f16> = vec![half::f16::ZERO; total_elems];
    let mut k_data: Vec<half::f16> = vec![half::f16::ZERO; total_kv_elems];
    let mut v_data: Vec<half::f16> = vec![half::f16::ZERO; total_kv_elems];

    // Fill only non-padded region with deterministic values
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_q {
                for d in 0..original_head_dim {
                    let idx = b * heads * seq_q * padded_head_dim
                        + h * seq_q * padded_head_dim
                        + s * padded_head_dim
                        + d;
                    let seed = (h * 1000 + s * original_head_dim + d) as f32;
                    q_data[idx] = half::f16::from_f32(((seed as usize % 17) as f32 - 8.0) * 0.05);
                }
            }
            for s in 0..seq_kv {
                for d in 0..original_head_dim {
                    let idx = b * heads * seq_kv * padded_head_dim
                        + h * seq_kv * padded_head_dim
                        + s * padded_head_dim
                        + d;
                    let seed = (h * 1000 + s * original_head_dim + d) as f32;
                    k_data[idx] = half::f16::from_f32(((seed as usize % 13) as f32 - 6.0) * 0.05);
                    v_data[idx] = half::f16::from_f32((seed as usize % 11) as f32 * 0.02);
                }
            }
        }
    }

    let out_data: Vec<half::f16> = vec![half::f16::ZERO; total_elems];

    let q_handle = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&q_data));
    let k_handle = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&k_data));
    let v_handle = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&v_data));
    let out_handle = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&out_data));

    let shape = vec![batch, heads, seq_q, padded_head_dim];
    let strides = vec![
        heads * seq_q * padded_head_dim,
        seq_q * padded_head_dim,
        padded_head_dim,
        1,
    ];
    let shape_kv = vec![batch, heads, seq_kv, padded_head_dim];
    let strides_kv = vec![
        heads * seq_kv * padded_head_dim,
        seq_kv * padded_head_dim,
        padded_head_dim,
        1,
    ];

    let line_size = 4usize;

    let q_ref = unsafe { TensorHandleRef::from_raw_parts(&q_handle, &strides, &shape, line_size) };
    let k_ref =
        unsafe { TensorHandleRef::from_raw_parts(&k_handle, &strides_kv, &shape_kv, line_size) };
    let v_ref =
        unsafe { TensorHandleRef::from_raw_parts(&v_handle, &strides_kv, &shape_kv, line_size) };
    let out_ref =
        unsafe { TensorHandleRef::from_raw_parts(&out_handle, &strides, &shape, line_size) };

    let attention_types = AttentionGlobalTypes {
        query: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        key: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        value: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        mask: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U8)),
        out: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
    };

    let attention_options = AttentionOptions {
        causal: false,
        accumulator_precision: AccumulatorPrecision::Strict(cubecl::ir::StorageType::Scalar(
            cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
        )),
    };

    println!("Launching multi-head attention...");

    let result = launch_ref::<cubecl_wgpu::WgpuRuntime>(
        Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(())),
        &client,
        &q_ref,
        &k_ref,
        &v_ref,
        &None,
        &out_ref,
        &attention_types,
        attention_options,
        Some(original_head_dim),
    );

    match result {
        Ok(()) => {
            println!("Kernel launched successfully!");

            let output_bytes = client.read_one(out_handle);
            let output_f16: &[half::f16] = bytemuck::cast_slice(&output_bytes);

            // Check values from different heads
            for h in 0..heads.min(3) {
                let head_start = h * seq_q * padded_head_dim;
                let vals: Vec<f32> = output_f16[head_start..head_start + 4]
                    .iter()
                    .map(|x| x.to_f32())
                    .collect();
                println!("Head {h} first 4 values: {:?}", vals);
            }

            // Check for NaN/Inf
            let has_nan = output_f16.iter().any(|x| x.to_f32().is_nan());
            let has_inf = output_f16.iter().any(|x| x.to_f32().is_infinite());
            assert!(!has_nan, "Output contains NaN");
            assert!(!has_inf, "Output contains Inf");

            let sum: f32 = output_f16.iter().map(|x| x.to_f32().abs()).sum();
            println!("Output sum: {sum}");
            assert!(sum > 0.001, "Expected non-zero output");

            println!("BlackboxAccelerated multi-head test PASSED!");
        }
        Err(e) => {
            panic!("Kernel failed: {e:?}");
        }
    }
}

/// Compare Unit (reference) vs BlackboxAccelerated (CMMA) outputs
/// Uses small dimensions so Unit completes quickly
#[test]
fn test_compare_unit_vs_blackbox() {
    let device = cubecl_wgpu::WgpuDevice::DefaultDevice;
    let client = cubecl_wgpu::WgpuRuntime::client(&device);

    // Small test case - Unit uses 4x4 tiles, BlackboxAccelerated uses CMMA
    // Use head_dim=64 which is CMMA-compatible (no padding needed)
    let batch = 1usize;
    let heads = 2usize;
    let seq_q = 32usize;
    let seq_kv = 32usize;
    let head_dim = 64usize;

    println!("\n=== Unit vs BlackboxAccelerated Comparison ===");
    println!("Shape: [{batch}, {heads}, {seq_q}, {head_dim}]");

    let total_q = batch * heads * seq_q * head_dim;
    let total_kv = batch * heads * seq_kv * head_dim;

    // Create deterministic input data
    let q_data: Vec<half::f16> = (0..total_q)
        .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.1))
        .collect();
    let k_data: Vec<half::f16> = (0..total_kv)
        .map(|i| half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.1))
        .collect();
    let v_data: Vec<half::f16> = (0..total_kv)
        .map(|i| half::f16::from_f32((i % 11) as f32 * 0.05))
        .collect();

    let shape_q = vec![batch, heads, seq_q, head_dim];
    let strides_q = vec![heads * seq_q * head_dim, seq_q * head_dim, head_dim, 1];
    let shape_kv = vec![batch, heads, seq_kv, head_dim];
    let strides_kv = vec![heads * seq_kv * head_dim, seq_kv * head_dim, head_dim, 1];
    let line_size = 4usize;

    let attention_types = AttentionGlobalTypes {
        query: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        key: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        value: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        mask: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U8)),
        out: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
    };

    let attention_options = AttentionOptions {
        causal: false,
        accumulator_precision: AccumulatorPrecision::Strict(cubecl::ir::StorageType::Scalar(
            cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
        )),
    };

    // Run Unit (reference)
    println!("\nRunning Unit (reference)...");
    let q_handle_unit = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&q_data));
    let k_handle_unit = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&k_data));
    let v_handle_unit = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&v_data));
    let out_unit: Vec<half::f16> = vec![half::f16::ZERO; total_q];
    let out_handle_unit =
        client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&out_unit));

    let q_ref_unit =
        unsafe { TensorHandleRef::from_raw_parts(&q_handle_unit, &strides_q, &shape_q, line_size) };
    let k_ref_unit = unsafe {
        TensorHandleRef::from_raw_parts(&k_handle_unit, &strides_kv, &shape_kv, line_size)
    };
    let v_ref_unit = unsafe {
        TensorHandleRef::from_raw_parts(&v_handle_unit, &strides_kv, &shape_kv, line_size)
    };
    let out_ref_unit = unsafe {
        TensorHandleRef::from_raw_parts(&out_handle_unit, &strides_q, &shape_q, line_size)
    };

    launch_ref::<cubecl_wgpu::WgpuRuntime>(
        Strategy::Unit(BlueprintStrategy::Inferred(())),
        &client,
        &q_ref_unit,
        &k_ref_unit,
        &v_ref_unit,
        &None,
        &out_ref_unit,
        &attention_types,
        attention_options.clone(),
        None,
    )
    .expect("Unit kernel failed");

    let unit_bytes = client.read_one(out_handle_unit);
    let unit_output: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&unit_bytes)
        .iter()
        .map(|x| x.to_f32())
        .collect();

    println!("Unit first 8: {:?}", &unit_output[..8]);

    // Run BlackboxAccelerated (CMMA)
    println!("\nRunning BlackboxAccelerated (CMMA)...");
    let q_handle_cmma = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&q_data));
    let k_handle_cmma = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&k_data));
    let v_handle_cmma = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&v_data));
    let out_cmma: Vec<half::f16> = vec![half::f16::ZERO; total_q];
    let out_handle_cmma =
        client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&out_cmma));

    let q_ref_cmma =
        unsafe { TensorHandleRef::from_raw_parts(&q_handle_cmma, &strides_q, &shape_q, line_size) };
    let k_ref_cmma = unsafe {
        TensorHandleRef::from_raw_parts(&k_handle_cmma, &strides_kv, &shape_kv, line_size)
    };
    let v_ref_cmma = unsafe {
        TensorHandleRef::from_raw_parts(&v_handle_cmma, &strides_kv, &shape_kv, line_size)
    };
    let out_ref_cmma = unsafe {
        TensorHandleRef::from_raw_parts(&out_handle_cmma, &strides_q, &shape_q, line_size)
    };

    launch_ref::<cubecl_wgpu::WgpuRuntime>(
        Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(())),
        &client,
        &q_ref_cmma,
        &k_ref_cmma,
        &v_ref_cmma,
        &None,
        &out_ref_cmma,
        &attention_types,
        attention_options,
        None,
    )
    .expect("BlackboxAccelerated kernel failed");

    let cmma_bytes = client.read_one(out_handle_cmma);
    let cmma_output: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&cmma_bytes)
        .iter()
        .map(|x| x.to_f32())
        .collect();

    println!("CMMA first 8: {:?}", &cmma_output[..8]);

    // Compare outputs
    println!("\n=== Comparison ===");
    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f32 = 0.0;
    let mut diff_count = 0usize;

    for (i, (u, c)) in unit_output.iter().zip(cmma_output.iter()).enumerate() {
        let diff = (u - c).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        sum_diff += diff;
        if diff > 0.01 && diff_count < 10 {
            println!("  Diff at {i}: unit={u:.6}, cmma={c:.6}, diff={diff:.6}");
            diff_count += 1;
        }
    }

    let avg_diff = sum_diff / unit_output.len() as f32;
    println!("\nMax diff: {max_diff:.6}");
    println!("Avg diff: {avg_diff:.6}");
    println!("Total elements: {}", unit_output.len());

    // Allow some tolerance for f16 precision differences
    assert!(max_diff < 0.1, "Max diff {max_diff} exceeds tolerance 0.1");
    println!("\nComparison PASSED!");
}

/// Compare Unit vs BlackboxAccelerated with PADDING (head_dim=40 -> padded to 64)
/// This tests whether the padding logic is correct
#[test]
fn test_compare_unit_vs_blackbox_with_padding() {
    let device = cubecl_wgpu::WgpuDevice::DefaultDevice;
    let client = cubecl_wgpu::WgpuRuntime::client(&device);

    // SD1.5-like: head_dim=40 requires padding to 64 for CMMA
    let batch = 1usize;
    let heads = 2usize;
    let seq_q = 32usize;
    let seq_kv = 32usize;
    let original_head_dim = 40usize;
    let padded_head_dim = 64usize;

    println!("\n=== Unit vs BlackboxAccelerated WITH PADDING ===");
    println!(
        "Shape: [{batch}, {heads}, {seq_q}, {original_head_dim}] -> padded to [{batch}, {heads}, {seq_q}, {padded_head_dim}]"
    );

    let total_original = batch * heads * seq_q * original_head_dim;
    let total_padded = batch * heads * seq_q * padded_head_dim;
    let total_kv_original = batch * heads * seq_kv * original_head_dim;
    let total_kv_padded = batch * heads * seq_kv * padded_head_dim;

    // Create input data at ORIGINAL dimensions
    let q_original: Vec<half::f16> = (0..total_original)
        .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.1))
        .collect();
    let k_original: Vec<half::f16> = (0..total_kv_original)
        .map(|i| half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.1))
        .collect();
    let v_original: Vec<half::f16> = (0..total_kv_original)
        .map(|i| half::f16::from_f32((i % 11) as f32 * 0.05))
        .collect();

    // Create PADDED versions (zeros in padding region)
    let mut q_padded: Vec<half::f16> = vec![half::f16::ZERO; total_padded];
    let mut k_padded: Vec<half::f16> = vec![half::f16::ZERO; total_kv_padded];
    let mut v_padded: Vec<half::f16> = vec![half::f16::ZERO; total_kv_padded];

    // Copy original data into padded tensors
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_q {
                for d in 0..original_head_dim {
                    let orig_idx = b * heads * seq_q * original_head_dim
                        + h * seq_q * original_head_dim
                        + s * original_head_dim
                        + d;
                    let pad_idx = b * heads * seq_q * padded_head_dim
                        + h * seq_q * padded_head_dim
                        + s * padded_head_dim
                        + d;
                    q_padded[pad_idx] = q_original[orig_idx];
                }
            }
            for s in 0..seq_kv {
                for d in 0..original_head_dim {
                    let orig_idx = b * heads * seq_kv * original_head_dim
                        + h * seq_kv * original_head_dim
                        + s * original_head_dim
                        + d;
                    let pad_idx = b * heads * seq_kv * padded_head_dim
                        + h * seq_kv * padded_head_dim
                        + s * padded_head_dim
                        + d;
                    k_padded[pad_idx] = k_original[orig_idx];
                    v_padded[pad_idx] = v_original[orig_idx];
                }
            }
        }
    }

    let shape_original = vec![batch, heads, seq_q, original_head_dim];
    let strides_original = vec![
        heads * seq_q * original_head_dim,
        seq_q * original_head_dim,
        original_head_dim,
        1,
    ];
    let shape_kv_original = vec![batch, heads, seq_kv, original_head_dim];
    let strides_kv_original = vec![
        heads * seq_kv * original_head_dim,
        seq_kv * original_head_dim,
        original_head_dim,
        1,
    ];

    let shape_padded = vec![batch, heads, seq_q, padded_head_dim];
    let strides_padded = vec![
        heads * seq_q * padded_head_dim,
        seq_q * padded_head_dim,
        padded_head_dim,
        1,
    ];
    let shape_kv_padded = vec![batch, heads, seq_kv, padded_head_dim];
    let strides_kv_padded = vec![
        heads * seq_kv * padded_head_dim,
        seq_kv * padded_head_dim,
        padded_head_dim,
        1,
    ];

    let line_size = 4usize;

    let attention_types = AttentionGlobalTypes {
        query: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        key: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        value: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        mask: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U8)),
        out: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
    };

    let attention_options = AttentionOptions {
        causal: false,
        accumulator_precision: AccumulatorPrecision::Strict(cubecl::ir::StorageType::Scalar(
            cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
        )),
    };

    // Run Unit on ORIGINAL dimensions (no padding)
    println!("\nRunning Unit on original dims [{batch}, {heads}, {seq_q}, {original_head_dim}]...");
    let q_handle_unit =
        client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&q_original));
    let k_handle_unit =
        client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&k_original));
    let v_handle_unit =
        client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&v_original));
    let out_unit: Vec<half::f16> = vec![half::f16::ZERO; total_original];
    let out_handle_unit =
        client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&out_unit));

    let q_ref_unit = unsafe {
        TensorHandleRef::from_raw_parts(
            &q_handle_unit,
            &strides_original,
            &shape_original,
            line_size,
        )
    };
    let k_ref_unit = unsafe {
        TensorHandleRef::from_raw_parts(
            &k_handle_unit,
            &strides_kv_original,
            &shape_kv_original,
            line_size,
        )
    };
    let v_ref_unit = unsafe {
        TensorHandleRef::from_raw_parts(
            &v_handle_unit,
            &strides_kv_original,
            &shape_kv_original,
            line_size,
        )
    };
    let out_ref_unit = unsafe {
        TensorHandleRef::from_raw_parts(
            &out_handle_unit,
            &strides_original,
            &shape_original,
            line_size,
        )
    };

    launch_ref::<cubecl_wgpu::WgpuRuntime>(
        Strategy::Unit(BlueprintStrategy::Inferred(())),
        &client,
        &q_ref_unit,
        &k_ref_unit,
        &v_ref_unit,
        &None,
        &out_ref_unit,
        &attention_types,
        attention_options.clone(),
        None, // No padding
    )
    .expect("Unit kernel failed");

    let unit_bytes = client.read_one(out_handle_unit);
    let unit_output: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&unit_bytes)
        .iter()
        .map(|x| x.to_f32())
        .collect();

    println!("Unit first 8: {:?}", &unit_output[..8]);

    // Run BlackboxAccelerated on PADDED dimensions
    println!(
        "\nRunning BlackboxAccelerated on padded dims [{batch}, {heads}, {seq_q}, {padded_head_dim}] with original_head_dim={original_head_dim}..."
    );
    let q_handle_cmma = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&q_padded));
    let k_handle_cmma = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&k_padded));
    let v_handle_cmma = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&v_padded));
    let out_cmma: Vec<half::f16> = vec![half::f16::ZERO; total_padded];
    let out_handle_cmma =
        client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&out_cmma));

    let q_ref_cmma = unsafe {
        TensorHandleRef::from_raw_parts(&q_handle_cmma, &strides_padded, &shape_padded, line_size)
    };
    let k_ref_cmma = unsafe {
        TensorHandleRef::from_raw_parts(
            &k_handle_cmma,
            &strides_kv_padded,
            &shape_kv_padded,
            line_size,
        )
    };
    let v_ref_cmma = unsafe {
        TensorHandleRef::from_raw_parts(
            &v_handle_cmma,
            &strides_kv_padded,
            &shape_kv_padded,
            line_size,
        )
    };
    let out_ref_cmma = unsafe {
        TensorHandleRef::from_raw_parts(&out_handle_cmma, &strides_padded, &shape_padded, line_size)
    };

    launch_ref::<cubecl_wgpu::WgpuRuntime>(
        Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(())),
        &client,
        &q_ref_cmma,
        &k_ref_cmma,
        &v_ref_cmma,
        &None,
        &out_ref_cmma,
        &attention_types,
        attention_options,
        Some(original_head_dim), // Pass original for correct softmax scale
    )
    .expect("BlackboxAccelerated kernel failed");

    let cmma_bytes = client.read_one(out_handle_cmma);
    let cmma_output_padded: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&cmma_bytes)
        .iter()
        .map(|x| x.to_f32())
        .collect();

    // Extract only the original (non-padded) values from CMMA output
    let mut cmma_output: Vec<f32> = Vec::with_capacity(total_original);
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_q {
                for d in 0..original_head_dim {
                    let pad_idx = b * heads * seq_q * padded_head_dim
                        + h * seq_q * padded_head_dim
                        + s * padded_head_dim
                        + d;
                    cmma_output.push(cmma_output_padded[pad_idx]);
                }
            }
        }
    }

    println!("CMMA first 8 (unpadded): {:?}", &cmma_output[..8]);

    // Compare outputs
    println!("\n=== Comparison ===");
    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f32 = 0.0;
    let mut diff_count = 0usize;

    for (i, (u, c)) in unit_output.iter().zip(cmma_output.iter()).enumerate() {
        let diff = (u - c).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        sum_diff += diff;
        if diff > 0.01 && diff_count < 10 {
            println!("  Diff at {i}: unit={u:.6}, cmma={c:.6}, diff={diff:.6}");
            diff_count += 1;
        }
    }

    let avg_diff = sum_diff / unit_output.len() as f32;
    println!("\nMax diff: {max_diff:.6}");
    println!("Avg diff: {avg_diff:.6}");
    println!("Total elements: {}", unit_output.len());

    // Allow some tolerance for f16 precision + padding approximation
    assert!(
        max_diff < 0.1,
        "Max diff {max_diff} exceeds tolerance 0.1 - PADDING LOGIC MAY BE BROKEN"
    );
    println!("\nPadding comparison PASSED!");
}

/// Compare with SD1.5-like dimensions: batch=2, heads=8, seq=256, head_dim=40
#[test]
fn test_compare_sd15_like_dimensions() {
    let device = cubecl_wgpu::WgpuDevice::DefaultDevice;
    let client = cubecl_wgpu::WgpuRuntime::client(&device);

    // SD1.5-like dimensions
    let batch = 2usize; // CFG uses batch=2
    let heads = 8usize;
    let seq_q = 256usize; // One of the mid-resolution layers
    let seq_kv = 256usize;
    let original_head_dim = 40usize;
    let padded_head_dim = 64usize;

    println!("\n=== SD1.5-like Dimensions Test ===");
    println!(
        "Shape: [{batch}, {heads}, {seq_q}, {original_head_dim}] -> padded to [{batch}, {heads}, {seq_q}, {padded_head_dim}]"
    );

    let total_original = batch * heads * seq_q * original_head_dim;
    let total_padded = batch * heads * seq_q * padded_head_dim;
    let total_kv_original = batch * heads * seq_kv * original_head_dim;
    let total_kv_padded = batch * heads * seq_kv * padded_head_dim;

    // Create deterministic input data
    let q_original: Vec<half::f16> = (0..total_original)
        .map(|i| half::f16::from_f32(((i % 17) as f32 - 8.0) * 0.1))
        .collect();
    let k_original: Vec<half::f16> = (0..total_kv_original)
        .map(|i| half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.1))
        .collect();
    let v_original: Vec<half::f16> = (0..total_kv_original)
        .map(|i| half::f16::from_f32((i % 11) as f32 * 0.05))
        .collect();

    // Create PADDED versions
    let mut q_padded: Vec<half::f16> = vec![half::f16::ZERO; total_padded];
    let mut k_padded: Vec<half::f16> = vec![half::f16::ZERO; total_kv_padded];
    let mut v_padded: Vec<half::f16> = vec![half::f16::ZERO; total_kv_padded];

    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_q {
                for d in 0..original_head_dim {
                    let orig_idx = b * heads * seq_q * original_head_dim
                        + h * seq_q * original_head_dim
                        + s * original_head_dim
                        + d;
                    let pad_idx = b * heads * seq_q * padded_head_dim
                        + h * seq_q * padded_head_dim
                        + s * padded_head_dim
                        + d;
                    q_padded[pad_idx] = q_original[orig_idx];
                }
            }
            for s in 0..seq_kv {
                for d in 0..original_head_dim {
                    let orig_idx = b * heads * seq_kv * original_head_dim
                        + h * seq_kv * original_head_dim
                        + s * original_head_dim
                        + d;
                    let pad_idx = b * heads * seq_kv * padded_head_dim
                        + h * seq_kv * padded_head_dim
                        + s * padded_head_dim
                        + d;
                    k_padded[pad_idx] = k_original[orig_idx];
                    v_padded[pad_idx] = v_original[orig_idx];
                }
            }
        }
    }

    let shape_original = vec![batch, heads, seq_q, original_head_dim];
    let strides_original = vec![
        heads * seq_q * original_head_dim,
        seq_q * original_head_dim,
        original_head_dim,
        1,
    ];
    let shape_kv_original = vec![batch, heads, seq_kv, original_head_dim];
    let strides_kv_original = vec![
        heads * seq_kv * original_head_dim,
        seq_kv * original_head_dim,
        original_head_dim,
        1,
    ];

    let shape_padded = vec![batch, heads, seq_q, padded_head_dim];
    let strides_padded = vec![
        heads * seq_q * padded_head_dim,
        seq_q * padded_head_dim,
        padded_head_dim,
        1,
    ];
    let shape_kv_padded = vec![batch, heads, seq_kv, padded_head_dim];
    let strides_kv_padded = vec![
        heads * seq_kv * padded_head_dim,
        seq_kv * padded_head_dim,
        padded_head_dim,
        1,
    ];

    let line_size = 4usize;

    let attention_types = AttentionGlobalTypes {
        query: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        key: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        value: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
        mask: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U8)),
        out: cubecl::ir::StorageType::Scalar(cubecl::ir::ElemType::Float(
            cubecl::ir::FloatKind::F16,
        )),
    };

    let attention_options = AttentionOptions {
        causal: false,
        accumulator_precision: AccumulatorPrecision::Strict(cubecl::ir::StorageType::Scalar(
            cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
        )),
    };

    // Run Unit
    println!("\nRunning Unit...");
    let q_handle_unit =
        client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&q_original));
    let k_handle_unit =
        client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&k_original));
    let v_handle_unit =
        client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&v_original));
    let out_unit: Vec<half::f16> = vec![half::f16::ZERO; total_original];
    let out_handle_unit =
        client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&out_unit));

    let q_ref_unit = unsafe {
        TensorHandleRef::from_raw_parts(
            &q_handle_unit,
            &strides_original,
            &shape_original,
            line_size,
        )
    };
    let k_ref_unit = unsafe {
        TensorHandleRef::from_raw_parts(
            &k_handle_unit,
            &strides_kv_original,
            &shape_kv_original,
            line_size,
        )
    };
    let v_ref_unit = unsafe {
        TensorHandleRef::from_raw_parts(
            &v_handle_unit,
            &strides_kv_original,
            &shape_kv_original,
            line_size,
        )
    };
    let out_ref_unit = unsafe {
        TensorHandleRef::from_raw_parts(
            &out_handle_unit,
            &strides_original,
            &shape_original,
            line_size,
        )
    };

    let start = std::time::Instant::now();
    launch_ref::<cubecl_wgpu::WgpuRuntime>(
        Strategy::Unit(BlueprintStrategy::Inferred(())),
        &client,
        &q_ref_unit,
        &k_ref_unit,
        &v_ref_unit,
        &None,
        &out_ref_unit,
        &attention_types,
        attention_options.clone(),
        None,
    )
    .expect("Unit kernel failed");

    let unit_bytes = client.read_one(out_handle_unit);
    println!("Unit took {:?}", start.elapsed());

    let unit_output: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&unit_bytes)
        .iter()
        .map(|x| x.to_f32())
        .collect();

    println!("Unit first 8: {:?}", &unit_output[..8]);

    // Run BlackboxAccelerated
    println!("\nRunning BlackboxAccelerated...");
    let q_handle_cmma = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&q_padded));
    let k_handle_cmma = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&k_padded));
    let v_handle_cmma = client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&v_padded));
    let out_cmma: Vec<half::f16> = vec![half::f16::ZERO; total_padded];
    let out_handle_cmma =
        client.create_from_slice(bytemuck::cast_slice::<half::f16, u8>(&out_cmma));

    let q_ref_cmma = unsafe {
        TensorHandleRef::from_raw_parts(&q_handle_cmma, &strides_padded, &shape_padded, line_size)
    };
    let k_ref_cmma = unsafe {
        TensorHandleRef::from_raw_parts(
            &k_handle_cmma,
            &strides_kv_padded,
            &shape_kv_padded,
            line_size,
        )
    };
    let v_ref_cmma = unsafe {
        TensorHandleRef::from_raw_parts(
            &v_handle_cmma,
            &strides_kv_padded,
            &shape_kv_padded,
            line_size,
        )
    };
    let out_ref_cmma = unsafe {
        TensorHandleRef::from_raw_parts(&out_handle_cmma, &strides_padded, &shape_padded, line_size)
    };

    let start = std::time::Instant::now();
    launch_ref::<cubecl_wgpu::WgpuRuntime>(
        Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(())),
        &client,
        &q_ref_cmma,
        &k_ref_cmma,
        &v_ref_cmma,
        &None,
        &out_ref_cmma,
        &attention_types,
        attention_options,
        Some(original_head_dim),
    )
    .expect("BlackboxAccelerated kernel failed");

    let cmma_bytes = client.read_one(out_handle_cmma);
    println!("CMMA took {:?}", start.elapsed());

    let cmma_output_padded: Vec<f32> = bytemuck::cast_slice::<u8, half::f16>(&cmma_bytes)
        .iter()
        .map(|x| x.to_f32())
        .collect();

    // Extract only non-padded values
    let mut cmma_output: Vec<f32> = Vec::with_capacity(total_original);
    for b in 0..batch {
        for h in 0..heads {
            for s in 0..seq_q {
                for d in 0..original_head_dim {
                    let pad_idx = b * heads * seq_q * padded_head_dim
                        + h * seq_q * padded_head_dim
                        + s * padded_head_dim
                        + d;
                    cmma_output.push(cmma_output_padded[pad_idx]);
                }
            }
        }
    }

    println!("CMMA first 8 (unpadded): {:?}", &cmma_output[..8]);

    // Compare
    println!("\n=== Comparison ===");
    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f32 = 0.0;
    let mut worst_idx = 0usize;

    for (i, (u, c)) in unit_output.iter().zip(cmma_output.iter()).enumerate() {
        let diff = (u - c).abs();
        if diff > max_diff {
            max_diff = diff;
            worst_idx = i;
        }
        sum_diff += diff;
    }

    let avg_diff = sum_diff / unit_output.len() as f32;
    println!("Max diff: {max_diff:.6} at index {worst_idx}");
    println!("  Unit[{worst_idx}] = {:.6}", unit_output[worst_idx]);
    println!("  CMMA[{worst_idx}] = {:.6}", cmma_output[worst_idx]);
    println!("Avg diff: {avg_diff:.6}");
    println!("Total elements: {}", unit_output.len());

    // Show first few large diffs
    let mut diff_count = 0;
    for (i, (u, c)) in unit_output.iter().zip(cmma_output.iter()).enumerate() {
        let diff = (u - c).abs();
        if diff > 0.01 && diff_count < 5 {
            println!("  Diff at {i}: unit={u:.6}, cmma={c:.6}, diff={diff:.6}");
            diff_count += 1;
        }
    }

    assert!(
        max_diff < 0.1,
        "Max diff {max_diff} exceeds tolerance - SD1.5-like dimensions FAIL"
    );
    println!("\nSD1.5-like comparison PASSED!");
}
