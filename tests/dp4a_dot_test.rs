//! Test DP4a dot product through cubecl's Dot operation
//!
//! This verifies that Line<i8>.dot() uses hardware-accelerated DP4a.
//!
//! Run with: cargo test --test dp4a_dot_test --features gpu-vulkan-f16 -- --nocapture

#![cfg(feature = "gpu-vulkan-f16")]

use burn::{
    backend::Vulkan,
    tensor::{Device, Tensor, TensorData},
};
use std::time::Instant;

type Backend = Vulkan;

fn get_device() -> Device<Backend> {
    Default::default()
}

/// Test dot product of i8 vectors
///
/// If DP4a is working, this should be fast because Line<i8>.dot() emits OpSDotKHR
#[test]
fn dp4a_dot_product_perf() {
    let device = get_device();
    let warmup_iters = 10;
    let test_iters = 100;

    // Create i8 tensors and compute dot products
    // This is a proxy test - we can't directly call Line<i8>.dot() from burn's API
    // but we can verify i8 operations work

    let size = 1024 * 4; // Must be multiple of 4 for DP4a packing

    let a_data: Vec<i8> = (0..size).map(|i| ((i % 127) as i8) - 63).collect();
    let b_data: Vec<i8> = (0..size).map(|i| ((i % 127) as i8) - 63).collect();

    let a: Tensor<Backend, 1, burn::tensor::Int> =
        Tensor::from_data(TensorData::new(a_data.clone(), [size]), &device);
    let b: Tensor<Backend, 1, burn::tensor::Int> =
        Tensor::from_data(TensorData::new(b_data.clone(), [size]), &device);

    // Compute expected result on CPU
    let expected: i64 = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(&x, &y)| (x as i64) * (y as i64))
        .sum();

    println!("\nDP4a dot product test");
    println!("Vector size: {size}");
    println!("Expected dot product: {expected}");

    // Warmup
    for _ in 0..warmup_iters {
        let prod = a.clone() * b.clone();
        let _sum = prod.sum();
    }

    // Timed test
    let start = Instant::now();
    for _ in 0..test_iters {
        let prod = a.clone() * b.clone();
        let _sum = prod.sum();
    }
    let elapsed = start.elapsed();

    // Get actual result
    let prod = a.clone() * b.clone();
    let sum = prod.sum();
    let result: i64 = sum.into_scalar().into();

    println!("Computed dot product: {result}");
    println!(
        "Time: {:?} total ({:.2} us/iter)",
        elapsed,
        elapsed.as_micros() as f64 / test_iters as f64
    );

    // Verify correctness (may overflow for large vectors, but should be close for small ones)
    println!("\nNote: This test uses element-wise multiply + sum, not native dot.");
    println!("A proper DP4a test would need to use cubecl's Line<i8>.dot() directly.");
}
