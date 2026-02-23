//! INT8 matmul performance verification
//!
//! Tests whether cubecl's INT8 matmul shows expected speedup over FP16.
//! If DP4a/integer dot product is being used, INT8 should be 2-4x faster.
//!
//! Run with: cargo test --test int8_matmul_perf --features gpu-vulkan-f16 -- --nocapture

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

/// Test i8 matmul performance relative to f16
#[test]
fn int8_vs_f16_matmul_perf() {
    let device = get_device();
    let warmup_iters = 10;
    let test_iters = 100;

    // Use sizes relevant to attention: batch=1, heads=8, seq=4096, head_dim=64
    // Q·K^T: [B*H, seq_q, head_dim] x [B*H, head_dim, seq_kv] = [B*H, seq_q, seq_kv]
    let m = 8 * 64; // batch * heads * some tiles = 512
    let n = 64; // head_dim
    let k = 4096; // seq_len

    println!("\nINT8 vs FP16 matmul performance test");
    println!("Matrix sizes: [{m}, {n}] x [{n}, {k}] = [{m}, {k}]");
    println!("Warmup: {warmup_iters} iters, Test: {test_iters} iters\n");

    // F16 test
    {
        let a_data: Vec<half::f16> = (0..m * n)
            .map(|i| half::f16::from_f32((i % 256) as f32 / 256.0 - 0.5))
            .collect();
        let b_data: Vec<half::f16> = (0..n * k)
            .map(|i| half::f16::from_f32((i % 256) as f32 / 256.0 - 0.5))
            .collect();

        let a: Tensor<Backend, 2, burn::tensor::Float> =
            Tensor::from_data(TensorData::new(a_data, [m, n]), &device);
        let b: Tensor<Backend, 2, burn::tensor::Float> =
            Tensor::from_data(TensorData::new(b_data, [n, k]), &device);

        // Warmup
        for _ in 0..warmup_iters {
            let _c = a.clone().matmul(b.clone());
        }

        // Timed test
        let start = Instant::now();
        for _ in 0..test_iters {
            let _c = a.clone().matmul(b.clone());
        }
        let f16_time = start.elapsed();

        println!(
            "FP16 matmul: {:?} total ({:.2} us/iter)",
            f16_time,
            f16_time.as_micros() as f64 / test_iters as f64
        );
    }

    // I8 test
    {
        let a_data: Vec<i8> = (0..m * n).map(|i| ((i % 127) as i8) - 63).collect();
        let b_data: Vec<i8> = (0..n * k).map(|i| ((i % 127) as i8) - 63).collect();

        let a: Tensor<Backend, 2, burn::tensor::Int> =
            Tensor::from_data(TensorData::new(a_data, [m, n]), &device);
        let b: Tensor<Backend, 2, burn::tensor::Int> =
            Tensor::from_data(TensorData::new(b_data, [n, k]), &device);

        // Warmup
        for _ in 0..warmup_iters {
            let _c = a.clone().matmul(b.clone());
        }

        // Timed test
        let start = Instant::now();
        for _ in 0..test_iters {
            let _c = a.clone().matmul(b.clone());
        }
        let i8_time = start.elapsed();

        println!(
            "INT8 matmul: {:?} total ({:.2} us/iter)",
            i8_time,
            i8_time.as_micros() as f64 / test_iters as f64
        );
    }

    println!("\nNote: If INT8 is 2-4x faster than FP16, DP4a is likely active.");
    println!("If similar or slower, we may need to verify SPIR-V codegen.");
}
