//! Burn tensor operations test
//!
//! Verifies that the Vulkan SPIR-V backend works for basic tensor operations.
//! Run with: cargo test --test tensor_ops -- --nocapture

// Only runs with a Vulkan backend
#![cfg(any(feature = "gpu-vulkan-f16", feature = "gpu-vulkan-bf16"))]

use burn::tensor::{Device, Tensor};
use hearth::types::Backend;

fn get_device() -> Device<Backend> {
    Default::default()
}

#[test]
fn tensor_creation() {
    let device = get_device();

    // Create a tensor from data
    let data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let tensor: Tensor<Backend, 2> = Tensor::from_floats(data, &device);

    let shape = tensor.shape();
    println!("Created tensor with shape: {:?}", shape.dims);

    assert_eq!(shape.dims, [2, 3]);
}

#[test]
fn tensor_arithmetic() {
    let device = get_device();

    let a: Tensor<Backend, 2> = Tensor::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let b: Tensor<Backend, 2> = Tensor::from_floats([[5.0, 6.0], [7.0, 8.0]], &device);

    // Addition
    let sum = a.clone() + b.clone();
    let sum_data: Vec<f32> = sum.into_data().convert::<f32>().to_vec().unwrap();
    println!("Addition result: {:?}", sum_data);
    assert_eq!(sum_data, vec![6.0, 8.0, 10.0, 12.0]);

    // Multiplication (element-wise)
    let prod = a.clone() * b.clone();
    let prod_data: Vec<f32> = prod.into_data().convert::<f32>().to_vec().unwrap();
    println!("Element-wise multiplication: {:?}", prod_data);
    assert_eq!(prod_data, vec![5.0, 12.0, 21.0, 32.0]);
}

#[test]
fn tensor_matmul() {
    let device = get_device();

    // [2, 3] x [3, 2] = [2, 2]
    let a: Tensor<Backend, 2> = Tensor::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let b: Tensor<Backend, 2> = Tensor::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);

    let result = a.matmul(b);
    let result_data: Vec<f32> = result.into_data().convert::<f32>().to_vec().unwrap();

    println!("Matrix multiplication result: {:?}", result_data);

    // Expected:
    // [1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6] = [22, 28]
    // [4*1 + 5*3 + 6*5, 4*2 + 5*4 + 6*6] = [49, 64]
    assert_eq!(result_data, vec![22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn tensor_reduction() {
    let device = get_device();

    let tensor: Tensor<Backend, 2> =
        Tensor::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

    // Sum all elements
    let sum = tensor.clone().sum();
    let sum_val: f32 = sum.into_data().convert::<f32>().to_vec().unwrap()[0];
    println!("Sum of all elements: {}", sum_val);
    assert!((sum_val - 21.0).abs() < 0.1);

    // Mean
    let mean = tensor.clone().mean();
    let mean_val: f32 = mean.into_data().convert::<f32>().to_vec().unwrap()[0];
    println!("Mean: {}", mean_val);
    assert!((mean_val - 3.5).abs() < 0.1);
}

#[test]
fn tensor_activation() {
    let device = get_device();

    let tensor: Tensor<Backend, 1> = Tensor::from_floats([-2.0, -1.0, 0.0, 1.0, 2.0], &device);

    // ReLU
    let relu_result = tensor.clone().clamp_min(0.0);
    let relu_data: Vec<f32> = relu_result.into_data().convert::<f32>().to_vec().unwrap();
    println!("ReLU result: {:?}", relu_data);
    assert_eq!(relu_data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn tensor_reshape() {
    let device = get_device();

    // Create [2, 3] tensor
    let tensor: Tensor<Backend, 2> =
        Tensor::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

    // Reshape to [3, 2]
    let reshaped: Tensor<Backend, 2> = tensor.reshape([3, 2]);
    let shape = reshaped.shape();
    println!("Reshaped to: {:?}", shape.dims);
    assert_eq!(shape.dims, [3, 2]);

    // Reshape to [6]
    let flat: Tensor<Backend, 1> = reshaped.reshape([6]);
    let flat_data: Vec<f32> = flat.into_data().convert::<f32>().to_vec().unwrap();
    println!("Flattened: {:?}", flat_data);
    assert_eq!(flat_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn tensor_larger_matmul() {
    let device = get_device();

    // Test with larger matrices to exercise GPU compute
    let size = 128;

    let a: Tensor<Backend, 2> = Tensor::ones([size, size], &device);
    let b: Tensor<Backend, 2> = Tensor::ones([size, size], &device);

    let result = a.matmul(b);

    // Each element should be `size` (sum of 1s)
    let sample: f32 = result
        .slice([0..1, 0..1])
        .into_data()
        .convert::<f32>()
        .to_vec()
        .unwrap()[0];
    println!(
        "128x128 matmul sample element: {} (expected {})",
        sample, size
    );
    assert!((sample - size as f32).abs() < 1.0);
}
