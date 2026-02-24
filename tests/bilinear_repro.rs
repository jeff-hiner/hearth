//! Minimal reproduction for the bilinear interpolation bug in the DPT decoder.
//!
//! The decoder's output_conv1 produces [1, 32, 296, 296] which is then
//! bilinear-upsampled to [1, 32, 518, 518]. The result has zeros in the
//! bottom ~half of the spatial dims despite valid input everywhere.
//!
//! Run: cargo test --test bilinear_repro -- --nocapture

use burn::{
    nn::{PaddingConfig2d, conv::Conv2dConfig},
    prelude::*,
    tensor::{
        Device, TensorData,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};
use hearth::types::Backend;

/// Check that bilinear upsample preserves nonzero values for multi-channel tensors.
#[test]
fn bilinear_ones_1ch_296_to_518() {
    let device: Device<Backend> = Default::default();
    let input: Tensor<Backend, 4> = Tensor::ones([1, 1, 296, 296], &device);
    let output = interpolate(
        input,
        [518, 518],
        InterpolateOptions::new(InterpolateMode::Bilinear),
    );
    let data: Vec<f32> = output.into_data().convert::<f32>().to_vec().unwrap();
    // Check several rows
    for row in [0, 100, 200, 300, 400, 500, 517] {
        let val = data[row * 518 + 259];
        println!("1ch ones: row {row} mid = {val:.6}");
        assert!(
            (val - 1.0).abs() < 0.01,
            "row {row}: expected ~1.0, got {val}"
        );
    }
}

#[test]
fn bilinear_ones_32ch_296_to_518() {
    let device: Device<Backend> = Default::default();
    let input: Tensor<Backend, 4> = Tensor::ones([1, 32, 296, 296], &device);
    let output = interpolate(
        input,
        [518, 518],
        InterpolateOptions::new(InterpolateMode::Bilinear),
    );
    let [_, _, h, w] = output.shape().dims();
    assert_eq!((h, w), (518, 518));

    // Check channel 0
    let ch0: Tensor<Backend, 1> = output
        .clone()
        .slice([0..1_usize, 0..1_usize])
        .reshape([518 * 518]);
    let data: Vec<f32> = ch0.into_data().convert::<f32>().to_vec().unwrap();
    for row in [0, 100, 200, 300, 400, 500, 517] {
        let val = data[row * 518 + 259];
        println!("32ch ones ch0: row {row} mid = {val:.6}");
        assert!(
            (val - 1.0).abs() < 0.01,
            "row {row}: expected ~1.0, got {val}"
        );
    }

    // Check channel 16
    let ch16: Tensor<Backend, 1> = output
        .slice([0..1_usize, 16..17_usize])
        .reshape([518 * 518]);
    let data: Vec<f32> = ch16.into_data().convert::<f32>().to_vec().unwrap();
    for row in [0, 100, 200, 300, 400, 500, 517] {
        let val = data[row * 518 + 259];
        println!("32ch ones ch16: row {row} mid = {val:.6}");
        assert!(
            (val - 1.0).abs() < 0.01,
            "row {row}: expected ~1.0, got {val}"
        );
    }
}

/// Test with a gradient pattern that makes spatial errors obvious.
/// Channel c, row r, col c_ gets value (r * 296 + c_) as f32, normalized.
#[test]
fn bilinear_gradient_32ch_296_to_518() {
    let device: Device<Backend> = Default::default();

    // Create a simple spatial gradient: value = row / 295
    // So row 0 = 0.0, row 295 = 1.0
    let h_in = 296;
    let w_in = 296;
    let ch = 32;
    let mut data = vec![0.0f32; ch * h_in * w_in];
    for c in 0..ch {
        for r in 0..h_in {
            let val = r as f32 / (h_in - 1) as f32; // 0.0 to 1.0
            for col in 0..w_in {
                data[c * h_in * w_in + r * w_in + col] = val;
            }
        }
    }

    let td = TensorData::new(data, [1, ch, h_in, w_in])
        .convert::<<Backend as burn::tensor::backend::Backend>::FloatElem>();
    let input: Tensor<Backend, 4> = Tensor::from_data(td, &device);

    let output = interpolate(
        input,
        [518, 518],
        InterpolateOptions::new(InterpolateMode::Bilinear),
    );

    // Check channel 0
    let ch0: Tensor<Backend, 1> = output
        .clone()
        .slice([0..1_usize, 0..1_usize])
        .reshape([518 * 518]);
    let vals: Vec<f32> = ch0.into_data().convert::<f32>().to_vec().unwrap();

    println!("gradient 32ch → 518x518, ch0:");
    for row in [0, 100, 200, 300, 400, 500, 517] {
        let val = vals[row * 518 + 259];
        // Expected: row maps to approximately row * (295/517) in input
        // Value at that input row ≈ input_row / 295
        // So expected ≈ row / 517 (roughly)
        let expected = row as f32 / 517.0;
        println!("  row {row}: got {val:.6}, expected ~{expected:.6}");
        assert!(
            (val - expected).abs() < 0.05,
            "row {row}: expected ~{expected:.4}, got {val:.4}"
        );
    }

    // Check channel 31 (same pattern, should give same results)
    let ch31: Tensor<Backend, 1> = output
        .slice([0..1_usize, 31..32_usize])
        .reshape([518 * 518]);
    let vals: Vec<f32> = ch31.into_data().convert::<f32>().to_vec().unwrap();
    println!("gradient 32ch → 518x518, ch31:");
    for row in [0, 100, 200, 300, 400, 500, 517] {
        let val = vals[row * 518 + 259];
        let expected = row as f32 / 517.0;
        println!("  row {row}: got {val:.6}, expected ~{expected:.6}");
        assert!(
            (val - expected).abs() < 0.05,
            "row {row}: expected ~{expected:.4}, got {val:.4}"
        );
    }
}

/// Test bilinear on a tensor that went through a Conv2d first (like output_conv1).
/// This tests whether fusion or non-contiguous memory from conv output matters.
#[test]
fn bilinear_after_conv2d_32ch() {
    let device: Device<Backend> = Default::default();

    // Create a 64→32 conv like output_conv1
    let config = Conv2dConfig::new([64, 32], [3, 3]).with_padding(PaddingConfig2d::Explicit(1, 1));
    let conv = config.init::<Backend>(&device);

    // Create input: [1, 64, 296, 296] with a spatial gradient
    let h = 296;
    let w = 296;
    let ch = 64;
    let mut data = vec![0.5f32; ch * h * w]; // constant 0.5
    // Add a row-varying component to channel 0
    for r in 0..h {
        for c in 0..w {
            data[r * w + c] = r as f32 / (h - 1) as f32;
        }
    }
    let td = TensorData::new(data, [1, ch, h, w])
        .convert::<<Backend as burn::tensor::backend::Backend>::FloatElem>();
    let input: Tensor<Backend, 4> = Tensor::from_data(td, &device);

    // Conv forward
    let conv_out = conv.forward(input);
    let [_, c_out, h_out, w_out] = conv_out.shape().dims();
    println!("conv output shape: [{c_out}, {h_out}, {w_out}]");

    // Now bilinear upsample
    // NOTE: Without a GPU sync between conv2d and interpolate, this fails
    // due to a Burn fusion bug. See src/depth/decoder.rs for the workaround.
    let upsampled = interpolate(
        conv_out,
        [518, 518],
        InterpolateOptions::new(InterpolateMode::Bilinear),
    );

    // Check that output has non-trivial values across the full spatial extent
    let ch0: Tensor<Backend, 1> = upsampled
        .slice([0..1_usize, 0..1_usize])
        .reshape([518 * 518]);
    let vals: Vec<f32> = ch0.into_data().convert::<f32>().to_vec().unwrap();

    println!("after conv+bilinear ch0:");
    let mut any_nonzero_late = false;
    for row in [0, 100, 200, 300, 400, 500, 517] {
        let val = vals[row * 518 + 259];
        println!("  row {row}: {val:.6}");
        if row >= 300 && val.abs() > 1e-4 {
            any_nonzero_late = true;
        }
    }
    assert!(
        any_nonzero_late,
        "All values in rows 300+ are zero — bilinear bug!"
    );
}

/// Test with a manually-created NHWC-contiguous tensor that's permuted to NCHW,
/// mimicking what conv2d output looks like (data is NHWC, view is NCHW).
#[test]
fn bilinear_nhwc_contiguous_32ch_296_to_518() {
    let device: Device<Backend> = Default::default();

    let h = 296_usize;
    let w = 296_usize;
    let ch = 32_usize;

    // Create NHWC data: value = row / 295 (spatial gradient)
    let mut data = vec![0.5f32; h * w * ch];
    for r in 0..h {
        for c in 0..w {
            let val = r as f32 / (h - 1) as f32;
            for k in 0..ch {
                data[r * w * ch + c * ch + k] = val;
            }
        }
    }

    // Build as NHWC [1, 296, 296, 32]
    let td = TensorData::new(data, [1, h, w, ch])
        .convert::<<Backend as burn::tensor::backend::Backend>::FloatElem>();
    let nhwc: Tensor<Backend, 4> = Tensor::from_data(td, &device);

    // Permute to NCHW [1, 32, 296, 296] — this matches how conv_forward returns its tensor
    let nchw = nhwc.permute([0, 3, 1, 2]);
    let [_, c_out, h_out, w_out] = nchw.shape().dims();
    println!("nchw shape: [{c_out}, {h_out}, {w_out}]");

    // Test BOTH nearest and bilinear
    let upsampled_nearest = interpolate(
        nchw.clone(),
        [518, 518],
        InterpolateOptions::new(InterpolateMode::Nearest),
    );

    // Check nearest first
    let ch0_n: Tensor<Backend, 1> = upsampled_nearest
        .slice([0..1_usize, 0..1_usize])
        .reshape([518 * 518]);
    let vals_n: Vec<f32> = ch0_n.into_data().convert::<f32>().to_vec().unwrap();
    println!("nhwc-contiguous NEAREST ch0:");
    for row in [0, 100, 200, 300, 400, 500, 517] {
        let val = vals_n[row * 518 + 259];
        let expected = row as f32 / 517.0;
        println!("  row {row}: got {val:.6}, expected ~{expected:.6}");
    }

    // Bilinear upsample (this goes through interpolate which does NCHW→NHWC internally)
    let upsampled = interpolate(
        nchw,
        [518, 518],
        InterpolateOptions::new(InterpolateMode::Bilinear),
    );

    // Check channel 0
    let ch0: Tensor<Backend, 1> = upsampled
        .slice([0..1_usize, 0..1_usize])
        .reshape([518 * 518]);
    let vals: Vec<f32> = ch0.into_data().convert::<f32>().to_vec().unwrap();

    println!("nhwc-contiguous bilinear ch0:");
    for row in [0, 100, 200, 300, 400, 500, 517] {
        let val = vals[row * 518 + 259];
        let expected = row as f32 / 517.0;
        println!("  row {row}: got {val:.6}, expected ~{expected:.6}");
        assert!(
            (val - expected).abs() < 0.05,
            "row {row}: expected ~{expected:.4}, got {val:.4}"
        );
    }
}
