//! DPT (Dense Prediction Transformer) decoder for Depth Anything V2.
//!
//! Takes four intermediate feature maps from the ViT encoder and produces
//! a single-channel depth map via reassembly, fusion, and an output head.
//!
//! Architecture:
//! 1. **Reassemble**: Strip CLS token, reshape to spatial, project to target dims,
//!    resize to target spatial resolution.
//! 2. **Fusion (RefineNet-style)**: Bottom-up feature fusion with residual conv units
//!    and bilinear upsampling.
//! 3. **Output head**: Two-stage convolution producing [B, 1, H, W] depth output.

use super::config::{DptSmall as D, VitSmall as V};
use crate::{
    model_loader::{LoadError, load_tensor_1d, load_tensor_4d},
    types::Backend,
};
use burn::{
    module::Param,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    prelude::*,
    tensor::{
        activation::relu,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};
use safetensors::SafeTensors;

/// DPT decoder that fuses multi-scale ViT features into a depth map.
#[derive(Debug)]
pub(crate) struct DptDecoder {
    /// 1x1 projection convolutions, one per feature level.
    projects: [Conv2d<Backend>; 4],
    /// Resize layers (ConvTranspose or Conv for spatial adjustment).
    resize_0: ConvTranspose2d<Backend>,
    /// Resize layer 1 (2x ConvTranspose).
    resize_1: ConvTranspose2d<Backend>,
    // resize_2 is Identity (no weights)
    /// Resize layer 3 (stride-2 Conv for downsampling).
    resize_3: Conv2d<Backend>,
    /// Scratch layer_rn convolutions (project reassembled features to `features` dim).
    layer_rn: [Conv2d<Backend>; 4],
    /// RefineNet fusion blocks (bottom-up).
    refinenets: [RefineNet; 4],
    /// Output convolution 1: features -> features/2.
    output_conv1: Conv2d<Backend>,
    /// Output convolution 2a: features/2 -> features/2.
    output_conv2_0: Conv2d<Backend>,
    /// Output convolution 2b: features/2 -> 1.
    output_conv2_2: Conv2d<Backend>,
}

impl DptDecoder {
    /// Load the DPT decoder from safetensors.
    ///
    /// Expects weights under the `depth_head.` prefix.
    pub(crate) fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let features = D::FEATURES;
        let out_ch = D::OUT_CHANNELS;

        // Project convolutions: Conv2d(384, out_ch[i], 1)
        let projects = [
            load_conv2d(
                tensors,
                &format!("{prefix}.projects.0"),
                1,
                V::EMBED_DIM,
                out_ch[0],
                1,
                device,
            )?,
            load_conv2d(
                tensors,
                &format!("{prefix}.projects.1"),
                1,
                V::EMBED_DIM,
                out_ch[1],
                1,
                device,
            )?,
            load_conv2d(
                tensors,
                &format!("{prefix}.projects.2"),
                1,
                V::EMBED_DIM,
                out_ch[2],
                1,
                device,
            )?,
            load_conv2d(
                tensors,
                &format!("{prefix}.projects.3"),
                1,
                V::EMBED_DIM,
                out_ch[3],
                1,
                device,
            )?,
        ];

        // Resize layers
        let resize_0 = load_conv_transpose2d(
            tensors,
            &format!("{prefix}.resize_layers.0"),
            out_ch[0],
            out_ch[0],
            4,
            4,
            device,
        )?;
        let resize_1 = load_conv_transpose2d(
            tensors,
            &format!("{prefix}.resize_layers.1"),
            out_ch[1],
            out_ch[1],
            2,
            2,
            device,
        )?;
        // resize_layers.2 is Identity
        let resize_3 = load_conv2d_with_stride(
            tensors,
            &format!("{prefix}.resize_layers.3"),
            3,
            out_ch[3],
            out_ch[3],
            2,
            device,
        )?;

        // Scratch layer_rn: Conv2d(out_ch[i], features, 3, pad=1, bias=False)
        let layer_rn = [
            load_conv2d_no_bias(
                tensors,
                &format!("{prefix}.scratch.layer1_rn"),
                3,
                out_ch[0],
                features,
                1,
                device,
            )?,
            load_conv2d_no_bias(
                tensors,
                &format!("{prefix}.scratch.layer2_rn"),
                3,
                out_ch[1],
                features,
                1,
                device,
            )?,
            load_conv2d_no_bias(
                tensors,
                &format!("{prefix}.scratch.layer3_rn"),
                3,
                out_ch[2],
                features,
                1,
                device,
            )?,
            load_conv2d_no_bias(
                tensors,
                &format!("{prefix}.scratch.layer4_rn"),
                3,
                out_ch[3],
                features,
                1,
                device,
            )?,
        ];

        // RefineNet fusion blocks
        let refinenets = [
            RefineNet::load(
                tensors,
                &format!("{prefix}.scratch.refinenet1"),
                features,
                device,
            )?,
            RefineNet::load(
                tensors,
                &format!("{prefix}.scratch.refinenet2"),
                features,
                device,
            )?,
            RefineNet::load(
                tensors,
                &format!("{prefix}.scratch.refinenet3"),
                features,
                device,
            )?,
            RefineNet::load(
                tensors,
                &format!("{prefix}.scratch.refinenet4"),
                features,
                device,
            )?,
        ];

        // Output head
        // output_conv1 is a single Conv2d(features, features/2, 3, pad=1) with bias
        let output_conv1 = load_conv2d(
            tensors,
            &format!("{prefix}.scratch.output_conv1"),
            3,
            features,
            features / 2,
            1,
            device,
        )?;
        // output_conv2 is Sequential: Conv2d(features/2, features/2, 3) -> ReLU -> Conv2d(features/2, 1, 1)
        let output_conv2_0 = load_conv2d(
            tensors,
            &format!("{prefix}.scratch.output_conv2.0"),
            3,
            features / 2,
            features / 2,
            1,
            device,
        )?;
        let output_conv2_2 = load_conv2d(
            tensors,
            &format!("{prefix}.scratch.output_conv2.2"),
            1,
            features / 2,
            1,
            1,
            device,
        )?;

        Ok(Self {
            projects,
            resize_0,
            resize_1,
            resize_3,
            layer_rn,
            refinenets,
            output_conv1,
            output_conv2_0,
            output_conv2_2,
        })
    }

    /// Forward pass: multi-scale features -> depth map.
    ///
    /// # Arguments
    /// * `features` - Four feature tensors from encoder layers, each `[B, N, 384]`
    /// * `patch_h` - Number of patches in height
    /// * `patch_w` - Number of patches in width
    ///
    /// # Returns
    /// Depth map `[B, 1, H_out, W_out]` where H_out = patch_h * patch_size, W_out = patch_w * patch_size.
    pub(crate) fn forward(
        &self,
        features: [Tensor<Backend, 3>; 4],
        patch_h: usize,
        patch_w: usize,
    ) -> Tensor<Backend, 4> {
        let target_h = patch_h * V::PATCH_SIZE;
        let target_w = patch_w * V::PATCH_SIZE;
        let [f0, f1, f2, f3] = features;

        // Reassemble: strip CLS, reshape to spatial, project, resize
        let layer_1 = self.reassemble(f0, patch_h, patch_w, 0);
        let layer_2 = self.reassemble(f1, patch_h, patch_w, 1);
        let layer_3 = self.reassemble(f2, patch_h, patch_w, 2);
        let layer_4 = self.reassemble(f3, patch_h, patch_w, 3);

        // Scratch layer_rn convolutions
        let layer_1_rn = self.layer_rn[0].forward(layer_1);
        let layer_2_rn = self.layer_rn[1].forward(layer_2);
        let layer_3_rn = self.layer_rn[2].forward(layer_3);
        let layer_4_rn = self.layer_rn[3].forward(layer_4);

        // Get target sizes for bilinear upsampling in refinenet blocks
        let size_3 = spatial_size(&layer_3_rn);
        let size_2 = spatial_size(&layer_2_rn);
        let size_1 = spatial_size(&layer_1_rn);

        // Bottom-up fusion
        let path_4 = self.refinenets[3].forward_single(layer_4_rn, Some(size_3));
        let path_3 = self.refinenets[2].forward_fuse(path_4, layer_3_rn, Some(size_2));
        let path_2 = self.refinenets[1].forward_fuse(path_3, layer_2_rn, Some(size_1));
        let path_1 = self.refinenets[0].forward_fuse(path_2, layer_1_rn, None);

        // Output head: conv -> bilinear upsample -> conv -> relu -> conv -> relu
        let out = self.output_conv1.forward(path_1);
        let out = interpolate(
            out,
            [target_h, target_w],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        let out = self.output_conv2_0.forward(out);
        let out = relu(out);
        let out = self.output_conv2_2.forward(out);
        relu(out)
    }

    /// Reassemble a single feature level.
    ///
    /// Steps: strip CLS token -> reshape to spatial -> project -> resize.
    fn reassemble(
        &self,
        features: Tensor<Backend, 3>,
        patch_h: usize,
        patch_w: usize,
        level: usize,
    ) -> Tensor<Backend, 4> {
        let [b, _n_plus_1, d] = features.shape().dims();

        // Strip CLS token: [B, 1+N, D] -> [B, N, D]
        let features = features.slice([0..b, 1.._n_plus_1, 0..d]);

        // Reshape to spatial: [B, N, D] -> [B, D, patch_h, patch_w]
        let features = features.swap_dims(1, 2).reshape([b, d, patch_h, patch_w]);

        // Project via 1x1 conv
        let features = self.projects[level].forward(features);

        // Resize
        match level {
            0 => self.resize_0.forward(features),
            1 => self.resize_1.forward(features),
            2 => features, // Identity
            3 => self.resize_3.forward(features),
            _ => unreachable!(),
        }
    }
}

/// Get the spatial dimensions (H, W) of a 4D tensor.
fn spatial_size(tensor: &Tensor<Backend, 4>) -> [usize; 2] {
    let dims = tensor.shape().dims::<4>();
    [dims[2], dims[3]]
}

/// RefineNet feature fusion block.
///
/// Architecture:
/// - If two inputs: `output = input1 + RCU(input2)`, then `RCU(output)`
/// - If one input: `RCU(input)`
/// - Bilinear upsample to target size (or 2x if no target specified)
/// - 1x1 output convolution
#[derive(Debug)]
struct RefineNet {
    /// First residual conv unit (applied to second input in fusion mode).
    rcu1: ResidualConvUnit,
    /// Second residual conv unit (applied to combined output).
    rcu2: ResidualConvUnit,
    /// 1x1 output convolution.
    out_conv: Conv2d<Backend>,
}

impl RefineNet {
    fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        features: usize,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let rcu1 =
            ResidualConvUnit::load(tensors, &format!("{prefix}.resConfUnit1"), features, device)?;
        let rcu2 =
            ResidualConvUnit::load(tensors, &format!("{prefix}.resConfUnit2"), features, device)?;
        let out_conv = load_conv2d(
            tensors,
            &format!("{prefix}.out_conv"),
            1,
            features,
            features,
            1,
            device,
        )?;

        Ok(Self {
            rcu1,
            rcu2,
            out_conv,
        })
    }

    /// Forward with two inputs (fusion mode).
    fn forward_fuse(
        &self,
        input1: Tensor<Backend, 4>,
        input2: Tensor<Backend, 4>,
        target_size: Option<[usize; 2]>,
    ) -> Tensor<Backend, 4> {
        let res = self.rcu1.forward(input2);
        let output = input1 + res;
        self.forward_common(output, target_size)
    }

    /// Forward with single input.
    fn forward_single(
        &self,
        input: Tensor<Backend, 4>,
        target_size: Option<[usize; 2]>,
    ) -> Tensor<Backend, 4> {
        self.forward_common(input, target_size)
    }

    /// Shared tail: RCU -> upsample -> 1x1 conv.
    fn forward_common(
        &self,
        input: Tensor<Backend, 4>,
        target_size: Option<[usize; 2]>,
    ) -> Tensor<Backend, 4> {
        let output = self.rcu2.forward(input);

        // Bilinear upsample
        let output = match target_size {
            Some(size) => interpolate(
                output,
                size,
                InterpolateOptions::new(InterpolateMode::Bilinear),
            ),
            None => {
                // Default: 2x upsample
                let [_, _, h, w] = output.shape().dims();
                interpolate(
                    output,
                    [h * 2, w * 2],
                    InterpolateOptions::new(InterpolateMode::Bilinear),
                )
            }
        };

        self.out_conv.forward(output)
    }
}

/// Residual convolution unit (pre-activation).
///
/// Architecture: `relu(x) -> conv -> relu -> conv + x`
#[derive(Debug)]
struct ResidualConvUnit {
    /// First 3x3 convolution.
    conv1: Conv2d<Backend>,
    /// Second 3x3 convolution.
    conv2: Conv2d<Backend>,
}

impl ResidualConvUnit {
    fn load(
        tensors: &SafeTensors<'_>,
        prefix: &str,
        features: usize,
        device: &Device<Backend>,
    ) -> Result<Self, LoadError> {
        let conv1 = load_conv2d(
            tensors,
            &format!("{prefix}.conv1"),
            3,
            features,
            features,
            1,
            device,
        )?;
        let conv2 = load_conv2d(
            tensors,
            &format!("{prefix}.conv2"),
            3,
            features,
            features,
            1,
            device,
        )?;

        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, x: Tensor<Backend, 4>) -> Tensor<Backend, 4> {
        let out = relu(x.clone());
        let out = self.conv1.forward(out);
        let out = relu(out);
        let out = self.conv2.forward(out);
        out + x
    }
}

// ---------------------------------------------------------------------------
// Loading helpers
// ---------------------------------------------------------------------------

/// Load a Conv2d with bias.
fn load_conv2d(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    kernel_size: usize,
    in_channels: usize,
    out_channels: usize,
    stride: usize,
    device: &Device<Backend>,
) -> Result<Conv2d<Backend>, LoadError> {
    let weight = load_tensor_4d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;

    let padding = kernel_size / 2;
    let config = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(padding, padding));

    let mut conv = config.init(device);
    conv.weight = Param::from_tensor(weight);
    conv.bias = Some(Param::from_tensor(bias));

    Ok(conv)
}

/// Load a Conv2d without bias.
fn load_conv2d_no_bias(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    kernel_size: usize,
    in_channels: usize,
    out_channels: usize,
    stride: usize,
    device: &Device<Backend>,
) -> Result<Conv2d<Backend>, LoadError> {
    let weight = load_tensor_4d(tensors, &format!("{prefix}.weight"), device)?;

    let padding = kernel_size / 2;
    let config = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(padding, padding))
        .with_bias(false);

    let mut conv = config.init(device);
    conv.weight = Param::from_tensor(weight);

    Ok(conv)
}

/// Load a Conv2d with custom stride and bias.
fn load_conv2d_with_stride(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    kernel_size: usize,
    in_channels: usize,
    out_channels: usize,
    stride: usize,
    device: &Device<Backend>,
) -> Result<Conv2d<Backend>, LoadError> {
    load_conv2d(
        tensors,
        prefix,
        kernel_size,
        in_channels,
        out_channels,
        stride,
        device,
    )
}

/// Load a ConvTranspose2d with bias.
fn load_conv_transpose2d(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    device: &Device<Backend>,
) -> Result<ConvTranspose2d<Backend>, LoadError> {
    let weight = load_tensor_4d(tensors, &format!("{prefix}.weight"), device)?;
    let bias = load_tensor_1d(tensors, &format!("{prefix}.bias"), device)?;

    let config =
        ConvTranspose2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
            .with_stride([stride, stride]);

    let mut conv_t = config.init(device);
    conv_t.weight = Param::from_tensor(weight);
    conv_t.bias = Some(Param::from_tensor(bias));

    Ok(conv_t)
}
