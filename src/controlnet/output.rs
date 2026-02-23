//! ControlNet output type containing residuals for UNet skip connections.

use crate::types::Backend;
use burn::prelude::*;

/// Output from a ControlNet forward pass.
///
/// Contains residual tensors that are added to the UNet's skip connections
/// and mid-block output to steer generation toward the conditioning signal.
#[derive(Debug)]
pub struct ControlNetOutput {
    /// Residuals matching UNet skip connections, in forward (push) order.
    ///
    /// The first element corresponds to the conv_in output, subsequent elements
    /// follow the same order as the UNet's skip collection.
    pub down_residuals: Vec<Tensor<Backend, 4>>,
    /// Residual for the mid block output.
    pub mid_residual: Tensor<Backend, 4>,
}

impl ControlNetOutput {
    /// Scale all residuals by a weight factor.
    pub fn scale(self, weight: f32) -> Self {
        Self {
            down_residuals: self
                .down_residuals
                .into_iter()
                .map(|t| t * weight)
                .collect(),
            mid_residual: self.mid_residual * weight,
        }
    }

    /// Sum multiple ControlNet outputs element-wise.
    ///
    /// Returns `None` if the input is empty.
    pub fn sum(outputs: Vec<Self>) -> Option<Self> {
        let mut iter = outputs.into_iter();
        let first = iter.next()?;

        Some(iter.fold(first, |acc, out| {
            let down_residuals = acc
                .down_residuals
                .into_iter()
                .zip(out.down_residuals)
                .map(|(a, b)| a + b)
                .collect();
            let mid_residual = acc.mid_residual + out.mid_residual;
            Self {
                down_residuals,
                mid_residual,
            }
        }))
    }
}
