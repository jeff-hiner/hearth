//! ComfyUI-style node system for building and executing inference graphs.
//!
//! # Architecture
//!
//! The node system has these layers:
//!
//! - **[`Node`] trait**: Each node declares typed input/output slots and an
//!   `execute` method that transforms inputs into outputs.
//! - **[`NodeValue`]**: A dynamically typed enum wrapping all edge types
//!   (tensors, model handles, scalars).
//! - **[`ExecutionGraph`] + [`Executor`]**: DAG executor that topo-sorts nodes,
//!   resolves inputs from upstream cached outputs, and runs nodes in order.
//! - **ComfyUI types** (`server::comfyui::types`): Strongly typed `ComfyNode`
//!   enum that maps `class_type` strings to typed input structs via serde.
//!
//! # Example
//!
//! ```ignore
//! use hearth::node::{ExecutionGraph, Executor, ExecutionContext};
//!
//! let mut graph = ExecutionGraph::new();
//! let empty = graph.add_node(Box::new(EmptyLatentImage::new(512, 512, 1)));
//! // ... add more nodes, connect edges ...
//! let mut ctx = ExecutionContext::new(device, models_dir, output_dir);
//! let outputs = Executor::run(&graph, &mut ctx)?;
//! ```

pub(crate) mod context;
pub(crate) mod error;
pub(crate) mod executor;
pub(crate) mod handle;
pub(crate) mod nodes;
pub(crate) mod value;
pub(crate) mod variant;

use context::ExecutionContext;
use error::NodeError;
pub use nodes::{
    CheckpointLoaderSimple, ClipTextEncode, EmptyLatentImage, KSampler, LoraLoader, SaveImage,
    VaeDecode, VaeEncode,
};
use value::NodeValue;

/// Description of a named input or output slot on a node.
#[derive(Debug, Clone, Copy)]
pub struct SlotDef {
    /// Slot name (used in ComfyUI JSON and error messages).
    pub name: &'static str,
    /// The value type this slot carries.
    pub value_type: ValueType,
    /// Whether the slot is optional (inputs only; outputs are always present).
    pub optional: bool,
}

impl SlotDef {
    /// Create a required slot.
    pub const fn required(name: &'static str, value_type: ValueType) -> Self {
        Self {
            name,
            value_type,
            optional: false,
        }
    }

    /// Create an optional slot.
    pub const fn optional(name: &'static str, value_type: ValueType) -> Self {
        Self {
            name,
            value_type,
            optional: true,
        }
    }
}

/// Type tag for node slots, used for validation and UI rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    /// Latent-space tensor.
    Latent,
    /// RGB image.
    Image,
    /// Single-channel mask.
    Mask,
    /// Encoded text conditioning.
    Conditioning,
    /// UNet / diffusion model handle.
    Model,
    /// VAE handle.
    Vae,
    /// CLIP text encoder handle.
    Clip,
    /// ControlNet handle.
    ControlNet,
    /// Scalar float.
    Float,
    /// Scalar integer.
    Int,
    /// String value.
    String,
    /// Sampling algorithm.
    Sampler,
    /// Sigma schedule selection.
    Scheduler,
    /// Pre-computed diffusion schedule.
    Schedule,
}

/// A resolved input ready for node execution.
///
/// Wraps a `NodeValue` reference with its slot name for error reporting.
#[derive(Debug)]
pub struct ResolvedInput<'a> {
    /// The slot name this input was resolved for.
    pub name: &'static str,
    /// The resolved value, or `None` if the slot is optional and unconnected.
    pub value: Option<&'a NodeValue>,
}

impl<'a> ResolvedInput<'a> {
    /// Get the value, returning a `MissingInput` error if absent.
    pub fn require(&self, node_type: &'static str) -> Result<&'a NodeValue, NodeError> {
        self.value.ok_or(NodeError::MissingInput {
            slot: self.name,
            node_type,
        })
    }
}

/// Trait for executable nodes in the graph.
///
/// Each node declares its input/output slots and provides an `execute` method.
/// The executor resolves inputs from upstream outputs, calls `execute`, and
/// caches the results.
pub trait Node: std::fmt::Debug + Send {
    /// ComfyUI-compatible class type string (e.g. `"KSampler"`).
    fn type_name(&self) -> &'static str;

    /// Input slot definitions.
    fn inputs(&self) -> &'static [SlotDef];

    /// Output slot definitions.
    fn outputs(&self) -> &'static [SlotDef];

    /// Peak scratch VRAM (bytes) needed during execution, beyond input
    /// tensors and model weights.
    ///
    /// The executor already knows input sizes. This reports only additional
    /// temporary VRAM the node's algorithm needs (intermediates, attention
    /// score matrices, etc.) which are freed when `execute` returns.
    fn scratch_vram(&self, _inputs: &[ResolvedInput]) -> u64 {
        0
    }

    /// Execute the node with resolved inputs.
    ///
    /// Returns one `NodeValue` per output slot, in order.
    fn execute(
        &self,
        inputs: &[ResolvedInput],
        ctx: &mut ExecutionContext,
    ) -> Result<Vec<NodeValue>, NodeError>;
}
