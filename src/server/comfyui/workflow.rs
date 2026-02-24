//! Parse ComfyUI workflow JSON into an [`ExecutionGraph`].

use super::types::ComfyNode;
use crate::node::{
    error::NodeError,
    executor::ExecutionGraph,
    nodes::{
        CheckpointLoaderSimple, ClipTextEncode, ControlNetApply, ControlNetLoader,
        EmptyLatentImage, KSampler, KSamplerAdvanced, LoraLoader, SaveImage, VaeDecode, VaeEncode,
    },
};
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
};

/// Convert a ComfyUI string node ID to a numeric u64 ID.
///
/// Parses as u64 if possible, otherwise hashes the string.
fn parse_or_hash(str_id: &str) -> u64 {
    str_id.parse().unwrap_or_else(|_| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        str_id.hash(&mut h);
        h.finish()
    })
}

/// Parse a ComfyUI workflow (map of string IDs → node defs) into an execution graph.
///
/// Each `ComfyNode` variant knows its widget fields (baked into the node at construction)
/// and its link fields (wired as edges in the second pass).
pub(super) fn parse_workflow(
    nodes: &HashMap<String, ComfyNode>,
) -> Result<ExecutionGraph, NodeError> {
    let mut graph = ExecutionGraph::new();
    let mut id_map: HashMap<&str, u64> = HashMap::new();

    // First pass: create nodes from widget values
    for (str_id, comfy_node) in nodes {
        let node: Box<dyn crate::node::Node> = match comfy_node {
            ComfyNode::CheckpointLoaderSimple(i) => {
                Box::new(CheckpointLoaderSimple::new(i.ckpt_name.clone()))
            }
            ComfyNode::ClipTextEncode(i) => Box::new(ClipTextEncode::new(i.text.clone())),
            ComfyNode::EmptyLatentImage(i) => Box::new(EmptyLatentImage::new(
                i.width as usize,
                i.height as usize,
                i.batch_size as usize,
            )),
            ComfyNode::KSampler(i) => Box::new(KSampler::new(
                i.seed,
                i.steps as usize,
                i.cfg as f32,
                i.sampler_name,
                i.scheduler,
                i.denoise as f32,
            )),
            ComfyNode::KSamplerAdvanced(i) => Box::new(KSamplerAdvanced::new(
                i.add_noise == "enable",
                i.start_at_step as usize,
                i.end_at_step as usize,
                i.seed,
                i.steps as usize,
                i.cfg as f32,
                i.sampler_name,
                i.scheduler,
            )),
            ComfyNode::LoraLoader(i) => Box::new(LoraLoader::new(
                i.lora_name.clone(),
                i.strength_model as f32,
                i.strength_clip as f32,
            )),
            ComfyNode::ControlNetLoader(i) => {
                Box::new(ControlNetLoader::new(i.control_net_name.clone()))
            }
            ComfyNode::ControlNetApply(i) => {
                Box::new(ControlNetApply::new(i.strength as f32, 0.0, 1.0))
            }
            ComfyNode::VAEDecode(_) => Box::new(VaeDecode),
            ComfyNode::VAEEncode(_) => Box::new(VaeEncode),
            ComfyNode::SaveImage(i) => Box::new(SaveImage::new(i.filename_prefix.clone())),
        };

        let numeric_id = parse_or_hash(str_id);
        graph.add_node_with_id(numeric_id, node);
        id_map.insert(str_id, numeric_id);
    }

    // Second pass: wire link fields as edges
    for (str_id, comfy_node) in nodes {
        let to_node = id_map[str_id.as_str()];
        for (to_slot, link) in comfy_node.links() {
            let from_node = *id_map
                .get(link.0.as_str())
                .ok_or_else(|| NodeError::Execution {
                    message: format!(
                        "node '{}' references unknown source node '{}'",
                        str_id, link.0
                    ),
                })?;
            graph.add_edge(from_node, link.1 as usize, to_node, to_slot);
        }
    }

    Ok(graph)
}
