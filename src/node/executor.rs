//! DAG executor for node graphs.
//!
//! Single-threaded execution: GPU ops must be serial (one device). The executor
//! topo-sorts nodes, iterates in order, resolves inputs from upstream cached
//! outputs, and caches results for downstream consumption.

use super::{
    Node, ResolvedInput,
    context::ExecutionContext,
    error::{NodeError, NodeId},
    value::NodeValue,
};
use std::collections::HashMap;
use tracing::{info, info_span};

/// An edge in the execution graph connecting an output slot to an input slot.
#[derive(Debug, Clone)]
pub(crate) struct Edge {
    /// Source node ID.
    pub from_node: NodeId,
    /// Source output slot index.
    pub from_slot: usize,
    /// Destination node ID.
    pub to_node: NodeId,
    /// Destination input slot index.
    pub to_slot: usize,
}

/// A node entry in the execution graph.
#[derive(Debug)]
struct GraphNode {
    /// The node implementation.
    node: Box<dyn Node>,
}

/// A directed acyclic graph of nodes to execute.
#[derive(Debug)]
pub(crate) struct ExecutionGraph {
    /// Nodes keyed by their ID.
    nodes: HashMap<NodeId, GraphNode>,
    /// All edges in the graph.
    edges: Vec<Edge>,
    /// Next node ID to assign.
    next_id: NodeId,
    /// Optional constant values for input slots that aren't connected to
    /// other nodes (e.g. widget values in ComfyUI).
    constants: HashMap<(NodeId, usize), NodeValue>,
}

impl ExecutionGraph {
    /// Create an empty execution graph.
    pub(crate) fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            next_id: 1,
            constants: HashMap::new(),
        }
    }

    /// Add a node to the graph and return its ID.
    pub(crate) fn add_node(&mut self, node: Box<dyn Node>) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, GraphNode { node });
        id
    }

    /// Add a node with a specific ID (used when parsing ComfyUI workflows).
    pub(crate) fn add_node_with_id(&mut self, id: NodeId, node: Box<dyn Node>) {
        self.nodes.insert(id, GraphNode { node });
        if id >= self.next_id {
            self.next_id = id + 1;
        }
    }

    /// Connect an output slot of one node to an input slot of another.
    pub(crate) fn add_edge(
        &mut self,
        from_node: NodeId,
        from_slot: usize,
        to_node: NodeId,
        to_slot: usize,
    ) {
        self.edges.push(Edge {
            from_node,
            from_slot,
            to_node,
            to_slot,
        });
    }

    /// Set a constant value for an input slot (widget values, injected images).
    pub(crate) fn set_constant(&mut self, node_id: NodeId, slot: usize, value: NodeValue) {
        self.constants.insert((node_id, slot), value);
    }

    /// Topological sort using Kahn's algorithm.
    ///
    /// Returns node IDs in execution order (dependencies first).
    fn topo_sort(&self) -> Result<Vec<NodeId>, NodeError> {
        // Build adjacency and in-degree
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut dependents: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for &id in self.nodes.keys() {
            in_degree.entry(id).or_insert(0);
        }

        for edge in &self.edges {
            *in_degree.entry(edge.to_node).or_insert(0) += 1;
            dependents
                .entry(edge.from_node)
                .or_default()
                .push(edge.to_node);
        }

        // Start with nodes that have no incoming edges
        let mut queue: Vec<NodeId> = in_degree
            .iter()
            .filter(|&(_, deg)| *deg == 0)
            .map(|(&id, _)| id)
            .collect();
        queue.sort(); // deterministic order

        let mut order = Vec::with_capacity(self.nodes.len());

        while let Some(id) = queue.pop() {
            order.push(id);
            if let Some(deps) = dependents.get(&id) {
                for &dep in deps {
                    let deg = in_degree
                        .get_mut(&dep)
                        .expect("node in edge but not in graph");
                    *deg -= 1;
                    if *deg == 0 {
                        // Insert sorted to keep deterministic
                        let pos = queue.partition_point(|&x| x > dep);
                        queue.insert(pos, dep);
                    }
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(NodeError::CycleDetected);
        }

        Ok(order)
    }
}

impl Default for ExecutionGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Executes an [`ExecutionGraph`] in topological order.
pub(crate) struct Executor;

impl Executor {
    /// Execute all nodes in the graph, returning cached outputs.
    ///
    /// Outputs are keyed by `(NodeId, slot_index)`.
    pub(crate) fn run(
        graph: &ExecutionGraph,
        ctx: &mut ExecutionContext,
    ) -> Result<HashMap<(NodeId, usize), NodeValue>, NodeError> {
        let order = graph.topo_sort()?;
        let mut cache: HashMap<(NodeId, usize), NodeValue> = HashMap::new();

        info!(num_nodes = order.len(), "executing graph");

        for &node_id in &order {
            let graph_node = graph
                .nodes
                .get(&node_id)
                .ok_or(NodeError::UnknownNode { id: node_id })?;
            let node = &graph_node.node;
            let input_defs = node.inputs();

            let _span = info_span!("node", id = node_id, ty = node.type_name()).entered();

            // Resolve inputs for this node
            let mut resolved = Vec::with_capacity(input_defs.len());
            for (slot_idx, slot_def) in input_defs.iter().enumerate() {
                // Check for an edge feeding this slot
                let edge = graph
                    .edges
                    .iter()
                    .find(|e| e.to_node == node_id && e.to_slot == slot_idx);

                let value = if let Some(edge) = edge {
                    // Get from upstream cache
                    let key = (edge.from_node, edge.from_slot);
                    cache.get(&key)
                } else {
                    // Check constants
                    graph.constants.get(&(node_id, slot_idx))
                };

                resolved.push(ResolvedInput {
                    name: slot_def.name,
                    value,
                });
            }

            // Execute node
            info!(node_type = node.type_name(), "executing");
            let outputs = node.execute(&resolved, ctx)?;

            // Validate output count
            let expected_outputs = node.outputs().len();
            if outputs.len() != expected_outputs {
                return Err(NodeError::Execution {
                    message: format!(
                        "node {} ({}) returned {} outputs, expected {}",
                        node_id,
                        node.type_name(),
                        outputs.len(),
                        expected_outputs,
                    ),
                });
            }

            // Cache outputs
            for (slot_idx, value) in outputs.into_iter().enumerate() {
                cache.insert((node_id, slot_idx), value);
            }
        }

        Ok(cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{SlotDef, ValueType};

    /// A trivial test node that outputs a constant float.
    #[derive(Debug)]
    struct ConstFloat(f32);

    impl Node for ConstFloat {
        fn type_name(&self) -> &'static str {
            "ConstFloat"
        }
        fn inputs(&self) -> &'static [SlotDef] {
            &[]
        }
        fn outputs(&self) -> &'static [SlotDef] {
            static OUTPUTS: [SlotDef; 1] = [SlotDef::required("value", ValueType::Float)];
            &OUTPUTS
        }
        fn execute(
            &self,
            _inputs: &[ResolvedInput],
            _ctx: &mut ExecutionContext,
        ) -> Result<Vec<NodeValue>, NodeError> {
            Ok(vec![NodeValue::Float(self.0)])
        }
    }

    /// A test node that adds two floats.
    #[derive(Debug)]
    struct AddFloats;

    impl Node for AddFloats {
        fn type_name(&self) -> &'static str {
            "AddFloats"
        }
        fn inputs(&self) -> &'static [SlotDef] {
            static INPUTS: [SlotDef; 2] = [
                SlotDef::required("a", ValueType::Float),
                SlotDef::required("b", ValueType::Float),
            ];
            &INPUTS
        }
        fn outputs(&self) -> &'static [SlotDef] {
            static OUTPUTS: [SlotDef; 1] = [SlotDef::required("result", ValueType::Float)];
            &OUTPUTS
        }
        fn execute(
            &self,
            inputs: &[ResolvedInput],
            _ctx: &mut ExecutionContext,
        ) -> Result<Vec<NodeValue>, NodeError> {
            let a = match inputs[0].require("AddFloats")? {
                NodeValue::Float(f) => *f,
                other => {
                    return Err(NodeError::TypeMismatch {
                        slot: "a",
                        expected: "FLOAT",
                        got: other.type_name(),
                    });
                }
            };
            let b = match inputs[1].require("AddFloats")? {
                NodeValue::Float(f) => *f,
                other => {
                    return Err(NodeError::TypeMismatch {
                        slot: "b",
                        expected: "FLOAT",
                        got: other.type_name(),
                    });
                }
            };
            Ok(vec![NodeValue::Float(a + b)])
        }
    }

    #[test]
    fn execute_simple_graph() {
        let mut graph = ExecutionGraph::new();
        let a = graph.add_node(Box::new(ConstFloat(3.0)));
        let b = graph.add_node(Box::new(ConstFloat(4.0)));
        let add = graph.add_node(Box::new(AddFloats));

        graph.add_edge(a, 0, add, 0);
        graph.add_edge(b, 0, add, 1);

        let mut ctx = ExecutionContext::new(
            Default::default(),
            std::path::PathBuf::from("models"),
            std::path::PathBuf::from("output"),
        );

        let outputs = Executor::run(&graph, &mut ctx).unwrap();
        match outputs.get(&(add, 0)) {
            Some(NodeValue::Float(f)) => assert!((f - 7.0).abs() < 1e-6),
            other => panic!("expected Float(7.0), got {other:?}"),
        }
    }

    #[test]
    fn detect_cycle() {
        let mut graph = ExecutionGraph::new();
        let a = graph.add_node(Box::new(ConstFloat(1.0)));
        let b = graph.add_node(Box::new(AddFloats));

        // Create a cycle: a -> b -> a
        graph.add_edge(a, 0, b, 0);
        graph.add_edge(b, 0, a, 0); // cycle!

        let mut ctx = ExecutionContext::new(
            Default::default(),
            std::path::PathBuf::from("models"),
            std::path::PathBuf::from("output"),
        );

        let result = Executor::run(&graph, &mut ctx);
        assert!(matches!(result, Err(NodeError::CycleDetected)));
    }

    #[test]
    fn constants_work() {
        let mut graph = ExecutionGraph::new();
        let add = graph.add_node(Box::new(AddFloats));

        graph.set_constant(add, 0, NodeValue::Float(10.0));
        graph.set_constant(add, 1, NodeValue::Float(20.0));

        let mut ctx = ExecutionContext::new(
            Default::default(),
            std::path::PathBuf::from("models"),
            std::path::PathBuf::from("output"),
        );

        let outputs = Executor::run(&graph, &mut ctx).unwrap();
        match outputs.get(&(add, 0)) {
            Some(NodeValue::Float(f)) => assert!((f - 30.0).abs() < 1e-6),
            other => panic!("expected Float(30.0), got {other:?}"),
        }
    }
}
