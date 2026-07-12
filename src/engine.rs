//! Vendor- and op-agnostic dispatch: recognize offloadable ops in a cgraph and route them to
//! the present NPU vendor. This is the only thing `backend/` (the ggml ABI shell) forwards to;
//! it knows nothing of matmul or Intel — [`crate::ops`] owns op recognition and tensor mapping,
//! [`crate::vendor`] owns lowering + submission (and each vendor its own artifact cache).
//!
//! (Named `engine`, not `core`, to avoid shadowing the `core` crate.)

use crate::ffi::*;
use crate::graph::{OpRegistry, OperationContract};
use crate::{ops, vendor};

const MAX_GRAPH_PARTITION_NODES: usize = 8;

/// `supports_op`: can the present NPU vendor run this node? The vendor may need the node's
/// operands (the Constant path compiles the weight blob here), so pass the node through.
pub unsafe fn supports(op: *const ggml_tensor) -> bool {
    let Some(v) = vendor::present() else {
        return false;
    };
    if let Some(matched) = ops::match_node(op) {
        if ops::can_lower(matched, op, v) {
            return true;
        }
    }
    if let Ok(spec) = OpRegistry::decode_tensor(op) {
        if v.node_support(op, &spec).is_supported() {
            return true;
        }
    }
    // Claim FFN glue (silu/mul) next to claimable matmuls, so the scheduler places the whole FFN on
    // us as one split for `compute` to fuse. A glue op has no standalone artifact to compile here;
    // it runs either fused (the common case) or via the single-node fallback in `compute`.
    ops::subgraph::claims_glue(op)
}

/// `graph_compute`: run the assigned split on the NPU. First try to compile the **whole split as
/// one fused program** (one submit, intra-graph fusion + pipelining); if it isn't a recognized
/// subgraph, fall back to dispatching each recognized node on its own.
pub unsafe fn compute(cgraph: *mut ggml_cgraph) -> bool {
    crate::trace::split(cgraph);
    let Some(v) = vendor::present() else {
        return false;
    };
    let graph = match crate::graph::capture(cgraph) {
        Some(graph) => {
            v.debug_note(&alloc::format!(
                "[npu ir2] captured split: nodes={} inputs={} outputs={}\n",
                graph.nodes.len(),
                graph.inputs.len(),
                graph.outputs.len()
            ));
            for (index, node) in graph.nodes.iter().enumerate() {
                v.debug_note(&alloc::format!(
                    "[npu ir2] node {index}: ggml_op={} wire={:?} inputs={}\n",
                    node.op,
                    crate::graph::GraphIr::wire_op(node),
                    node.inputs.len()
                ));
            }
            let partitions = graph.execution_partitions(MAX_GRAPH_PARTITION_NODES, 1);
            if partitions.len() > 1 {
                v.debug_note(&alloc::format!(
                    "[npu ir2] executing {} bounded partitions\n",
                    partitions.len()
                ));
                if partitions
                    .iter()
                    .any(|part| !v.graph_support(part).is_supported())
                {
                    v.debug_note("[npu ir2] bounded partition rejected before execution\n");
                    return false;
                }
                for (index, part) in partitions.iter().enumerate() {
                    v.debug_note(&alloc::format!(
                        "[npu ir2] partition {index}: nodes={} inputs={} outputs={}\n",
                        part.nodes.len(),
                        part.inputs.len(),
                        part.outputs.len()
                    ));
                    if !v.run_graph(part) {
                        return false;
                    }
                }
                return true;
            }
            let stateful = graph.nodes.iter().any(|node| node.op == GGML_OP_SSM_CONV);
            if stateful && v.graph_support(&graph).is_supported() {
                return v.run_graph(&graph);
            }
            Some(graph)
        }
        None => {
            v.debug_note("[npu ir2] capture failed for assigned split\n");
            None
        }
    };
    if let Some(sg) = ops::subgraph::recognize(cgraph) {
        // A recognized subgraph spans ops the per-node path can't run on its own (e.g. silu/mul),
        // so it must run as one program or fail loudly — never fall through to a partial dispatch.
        return v.run_subgraph(&sg);
    }
    if let Some(graph) = graph {
        if v.graph_support(&graph).is_supported() {
            return v.run_graph(&graph);
        }
    }
    let n = ggml_graph_n_nodes(cgraph);
    for i in 0..n {
        let node = ggml_graph_node(cgraph, i);
        if node.is_null() {
            continue;
        }
        let t = node as *const ggml_tensor;
        if let Some(op) = ops::match_node(t) {
            if !ops::run(op, node, v) {
                return false;
            }
        } else if ops::subgraph::is_glue_op(t) {
            // A claimed glue op that didn't fuse (this split isn't a full FFN): run it standalone.
            // We claimed it, so it must run — fail loudly if we somehow can't build it.
            match ops::subgraph::recognize_glue(t) {
                Some(sg) if v.run_subgraph(&sg) => {}
                _ => return false,
            }
        } else {
            // Every node in this scheduler split was assigned to this backend. Returning success
            // without executing an unknown node silently corrupts the graph.
            return false;
        }
    }
    true
}

pub fn synchronize() -> bool {
    vendor::present().is_none_or(|v| v.synchronize())
}
