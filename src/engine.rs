//! Vendor- and op-agnostic dispatch: recognize offloadable ops in a cgraph and route them to
//! the present NPU vendor. This is the only thing `backend/` (the ggml ABI shell) forwards to;
//! it knows nothing of matmul or Intel — [`crate::ops`] owns op recognition and tensor mapping,
//! [`crate::vendor`] owns lowering + submission (and each vendor its own artifact cache).
//!
//! (Named `engine`, not `core`, to avoid shadowing the `core` crate.)

use crate::ffi::*;
use crate::graph::{
    ExecutionPartition, GraphIr, OpRegistry, OperationContract, PartitionPolicy, PartitionReason,
};
use crate::vendor::{LoweringSupport, Vendor};
use crate::{ops, vendor};

fn feasible_partitions(
    graph: &GraphIr,
    vendor: &dyn Vendor,
) -> Result<alloc::vec::Vec<ExecutionPartition>, &'static str> {
    let semantic_regions = graph.execution_regions(PartitionPolicy::default());
    if semantic_regions.is_empty() {
        return Err("empty_graph");
    }
    let mut result = alloc::vec::Vec::new();
    for region in semantic_regions {
        if vendor.graph_support(&region.graph).is_supported() {
            result.push(region);
            continue;
        }
        let mut start = 0usize;
        while start < region.graph.nodes.len() {
            let mut chosen = None;
            let mut last_rejection = "vendor_rejected_region";
            for end in ((start + 1)..=region.graph.nodes.len()).rev() {
                let Some(candidate) = region.graph.partition_range(start, end) else {
                    continue;
                };
                match vendor.graph_support(&candidate) {
                    LoweringSupport::Supported => {
                        chosen = Some(ExecutionPartition {
                            graph: candidate,
                            reason: PartitionReason::FeasibilityLimit,
                            start: region.start + start,
                        });
                        break;
                    }
                    LoweringSupport::Unsupported(reason) => last_rejection = reason,
                }
            }
            let Some(partition) = chosen else {
                return Err(last_rejection);
            };
            start += partition.graph.nodes.len();
            result.push(partition);
        }
    }
    Ok(result)
}

fn run_partitions(vendor: &dyn Vendor, graph: &GraphIr) -> bool {
    let partitions = match feasible_partitions(graph, vendor) {
        Ok(partitions) => partitions,
        Err(reason) => {
            vendor.debug_note(&alloc::format!(
                "[npu ir2] graph rejected without fallback: reason={reason}\n"
            ));
            crate::trace::graph("graph_rejected", graph, reason);
            return false;
        }
    };
    for partition in &partitions {
        crate::trace::graph(
            "graph_partition",
            &partition.graph,
            partition.reason.as_str(),
        );
    }
    // Capability is preflighted for every partition before the first state mutation or output.
    for (index, partition) in partitions.iter().enumerate() {
        vendor.debug_note(&alloc::format!(
            "[npu ir2] partition {index}: nodes={} inputs={} outputs={} reason={}\n",
            partition.graph.nodes.len(),
            partition.graph.inputs.len(),
            partition.graph.outputs.len(),
            partition.reason.as_str(),
        ));
        if !vendor.run_graph(&partition.graph) {
            vendor.debug_note(&alloc::format!(
                "[npu ir2] claimed partition execution failed: index={index} reason=runtime_failure\n"
            ));
            crate::trace::graph("graph_runtime_failure", &partition.graph, "runtime_failure");
            return false;
        }
    }
    true
}

/// `supports_op`: can the present NPU vendor run this node? The vendor may need the node's
/// operands (the Constant path compiles the weight blob here), so pass the node through.
pub unsafe fn supports(op: *const ggml_tensor) -> bool {
    let Some(v) = vendor::present() else {
        return false;
    };
    if let Ok(spec) = OpRegistry::decode_tensor(op) {
        let support = v.node_support(op, &spec);
        if support.is_supported() && crate::policy::supports_region(op, spec) {
            if let Some(matched) = ops::match_node(op) {
                if ops::can_lower(matched, op, v) {
                    return true;
                }
            }
            return true;
        }
        if let LoweringSupport::Unsupported(reason) = support {
            crate::trace::perf(
                "placement_rejection",
                &alloc::format!(
                    ",\"reason\":\"{reason}\",\"op\":{},\"wire\":{}",
                    (*op).op,
                    spec.wire as i32,
                ),
            );
        }
    }
    // Claim FFN glue (silu/mul) next to claimable matmuls, so the scheduler places the whole FFN on
    // us as one split for `compute` to fuse. A glue op has no standalone artifact to compile here;
    // it runs either fused (the common case) or via the single-node fallback in `compute`.
    ops::subgraph::claims_glue(op)
        && OpRegistry::decode_tensor(op).is_ok_and(|spec| crate::policy::supports_region(op, spec))
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
            crate::trace::graph("graph_captured", &graph, "scheduler_split");
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
            let semantic_regions = graph.execution_regions(PartitionPolicy::default());
            if semantic_regions.len() > 1 {
                v.debug_note(&alloc::format!(
                    "[npu ir2] executing {} semantic regions\n",
                    semantic_regions.len()
                ));
                return run_partitions(v, &graph);
            }
            let stateful = graph.nodes.iter().any(|node| node.op == GGML_OP_SSM_CONV);
            if stateful && v.graph_support(&graph).is_supported() {
                crate::trace::graph("graph_partition", &graph, "stateful_whole_graph");
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
        return run_partitions(v, &graph);
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
