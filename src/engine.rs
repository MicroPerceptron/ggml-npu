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
        let nodes = partition.graph.nodes.len() as u64;
        crate::metrics::record_npu_program(nodes);
        crate::trace::perf(
            "npu_program",
            &alloc::format!(
                ",\"nodes\":{nodes},\"unitary\":{},\"reason\":\"{}\",\"index\":{index}",
                if nodes <= 1 { "true" } else { "false" },
                partition.reason.as_str(),
            ),
        );
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

/// `supports_op`: can the present NPU vendor **execute** this node correctly?
///
/// Capability only — profitability / region expansion live in [`crate::policy::should_pull`]
/// (`offload_op`). Never claim a node the vendor lowerer rejects, and never claim a matched
/// single-op path when `can_lower` fails (#8).
pub unsafe fn supports(op: *const ggml_tensor) -> bool {
    let Some(v) = vendor::present() else {
        return false;
    };
    if let Ok(spec) = OpRegistry::decode_tensor(op) {
        match v.node_support(op, &spec) {
            LoweringSupport::Supported => {
                // Dedicated single-op lowerers must accept the node; otherwise refuse rather than
                // over-claim and hope graph fusion covers it.
                if let Some(matched) = ops::match_node(op) {
                    return ops::can_lower(matched, op, v);
                }
                return true;
            }
            LoweringSupport::Unsupported(reason) => {
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
    }
    // FFN glue next to claimable matmuls: still requires vendor node feasibility so we never
    // advertise glue the lowerer cannot run. Fusion is preferred; compute fails loud otherwise.
    if ops::subgraph::claims_glue(op) {
        if let Ok(spec) = OpRegistry::decode_tensor(op) {
            return v.node_support(op, &spec).is_supported();
        }
    }
    false
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
                let nodes = graph.nodes.len() as u64;
                crate::metrics::record_npu_program(nodes);
                crate::trace::perf(
                    "npu_program",
                    &alloc::format!(
                        ",\"nodes\":{nodes},\"unitary\":{},\"reason\":\"stateful_whole_graph\",\"index\":0",
                        if nodes <= 1 { "true" } else { "false" },
                    ),
                );
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
        let nodes = sg.nodes.len() as u64;
        crate::metrics::record_npu_program(nodes);
        crate::trace::perf(
            "npu_program",
            &alloc::format!(
                ",\"nodes\":{nodes},\"unitary\":{},\"reason\":\"recognized_subgraph\",\"index\":0",
                if nodes <= 1 { "true" } else { "false" },
            ),
        );
        let ok = v.run_subgraph(&sg);
        if !ok {
            v.debug_note(
                "[npu ir2] claimed recognized subgraph execution failed: reason=runtime_failure\n",
            );
            crate::trace::perf(
                "claimed_execution_failure",
                ",\"reason\":\"recognized_subgraph_runtime_failure\"",
            );
        }
        return ok;
    }
    if let Some(graph) = graph {
        return run_partitions(v, &graph);
    }
    let n = ggml_graph_n_nodes(cgraph);
    if n <= 0 {
        // Nothing assigned — vacuously done (not a claimed-but-skipped execution).
        return true;
    }
    let mut executed = 0u32;
    for i in 0..n {
        let node = ggml_graph_node(cgraph, i);
        if node.is_null() {
            v.debug_note(&alloc::format!(
                "[npu ir2] claimed split contains null node at index {i}\n"
            ));
            crate::trace::perf(
                "claimed_execution_failure",
                &alloc::format!(",\"reason\":\"null_claimed_node\",\"index\":{i}"),
            );
            return false;
        }
        let t = node as *const ggml_tensor;
        if let Some(op) = ops::match_node(t) {
            crate::metrics::record_npu_program(1);
            crate::trace::perf(
                "npu_program",
                ",\"nodes\":1,\"unitary\":true,\"reason\":\"per_node\",\"index\":0",
            );
            if !ops::run(op, node, v) {
                v.debug_note(&alloc::format!(
                    "[npu ir2] claimed per-node execution failed: index={i}\n"
                ));
                crate::trace::perf(
                    "claimed_execution_failure",
                    &alloc::format!(",\"reason\":\"per_node_runtime_failure\",\"index\":{i}"),
                );
                return false;
            }
            executed += 1;
        } else if ops::subgraph::is_glue_op(t) {
            // A claimed glue op that didn't fuse (this split isn't a full FFN): run it standalone.
            // We claimed it, so it must run — fail loudly if we somehow can't build it.
            let Some(sg) = ops::subgraph::recognize_glue(t) else {
                v.debug_note(&alloc::format!(
                    "[npu ir2] claimed glue node could not be built: index={i}\n"
                ));
                crate::trace::perf(
                    "claimed_execution_failure",
                    &alloc::format!(",\"reason\":\"glue_build_failure\",\"index\":{i}"),
                );
                return false;
            };
            crate::metrics::record_npu_program(1);
            crate::trace::perf(
                "npu_program",
                ",\"nodes\":1,\"unitary\":true,\"reason\":\"glue_fallback\",\"index\":0",
            );
            if !v.run_subgraph(&sg) {
                v.debug_note(&alloc::format!(
                    "[npu ir2] claimed glue execution failed: index={i}\n"
                ));
                crate::trace::perf(
                    "claimed_execution_failure",
                    &alloc::format!(",\"reason\":\"glue_runtime_failure\",\"index\":{i}"),
                );
                return false;
            }
            executed += 1;
        } else {
            // Every node in this scheduler split was assigned to this backend. Returning success
            // without executing an unknown node silently corrupts the graph.
            v.debug_note(&alloc::format!(
                "[npu ir2] claimed unknown node rejected: index={i} ggml_op={}\n",
                (*t).op
            ));
            crate::trace::perf(
                "claimed_execution_failure",
                &alloc::format!(
                    ",\"reason\":\"unknown_claimed_node\",\"index\":{i},\"op\":{}",
                    (*t).op
                ),
            );
            return false;
        }
    }
    if executed == 0 {
        v.debug_note("[npu ir2] claimed split executed zero nodes\n");
        crate::trace::perf(
            "claimed_execution_failure",
            ",\"reason\":\"zero_nodes_executed\"",
        );
        return false;
    }
    true
}

pub fn synchronize() -> bool {
    vendor::present().is_none_or(|v| v.synchronize())
}
