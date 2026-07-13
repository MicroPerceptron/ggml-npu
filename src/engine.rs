//! Vendor- and op-agnostic dispatch: recognize offloadable ops in a cgraph and route them to
//! the present NPU vendor. This is the only thing `backend/` (the ggml ABI shell) forwards to;
//! it knows nothing of matmul or Intel — [`crate::ops`] owns op recognition and tensor mapping,
//! [`crate::vendor`] owns lowering + submission (and each vendor its own artifact cache).
//!
//! (Named `engine`, not `core`, to avoid shadowing the `core` crate.)

use crate::ffi::*;
use crate::graph::{
    ExecutionPartition, GraphIr, Node, OpRegistry, OperationContract, PartitionPolicy,
    PartitionReason, WireOp,
};
use crate::vendor::{LoweringSupport, Vendor};
use crate::{ops, vendor};

/// Host-realize layout metadata that landed inside a claimed multi-op split.
///
/// View/Reshape/Transpose/Permute are aliasing header ops — ggml already configured the
/// destination tensor when the node was created, so compute is a no-op. Cont materializes a
/// contiguous copy on the host because OpenVINO cannot lower layout-only programs alone.
fn host_realize_layout_graph(graph: &GraphIr) -> Result<(), &'static str> {
    for node in &graph.nodes {
        match GraphIr::wire_op(node) {
            Some(WireOp::View | WireOp::Reshape | WireOp::Transpose | WireOp::Permute) => {}
            Some(WireOp::Cont) => host_realize_cont(graph, node)?,
            Some(_) => return Err("non_layout_in_layout_glue"),
            None => return Err("unmapped_layout_glue_node"),
        }
    }
    Ok(())
}

fn tensor_logical_nbytes(desc: &crate::graph::TensorDesc) -> Option<usize> {
    let traits = unsafe { ggml_get_type_traits(desc.dtype) };
    if traits.is_null() {
        return None;
    }
    let type_size = unsafe { (*traits).type_size };
    let block = usize::try_from(unsafe { (*traits).blck_size }).ok()?;
    let mut elements = 1usize;
    for dim in desc.dims {
        let crate::graph::Dim::Static(n) = dim else {
            return None;
        };
        elements = elements.checked_mul(n.max(0) as usize)?;
    }
    elements.checked_div(block)?.checked_mul(type_size)
}

fn host_realize_cont(graph: &GraphIr, node: &Node) -> Result<(), &'static str> {
    let Some(&src_id) = node.inputs.first() else {
        return Err("cont_missing_input");
    };
    let src = graph
        .values
        .get(src_id as usize)
        .ok_or("cont_bad_input")?;
    let dst = graph
        .values
        .get(node.output as usize)
        .ok_or("cont_bad_output")?;
    if src.data == 0 || dst.data == 0 {
        return Err("cont_null_data");
    }
    let nbytes = tensor_logical_nbytes(dst).ok_or("cont_nbytes")?;
    if tensor_logical_nbytes(src) != Some(nbytes) {
        return Err("cont_size_mismatch");
    }
    if src.data == dst.data {
        return Ok(());
    }
    unsafe {
        core::ptr::copy_nonoverlapping(src.data as *const u8, dst.data as *mut u8, nbytes);
    }
    Ok(())
}

fn layout_glue_partition(
    parent: &GraphIr,
    region: &ExecutionPartition,
    start: usize,
    end: usize,
) -> Option<ExecutionPartition> {
    let graph = region.graph.partition_range(start, end)?;
    if !parent.has_lowerable_compute() || !graph.is_layout_only() {
        return None;
    }
    Some(ExecutionPartition {
        graph,
        reason: PartitionReason::LayoutGlueHost,
        start: region.start + start,
    })
}

fn leading_layout_only_end(graph: &GraphIr, start: usize) -> Option<usize> {
    if start >= graph.nodes.len() || !GraphIr::node_is_layout_metadata(&graph.nodes[start]) {
        return None;
    }
    let mut end = start + 1;
    while end < graph.nodes.len() && GraphIr::node_is_layout_metadata(&graph.nodes[end]) {
        end += 1;
    }
    Some(end)
}

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
        // Layout-only region beside real compute in the same claimed split: host-realize as
        // glue. Standalone layout-only splits stay rejected (prefer unclaim via supports_op).
        if let Some(glue) = layout_glue_partition(graph, &region, 0, region.graph.nodes.len()) {
            result.push(glue);
            continue;
        }
        if vendor.graph_support(&region.graph).is_supported() {
            result.push(region);
            continue;
        }
        let mut start = 0usize;
        while start < region.graph.nodes.len() {
            if let Some(end) = leading_layout_only_end(&region.graph, start) {
                let Some(glue) = layout_glue_partition(graph, &region, start, end) else {
                    return Err("layout_only_unclaimed");
                };
                start = end;
                result.push(glue);
                continue;
            }
            let mut chosen = None;
            let mut last_rejection = "vendor_rejected_region";
            for end in ((start + 1)..=region.graph.nodes.len()).rev() {
                let Some(candidate) = region.graph.partition_range(start, end) else {
                    continue;
                };
                if candidate.is_layout_only() {
                    // Do not ask the vendor to compile layout-only prefixes; peel them as glue
                    // when the parent split has compute, otherwise refuse (standalone unclaim).
                    last_rejection = "layout_only_unclaimed";
                    continue;
                }
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
        if partition.reason == PartitionReason::LayoutGlueHost {
            match host_realize_layout_graph(&partition.graph) {
                Ok(()) => {
                    vendor.debug_note(&alloc::format!(
                        "[npu ir2] host-realized layout glue: index={index} nodes={}\n",
                        partition.graph.nodes.len()
                    ));
                    continue;
                }
                Err(reason) => {
                    vendor.debug_note(&alloc::format!(
                        "[npu ir2] claimed layout glue failed: index={index} reason={reason}\n"
                    ));
                    crate::trace::graph("graph_runtime_failure", &partition.graph, reason);
                    return false;
                }
            }
        }
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
/// (`offload_op`). Never claim a node the vendor lowerer rejects. Const-weight matmuls that pass
/// `node_support` remain claimable even when the dedicated Constant-path `can_lower` probe fails
/// — Graph IR still lowers them in fused prefill splits; refusing here silently handed large-M
/// FFN/ShortConv/QKV work to Vulkan (#5 / #8).
pub unsafe fn supports(op: *const ggml_tensor) -> bool {
    let Some(v) = vendor::present() else {
        return false;
    };
    if let Ok(spec) = OpRegistry::decode_tensor(op) {
        match v.node_support(op, &spec) {
            LoweringSupport::Supported => {
                if let Some(matched) = ops::match_node(op) {
                    if ops::can_lower(matched, op, v) {
                        return true;
                    }
                    // Constant-path probe missed; keep Graph IR eligibility for bakeable weights.
                    if ops::subgraph::is_const_weight_matmul(op) {
                        crate::trace::perf(
                            "placement_rejection",
                            &alloc::format!(
                                ",\"reason\":\"constant_path_probe_miss_graph_ir_ok\",\"op\":{},\"wire\":{}",
                                (*op).op,
                                spec.wire as i32,
                            ),
                        );
                        return true;
                    }
                    crate::trace::perf(
                        "placement_rejection",
                        &alloc::format!(
                            ",\"reason\":\"constant_path_can_lower\",\"op\":{},\"wire\":{}",
                            (*op).op,
                            spec.wire as i32,
                        ),
                    );
                    return false;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{
        Dim, Node, OpRegistry, OperationContract, TensorDesc, TensorRole, OP_PARAM_BYTES,
    };
    use crate::ops::{Op, Operand};
    use crate::vendor::{DeviceInfo, LoweringSupport, Vendor};
    use alloc::vec::Vec;
    use core::ffi::c_void;

    /// Vendor that claims every graph and then fails execution — models claim-then-abort.
    struct ClaimThenFailVendor;

    impl Vendor for ClaimThenFailVendor {
        fn name(&self) -> &'static str {
            "claim-then-fail"
        }
        fn probe(&self) -> Vec<DeviceInfo> {
            Vec::new()
        }
        fn can_lower(&self, _op: Op, _inputs: &[Operand]) -> bool {
            true
        }
        fn run(&self, _op: Op, _inputs: &[Operand], _output: &Operand) -> bool {
            false
        }
        fn run_subgraph(&self, _sg: &crate::ops::subgraph::Subgraph) -> bool {
            false
        }
        fn graph_support(&self, _graph: &GraphIr) -> LoweringSupport {
            LoweringSupport::Supported
        }
        fn run_graph(&self, _graph: &GraphIr) -> bool {
            false
        }
        unsafe fn alloc_host(&self, _size: usize) -> *mut c_void {
            core::ptr::null_mut()
        }
        unsafe fn free_host(&self, _ptr: *mut c_void, _size: usize) {}
    }

    /// Vendor that refuses every graph — models honest unclaim / feasibility rejection.
    struct UnclaimVendor;

    impl Vendor for UnclaimVendor {
        fn name(&self) -> &'static str {
            "unclaim"
        }
        fn probe(&self) -> Vec<DeviceInfo> {
            Vec::new()
        }
        fn can_lower(&self, _op: Op, _inputs: &[Operand]) -> bool {
            false
        }
        fn run(&self, _op: Op, _inputs: &[Operand], _output: &Operand) -> bool {
            false
        }
        fn run_subgraph(&self, _sg: &crate::ops::subgraph::Subgraph) -> bool {
            false
        }
        fn graph_support(&self, _graph: &GraphIr) -> LoweringSupport {
            LoweringSupport::Unsupported("layout_only_unclaimed")
        }
        fn run_graph(&self, _graph: &GraphIr) -> bool {
            // Must never be reached after an honest unclaim; if it is, fail loud.
            false
        }
        unsafe fn alloc_host(&self, _size: usize) -> *mut c_void {
            core::ptr::null_mut()
        }
        unsafe fn free_host(&self, _ptr: *mut c_void, _size: usize) {}
    }

    fn single_matmul_graph() -> GraphIr {
        let mul_spec = OpRegistry::decode_node(&Node {
            op: crate::ffi::GGML_OP_MUL_MAT,
            spec: Err(crate::graph::OpReject::UnsupportedOp),
            inputs: alloc::vec![0, 1],
            output: 2,
            op_params: [0; OP_PARAM_BYTES],
        })
        .expect("MatMul must decode");
        let tensor = |role, data| TensorDesc {
            dtype: crate::ffi::GGML_TYPE_F32,
            dims: [
                Dim::Static(4),
                Dim::Static(4),
                Dim::Static(1),
                Dim::Static(1),
            ],
            strides: [4, 16, 16, 16],
            role,
            data,
            owned: false,
            shared: None,
        };
        GraphIr {
            generation: 1,
            values: alloc::vec![
                tensor(TensorRole::Constant, 1),
                tensor(TensorRole::RuntimeInput, 2),
                tensor(TensorRole::Output, 3),
            ],
            nodes: alloc::vec![Node {
                op: crate::ffi::GGML_OP_MUL_MAT,
                spec: Ok(mul_spec),
                inputs: alloc::vec![0, 1],
                output: 2,
                op_params: [0; OP_PARAM_BYTES],
            }],
            inputs: alloc::vec![0, 1],
            outputs: alloc::vec![2],
        }
    }

    /// Supports only graphs that contain real compute — refuses layout-only the way OpenVINO does.
    struct ComputeOnlyVendor {
        ran: core::sync::atomic::AtomicUsize,
    }

    impl Vendor for ComputeOnlyVendor {
        fn name(&self) -> &'static str {
            "compute-only"
        }
        fn probe(&self) -> Vec<DeviceInfo> {
            Vec::new()
        }
        fn can_lower(&self, _op: Op, _inputs: &[Operand]) -> bool {
            true
        }
        fn run(&self, _op: Op, _inputs: &[Operand], _output: &Operand) -> bool {
            true
        }
        fn run_subgraph(&self, _sg: &crate::ops::subgraph::Subgraph) -> bool {
            true
        }
        fn graph_support(&self, graph: &GraphIr) -> LoweringSupport {
            if graph.is_layout_only() {
                LoweringSupport::Unsupported("layout_only_unclaimed")
            } else if graph.has_lowerable_compute() {
                LoweringSupport::Supported
            } else {
                LoweringSupport::Unsupported("empty_or_unknown")
            }
        }
        fn run_graph(&self, graph: &GraphIr) -> bool {
            assert!(
                !graph.is_layout_only(),
                "vendor must not be asked to run layout-only glue"
            );
            self.ran
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            true
        }
        unsafe fn alloc_host(&self, _size: usize) -> *mut c_void {
            core::ptr::null_mut()
        }
        unsafe fn free_host(&self, _ptr: *mut c_void, _size: usize) {}
    }

    fn matmul_plus_disconnected_views_graph() -> GraphIr {
        let mul_spec = OpRegistry::decode_node(&Node {
            op: crate::ffi::GGML_OP_MUL_MAT,
            spec: Err(crate::graph::OpReject::UnsupportedOp),
            inputs: alloc::vec![0, 1],
            output: 2,
            op_params: [0; OP_PARAM_BYTES],
        })
        .expect("MatMul must decode");
        let view_spec = OpRegistry::decode_node(&Node {
            op: crate::ffi::GGML_OP_VIEW,
            spec: Err(crate::graph::OpReject::UnsupportedOp),
            inputs: alloc::vec![0],
            output: 1,
            op_params: [0; OP_PARAM_BYTES],
        })
        .expect("View must decode");
        let tensor = |role, data| TensorDesc {
            dtype: crate::ffi::GGML_TYPE_F32,
            dims: [
                Dim::Static(4),
                Dim::Static(4),
                Dim::Static(1),
                Dim::Static(1),
            ],
            strides: [4, 16, 16, 16],
            role,
            data,
            owned: false,
            shared: None,
        };
        // MatMul on values 0/1→2, then a disconnected View on independent values 3→4.
        // Models the LFM claimed split that partitions into compute + layout-only regions.
        GraphIr {
            generation: 1,
            values: alloc::vec![
                tensor(TensorRole::Constant, 1),
                tensor(TensorRole::RuntimeInput, 2),
                tensor(TensorRole::Output, 3),
                tensor(TensorRole::RuntimeInput, 4),
                tensor(TensorRole::Output, 5),
            ],
            nodes: alloc::vec![
                Node {
                    op: crate::ffi::GGML_OP_MUL_MAT,
                    spec: Ok(mul_spec),
                    inputs: alloc::vec![0, 1],
                    output: 2,
                    op_params: [0; OP_PARAM_BYTES],
                },
                Node {
                    op: crate::ffi::GGML_OP_VIEW,
                    spec: Ok(view_spec),
                    inputs: alloc::vec![3],
                    output: 4,
                    op_params: [0; OP_PARAM_BYTES],
                },
            ],
            inputs: alloc::vec![0, 1, 3],
            outputs: alloc::vec![2, 4],
        }
    }

    fn view_only_graph() -> GraphIr {
        let view_spec = OpRegistry::decode_node(&Node {
            op: crate::ffi::GGML_OP_VIEW,
            spec: Err(crate::graph::OpReject::UnsupportedOp),
            inputs: alloc::vec![0],
            output: 1,
            op_params: [0; OP_PARAM_BYTES],
        })
        .expect("View must decode");
        let tensor = |role, data| TensorDesc {
            dtype: crate::ffi::GGML_TYPE_F32,
            dims: [
                Dim::Static(4),
                Dim::Static(1),
                Dim::Static(1),
                Dim::Static(1),
            ],
            strides: [4, 4, 4, 4],
            role,
            data,
            owned: false,
            shared: None,
        };
        GraphIr {
            generation: 1,
            values: alloc::vec![
                tensor(TensorRole::RuntimeInput, 1),
                tensor(TensorRole::Output, 2),
            ],
            nodes: alloc::vec![Node {
                op: crate::ffi::GGML_OP_VIEW,
                spec: Ok(view_spec),
                inputs: alloc::vec![0],
                output: 1,
                op_params: [0; OP_PARAM_BYTES],
            }],
            inputs: alloc::vec![0],
            outputs: alloc::vec![1],
        }
    }

    #[test]
    fn claimed_partition_runtime_failure_is_not_success() {
        // After claim, compile/exec failure must not soft-succeed (#8).
        let graph = single_matmul_graph();
        assert!(!run_partitions(&ClaimThenFailVendor, &graph));
    }

    #[test]
    fn unclaim_rejects_graph_without_pretending_success() {
        // Feasibility refusal is not a claimed success — run_partitions returns false so the
        // backend can map to GGML_STATUS_FAILED rather than silent SUCCESS (#8).
        let graph = single_matmul_graph();
        assert!(feasible_partitions(&graph, &UnclaimVendor).is_err());
        assert!(!run_partitions(&UnclaimVendor, &graph));
    }

    #[test]
    fn layout_glue_inside_claimed_split_is_host_realized() {
        // Mixed claimed split with a disconnected layout-only region must not abort the whole
        // graph as "rejected layout-only" — host-realize glue and still run compute (#5/#8).
        let graph = matmul_plus_disconnected_views_graph();
        assert!(graph.has_lowerable_compute());
        assert!(!graph.is_layout_only());
        let regions = graph.execution_regions(PartitionPolicy::default());
        assert!(
            regions.len() >= 2,
            "expected disconnected compute + layout regions"
        );
        assert!(regions.iter().any(|r| r.graph.is_layout_only()));
        assert!(regions.iter().any(|r| r.graph.has_lowerable_compute()));

        let vendor = ComputeOnlyVendor {
            ran: core::sync::atomic::AtomicUsize::new(0),
        };
        let parts = feasible_partitions(&graph, &vendor).expect("mixed split must partition");
        assert!(
            parts
                .iter()
                .any(|p| p.reason == PartitionReason::LayoutGlueHost),
            "layout-only region must be scheduled as host glue"
        );
        assert!(
            parts.iter().any(|p| p.graph.has_lowerable_compute()),
            "compute region must remain"
        );
        assert!(run_partitions(&vendor, &graph));
        assert_eq!(
            vendor.ran.load(core::sync::atomic::Ordering::Relaxed),
            1,
            "vendor should run exactly the compute partition"
        );
    }

    #[test]
    fn standalone_layout_only_split_stays_unclaimed() {
        // Lone View/Reshape/… partitions must still refuse — prefer unclaim, no soft success.
        let graph = view_only_graph();
        assert!(graph.is_layout_only());
        let vendor = ComputeOnlyVendor {
            ran: core::sync::atomic::AtomicUsize::new(0),
        };
        assert!(feasible_partitions(&graph, &vendor).is_err());
        assert!(!run_partitions(&vendor, &graph));
        assert_eq!(vendor.ran.load(core::sync::atomic::Ordering::Relaxed), 0);
    }

    /// Models Intel OpenVINO: unitary ShortConv state ops (and gather+conv) lower, but a single
    /// program that fuses `SSM_CONV` with `SET_ROWS` is rejected (#9).
    struct ShortConvAdjacentVendor {
        ran: core::sync::atomic::AtomicUsize,
    }

    fn wire_ops(graph: &GraphIr) -> alloc::vec::Vec<WireOp> {
        graph
            .nodes
            .iter()
            .filter_map(GraphIr::wire_op)
            .collect()
    }

    impl Vendor for ShortConvAdjacentVendor {
        fn name(&self) -> &'static str {
            "shortconv-adjacent"
        }
        fn probe(&self) -> Vec<DeviceInfo> {
            Vec::new()
        }
        fn can_lower(&self, _op: Op, _inputs: &[Operand]) -> bool {
            true
        }
        fn run(&self, _op: Op, _inputs: &[Operand], _output: &Operand) -> bool {
            true
        }
        fn run_subgraph(&self, _sg: &crate::ops::subgraph::Subgraph) -> bool {
            true
        }
        fn graph_support(&self, graph: &GraphIr) -> LoweringSupport {
            if graph.is_layout_only() {
                return LoweringSupport::Unsupported("layout_only_unclaimed");
            }
            let ops = wire_ops(graph);
            let has_ssm = ops.iter().any(|op| *op == WireOp::SsmConv);
            let has_set = ops.iter().any(|op| *op == WireOp::SetRows);
            if has_ssm && has_set {
                return LoweringSupport::Unsupported("ssm_setrows_fusion_rejected");
            }
            if graph.has_lowerable_compute() {
                LoweringSupport::Supported
            } else {
                LoweringSupport::Unsupported("empty_or_unknown")
            }
        }
        fn run_graph(&self, graph: &GraphIr) -> bool {
            assert!(
                !graph.is_layout_only(),
                "stateful ShortConv must not host-realize as layout glue"
            );
            let ops = wire_ops(graph);
            assert!(
                !(ops.iter().any(|op| *op == WireOp::SsmConv)
                    && ops.iter().any(|op| *op == WireOp::SetRows)),
                "vendor must not be asked to run fused SSM_CONV+SET_ROWS"
            );
            self.ran
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            true
        }
        unsafe fn alloc_host(&self, _size: usize) -> *mut c_void {
            core::ptr::null_mut()
        }
        unsafe fn free_host(&self, _ptr: *mut c_void, _size: usize) {}
    }

    fn shortconv_gather_conv_write_graph() -> GraphIr {
        // Same connected DAG as graph::connected_shortconv_gather_conv_write_stays_one_region.
        let get_spec = OpRegistry::decode_node(&Node {
            op: crate::ffi::GGML_OP_GET_ROWS,
            spec: Err(crate::graph::OpReject::UnsupportedOp),
            inputs: alloc::vec![0, 1],
            output: 2,
            op_params: [0; OP_PARAM_BYTES],
        })
        .expect("GetRows must decode");
        let ssm_spec = OpRegistry::decode_node(&Node {
            op: crate::ffi::GGML_OP_SSM_CONV,
            spec: Err(crate::graph::OpReject::UnsupportedOp),
            inputs: alloc::vec![2, 3],
            output: 4,
            op_params: [0; OP_PARAM_BYTES],
        })
        .expect("SsmConv must decode");
        let set_spec = OpRegistry::decode_node(&Node {
            op: crate::ffi::GGML_OP_SET_ROWS,
            spec: Err(crate::graph::OpReject::UnsupportedOp),
            inputs: alloc::vec![4, 5, 0],
            output: 6,
            op_params: [0; OP_PARAM_BYTES],
        })
        .expect("SetRows must decode");
        let tensor = |role, data| TensorDesc {
            dtype: crate::ffi::GGML_TYPE_F32,
            dims: [
                Dim::Static(4),
                Dim::Static(2),
                Dim::Static(1),
                Dim::Static(1),
            ],
            strides: [4, 8, 8, 8],
            role,
            data,
            owned: role == TensorRole::MutableState,
            shared: None,
        };
        GraphIr {
            generation: 1,
            values: alloc::vec![
                tensor(TensorRole::MutableState, 1),
                tensor(TensorRole::RuntimeInput, 2),
                tensor(TensorRole::Intermediate, 3),
                tensor(TensorRole::Constant, 4),
                tensor(TensorRole::Intermediate, 5),
                tensor(TensorRole::RuntimeInput, 6),
                tensor(TensorRole::MutableState, 7),
            ],
            nodes: alloc::vec![
                Node {
                    op: crate::ffi::GGML_OP_GET_ROWS,
                    spec: Ok(get_spec),
                    inputs: alloc::vec![0, 1],
                    output: 2,
                    op_params: [0; OP_PARAM_BYTES],
                },
                Node {
                    op: crate::ffi::GGML_OP_SSM_CONV,
                    spec: Ok(ssm_spec),
                    inputs: alloc::vec![2, 3],
                    output: 4,
                    op_params: [0; OP_PARAM_BYTES],
                },
                Node {
                    op: crate::ffi::GGML_OP_SET_ROWS,
                    spec: Ok(set_spec),
                    inputs: alloc::vec![4, 5, 0],
                    output: 6,
                    op_params: [0; OP_PARAM_BYTES],
                },
            ],
            inputs: alloc::vec![0, 1, 3, 5],
            outputs: alloc::vec![6],
        }
    }

    #[test]
    fn shortconv_gather_conv_write_splits_to_adjacent_npu_not_host_glue() {
        // Connected gather→SSM_CONV→SetRows is one semantic region, but Intel rejects fusing
        // SSM_CONV+SET_ROWS. Engine must peel FeasibilityLimit adjacent NPU programs — never
        // host-realize stateful edges as layout_glue_host, and never soft-succeed (#9).
        let graph = shortconv_gather_conv_write_graph();
        assert!(!graph.is_layout_only());
        assert!(graph.has_lowerable_compute());
        let regions = graph.execution_regions(PartitionPolicy::default());
        assert_eq!(regions.len(), 1, "connected ShortConv state must stay one region");
        assert_eq!(regions[0].reason, PartitionReason::WholeGraph);

        let vendor = ShortConvAdjacentVendor {
            ran: core::sync::atomic::AtomicUsize::new(0),
        };
        assert!(
            !vendor.graph_support(&graph).is_supported(),
            "whole fused gather/conv/write must be refused by the Intel-shaped vendor"
        );
        let parts = feasible_partitions(&graph, &vendor).expect("must peel adjacent programs");
        assert!(
            parts.len() >= 2,
            "expected adjacent FeasibilityLimit programs, got {}",
            parts.len()
        );
        assert!(
            parts
                .iter()
                .all(|p| p.reason != PartitionReason::LayoutGlueHost),
            "stateful ShortConv must not be scheduled as layout_glue_host"
        );
        assert!(
            parts
                .iter()
                .all(|p| p.reason == PartitionReason::FeasibilityLimit),
            "adjacent peels must be feasibility_limit"
        );
        assert!(
            parts.iter().all(|p| p.graph.has_lowerable_compute()),
            "every peeled program must remain lowerable compute"
        );
        assert!(run_partitions(&vendor, &graph));
        assert_eq!(
            vendor.ran.load(core::sync::atomic::Ordering::Relaxed),
            parts.len(),
            "each adjacent ShortConv program must actually run on the vendor"
        );
    }

    #[test]
    fn shortconv_stateful_region_is_never_classified_layout_only() {
        // Host-realize is only for layout metadata beside compute. GetRows/SsmConv/SetRows are
        // lowerable compute / stateful — claiming them then host-realizing would be soft fallback.
        let graph = shortconv_gather_conv_write_graph();
        assert!(!graph.is_layout_only());
        for node in &graph.nodes {
            assert!(
                !GraphIr::node_is_layout_metadata(node),
                "ShortConv state edge must not look like layout metadata"
            );
        }
        assert!(layout_glue_partition(
            &graph,
            &ExecutionPartition {
                graph: graph.clone(),
                reason: PartitionReason::WholeGraph,
                start: 0,
            },
            0,
            graph.nodes.len(),
        )
        .is_none());
    }
}
