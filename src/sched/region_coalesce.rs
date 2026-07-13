//! Prefill-oriented same-backend region coalesce (Rust core).
//!
//! Conceptual inspiration: axiom/kore bit-row set algebra for DAG intervals —
//! not a Q1 tensor copy. This pass finds short foreign "islands" sandwiched
//! between the same backend and recolors them when every island node is
//! feasible on the surrounding backend (via the caller-supplied mask).
//!
//! With cut-cost mode (`OPT_WEIGHTED_CUT_COST` / v2 ABI), absorbs are gated by
//! two-objective scoring (#48): **latency** (handoff nbytes × backend factor)
//! and **throughput** (op-class compute weight — MatMul/FA heavy vs glue light).
//! Anti-unitary policy grows onto existing NPU neighbors (including asymmetric
//! edges), preserves fat / heavy-compute NPU islands, dissolves thin unitaries,
//! and modestly boosts max-island across NPU boundaries / light glue.
//!
//! Profitability / claim policy stays in `offload_op` / `policy::should_pull`.
//! This only widens homogeneous splits the scheduler already painted.

use core::ffi::{c_char, c_int, c_void, CStr};
use core::ptr;

use super::cut_cost::{
    decide_absorb, effective_max_island, note_absorb_outcome, npu_absorb_target,
    reset_cut_cost_stats, snapshot_cut_cost_stats, total_cut_cost, AbsorbDecision, BackendKind,
    CutCostStats, IslandCompute, OPT_COMPUTE_WEIGHT, OPT_WEIGHTED_CUT_COST,
};

/// Skip this node in interval scans (view / reshape glue that does not cut).
pub const FLAG_VIEW: u32 = 1 << 0;
/// Do not recolor (graph input, weight buffer, or other pinned placement).
pub const FLAG_PINNED: u32 = 1 << 1;

/// POD node row packed by the llama.cpp scheduler hook (v1 ABI).
///
/// Layout must stay in sync with `struct ggml_sched_region_coalesce_node` in
/// MicroPerceptron/llama.cpp `ggml-backend` (see
/// `docs/sched-region-coalesce.md`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedRegionCoalesceNode {
    /// In/out: assigned backend index; `-1` means unassigned (left untouched).
    pub backend_id: i32,
    /// Bit `i` set ⇒ backend `i` can execute this op (`supports_op`).
    pub feasible_mask: u32,
    pub flags: u32,
}

/// Extended POD row for weighted cut-cost coalesce (v2 ABI).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedCutCostNode {
    pub backend_id: i32,
    pub feasible_mask: u32,
    pub flags: u32,
    /// `ggml_nbytes(node)` capped to `u32::MAX` (0 ⇒ unknown / unweighted fallback).
    pub nbytes: u32,
    /// `ggml_op` ordinal; 0 ⇒ unknown → medium compute weight.
    pub op: u32,
}

const PROC_NAME: &[u8] = b"ggml_backend_sched_region_coalesce";
const PROC_NAME_V2: &[u8] = b"ggml_backend_sched_region_coalesce_v2";
const PROC_NAME_STATS: &[u8] = b"ggml_backend_sched_cut_cost_stats";
const PROC_NAME_STATS_RESET: &[u8] = b"ggml_backend_sched_cut_cost_stats_reset";

/// Pure analysis (v1): mutate `nodes[..]` backend ids; return how many were recolored.
pub fn coalesce_regions(
    nodes: &mut [SchedRegionCoalesceNode],
    n_backends: usize,
    max_island: usize,
) -> usize {
    let mut ext: alloc::vec::Vec<SchedCutCostNode> = nodes
        .iter()
        .map(|n| SchedCutCostNode {
            backend_id: n.backend_id,
            feasible_mask: n.feasible_mask,
            flags: n.flags,
            nbytes: 0,
            op: 0,
        })
        .collect();
    // Unknown kinds ⇒ cut-cost gate off; classic island absorb.
    let kinds = alloc::vec![BackendKind::Unknown; n_backends];
    let n = coalesce_regions_weighted(&mut ext, n_backends, max_island, &kinds, false);
    for (dst, src) in nodes.iter_mut().zip(ext.iter()) {
        dst.backend_id = src.backend_id;
    }
    n
}

/// Weighted / NPU-aware coalesce (v2 core).
pub fn coalesce_regions_weighted(
    nodes: &mut [SchedCutCostNode],
    n_backends: usize,
    max_island: usize,
    kinds: &[BackendKind],
    weighted: bool,
) -> usize {
    if nodes.is_empty() || n_backends < 2 || n_backends > 32 {
        return 0;
    }
    if kinds.len() < n_backends {
        return 0;
    }
    let max_island = max_island.max(1);
    if weighted {
        reset_cut_cost_stats();
    }

    // Compact non-view indices — interval algebra over execution order.
    let mut ops: alloc::vec::Vec<usize> = alloc::vec::Vec::new();
    for (i, n) in nodes.iter().enumerate() {
        if n.flags & FLAG_VIEW == 0 {
            ops.push(i);
        }
    }
    if ops.len() < 3 {
        return 0;
    }

    let mut reassigned = 0usize;
    let mut changed = true;
    let mut guard = 0usize;
    while changed && guard < ops.len().saturating_add(1) {
        changed = false;
        guard += 1;

        // Pass A (weighted): grow existing NPU regions — sandwich + asymmetric
        // absorb toward an NPU neighbor. Never invent NPU when neither side is NPU.
        if weighted {
            if try_absorb_pass(
                nodes,
                &ops,
                kinds,
                max_island,
                weighted,
                AbsorbMode::IntoNpu,
                &mut reassigned,
            ) {
                changed = true;
                continue;
            }
        }

        // Pass B: classic same-backend sandwich (incl. dissolve thin NPU, fat preserve).
        if try_absorb_pass(
            nodes,
            &ops,
            kinds,
            max_island,
            weighted,
            AbsorbMode::Sandwich,
            &mut reassigned,
        ) {
            changed = true;
        }
    }

    // Views stay untouched here. The llama.cpp hook (1) applies NPU / same-buft
    // always, and other targets only when committed buffers are supported by
    // the destination (buffer-safe Vulkan sandwiches), then (2) re-syncs view
    // ops to view_src.

    let _ = total_cut_cost_of_nodes(nodes, &ops, kinds); // keep scoring path warm / testable
    reassigned
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AbsorbMode {
    /// Prefer absorbing islands onto an existing NPU neighbor (anti-unitary).
    IntoNpu,
    /// Classic left==right sandwich absorb.
    Sandwich,
}

fn try_absorb_pass(
    nodes: &mut [SchedCutCostNode],
    ops: &[usize],
    kinds: &[BackendKind],
    max_island: usize,
    weighted: bool,
    mode: AbsorbMode,
    reassigned: &mut usize,
) -> bool {
    let mut run_start = 0usize;
    while run_start < ops.len() {
        let bid = nodes[ops[run_start]].backend_id;
        let mut run_end = run_start + 1;
        while run_end < ops.len() && nodes[ops[run_end]].backend_id == bid {
            run_end += 1;
        }
        let run_len = run_end - run_start;

        if run_start > 0 && run_end < ops.len() {
            let left = nodes[ops[run_start - 1]].backend_id;
            let right = nodes[ops[run_end]].backend_id;
            if left >= 0 && right >= 0 && left != bid {
                let target = match mode {
                    AbsorbMode::Sandwich => {
                        if left == right {
                            Some(left)
                        } else {
                            None
                        }
                    }
                    AbsorbMode::IntoNpu => {
                        // NPU sandwiches grow here first; non-NPU sandwiches /
                        // thin-NPU dissolve stay in Pass B (fat preserve).
                        if left == right && kind_at_npu(kinds, left) {
                            Some(left)
                        } else if left != right {
                            npu_absorb_target(left, right, kinds, weighted)
                        } else {
                            None
                        }
                    }
                };

                if let Some(target) = target {
                    // IntoNpu mode: only when target is NPU (and island is not already).
                    if mode == AbsorbMode::IntoNpu {
                        if !kind_at_npu(kinds, target) || kind_at_npu(kinds, bid) {
                            run_start = run_end;
                            continue;
                        }
                    }

                    let left_edge_nb = nodes[ops[run_start]].nbytes;
                    let right_edge_nb = nodes[ops[run_end]].nbytes;
                    let target_u = target as u32;
                    if target_u >= 32 || run_len > max_island.saturating_mul(3).saturating_add(12)
                    {
                        // Hard ceiling: even heavy-edge boost never exceeds this.
                        run_start = run_end;
                        continue;
                    }
                    let bit = 1u32 << target_u;
                    let mut ok = true;
                    let mut island_bytes = 0u64;
                    let mut compute_pairs: alloc::vec::Vec<(u32, u32)> =
                        alloc::vec::Vec::with_capacity(run_len);
                    for &op_i in &ops[run_start..run_end] {
                        let n = &nodes[op_i];
                        if n.flags & FLAG_PINNED != 0 || n.feasible_mask & bit == 0 {
                            ok = false;
                            break;
                        }
                        island_bytes = island_bytes.saturating_add(n.nbytes as u64);
                        compute_pairs.push((n.nbytes, n.op));
                    }
                    if ok {
                        let compute = IslandCompute::from_nodes(compute_pairs);
                        let eff_max = effective_max_island(
                            max_island,
                            target,
                            kinds,
                            left_edge_nb,
                            right_edge_nb,
                            weighted,
                            compute,
                        );
                        if run_len <= eff_max {
                            let decision = decide_absorb(
                                bid,
                                target,
                                kinds,
                                island_bytes,
                                run_len,
                                left_edge_nb,
                                right_edge_nb,
                                weighted,
                                compute,
                            );
                            if decision == AbsorbDecision::Absorb {
                                let mut flipped = 0usize;
                                for &op_i in &ops[run_start..run_end] {
                                    if nodes[op_i].backend_id != target {
                                        nodes[op_i].backend_id = target;
                                        flipped += 1;
                                    }
                                }
                                if flipped > 0 {
                                    note_absorb_outcome(
                                        decision, bid, target, kinds, flipped, compute,
                                    );
                                    *reassigned += flipped;
                                    return true;
                                }
                            } else {
                                note_absorb_outcome(decision, bid, target, kinds, 0, compute);
                            }
                        }
                    }
                }
            }
        }
        run_start = run_end;
    }
    false
}

#[inline]
fn kind_at_npu(kinds: &[BackendKind], backend_id: i32) -> bool {
    let u = backend_id as usize;
    u < kinds.len() && kinds[u].is_npu()
}

/// Attribution: sum cut costs over non-view ops after coloring.
pub fn total_cut_cost_of_nodes(
    nodes: &[SchedCutCostNode],
    ops: &[usize],
    kinds: &[BackendKind],
) -> u64 {
    let mut ids = alloc::vec::Vec::with_capacity(ops.len());
    let mut nb = alloc::vec::Vec::with_capacity(ops.len());
    for &i in ops {
        ids.push(nodes[i].backend_id);
        nb.push(nodes[i].nbytes);
    }
    total_cut_cost(&ids, &nb, kinds)
}

/// Count backend flips along non-view nodes (debug / tests).
pub fn count_backend_flips(nodes: &[SchedRegionCoalesceNode]) -> usize {
    let mut flips = 0usize;
    let mut prev = None;
    for n in nodes {
        if n.flags & FLAG_VIEW != 0 || n.backend_id < 0 {
            continue;
        }
        if let Some(p) = prev {
            if p != n.backend_id {
                flips += 1;
            }
        }
        prev = Some(n.backend_id);
    }
    flips
}

pub fn count_backend_flips_cut(nodes: &[SchedCutCostNode]) -> usize {
    let mut flips = 0usize;
    let mut prev = None;
    for n in nodes {
        if n.flags & FLAG_VIEW != 0 || n.backend_id < 0 {
            continue;
        }
        if let Some(p) = prev {
            if p != n.backend_id {
                flips += 1;
            }
        }
        prev = Some(n.backend_id);
    }
    flips
}

/// C ABI entry: `ggml_backend_sched_region_coalesce` (v1, length-only).
#[no_mangle]
pub unsafe extern "C" fn ggml_backend_sched_region_coalesce(
    nodes: *mut SchedRegionCoalesceNode,
    n_nodes: c_int,
    n_backends: c_int,
    max_island: c_int,
) -> c_int {
    if nodes.is_null() || n_nodes < 0 || n_backends < 0 || max_island < 0 {
        return -1;
    }
    if n_nodes == 0 {
        return 0;
    }
    let slice = unsafe { core::slice::from_raw_parts_mut(nodes, n_nodes as usize) };
    coalesce_regions(slice, n_backends as usize, max_island as usize) as c_int
}

/// C ABI entry: `ggml_backend_sched_region_coalesce_v2` (nbytes + kinds + options).
///
/// `backend_kinds` length must be `n_backends` (`ggml_backend_dev_type` ordinals).
/// `options` bit0 = [`OPT_WEIGHTED_CUT_COST`]; bit1 = [`OPT_COMPUTE_WEIGHT`]
/// (informational — compute weights apply whenever weighted and `op` is packed).
#[no_mangle]
pub unsafe extern "C" fn ggml_backend_sched_region_coalesce_v2(
    nodes: *mut SchedCutCostNode,
    n_nodes: c_int,
    n_backends: c_int,
    max_island: c_int,
    backend_kinds: *const u8,
    options: u32,
) -> c_int {
    if nodes.is_null()
        || backend_kinds.is_null()
        || n_nodes < 0
        || n_backends < 0
        || max_island < 0
    {
        return -1;
    }
    if n_nodes == 0 {
        return 0;
    }
    let n_b = n_backends as usize;
    let slice = unsafe { core::slice::from_raw_parts_mut(nodes, n_nodes as usize) };
    let kind_bytes = unsafe { core::slice::from_raw_parts(backend_kinds, n_b) };
    let mut kinds = alloc::vec::Vec::with_capacity(n_b);
    for &k in kind_bytes {
        kinds.push(BackendKind::from_u8(k));
    }
    let weighted = options & OPT_WEIGHTED_CUT_COST != 0;
    // OPT_COMPUTE_WEIGHT is advisory: scoring uses packed `op` whenever present.
    let _ = options & OPT_COMPUTE_WEIGHT;
    coalesce_regions_weighted(slice, n_b, max_island as usize, &kinds, weighted) as c_int
}

/// Snapshot cut-cost absorb/dissolve counters (POD out-param).
#[no_mangle]
pub unsafe extern "C" fn ggml_backend_sched_cut_cost_stats(out: *mut CutCostStats) {
    if out.is_null() {
        return;
    }
    unsafe {
        *out = snapshot_cut_cost_stats();
    }
}

/// Reset cut-cost absorb/dissolve counters.
#[no_mangle]
pub unsafe extern "C" fn ggml_backend_sched_cut_cost_stats_reset() {
    reset_cut_cost_stats();
}

pub(crate) unsafe fn proc_address(name: *const c_char) -> *mut c_void {
    if name.is_null() {
        return ptr::null_mut();
    }
    let bytes = unsafe { CStr::from_ptr(name) }.to_bytes();
    if bytes == PROC_NAME {
        ggml_backend_sched_region_coalesce as *mut c_void
    } else if bytes == PROC_NAME_V2 {
        ggml_backend_sched_region_coalesce_v2 as *mut c_void
    } else if bytes == PROC_NAME_STATS {
        ggml_backend_sched_cut_cost_stats as *mut c_void
    } else if bytes == PROC_NAME_STATS_RESET {
        ggml_backend_sched_cut_cost_stats_reset as *mut c_void
    } else {
        ptr::null_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::cut_cost::BackendKind;

    fn node(backend: i32, feasible: u32, flags: u32) -> SchedRegionCoalesceNode {
        SchedRegionCoalesceNode {
            backend_id: backend,
            feasible_mask: feasible,
            flags,
        }
    }

    fn cut_node(backend: i32, feasible: u32, flags: u32, nbytes: u32) -> SchedCutCostNode {
        SchedCutCostNode {
            backend_id: backend,
            feasible_mask: feasible,
            flags,
            nbytes,
            op: 0,
        }
    }

    fn cut_node_op(
        backend: i32,
        feasible: u32,
        flags: u32,
        nbytes: u32,
        op: u32,
    ) -> SchedCutCostNode {
        SchedCutCostNode {
            backend_id: backend,
            feasible_mask: feasible,
            flags,
            nbytes,
            op,
        }
    }

    #[test]
    fn absorbs_single_foreign_island() {
        let both = 0b11;
        let mut nodes = [
            node(0, both, 0),
            node(0, both, 0),
            node(1, both, 0),
            node(0, both, 0),
            node(0, both, 0),
        ];
        assert_eq!(count_backend_flips(&nodes), 2);
        let n = coalesce_regions(&mut nodes, 2, 4);
        assert_eq!(n, 1);
        assert!(nodes.iter().all(|x| x.backend_id == 0));
        assert_eq!(count_backend_flips(&nodes), 0);
    }

    #[test]
    fn refuses_infeasible_island() {
        let mut nodes = [
            node(0, 0b01, 0),
            node(1, 0b10, 0),
            node(0, 0b01, 0),
        ];
        let n = coalesce_regions(&mut nodes, 2, 4);
        assert_eq!(n, 0);
        assert_eq!(nodes[1].backend_id, 1);
    }

    #[test]
    fn respects_pinned_flag() {
        let both = 0b11;
        let mut nodes = [
            node(0, both, 0),
            node(1, both, FLAG_PINNED),
            node(0, both, 0),
        ];
        assert_eq!(coalesce_regions(&mut nodes, 2, 4), 0);
        assert_eq!(nodes[1].backend_id, 1);
    }

    #[test]
    fn skips_views_in_intervals() {
        let both = 0b11;
        let mut nodes = [
            node(0, both, 0),
            node(1, both, FLAG_VIEW),
            node(1, both, 0),
            node(0, both, 0),
        ];
        assert_eq!(coalesce_regions(&mut nodes, 2, 4), 1);
        assert_eq!(nodes[0].backend_id, 0);
        assert_eq!(nodes[1].backend_id, 1);
        assert_eq!(nodes[2].backend_id, 0);
        assert_eq!(nodes[3].backend_id, 0);
    }

    #[test]
    fn leaves_sandwiched_views_for_host_view_src_sync() {
        let both = 0b11;
        let mut nodes = [
            node(0, both, 0),
            node(1, both, FLAG_VIEW),
            node(0, both, 0),
        ];
        assert_eq!(coalesce_regions(&mut nodes, 2, 4), 0);
        assert_eq!(nodes[1].backend_id, 1);
    }

    #[test]
    fn checkerboard_collapses_with_fixpoint() {
        let both = 0b11;
        let mut nodes = [
            node(0, both, 0),
            node(1, both, 0),
            node(0, both, 0),
            node(1, both, 0),
            node(0, both, 0),
        ];
        let before = count_backend_flips(&nodes);
        assert_eq!(before, 4);
        let n = coalesce_regions(&mut nodes, 2, 1);
        assert!(n >= 2);
        assert_eq!(count_backend_flips(&nodes), 0);
        assert!(nodes.iter().all(|x| x.backend_id == 0));
    }

    #[test]
    fn idle_when_disabled_shape() {
        let mut nodes = [node(0, 0b01, 0), node(0, 0b01, 0)];
        assert_eq!(coalesce_regions(&mut nodes, 2, 4), 0);
    }

    #[test]
    fn c_abi_null_is_error() {
        let rc = unsafe { ggml_backend_sched_region_coalesce(ptr::null_mut(), 1, 2, 4) };
        assert_eq!(rc, -1);
    }

    #[test]
    fn weighted_preserves_fat_npu_island_inside_gpu() {
        // backend 0 = GPU, backend 1 = NPU; 3-op NPU island stays
        let both = 0b11;
        let kinds = [BackendKind::Gpu, BackendKind::Npu];
        let mut nodes = [
            cut_node(0, both, 0, 4096),
            cut_node(1, both, 0, 8192),
            cut_node(1, both, 0, 8192),
            cut_node(1, both, 0, 8192), // 3-op NPU island
            cut_node(0, both, 0, 4096),
        ];
        let n = coalesce_regions_weighted(&mut nodes, 2, 4, &kinds, true);
        assert_eq!(n, 0);
        assert_eq!(nodes[1].backend_id, 1);
        assert_eq!(nodes[2].backend_id, 1);
        assert_eq!(nodes[3].backend_id, 1);
    }

    #[test]
    fn weighted_dissolves_thin_two_op_npu_into_gpu() {
        let both = 0b11;
        let kinds = [BackendKind::Gpu, BackendKind::Npu];
        let mut nodes = [
            cut_node(0, both, 0, 4096),
            cut_node(1, both, 0, 8192),
            cut_node(1, both, 0, 8192), // thin 2-op NPU
            cut_node(0, both, 0, 4096),
        ];
        let n = coalesce_regions_weighted(&mut nodes, 2, 4, &kinds, true);
        assert_eq!(n, 2);
        assert!(nodes.iter().all(|x| x.backend_id == 0));
    }

    #[test]
    fn weighted_dissolves_tiny_npu_unitary_into_gpu() {
        let both = 0b11;
        let kinds = [BackendKind::Gpu, BackendKind::Npu];
        let mut nodes = [
            cut_node(0, both, 0, 4096),
            cut_node(1, both, 0, 1024), // tiny unitary NPU
            cut_node(0, both, 0, 4096),
        ];
        let n = coalesce_regions_weighted(&mut nodes, 2, 4, &kinds, true);
        assert_eq!(n, 1);
        assert!(nodes.iter().all(|x| x.backend_id == 0));
    }

    #[test]
    fn weighted_absorbs_gpu_island_into_npu() {
        let both = 0b11;
        let kinds = [BackendKind::Npu, BackendKind::Gpu];
        let mut nodes = [
            cut_node(0, both, 0, 4096),
            cut_node(1, both, 0, 1024), // GPU glue between NPU
            cut_node(0, both, 0, 4096),
        ];
        let n = coalesce_regions_weighted(&mut nodes, 2, 4, &kinds, true);
        assert_eq!(n, 1);
        assert!(nodes.iter().all(|x| x.backend_id == 0));
    }

    #[test]
    fn weighted_asymmetric_absorbs_into_npu_neighbor() {
        // NPU | GPU | CPU — not a sandwich, but grow onto NPU.
        let all = 0b111;
        let kinds = [BackendKind::Npu, BackendKind::Gpu, BackendKind::Cpu];
        let mut nodes = [
            cut_node(0, all, 0, 4096),
            cut_node(1, all, 0, 1024),
            cut_node(2, all, 0, 4096),
        ];
        let n = coalesce_regions_weighted(&mut nodes, 3, 4, &kinds, true);
        assert_eq!(n, 1);
        assert_eq!(nodes[0].backend_id, 0);
        assert_eq!(nodes[1].backend_id, 0); // absorbed into NPU
        assert_eq!(nodes[2].backend_id, 2);
    }

    #[test]
    fn weighted_does_not_invent_npu_without_neighbor() {
        // GPU | CPU | GPU — both GPU, CPU feasible on NPU too, but no NPU neighbor.
        let gpu_cpu_npu = 0b111; // bits 0,1,2
        let kinds = [BackendKind::Gpu, BackendKind::Cpu, BackendKind::Npu];
        let mut nodes = [
            cut_node(0, gpu_cpu_npu, 0, 4096),
            cut_node(1, gpu_cpu_npu, 0, 1024), // feasible on NPU (bit2) but no NPU neighbor
            cut_node(0, gpu_cpu_npu, 0, 4096),
        ];
        let n = coalesce_regions_weighted(&mut nodes, 3, 4, &kinds, true);
        // Sandwich absorbs into GPU (0), not invent NPU.
        assert_eq!(n, 1);
        assert!(nodes.iter().all(|x| x.backend_id == 0));
    }

    #[test]
    fn weighted_off_dissolves_npu_like_v1() {
        let both = 0b11;
        let kinds = [BackendKind::Gpu, BackendKind::Npu];
        let mut nodes = [
            cut_node(0, both, 0, 4096),
            cut_node(1, both, 0, 8192),
            cut_node(0, both, 0, 4096),
        ];
        let n = coalesce_regions_weighted(&mut nodes, 2, 4, &kinds, false);
        assert_eq!(n, 1);
        assert!(nodes.iter().all(|x| x.backend_id == 0));
    }

    #[test]
    fn v2_abi_null_kinds_is_error() {
        let mut nodes = [cut_node(0, 0b01, 0, 0)];
        let rc = unsafe {
            ggml_backend_sched_region_coalesce_v2(
                nodes.as_mut_ptr(),
                1,
                2,
                4,
                ptr::null(),
                OPT_WEIGHTED_CUT_COST,
            )
        };
        assert_eq!(rc, -1);
    }

    #[test]
    fn heavy_npu_boundary_boosts_island() {
        let both = 0b11;
        let kinds = [BackendKind::Npu, BackendKind::Gpu];
        // Five-node GPU island between NPU — base max_island=4 would refuse;
        // 1 MiB edge boosts to allow absorb.
        let mut nodes = [
            cut_node(0, both, 0, 1 << 20),
            cut_node(1, both, 0, 64),
            cut_node(1, both, 0, 64),
            cut_node(1, both, 0, 64),
            cut_node(1, both, 0, 64),
            cut_node(1, both, 0, 64),
            cut_node(0, both, 0, 1 << 20),
        ];
        let n = coalesce_regions_weighted(&mut nodes, 2, 4, &kinds, true);
        assert!(n >= 5);
        assert!(nodes.iter().all(|x| x.backend_id == 0));
    }

    #[test]
    fn stats_exported_after_weighted_pass() {
        let both = 0b11;
        let kinds = [BackendKind::Npu, BackendKind::Gpu];
        let mut nodes = [
            cut_node(0, both, 0, 4096),
            cut_node(1, both, 0, 1024),
            cut_node(0, both, 0, 4096),
        ];
        let _ = coalesce_regions_weighted(&mut nodes, 2, 4, &kinds, true);
        let s = snapshot_cut_cost_stats();
        assert!(s.absorbed_into_npu >= 1);
    }

    #[test]
    fn weighted_preserves_heavy_matmul_unitary() {
        use crate::ffi::GGML_OP_MUL_MAT;
        let both = 0b11;
        let kinds = [BackendKind::Gpu, BackendKind::Npu];
        let mut nodes = [
            cut_node(0, both, 0, 4096),
            // 32 KiB MatMul unitary — below fat 64 KiB floor, heavy-compute preserve.
            cut_node_op(1, both, 0, 32 * 1024, GGML_OP_MUL_MAT as u32),
            cut_node(0, both, 0, 4096),
        ];
        let n = coalesce_regions_weighted(&mut nodes, 2, 4, &kinds, true);
        assert_eq!(n, 0);
        assert_eq!(nodes[1].backend_id, 1);
        let s = snapshot_cut_cost_stats();
        assert!(s.heavy_compute_preserved >= 1 || s.preserved_fat_npu >= 1);
    }

    #[test]
    fn weighted_absorbs_light_glue_with_op_class() {
        use crate::ffi::GGML_OP_ADD;
        let both = 0b11;
        let kinds = [BackendKind::Npu, BackendKind::Gpu];
        let mut nodes = [
            cut_node(0, both, 0, 8192),
            cut_node_op(1, both, 0, 1024, GGML_OP_ADD as u32),
            cut_node_op(1, both, 0, 1024, GGML_OP_ADD as u32),
            cut_node(0, both, 0, 8192),
        ];
        let n = coalesce_regions_weighted(&mut nodes, 2, 4, &kinds, true);
        assert_eq!(n, 2);
        assert!(nodes.iter().all(|x| x.backend_id == 0));
        let s = snapshot_cut_cost_stats();
        assert!(s.light_glue_into_npu >= 2);
    }
}
