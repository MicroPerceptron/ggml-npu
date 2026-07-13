//! Prefill-oriented same-backend region coalesce (Rust core).
//!
//! Conceptual inspiration: axiom/kore bit-row set algebra for DAG intervals —
//! not a Q1 tensor copy. This pass finds short foreign "islands" sandwiched
//! between the same backend and recolors them when every island node is
//! feasible on the surrounding backend (via the caller-supplied mask).
//!
//! Profitability / claim policy stays in `offload_op` / `policy::should_pull`.
//! This only widens homogeneous splits the scheduler already painted.

use core::ffi::{c_char, c_int, c_void, CStr};
use core::ptr;

/// Skip this node in interval scans (view / reshape glue that does not cut).
pub const FLAG_VIEW: u32 = 1 << 0;
/// Do not recolor (graph input, weight buffer, or other pinned placement).
pub const FLAG_PINNED: u32 = 1 << 1;

/// POD node row packed by the llama.cpp scheduler hook.
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

const PROC_NAME: &[u8] = b"ggml_backend_sched_region_coalesce";

/// Pure analysis: mutate `nodes[..]` backend ids; return how many were recolored.
///
/// `n_backends` must be in `1..=32` (mask width). `max_island` is the longest
/// foreign run that may be absorbed (clamped to at least 1).
pub fn coalesce_regions(
    nodes: &mut [SchedRegionCoalesceNode],
    n_backends: usize,
    max_island: usize,
) -> usize {
    if nodes.is_empty() || n_backends < 2 || n_backends > 32 {
        return 0;
    }
    let max_island = max_island.max(1);

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
        let mut run_start = 0usize;
        while run_start < ops.len() {
            let bid = nodes[ops[run_start]].backend_id;
            let mut run_end = run_start + 1;
            while run_end < ops.len() && nodes[ops[run_end]].backend_id == bid {
                run_end += 1;
            }
            let run_len = run_end - run_start;

            if run_start > 0 && run_end < ops.len() && run_len <= max_island {
                let left = nodes[ops[run_start - 1]].backend_id;
                let right = nodes[ops[run_end]].backend_id;
                if left == right && left != bid && left >= 0 {
                    let left_u = left as u32;
                    if left_u < 32 {
                        let bit = 1u32 << left_u;
                        let mut ok = true;
                        for &op_i in &ops[run_start..run_end] {
                            let n = &nodes[op_i];
                            if n.flags & FLAG_PINNED != 0 || n.feasible_mask & bit == 0 {
                                ok = false;
                                break;
                            }
                        }
                        if ok {
                            for &op_i in &ops[run_start..run_end] {
                                if nodes[op_i].backend_id != left {
                                    nodes[op_i].backend_id = left;
                                    reassigned += 1;
                                }
                            }
                            changed = true;
                            break;
                        }
                    }
                }
            }
            run_start = run_end;
        }
    }

    // Views stay untouched here. The llama.cpp hook (1) applies only safe
    // targets (NPU / same buft — not Vulkan sandwiches) and (2) re-syncs view
    // ops to view_src. Nearest-neighbor view painting broke weight views and
    // still SIGSEGV'd in ggml_vk_build_graph on pp128.

    // Edge-cost stub site: future weighted cuts can score handoff bytes here
    // without changing the C ABI (see docs/sched-region-coalesce.md).
    let _edge_cost_stub = 0usize;
    let _ = (_edge_cost_stub, n_backends);

    reassigned
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

/// C ABI entry: `ggml_backend_sched_region_coalesce`.
///
/// Returns the number of nodes whose `backend_id` changed, or `-1` on bad args.
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

pub(crate) unsafe fn proc_address(name: *const c_char) -> *mut c_void {
    if name.is_null() {
        return ptr::null_mut();
    }
    if unsafe { CStr::from_ptr(name) }.to_bytes() == PROC_NAME {
        ggml_backend_sched_region_coalesce as *mut c_void
    } else {
        ptr::null_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(backend: i32, feasible: u32, flags: u32) -> SchedRegionCoalesceNode {
        SchedRegionCoalesceNode {
            backend_id: backend,
            feasible_mask: feasible,
            flags,
        }
    }

    #[test]
    fn absorbs_single_foreign_island() {
        // A A B A A  with B feasible on A → all A
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
            node(1, 0b10, 0), // only backend 1
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
            node(1, both, FLAG_VIEW), // ignored for island scan; left for host sync
            node(1, both, 0),
            node(0, both, 0),
        ];
        // Non-view sequence: 0, 1, 0 → absorb middle; view row unchanged here
        assert_eq!(coalesce_regions(&mut nodes, 2, 4), 1);
        assert_eq!(nodes[0].backend_id, 0);
        assert_eq!(nodes[1].backend_id, 1);
        assert_eq!(nodes[2].backend_id, 0);
        assert_eq!(nodes[3].backend_id, 0);
    }

    #[test]
    fn leaves_sandwiched_views_for_host_view_src_sync() {
        // Host reapplies view backends from view_src after guarded apply.
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
}
