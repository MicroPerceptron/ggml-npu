//! Vendor- and op-agnostic dispatch: recognize offloadable ops in a cgraph and route them to
//! the present NPU vendor. This is the only thing `backend/` (the ggml ABI shell) forwards to;
//! it knows nothing of matmul or Intel — [`crate::ops`] owns op recognition and tensor mapping,
//! [`crate::vendor`] owns lowering + submission (and each vendor its own artifact cache).
//!
//! (Named `engine`, not `core`, to avoid shadowing the `core` crate.)

use crate::ffi::*;
use crate::{ops, vendor};

/// `supports_op`: can the present NPU vendor run this node?
pub unsafe fn supports(op: *const ggml_tensor) -> bool {
    let Some(matched) = ops::match_node(op) else {
        return false;
    };
    vendor::present().is_some_and(|v| v.can_lower(matched))
}

/// `graph_compute`: run every recognized op in the assigned split on the NPU.
pub unsafe fn compute(cgraph: *mut ggml_cgraph) -> bool {
    let Some(v) = vendor::present() else {
        return true; // no NPU present; nothing was placed on us
    };
    let n = ggml_graph_n_nodes(cgraph);
    for i in 0..n {
        let node = ggml_graph_node(cgraph, i);
        if node.is_null() {
            continue;
        }
        let Some(op) = ops::match_node(node as *const ggml_tensor) else {
            continue; // not one of ours; leave it (shouldn't happen within our split)
        };
        if !ops::run(op, node, v) {
            return false;
        }
    }
    true
}
