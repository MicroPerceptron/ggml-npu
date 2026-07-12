//! Conservative heterogeneous placement policy.
//!
//! Pre-placed NPU tensors use the normal capability path. `offload_op` uses this cost floor before
//! pulling work from CPU/GPU memory, avoiding per-op ping-pong for tiny latency-bound kernels.

use core::ffi::CStr;

use crate::ffi::*;
use crate::graph::{OpRegistry, OpSpec, OperationContract, PlacementClass};
use crate::host::{ConfigService, HostEnvironment};

fn env_u64(name: &'static CStr, default: u64) -> u64 {
    let Some(value) = crate::host::builtin().config().value(name) else {
        return default;
    };
    let bytes = value.to_bytes();
    let mut value = 0u64;
    if bytes.is_empty() {
        return default;
    }
    for &b in bytes {
        if !b.is_ascii_digit() {
            return default;
        }
        value = match value
            .checked_mul(10)
            .and_then(|x| x.checked_add((b - b'0') as u64))
        {
            Some(v) => v,
            None => return default,
        };
    }
    value
}

unsafe fn name_contains(t: *const ggml_tensor, needle: &[u8]) -> bool {
    if t.is_null() || needle.is_empty() {
        return false;
    }
    let name = &(*t).name;
    name.windows(needle.len())
        .any(|window| window.iter().zip(needle).all(|(&a, &b)| a as u8 == b))
}

unsafe fn has_shortconv_ancestor(t: *const ggml_tensor, depth: usize) -> bool {
    if t.is_null() {
        return false;
    }
    if (*t).op == GGML_OP_SSM_CONV || name_contains(t, b".conv.") {
        return true;
    }
    depth != 0
        && (*t)
            .src
            .iter()
            .copied()
            .filter(|src| !src.is_null())
            .any(|src| has_shortconv_ancestor(src, depth - 1))
}

unsafe fn is_shortconv_candidate(t: *const ggml_tensor) -> bool {
    // Explicit architecture boundaries stop ancestry from leaking through the residual stream
    // into the following FFN/attention block.
    !name_contains(t, b"ffn")
        && !name_contains(t, b"self_attn")
        && !name_contains(t, b"operator_norm")
        && has_shortconv_ancestor(t, 6)
}

unsafe fn has_anchor_ancestor(t: *const ggml_tensor, depth: usize) -> bool {
    if t.is_null() || depth == 0 {
        return false;
    }
    (*t).src
        .iter()
        .copied()
        .filter(|src| !src.is_null())
        .any(|src| {
            let anchored = OpRegistry::decode_tensor(src).is_ok_and(|spec| {
                matches!(
                    spec.placement,
                    PlacementClass::Anchor
                        | PlacementClass::StandaloneIfProfitable
                        | PlacementClass::Stateful
                )
            });
            anchored || has_anchor_ancestor(src, depth - 1)
        })
}

unsafe fn has_disabled_moe_ancestor(t: *const ggml_tensor, depth: usize) -> bool {
    if t.is_null() || depth == 0 {
        return false;
    }
    (*t).src
        .iter()
        .copied()
        .filter(|src| !src.is_null())
        .any(|src| {
            let routed = OpRegistry::decode_tensor(src).is_ok_and(|spec| spec.traits.moe);
            routed || has_disabled_moe_ancestor(src, depth - 1)
        })
}

fn output_elements(node: &ggml_tensor) -> u64 {
    node.ne.iter().fold(1u64, |product, &extent| {
        product.saturating_mul(extent.max(0) as u64)
    })
}

fn region_supported(
    spec: OpSpec,
    elements: u64,
    moe_enabled: bool,
    disabled_moe_ancestor: bool,
    anchor_ancestor: bool,
    min_region_elements: u64,
) -> bool {
    if spec.traits.moe && !moe_enabled {
        return false;
    }
    match spec.placement {
        PlacementClass::Anchor | PlacementClass::Stateful => true,
        PlacementClass::StandaloneIfProfitable => elements >= min_region_elements,
        PlacementClass::FusibleOnly | PlacementClass::Metadata => {
            (moe_enabled || !disabled_moe_ancestor) && anchor_ancestor
        }
    }
}

/// Capability placement for scheduler expansion. Cheap metadata/elementwise work may extend an
/// eligible accelerator region, but cannot cross a routed-MoE CPU boundary or anchor a split by
/// itself. This is shape/graph based and deliberately contains no model or tensor-name checks.
pub(crate) unsafe fn supports_region(node: *const ggml_tensor, spec: OpSpec) -> bool {
    if node.is_null() {
        return false;
    }
    let moe_enabled = crate::host::builtin().config().is_set(c"GGML_NPU_MOE");
    region_supported(
        spec,
        output_elements(&*node),
        moe_enabled,
        !moe_enabled && has_disabled_moe_ancestor(node, 12),
        has_anchor_ancestor(node, 12),
        env_u64(c"GGML_NPU_MIN_REGION_ELEMENTS", 32_768),
    )
}

pub(crate) unsafe fn should_pull(node: *const ggml_tensor) -> bool {
    if node.is_null() {
        return false;
    }
    let x = &*node;
    let Ok(spec) = OpRegistry::decode_tensor(node) else {
        return false;
    };
    if crate::host::builtin()
        .config()
        .is_set(c"GGML_NPU_LFM_SHORTCONV")
        && (spec.placement != PlacementClass::Stateful || x.op == GGML_OP_SSM_CONV)
        && is_shortconv_candidate(node)
    {
        return true;
    }
    if crate::host::builtin()
        .config()
        .is_set(c"GGML_NPU_FORCE_GRAPH")
    {
        // Anchors may start a split. Cheap glue/layout may only extend a split downstream from an
        // anchor; this prevents embedding gathers or isolated unary nodes from becoming NPU
        // dispatches merely because force mode was requested.
        return match spec.placement {
            PlacementClass::Anchor | PlacementClass::StandaloneIfProfitable => true,
            PlacementClass::Stateful => x.op == GGML_OP_SSM_CONV,
            PlacementClass::FusibleOnly | PlacementClass::Metadata => has_anchor_ancestor(node, 6),
        };
    }
    let min_work = env_u64(c"GGML_NPU_MIN_OFFLOAD_WORK", 4_000_000);
    let work = match x.op {
        GGML_OP_MUL_MAT | GGML_OP_MUL_MAT_ID => {
            let a = x.src[0];
            if a.is_null() {
                return false;
            }
            (x.ne[0].max(0) as u64)
                .saturating_mul(x.ne[1].max(0) as u64)
                .saturating_mul((*a).ne[0].max(0) as u64)
        }
        GGML_OP_SSM_CONV => {
            x.ne.iter()
                .fold(1u64, |a, &b| a.saturating_mul(b.max(0) as u64))
        }
        // Elementwise/state nodes should move only as part of a larger split formed by an
        // expensive anchor; pulling them independently creates synchronization islands.
        _ => return false,
    };
    // Orphan/attention-side matmuls (not FFN-local) pay full handoff cost as unitary programs.
    // Opt-in higher floor via GGML_NPU_MIN_UNITARY_OFFLOAD_WORK (defaults to MIN_OFFLOAD_WORK).
    let min_work = if x.op == GGML_OP_MUL_MAT
        && !crate::ops::subgraph::is_ffn_shaped_matmul(node)
    {
        env_u64(c"GGML_NPU_MIN_UNITARY_OFFLOAD_WORK", min_work).max(min_work)
    } else {
        min_work
    };
    work >= min_work
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(placement: PlacementClass, moe: bool) -> OpSpec {
        let mut spec = crate::graph::OpRegistry::decode_node(&crate::graph::Node {
            op: GGML_OP_ADD,
            spec: Err(crate::graph::OpReject::UnsupportedOp),
            inputs: alloc::vec![0, 1],
            output: 2,
            op_params: [0; crate::graph::OP_PARAM_BYTES],
        })
        .unwrap();
        spec.placement = placement;
        spec.traits.moe = moe;
        spec
    }

    #[test]
    fn moe_disabled_region_does_not_pull_adjacent_glue() {
        let glue = spec(PlacementClass::FusibleOnly, false);
        assert!(!region_supported(glue, 4096, false, true, true, 32_768));
        assert!(region_supported(glue, 4096, false, false, true, 32_768));
        assert!(!region_supported(glue, 4096, false, false, false, 32_768));
    }

    #[test]
    fn standalone_region_obeys_deterministic_cost_floor() {
        let standalone = spec(PlacementClass::StandaloneIfProfitable, false);
        assert!(!region_supported(
            standalone, 32_767, false, false, false, 32_768
        ));
        assert!(region_supported(
            standalone, 32_768, false, false, false, 32_768
        ));
    }
}
