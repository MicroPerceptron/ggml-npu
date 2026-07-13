//! Conservative heterogeneous placement / profitability policy.
//!
//! Capability (`supports_op` / vendor `node_support` / `graph_feasible`) answers “can we execute
//! correctly?”. This module answers “is it worth pulling?” via [`should_pull`] (`offload_op`).
//! [`supports_region`] is region-expansion eligibility for fusible/metadata neighborhoods — it must
//! **not** gate `supports_op`, or the backend over/under-claims relative to the compiler (#8).

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

/// LFM ShortConv recurrent cache is `cache_r_lN` (see llama.cpp `pattern_r_cache`). Attention KV
/// uses `cache_k` / `cache_v` and must stay out of this opt-in pull path (#9 / #10).
unsafe fn touches_shortconv_state(t: *const ggml_tensor) -> bool {
    if t.is_null() {
        return false;
    }
    if name_contains(t, b"cache_r") {
        return true;
    }
    (*t).src
        .iter()
        .copied()
        .filter(|src| !src.is_null())
        .any(|src| name_contains(src, b"cache_r"))
}

fn is_shortconv_state_op(op: ggml_op) -> bool {
    matches!(
        op,
        GGML_OP_SSM_CONV | GGML_OP_SET_ROWS | GGML_OP_GET_ROWS | GGML_OP_CPY
    )
}

unsafe fn is_shortconv_candidate(t: *const ggml_tensor) -> bool {
    // Explicit architecture boundaries stop ancestry from leaking through the residual stream
    // into the following FFN/attention block. Depth 12 covers LFM ShortConv gather/state spans
    // that previously fragmented across CPU/NPU when ancestry stopped at 6 (#5).
    // Gather/write of `cache_r*` often sits *before* SSM_CONV in the DAG (no conv ancestor), so
    // also admit those state edges by cache name (#9).
    !name_contains(t, b"ffn")
        && !name_contains(t, b"self_attn")
        && !name_contains(t, b"operator_norm")
        && (has_shortconv_ancestor(t, 12) || touches_shortconv_state(t))
}

fn shortconv_should_pull(placement: PlacementClass, op: ggml_op) -> bool {
    match placement {
        // Stateful SET_ROWS / SSM_CONV are first-class ShortConv state edges; other Stateful
        // ops stay gated so attention-side state does not ride this opt-in.
        PlacementClass::Stateful => is_shortconv_state_op(op),
        _ => true,
    }
}

/// Attention KV cache tensors are `cache_k*` / `cache_v*` (llama.cpp KV). Distinct from ShortConv
/// `cache_r*` (#9). Used only to keep SetRows/CPY contiguous with claimable QKV+RoPE (#10).
unsafe fn touches_attn_kv_cache(t: *const ggml_tensor) -> bool {
    if t.is_null() {
        return false;
    }
    if name_contains(t, b"cache_k") || name_contains(t, b"cache_v") {
        return true;
    }
    (*t).src
        .iter()
        .copied()
        .filter(|src| !src.is_null())
        .any(|src| name_contains(src, b"cache_k") || name_contains(src, b"cache_v"))
}

unsafe fn has_claimable_matmul_ancestor(t: *const ggml_tensor, depth: usize) -> bool {
    if t.is_null() || depth == 0 {
        return false;
    }
    (*t).src
        .iter()
        .copied()
        .filter(|src| !src.is_null())
        .any(|src| {
            crate::ops::subgraph::is_claimable_matmul(src)
                || has_claimable_matmul_ancestor(src, depth - 1)
        })
}

unsafe fn has_rope_ancestor(t: *const ggml_tensor, depth: usize) -> bool {
    if t.is_null() || depth == 0 {
        return false;
    }
    (*t).src
        .iter()
        .copied()
        .filter(|src| !src.is_null())
        .any(|src| (*src).op == GGML_OP_ROPE || has_rope_ancestor(src, depth - 1))
}

/// Prefill QKV continuity pull (#10): RoPE / qk-norm / KV SetRows that sit under constant-weight
/// QKV projections, without opting into SoftMax/FA (`GGML_NPU_ATTENTION=full`).
fn qkv_continuity_should_pull(op: ggml_op) -> bool {
    matches!(
        op,
        GGML_OP_ROPE | GGML_OP_RMS_NORM | GGML_OP_SET_ROWS | GGML_OP_GET_ROWS | GGML_OP_CPY
    )
}

unsafe fn is_qkv_continuity_candidate(t: *const ggml_tensor) -> bool {
    if t.is_null() || !qkv_continuity_should_pull((*t).op) {
        return false;
    }
    // SoftMax/FA are never continuity pulls — capability keeps them behind ATTENTION=full.
    match (*t).op {
        GGML_OP_ROPE | GGML_OP_RMS_NORM => has_claimable_matmul_ancestor(t, 12),
        GGML_OP_SET_ROWS | GGML_OP_GET_ROWS | GGML_OP_CPY => {
            touches_attn_kv_cache(t)
                && (has_claimable_matmul_ancestor(t, 12) || has_rope_ancestor(t, 12))
        }
        _ => false,
    }
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

/// True when `t` sits under FFN fusible producers (silu / SwiGLU / silu·up / FFN-shaped matmul).
/// Used to enlarge layout/glue regions around FFN without pulling attention-side orphans (#5).
unsafe fn has_ffn_fusible_ancestor(t: *const ggml_tensor, depth: usize) -> bool {
    if t.is_null() || depth == 0 {
        return false;
    }
    (*t).src
        .iter()
        .copied()
        .filter(|src| !src.is_null())
        .any(|src| {
            crate::ops::subgraph::is_ffn_shaped_matmul(src)
                || crate::ops::subgraph::is_silu(src)
                || crate::ops::subgraph::is_swiglu(src)
                || crate::ops::subgraph::is_ffn_shaped_activation(src)
                || has_ffn_fusible_ancestor(src, depth - 1)
        })
}

/// Dependency-aware fusible/metadata pull: FFN glue claims, or layout under an FFN fusible span.
fn fusible_neighborhood_should_pull(
    placement: PlacementClass,
    claims_glue: bool,
    ffn_fusible_ancestor: bool,
) -> bool {
    match placement {
        PlacementClass::FusibleOnly | PlacementClass::Metadata => {
            claims_glue || ffn_fusible_ancestor
        }
        _ => false,
    }
}

unsafe fn should_pull_fusible_neighborhood(node: *const ggml_tensor, spec: OpSpec) -> bool {
    fusible_neighborhood_should_pull(
        spec.placement,
        crate::ops::subgraph::claims_glue(node),
        has_ffn_fusible_ancestor(node, 12),
    )
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

/// Region-expansion eligibility (fusible/metadata near an anchor), not binary capability.
/// Cheap metadata/elementwise work may extend an eligible accelerator region, but cannot cross a
/// routed-MoE CPU boundary or anchor a split by itself. Shape/graph based; no tensor-name checks.
/// Callers: profitability / neighborhood logic only — never `supports_op` (#8).
pub(crate) unsafe fn supports_region(node: *const ggml_tensor, spec: OpSpec) -> bool {
    if node.is_null() {
        return false;
    }
    // Match compile capability: only `GGML_NPU_MOE=1` opts in (not mere presence / `=0`).
    let moe_enabled = crate::host::builtin().config().is_opt_in_one(c"GGML_NPU_MOE");
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
        && shortconv_should_pull(spec.placement, x.op)
        && is_shortconv_candidate(node)
    {
        return true;
    }
    // Default (no ATTENTION=full): keep prefill QKV-proj regions contiguous by pulling RoPE,
    // qk-norm, and KV-cache SetRows that sit under claimable projections (#10). SoftMax/FA stay
    // capability-gated and are never selected here.
    if is_qkv_continuity_candidate(node) {
        return true;
    }
    if crate::host::builtin()
        .config()
        .is_set(c"GGML_NPU_FORCE_GRAPH")
    {
        // Anchors may start a split. Cheap glue/layout may only extend a split downstream from an
        // anchor; this prevents embedding gathers or isolated unary nodes from becoming NPU
        // dispatches merely because force mode was requested. Depth matches supports_region (#5).
        return match spec.placement {
            PlacementClass::Anchor | PlacementClass::StandaloneIfProfitable => true,
            PlacementClass::Stateful => is_shortconv_state_op(x.op),
            PlacementClass::FusibleOnly | PlacementClass::Metadata => has_anchor_ancestor(node, 12),
        };
    }
    let min_work = env_u64(c"GGML_NPU_MIN_OFFLOAD_WORK", 4_000_000);
    let work = match x.op {
        GGML_OP_MUL_MAT | GGML_OP_MUL_MAT_ID => {
            let a = x.src[0];
            if a.is_null() {
                return false;
            }
            // Runtime-weight matmuls cannot bake into fused NPU programs; pulling them creates
            // uneconomic unitary islands. Const-weight FFN/attention projections remain eligible.
            if x.op == GGML_OP_MUL_MAT && !crate::ops::subgraph::is_claimable_matmul(node) {
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
        // Dependency-aware FFN glue/layout pull: enlarge fused partitions around ShortConv/FFN
        // instead of leaving claimable silu/mul/residual on CPU while matmuls go NPU (#5).
        _ => return should_pull_fusible_neighborhood(node, spec),
    };
    // Orphan/attention-side matmuls (not FFN-local) pay full handoff cost as unitary programs.
    // Default unitary floor is 4× MIN_OFFLOAD_WORK so uneconomic 1-node MUL_MAT pulls stay off
    // NPU unless explicitly lowered via GGML_NPU_MIN_UNITARY_OFFLOAD_WORK (#5).
    let min_work = if x.op == GGML_OP_MUL_MAT
        && !crate::ops::subgraph::is_ffn_shaped_matmul(node)
    {
        unitary_offload_floor(min_work)
    } else {
        min_work
    };
    work >= min_work
}

/// Higher offload floor for non-FFN unitary matmul pulls. `GGML_NPU_MIN_UNITARY_OFFLOAD_WORK`
/// overrides the default (`4 * min_work`) but never drops below the general floor.
fn unitary_offload_floor(min_work: u64) -> u64 {
    let default = min_work.saturating_mul(4).max(min_work);
    env_u64(c"GGML_NPU_MIN_UNITARY_OFFLOAD_WORK", default).max(min_work)
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
    fn supports_region_rejects_null_and_stays_off_supports_op_path() {
        // supports_region is region-expansion only; engine::supports must not call it (#8).
        assert!(!unsafe { supports_region(core::ptr::null(), spec(PlacementClass::Anchor, false)) });
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

    #[test]
    fn unitary_offload_floor_defaults_to_four_times_general_floor() {
        // No env override in unit tests: default is 4× and never below min_work.
        assert_eq!(unitary_offload_floor(4_000_000), 16_000_000);
        assert_eq!(unitary_offload_floor(0), 0);
        assert_eq!(unitary_offload_floor(u64::MAX / 2 + 1), u64::MAX);
    }

    #[test]
    fn qkv_continuity_pulls_rope_and_kv_writes_not_softmax() {
        // Pure op allowlist for prefill QKV continuity (#10) — SoftMax/FA never ride this path.
        assert!(qkv_continuity_should_pull(GGML_OP_ROPE));
        assert!(qkv_continuity_should_pull(GGML_OP_RMS_NORM));
        assert!(qkv_continuity_should_pull(GGML_OP_SET_ROWS));
        assert!(qkv_continuity_should_pull(GGML_OP_CPY));
        assert!(!qkv_continuity_should_pull(GGML_OP_SOFT_MAX));
        assert!(!qkv_continuity_should_pull(GGML_OP_FLASH_ATTN_EXT));
        assert!(!qkv_continuity_should_pull(GGML_OP_MUL_MAT));
    }

    #[test]
    fn fusible_neighborhood_pulls_claimed_glue_and_ffn_span_layout() {
        // Claimed FFN glue always pulls; layout/metadata only under an FFN fusible ancestor.
        assert!(fusible_neighborhood_should_pull(
            PlacementClass::FusibleOnly,
            true,
            false
        ));
        assert!(fusible_neighborhood_should_pull(
            PlacementClass::Metadata,
            false,
            true
        ));
        assert!(!fusible_neighborhood_should_pull(
            PlacementClass::FusibleOnly,
            false,
            false
        ));
        assert!(!fusible_neighborhood_should_pull(
            PlacementClass::Anchor,
            true,
            true
        ));
    }

    #[test]
    fn shortconv_opt_in_pulls_stateful_set_rows_and_ssm_conv() {
        assert!(shortconv_should_pull(
            PlacementClass::Stateful,
            GGML_OP_SSM_CONV
        ));
        assert!(shortconv_should_pull(
            PlacementClass::Stateful,
            GGML_OP_SET_ROWS
        ));
        assert!(shortconv_should_pull(
            PlacementClass::FusibleOnly,
            GGML_OP_GET_ROWS
        ));
        assert!(shortconv_should_pull(
            PlacementClass::FusibleOnly,
            GGML_OP_CPY
        ));
        // Unknown Stateful ops stay refused under the ShortConv opt-in.
        assert!(!shortconv_should_pull(
            PlacementClass::Stateful,
            GGML_OP_FLASH_ATTN_EXT
        ));
    }
}
