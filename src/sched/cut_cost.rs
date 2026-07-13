//! Weighted DAG cut-cost helpers for region coalesce (#48).
//!
//! Heuristic edge cost = tensor nbytes × backend-pair handoff factor.
//! Prefer growing / preserving fat NPU regions; dissolve thin unitaries;
//! never invent NPU claims (feasibility still comes from the caller's
//! `feasible_mask`).
//!
//! Not a per-ioctl bandwidth oracle. Factors are tunable constants.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

/// Matches `enum ggml_backend_dev_type` ordinals in ggml-backend.h.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Cpu = 0,
    Gpu = 1,
    Igpu = 2,
    Accel = 3,
    Meta = 4,
    Npu = 5,
    Unknown = 255,
}

impl BackendKind {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Cpu,
            1 => Self::Gpu,
            2 => Self::Igpu,
            3 => Self::Accel,
            4 => Self::Meta,
            5 => Self::Npu,
            _ => Self::Unknown,
        }
    }

    #[inline]
    pub fn is_npu(self) -> bool {
        matches!(self, Self::Npu)
    }
}

/// Options bit for `ggml_backend_sched_region_coalesce_v2`.
pub const OPT_WEIGHTED_CUT_COST: u32 = 1 << 0;

/// Fixed-point handoff factor (×16). Higher ⇒ more expensive cross-backend edge.
#[inline]
pub fn handoff_factor(from: BackendKind, to: BackendKind) -> u32 {
    if from == to {
        return 0;
    }
    match (from, to) {
        (BackendKind::Npu, BackendKind::Gpu) | (BackendKind::Gpu, BackendKind::Npu) => 48, // 3.0×
        (BackendKind::Npu, BackendKind::Igpu) | (BackendKind::Igpu, BackendKind::Npu) => 40, // 2.5×
        (BackendKind::Npu, BackendKind::Cpu) | (BackendKind::Cpu, BackendKind::Npu) => 32, // 2.0×
        (BackendKind::Npu, BackendKind::Accel) | (BackendKind::Accel, BackendKind::Npu) => 28, // 1.75×
        (BackendKind::Gpu, BackendKind::Cpu) | (BackendKind::Cpu, BackendKind::Gpu) => 24, // 1.5×
        (BackendKind::Gpu, BackendKind::Accel) | (BackendKind::Accel, BackendKind::Gpu) => 20, // 1.25×
        _ => 16, // 1.0× default hop
    }
}

/// Cost of one cut edge in fixed-point (nbytes × factor).
#[inline]
pub fn edge_cost(nbytes: u32, from: BackendKind, to: BackendKind) -> u64 {
    let f = handoff_factor(from, to) as u64;
    if f == 0 {
        return 0;
    }
    (nbytes as u64).saturating_mul(f)
}

/// Sum cut costs along a linear backend coloring (non-view ops already filtered).
///
/// When `portable-simd` is on, scores adjacent edges in SIMD batches of 4 for
/// the nbytes×factor product; kinds still resolve per-lane (table lookup).
pub fn total_cut_cost(
    backend_ids: &[i32],
    nbytes: &[u32],
    kinds: &[BackendKind],
) -> u64 {
    debug_assert_eq!(backend_ids.len(), nbytes.len());
    if backend_ids.len() < 2 {
        return 0;
    }

    #[cfg(feature = "portable-simd")]
    {
        total_cut_cost_simd(backend_ids, nbytes, kinds)
    }
    #[cfg(not(feature = "portable-simd"))]
    {
        total_cut_cost_scalar(backend_ids, nbytes, kinds)
    }
}

fn total_cut_cost_scalar(
    backend_ids: &[i32],
    nbytes: &[u32],
    kinds: &[BackendKind],
) -> u64 {
    let mut sum = 0u64;
    for i in 1..backend_ids.len() {
        let a = backend_ids[i - 1];
        let b = backend_ids[i];
        if a < 0 || b < 0 || a == b {
            continue;
        }
        let ka = kind_at(kinds, a);
        let kb = kind_at(kinds, b);
        // Edge weight uses the destination tensor bytes (activation crossing the cut).
        sum = sum.saturating_add(edge_cost(nbytes[i], ka, kb));
    }
    sum
}

#[cfg(feature = "portable-simd")]
fn total_cut_cost_simd(
    backend_ids: &[i32],
    nbytes: &[u32],
    kinds: &[BackendKind],
) -> u64 {
    use core::simd::u32x4;

    let mut sum = 0u64;
    let n = backend_ids.len();
    let mut i = 1usize;
    // Process edges in groups of 4: product nbytes[i] × factor, then horizontal add.
    while i + 3 < n {
        let mut factors = [0u32; 4];
        let mut bytes = [0u32; 4];
        for lane in 0..4 {
            let idx = i + lane;
            let a = backend_ids[idx - 1];
            let b = backend_ids[idx];
            if a >= 0 && b >= 0 && a != b {
                factors[lane] = handoff_factor(kind_at(kinds, a), kind_at(kinds, b));
                bytes[lane] = nbytes[idx];
            }
        }
        let fb = u32x4::from_array(bytes) * u32x4::from_array(factors);
        let arr = fb.to_array();
        sum = sum
            .saturating_add(arr[0] as u64)
            .saturating_add(arr[1] as u64)
            .saturating_add(arr[2] as u64)
            .saturating_add(arr[3] as u64);
        i += 4;
    }
    while i < n {
        let a = backend_ids[i - 1];
        let b = backend_ids[i];
        if a >= 0 && b >= 0 && a != b {
            sum = sum.saturating_add(edge_cost(
                nbytes[i],
                kind_at(kinds, a),
                kind_at(kinds, b),
            ));
        }
        i += 1;
    }
    sum
}

#[inline]
fn kind_at(kinds: &[BackendKind], backend_id: i32) -> BackendKind {
    let u = backend_id as usize;
    if u < kinds.len() {
        kinds[u]
    } else {
        BackendKind::Unknown
    }
}

/// Minimum island payload (bytes) worth keeping on NPU instead of dissolving
/// into Vulkan/CPU. Below this, classic coalesce may absorb (avoids 400-disp
/// unitary explosion from preserving every tiny feasible NPU sandwich).
pub const NPU_PRESERVE_MIN_BYTES: u64 = 64 * 1024;
/// Prefer fat islands: require ≥3 non-view ops (2-op islands dissolve unless
/// they meet the byte floor). MVP used 2 and inflated pp16 dispatch count.
pub const NPU_PRESERVE_MIN_OPS: usize = 3;

/// Decision for absorbing a foreign island into a target backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbsorbDecision {
    Absorb,
    RefusePreserveNpu,
    RefuseNoGain,
}

/// Process-wide cut-cost attribution (reset at the start of each weighted pass).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CutCostStats {
    pub absorbed_into_npu: u64,
    pub dissolved_npu: u64,
    pub preserved_fat_npu: u64,
    pub absorbed_other: u64,
}

static STAT_ABSORBED_INTO_NPU: AtomicU64 = AtomicU64::new(0);
static STAT_DISSOLVED_NPU: AtomicU64 = AtomicU64::new(0);
static STAT_PRESERVED_FAT_NPU: AtomicU64 = AtomicU64::new(0);
static STAT_ABSORBED_OTHER: AtomicU64 = AtomicU64::new(0);

pub fn reset_cut_cost_stats() {
    STAT_ABSORBED_INTO_NPU.store(0, Ordering::Relaxed);
    STAT_DISSOLVED_NPU.store(0, Ordering::Relaxed);
    STAT_PRESERVED_FAT_NPU.store(0, Ordering::Relaxed);
    STAT_ABSORBED_OTHER.store(0, Ordering::Relaxed);
}

pub fn snapshot_cut_cost_stats() -> CutCostStats {
    CutCostStats {
        absorbed_into_npu: STAT_ABSORBED_INTO_NPU.load(Ordering::Relaxed),
        dissolved_npu: STAT_DISSOLVED_NPU.load(Ordering::Relaxed),
        preserved_fat_npu: STAT_PRESERVED_FAT_NPU.load(Ordering::Relaxed),
        absorbed_other: STAT_ABSORBED_OTHER.load(Ordering::Relaxed),
    }
}

#[inline]
fn record_decision(
    decision: AbsorbDecision,
    island_backend: i32,
    target_backend: i32,
    kinds: &[BackendKind],
    recolored_ops: usize,
) {
    match decision {
        AbsorbDecision::Absorb => {
            if kind_at(kinds, target_backend).is_npu() {
                STAT_ABSORBED_INTO_NPU.fetch_add(recolored_ops as u64, Ordering::Relaxed);
            } else if kind_at(kinds, island_backend).is_npu() {
                STAT_DISSOLVED_NPU.fetch_add(recolored_ops as u64, Ordering::Relaxed);
            } else {
                STAT_ABSORBED_OTHER.fetch_add(recolored_ops as u64, Ordering::Relaxed);
            }
        }
        AbsorbDecision::RefusePreserveNpu => {
            STAT_PRESERVED_FAT_NPU.fetch_add(1, Ordering::Relaxed);
        }
        AbsorbDecision::RefuseNoGain => {}
    }
}

/// True when an NPU island is worth keeping instead of dissolving into surround.
#[inline]
pub fn is_fat_npu_island(island_op_count: usize, island_nbytes_sum: u64) -> bool {
    island_op_count >= NPU_PRESERVE_MIN_OPS || island_nbytes_sum >= NPU_PRESERVE_MIN_BYTES
}

/// Prefer an existing NPU neighbor as absorb target (anti-unitary growth).
///
/// Does **not** invent NPU placement when neither neighbor is NPU — that would
/// paint new NPU islands inside Vulkan/CPU runs.
pub fn npu_absorb_target(
    left: i32,
    right: i32,
    kinds: &[BackendKind],
    weighted: bool,
) -> Option<i32> {
    if !weighted || left < 0 || right < 0 {
        return None;
    }
    let left_npu = kind_at(kinds, left).is_npu();
    let right_npu = kind_at(kinds, right).is_npu();
    match (left_npu, right_npu) {
        (true, true) => Some(left), // sandwich / both NPU: grow left (same id when equal)
        (true, false) => Some(left),
        (false, true) => Some(right),
        (false, false) => None,
    }
}

/// NPU-aware absorb gate for island → `target_backend`.
///
/// - Growing onto NPU: absorb when feasible (caller already checked mask).
/// - Dissolving an NPU island into non-NPU: refuse only when the island is
///   "fat" (≥ [`NPU_PRESERVE_MIN_OPS`] ops or ≥ [`NPU_PRESERVE_MIN_BYTES`]);
///   thin unitaries / 2-op scraps may dissolve so buft-safe coalesce duty-cycle
///   wins are not undone.
/// - Other sandwiches: absorb when estimated cut cost drops (or nbytes unknown).
pub fn decide_absorb(
    island_backend: i32,
    target_backend: i32,
    kinds: &[BackendKind],
    island_nbytes_sum: u64,
    island_op_count: usize,
    left_edge_nbytes: u32,
    right_edge_nbytes: u32,
    weighted: bool,
) -> AbsorbDecision {
    if !weighted {
        return AbsorbDecision::Absorb;
    }
    let island_k = kind_at(kinds, island_backend);
    let target_k = kind_at(kinds, target_backend);

    // Prefer fat NPU regions: keep multi-op / heavy islands off Vulkan/CPU.
    if island_k.is_npu() && !target_k.is_npu() {
        if is_fat_npu_island(island_op_count, island_nbytes_sum) {
            return AbsorbDecision::RefusePreserveNpu;
        }
        // Thin unitary / small 2-op NPU: allow classic dissolve into surround.
        return AbsorbDecision::Absorb;
    }

    // Absorbs onto NPU always welcome when feasible.
    if target_k.is_npu() {
        return AbsorbDecision::Absorb;
    }

    // Non-NPU sandwich: require estimated cost improvement when we have sizes.
    if island_nbytes_sum == 0 && left_edge_nbytes == 0 && right_edge_nbytes == 0 {
        return AbsorbDecision::Absorb; // fall back to length-only coalesce
    }
    let before = edge_cost(left_edge_nbytes, target_k, island_k)
        .saturating_add(edge_cost(right_edge_nbytes, island_k, target_k));
    // After absorb there is no cut at this sandwich.
    let after = 0u64;
    if after < before {
        AbsorbDecision::Absorb
    } else {
        AbsorbDecision::RefuseNoGain
    }
}

/// Record an absorb attempt for profile counters (call after decide + recolor).
pub fn note_absorb_outcome(
    decision: AbsorbDecision,
    island_backend: i32,
    target_backend: i32,
    kinds: &[BackendKind],
    recolored_ops: usize,
) {
    record_decision(decision, island_backend, target_backend, kinds, recolored_ops);
}

/// Effective max island length: boost when absorbing into NPU across heavy edges.
pub fn effective_max_island(
    base: usize,
    target_backend: i32,
    kinds: &[BackendKind],
    left_edge_nbytes: u32,
    right_edge_nbytes: u32,
    weighted: bool,
) -> usize {
    if !weighted {
        return base;
    }
    if !kind_at(kinds, target_backend).is_npu() {
        return base;
    }
    let edge = left_edge_nbytes.max(right_edge_nbytes);
    // ≥256 KiB boundary → allow up to 2× island; ≥1 MiB → 3× (capped).
    // Anti-unitary: modest +2 on light NPU growth edges. Larger light boosts
    // (2×) inflated pp16 disp toward the preserve-all failure mode (~412).
    if edge >= 1 << 20 {
        base.saturating_mul(3).min(base.saturating_add(12))
    } else if edge >= 1 << 18 {
        base.saturating_mul(2).min(base.saturating_add(8))
    } else {
        base.saturating_add(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npu_gpu_factor_exceeds_default() {
        assert!(handoff_factor(BackendKind::Npu, BackendKind::Gpu) > 16);
        assert_eq!(handoff_factor(BackendKind::Cpu, BackendKind::Cpu), 0);
    }

    #[test]
    fn refuse_dissolving_fat_npu_island() {
        let kinds = [BackendKind::Gpu, BackendKind::Npu];
        let d = decide_absorb(1, 0, &kinds, 128 * 1024, 1, 4096, 4096, true);
        assert_eq!(d, AbsorbDecision::RefusePreserveNpu);
        let multi = decide_absorb(1, 0, &kinds, 1024, 3, 4096, 4096, true);
        assert_eq!(multi, AbsorbDecision::RefusePreserveNpu);
    }

    #[test]
    fn allow_dissolving_tiny_npu_unitary() {
        let kinds = [BackendKind::Gpu, BackendKind::Npu];
        let d = decide_absorb(1, 0, &kinds, 4096, 1, 4096, 4096, true);
        assert_eq!(d, AbsorbDecision::Absorb);
    }

    #[test]
    fn allow_dissolving_thin_two_op_npu() {
        let kinds = [BackendKind::Gpu, BackendKind::Npu];
        // 2-op below byte floor dissolves (anti-unitary / dispatch hygiene).
        let d = decide_absorb(1, 0, &kinds, 8192, 2, 4096, 4096, true);
        assert_eq!(d, AbsorbDecision::Absorb);
    }

    #[test]
    fn absorb_into_npu() {
        let kinds = [BackendKind::Npu, BackendKind::Gpu];
        let d = decide_absorb(1, 0, &kinds, 1024, 1, 1024, 1024, true);
        assert_eq!(d, AbsorbDecision::Absorb);
    }

    #[test]
    fn npu_target_asymmetric_picks_npu_neighbor() {
        let kinds = [BackendKind::Npu, BackendKind::Gpu, BackendKind::Cpu];
        assert_eq!(npu_absorb_target(0, 2, &kinds, true), Some(0));
        assert_eq!(npu_absorb_target(2, 0, &kinds, true), Some(0));
        assert_eq!(npu_absorb_target(1, 2, &kinds, true), None);
        assert_eq!(npu_absorb_target(0, 2, &kinds, false), None);
    }

    #[test]
    fn unweighted_always_absorbs() {
        let kinds = [BackendKind::Gpu, BackendKind::Npu];
        assert_eq!(
            decide_absorb(1, 0, &kinds, 128 * 1024, 2, 4096, 4096, false),
            AbsorbDecision::Absorb
        );
    }

    #[test]
    fn total_cut_cost_counts_hops() {
        let kinds = [BackendKind::Npu, BackendKind::Gpu];
        let ids = [0i32, 1, 0];
        let nb = [100u32, 200, 300];
        let c = total_cut_cost(&ids, &nb, &kinds);
        let expect = edge_cost(200, BackendKind::Npu, BackendKind::Gpu)
            + edge_cost(300, BackendKind::Gpu, BackendKind::Npu);
        assert_eq!(c, expect);
    }

    #[test]
    fn npu_max_island_boost_on_heavy_edge() {
        let kinds = [BackendKind::Npu, BackendKind::Gpu];
        let boosted = effective_max_island(4, 0, &kinds, 1 << 20, 64, true);
        assert!(boosted > 4);
        assert_eq!(effective_max_island(4, 1, &kinds, 1 << 20, 64, true), 4);
        // Light NPU growth edge still gets +2 anti-unitary headroom.
        assert_eq!(effective_max_island(4, 0, &kinds, 64, 64, true), 6);
    }

    #[test]
    fn stats_track_absorb_and_dissolve() {
        reset_cut_cost_stats();
        let kinds = [BackendKind::Npu, BackendKind::Gpu];
        note_absorb_outcome(AbsorbDecision::Absorb, 1, 0, &kinds, 2);
        note_absorb_outcome(AbsorbDecision::Absorb, 0, 1, &kinds, 1);
        note_absorb_outcome(AbsorbDecision::RefusePreserveNpu, 0, 1, &kinds, 0);
        let s = snapshot_cut_cost_stats();
        assert_eq!(s.absorbed_into_npu, 2);
        assert_eq!(s.dissolved_npu, 1);
        assert_eq!(s.preserved_fat_npu, 1);
    }
}
