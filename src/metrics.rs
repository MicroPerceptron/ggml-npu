//! Minimal no-std runtime observability seam.

#![allow(dead_code)]

use core::sync::atomic::{AtomicU64, Ordering};

/// Process-wide NPU program-size attribution for scheduler handoff work (#5).
static NPU_PROGRAMS: AtomicU64 = AtomicU64::new(0);
static NPU_UNITARY_PROGRAMS: AtomicU64 = AtomicU64::new(0);
static NPU_PROGRAM_NODES: AtomicU64 = AtomicU64::new(0);

/// Record one executed NPU program (fused partition / subgraph) for handoff attribution.
pub fn record_npu_program(nodes: u64) {
    NPU_PROGRAMS.fetch_add(1, Ordering::Relaxed);
    NPU_PROGRAM_NODES.fetch_add(nodes, Ordering::Relaxed);
    if nodes <= 1 {
        NPU_UNITARY_PROGRAMS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Snapshot handoff attribution counters (programs, unitaries, total nodes).
pub fn npu_program_attribution() -> (u64, u64, u64) {
    (
        NPU_PROGRAMS.load(Ordering::Relaxed),
        NPU_UNITARY_PROGRAMS.load(Ordering::Relaxed),
        NPU_PROGRAM_NODES.load(Ordering::Relaxed),
    )
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DispatchMetrics {
    pub stage_in_ns: u64,
    pub submit_ns: u64,
    pub wait_ns: u64,
    pub copy_out_ns: u64,
    /// Device-native allocation bound without a payload copy.
    pub direct_bindings: u64,
    /// Shared allocation already understood by both backends.
    pub shared_bindings: u64,
    /// Foreign allocation imported into the consuming backend without a payload copy.
    pub imported_bindings: u64,
    /// Per-dispatch payload-copy occurrences. This is never counted as interop success.
    pub staged_occurrences: u64,
    pub staged_bytes: u64,
    pub bytes_in: u64,
    pub bytes_out: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricEvent {
    Dispatch(DispatchMetrics),
    CacheHit {
        cache: &'static str,
        bytes: u64,
    },
    CacheMiss {
        cache: &'static str,
    },
    CacheBypass {
        cache: &'static str,
        reason: &'static str,
        bytes: u64,
    },
    CacheCorruption {
        cache: &'static str,
    },
    CacheLatency {
        operation: &'static str,
        elapsed_ns: u64,
        bytes: u64,
    },
    CompressionRatio {
        uncompressed_bytes: u64,
        compressed_bytes: u64,
    },
    CacheBytes {
        cache: &'static str,
        current_bytes: u64,
        budget_bytes: u64,
    },
    Allocation {
        domain: &'static str,
        bytes: u64,
    },
    StagingAllocation {
        /// Process-local allocation identity used to avoid conflating occurrences with objects.
        allocation: u64,
        bytes: u64,
    },
    SharedAllocation {
        allocation: u64,
        bytes: u64,
        imported: bool,
    },
    MemoryState {
        resident_bytes: u64,
        resident_high_water: u64,
        accounted_high_water: u64,
        resident_limit: u64,
    },
    CacheEviction {
        cache: &'static str,
        reason: &'static str,
        bytes: u64,
    },
    Compiler {
        stage: &'static str,
        elapsed_ns: u64,
    },
}

pub trait MetricsSink: Sync {
    fn event(&self, event: MetricEvent);
}

pub struct NoopMetrics;

impl MetricsSink for NoopMetrics {
    #[inline(always)]
    fn event(&self, _event: MetricEvent) {}
}

pub static NOOP_METRICS: NoopMetrics = NoopMetrics;
