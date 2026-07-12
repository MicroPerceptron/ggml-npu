//! Minimal no-std runtime observability seam.

#![allow(dead_code)]

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
