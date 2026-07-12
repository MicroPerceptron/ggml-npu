//! Minimal no-std runtime observability seam.

#![allow(dead_code)]

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DispatchMetrics {
    pub stage_in_ns: u64,
    pub submit_ns: u64,
    pub wait_ns: u64,
    pub copy_out_ns: u64,
    pub owned_bindings: u64,
    pub bounced_bindings: u64,
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
    Allocation {
        domain: &'static str,
        bytes: u64,
    },
    MemoryState {
        resident_bytes: u64,
        resident_high_water: u64,
        accounted_high_water: u64,
        resident_limit: u64,
    },
    DirectMemoryBreakdown {
        executable_bytes: u64,
        binding_bytes: u64,
        hpi_bytes: u64,
        command_bytes: u64,
        preemption_bytes: u64,
    },
    CacheEviction {
        cache: &'static str,
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
