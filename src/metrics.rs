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
